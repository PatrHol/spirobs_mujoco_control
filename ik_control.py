#!/usr/bin/env python3
"""
Интерактивное управление кончиком Spirobs в плоскости ZY.

Возможности:
  1. Автоматически определяет структуру модели (суставы, кончик, база).
  2. Строит рабочую зону — диапазон достижимых Y и Z (случайная выборка).
  3. Показывает пример достижимой точки.
  4. Принимает целевую точку от пользователя (интерактивно или через --target).
  5. Решает IK. Если точка недостижима — сообщает и подсказывает ближайшую.
  6. Открывает viewer и удерживает робота в достигнутой позе.

Использование:
    python ik_control.py                       # интерактивный ввод Y и Z
    python ik_control.py --target 0.15 0.70    # задать сразу
    python ik_control.py --xml path/to/robot.xml
"""

import argparse
import glob
import sys
import time
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer


# ============================================================
# ПОИСК XML
# ============================================================
def find_latest_xml(search_dir: Path) -> Path | None:
    """Ищет последний сгенерированный robot.xml в exports/xml_*/."""
    pattern = str(search_dir / "xml_*" / "robot.xml")
    files = sorted(glob.glob(pattern))
    return Path(files[-1]) if files else None


# ============================================================
# КЛАСС УПРАВЛЕНИЯ
# ============================================================
class SpirobsController:
    """Обёртка над MuJoCo-моделью с FK и IK."""

    def __init__(self, xml_path: Path):
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.xml_path = xml_path

        # --- 1. Индексы hinge-суставов ---
        self.joint_ids = [
            jid for jid in range(self.model.njnt)
            if self.model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_HINGE
        ]
        self.n_joints = len(self.joint_ids)
        if self.n_joints == 0:
            raise RuntimeError("В модели нет hinge-суставов.")

        # --- 2. Плоскость движения (ось вращения) ---
        axes = np.array([self.model.jnt_axis[jid] for jid in self.joint_ids])
        self.motion_axis = int(np.argmax(np.mean(np.abs(axes), axis=0)))

        # --- 3. Адреса в qpos ---
        self.qpos_adr = np.array([self.model.jnt_qposadr[jid] for jid in self.joint_ids])

        # --- 4. Ограничения углов ---
        self.q_min = np.zeros(self.n_joints)
        self.q_max = np.zeros(self.n_joints)
        for i, jid in enumerate(self.joint_ids):
            if self.model.jnt_limited[jid]:
                self.q_min[i] = self.model.jnt_range[jid][0]
                self.q_max[i] = self.model.jnt_range[jid][1]
            else:
                self.q_min[i], self.q_max[i] = -np.pi, np.pi

        # --- 5. Кончик ---
        has_child = np.zeros(self.model.nbody, dtype=bool)
        for b in range(1, self.model.nbody):
            has_child[self.model.body_parentid[b]] = True
        leaves = [b for b in range(1, self.model.nbody) if not has_child[b]]
        self.tip_body_id = leaves[-1] if leaves else self.model.nbody - 1

        # --- 6. База ---
        self.base_pos = self.model.body_pos[1].copy() if self.model.nbody > 1 else np.zeros(3)

        # --- 7. Текущее состояние ---
        self.q = np.zeros(self.n_joints)

        # --- 8. Печатаем структуру ---
        axis_names = ["X", "Y", "Z"]
        plane = "".join(a for i, a in enumerate(axis_names) if i != self.motion_axis)
        print("=" * 60)
        print(f"Модель:        {xml_path.name}")
        print(f"Суставов:      {self.n_joints}")
        print(f"Ось вращения:  {axis_names[self.motion_axis]} (плоскость {plane})")
        print(f"Ограничения:   [{self.q_min.min():.3f}, {self.q_max.max():.3f}] рад "
              f"({np.degrees(self.q_min.min()):.1f}°..{np.degrees(self.q_max.max()):.1f}°)")
        print(f"База:          z = {self.base_pos[2]:.3f} м")
        print("=" * 60)

    # --------------------------------------------------------
    def forward_kinematics(self, q):
        """По углам q возвращает (y, z) кончика."""
        self.data.qpos[self.qpos_adr] = q
        mujoco.mj_kinematics(self.model, self.data)
        p = self.data.xpos[self.tip_body_id]
        if self.motion_axis == 0:
            return p[1], p[2]
        elif self.motion_axis == 1:
            return p[0], p[2]
        else:
            return p[0], p[1]

    # --------------------------------------------------------
    def compute_jacobian(self, q, eps=1e-6):
        """Численный Якобиан 2×n."""
        J = np.zeros((2, self.n_joints))
        y0, z0 = self.forward_kinematics(q)
        for j in range(self.n_joints):
            q_eps = q.copy()
            q_eps[j] += eps
            y1, z1 = self.forward_kinematics(q_eps)
            J[0, j] = (y1 - y0) / eps
            J[1, j] = (z1 - z0) / eps
        return J

    # --------------------------------------------------------
    def inverse_kinematics(self, target_y, target_z,
                           q_init=None, max_iter=300, tol=1e-4, damping=1e-3):
        """Решает IK. Возвращает (q, err_norm)."""
        q = (q_init if q_init is not None else self.q).copy()

        for _ in range(max_iter):
            y, z = self.forward_kinematics(q)
            error = np.array([target_y - y, target_z - z])
            err_norm = np.linalg.norm(error)
            if err_norm < tol:
                return q, err_norm

            J = self.compute_jacobian(q)
            JJT = J @ J.T + damping * np.eye(2)
            dq = J.T @ np.linalg.solve(JJT, error)

            step_norm = np.linalg.norm(dq)
            max_step = 0.1
            if step_norm > max_step:
                dq *= max_step / step_norm

            q = np.clip(q + dq, self.q_min, self.q_max)

        y, z = self.forward_kinematics(q)
        return q, np.linalg.norm([target_y - y, target_z - z])

    # --------------------------------------------------------
    def set_joint_angles(self, q):
        """Записывает углы и обновляет кинематику."""
        self.data.qpos[self.qpos_adr] = q
        mujoco.mj_forward(self.model, self.data)
        self.q = q.copy()

    # --------------------------------------------------------
    def sample_workspace(self, n_samples=3000):
        """
        Приблизительная рабочая зона: генерируем n_samples случайных
        конфигураций в пределах q_min..q_max и считаем положение кончика.
        Возвращает (ys, zs).
        """
        ys = np.empty(n_samples)
        zs = np.empty(n_samples)
        for i in range(n_samples):
            q = np.random.uniform(self.q_min, self.q_max)
            ys[i], zs[i] = self.forward_kinematics(q)
        return ys, zs


# ============================================================
# ВЗАИМОДЕЙСТВИЕ С ПОЛЬЗОВАТЕЛЕМ
# ============================================================
def prompt_target(default_y, default_z, y_min, y_max, z_min, z_max):
    """Спрашивает у пользователя целевую точку Y, Z."""
    print()
    print("=" * 60)
    print("ЗАДАНИЕ ЦЕЛЕВОЙ ТОЧКИ")
    print("=" * 60)
    print(f"Рабочая зона:  Y ∈ [{y_min:.3f}, {y_max:.3f}] м")
    print(f"               Z ∈ [{z_min:.3f}, {z_max:.3f}] м")
    print(f"Пример точки:  Y = {default_y:.3f}, Z = {default_z:.3f}")
    print("(Нажмите Enter, чтобы принять значение по умолчанию)")
    print()

    while True:
        try:
            s = input(f"Y (м) [{default_y:.3f}]: ").strip()
            y = float(s) if s else default_y
        except ValueError:
            print("  Ошибка: нужно число. Попробуйте снова.")
            continue

        try:
            s = input(f"Z (м) [{default_z:.3f}]: ").strip()
            z = float(s) if s else default_z
        except ValueError:
            print("  Ошибка: нужно число. Попробуйте снова.")
            continue

        return y, z


# ============================================================
# ОСНОВНАЯ ПРОГРАММА
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(description="Управление Spirobs в плоскости ZY")
    p.add_argument("--xml", default=None, help="Путь к XML (иначе — последний в exports/)")
    p.add_argument("--target", "-t", nargs=2, type=float, default=None,
                   metavar=("Y", "Z"), help="Целевая точка (без интерактива)")
    p.add_argument("--samples", type=int, default=3000,
                   help="Число сэмплов для оценки рабочей зоны")
    p.add_argument("--tol-mm", type=float, default=5.0,
                   help="Порог достижимости (мм). Ошибка IK выше — считаем недостижимой.")
    return p.parse_args()


def main():
    args = parse_args()

    # --- 1. Находим XML ---
    if args.xml:
        xml_path = Path(args.xml).expanduser().resolve()
        if not xml_path.exists():
            print(f"Файл не найден: {xml_path}")
            sys.exit(1)
    else:
        candidates = [
            Path.home() / "dev/Open-Spiral-Robots-v1/design-tool/exports",
            Path.cwd() / "exports",
        ]
        exports_dir = next((c for c in candidates if c.exists()), None)
        if exports_dir is None:
            print("Не найдена папка exports/. Укажите XML через --xml.")
            sys.exit(1)
        xml_path = find_latest_xml(exports_dir)
        if xml_path is None:
            print(f"В {exports_dir} нет xml_*/robot.xml")
            sys.exit(1)
        print(f"Найден XML: {xml_path}\n")

    # --- 2. Загружаем контроллер ---
    ctrl = SpirobsController(xml_path)

    # --- 3. Считаем рабочую зону ---
    print(f"\nОцениваю рабочую зону ({args.samples} сэмплов)...")
    t0 = time.time()
    ys, zs = ctrl.sample_workspace(args.samples)
    y_min, y_max = ys.min(), ys.max()
    z_min, z_max = zs.min(), zs.max()
    print(f"  Готово за {time.time() - t0:.2f} с")

    # Пример достижимой точки — конфигурация "все суставы в нуле" (прямо вверх)
    example_y, example_z = ctrl.forward_kinematics(np.zeros(ctrl.n_joints))

    # --- 4. Получаем целевую точку ---
    if args.target is not None:
        target_y, target_z = args.target
        print(f"\nЦель из аргументов: Y={target_y:.3f}, Z={target_z:.3f}")
    else:
        target_y, target_z = prompt_target(
            example_y, example_z, y_min, y_max, z_min, z_max
        )
        print(f"\nЦель: Y={target_y:.3f}, Z={target_z:.3f}")

    # --- 5. Быстрая проверка попадания в bounding box ---
    margin = 0.02  # 2 см допуска на границы
    inside_box = (
        y_min - margin <= target_y <= y_max + margin and
        z_min - margin <= target_z <= z_max + margin
    )
    if not inside_box:
        print("\n" + "!" * 60)
        print("ТОЧКА НЕДОСТИЖИМА: выходит за пределы рабочей зоны.")
        print(f"  Ваша точка:  Y={target_y:.3f}, Z={target_z:.3f}")
        print(f"  Возможный Y: [{y_min:.3f}, {y_max:.3f}]")
        print(f"  Возможный Z: [{z_min:.3f}, {z_max:.3f}]")
        print("!" * 60)
        sys.exit(0)

    # --- 6. Решаем IK ---
    print("\nРешаю обратную кинематику...")
    t0 = time.time()
    q_sol, err = ctrl.inverse_kinematics(target_y, target_z)
    dt = time.time() - t0
    err_mm = err * 1000.0
    y_actual, z_actual = ctrl.forward_kinematics(q_sol)

    # --- 7. Проверяем достижимость ---
    if err_mm > args.tol_mm:
        print("\n" + "!" * 60)
        print(f"ТОЧКА НЕДОСТИЖИМА (ошибка {err_mm:.1f} мм > порога {args.tol_mm:.1f} мм).")
        print(f"  Запрошено:       Y={target_y:.3f}, Z={target_z:.3f}")
        print(f"  Ближайшее:       Y={y_actual:.3f}, Z={z_actual:.3f}")
        print(f"  Ошибка:          {err_mm:.1f} мм")
        print("  Попробуйте точку внутри рабочей зоны.")
        print("!" * 60)
        sys.exit(0)

    # --- 8. Успех ---
    print(f"  Сошлось за {dt:.2f} с, ошибка = {err_mm:.2f} мм")
    print(f"  Достигнуто: Y={y_actual:.4f}, Z={z_actual:.4f}")
    print(f"  (цель была: Y={target_y:.4f}, Z={target_z:.4f})")

    ctrl.set_joint_angles(q_sol)

    # --- 9. Viewer ---
    print("\nЗапуск viewer (Esc — выход)...")
    with mujoco.viewer.launch_passive(ctrl.model, ctrl.data) as viewer:
        # Рисуем красную сферу — целевую точку
        scn = viewer.user_scn
        scn.ngeom = 0
        if ctrl.motion_axis == 0:
            pos = np.array([0.0, target_y, target_z])
        elif ctrl.motion_axis == 1:
            pos = np.array([target_y, 0.0, target_z])
        else:
            pos = np.array([target_y, target_z, 0.0])
        mujoco.mjv_initGeom(
            scn.geoms[0],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([0.02, 0.02, 0.02]),
            pos,
            np.eye(3).flatten(),
            np.array([1.0, 0.0, 0.0, 0.5]),  # красный, полупрозрачный
        )
        scn.ngeom = 1

        # Держим позу: перезаписываем qpos каждый кадр (иначе stiffness вернёт к нулю)
        while viewer.is_running():
            ctrl.data.qpos[ctrl.qpos_adr] = q_sol
            ctrl.data.qvel[:] = 0
            mujoco.mj_forward(ctrl.model, ctrl.data)
            viewer.sync()
            time.sleep(1.0 / 240.0)


if __name__ == "__main__":
    main()
