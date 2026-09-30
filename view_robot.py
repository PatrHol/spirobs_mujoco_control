#!/usr/bin/env python3

"""Просмотр Spirobs XML в MuJoCo."""
import sys
import time
import mujoco
import mujoco.viewer

xml_path = sys.argv[1] if len(sys.argv) > 1 else "robot.xml"
print(f"Загружаю: {xml_path}")

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

print(f"\n=== Модель ===")
print(f"  nbody    = {model.nbody}     (тел)")
print(f"  njnt     = {model.njnt}      (суставов)")
print(f"  nu       = {model.nu}        (актуаторов)")
print(f"  ntendon  = {model.ntendon}   (тросов)")
print(f"  nsite    = {model.nsite}     (site-точек)")

print("\n=== Суставы ===")
for i in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or f"joint_{i}"
    jtype = model.jnt_type[i]
    jtype_name = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}.get(jtype, "?")
    axis = model.jnt_axis[i]
    print(f"  [{i}] {name}  type={jtype_name}  axis={axis}")

print("\n=== Актуаторы ===")
for i in range(model.nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"act_{i}"
    print(f"  [{i}] {name}")

print("\n=== Тросы ===")
for i in range(model.ntendon):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_TENDON, i) or f"tendon_{i}"
    print(f"  [{i}] {name}")

print("\nЗапускаю viewer...")
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(1/240)
