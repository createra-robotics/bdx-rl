#!/usr/bin/env python3
"""MuJoCo BD-X — PD ctrl. Base is fixed to world, neck/head locked, gains from Isaac."""

import math, os
import numpy as np
import mujoco, mujoco.viewer
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(SCRIPT_DIR, "../xmls/scene.xml")

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

# --- PD gains (from Isaac Lab BDX config) ---
kp_default = 80.0
kd_default = 10.0
kp_ankle = 40.0
kd_ankle = 2.0

# Build per-actuator kp / per-joint kd arrays
nu = model.nu
nv = model.nv

kp = np.full(nu, kp_default)
kd = np.zeros(nv)

for i in range(nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    if name and "Ankle" in name:
        kp[i] = kp_ankle

for j in range(nv):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
    if name is None:
        continue
    if "Ankle" in name:
        kd[j] = kd_ankle
    elif any(k in name for k in ("Hip_Yaw", "Hip_Roll", "Hip_Pitch", "Knee")):
        kd[j] = kd_default

model.dof_damping[:] = kd
model.actuator_gainprm[:, 0] = kp
model.actuator_biasprm[:, 1] = -kp

# --- print diagnostics ---
nq = model.nq
print(f"nq={nq}  nv={nv}  nu={nu}")
print(f"kp={kp}")
print(f"kd={kd}")

# --- helpers ---
act_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
             for i in range(model.nu)]
act_idx = {n: i for i, n in enumerate(act_names)}

sim_dt = 0.002
model.opt.timestep = sim_dt

mujoco.mj_forward(model, data)

step = 0
dt = model.opt.timestep

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.lookat[:] = [0, 0, 0.3]
    viewer.cam.distance = 1.8
    viewer.cam.azimuth = 30
    viewer.cam.elevation = -20

    while viewer.is_running():
        start_time = time.time()
        t = step * dt
        # sin_val = 0.2 * math.sin(2 * math.pi * 0.5 * t)

        # data.ctrl[:] = 0.0
        # data.ctrl[act_idx["Left_Hip_Yaw_servo"]] = sin_val
        # data.ctrl[act_idx["Right_Hip_Yaw_servo"]] = -sin_val
        # data.ctrl[act_idx["Left_Hip_Roll_servo"]] = sin_val * 0.5
        # data.ctrl[act_idx["Right_Hip_Roll_servo"]] = sin_val * 0.5

        mujoco.mj_step(model, data)

        step += 1
        viewer.sync()
        elapsed = time.time() - start_time
        if elapsed < sim_dt:
            time.sleep(sim_dt - elapsed)
