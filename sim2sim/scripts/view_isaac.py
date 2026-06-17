#!/usr/bin/env python3
"""Isaac Sim BD-X — fixed base, ctrl=0, output q/target/err/tau for comparison.

Usage:
  python view_isaac.py
"""

from isaaclab.app import AppLauncher
import argparse
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import isaaclab_tasks  # noqa
import BDX.tasks  # noqa
from BDX.tasks.bdx_locomotion.bdx_env_cfg import BdxrEnvCfg

JOINT_NAMES = [
    "Left_Hip_Yaw", "Right_Hip_Yaw", "Left_Hip_Roll", "Right_Hip_Roll", "Left_Hip_Pitch",
    "Right_Hip_Pitch", "Left_Knee", "Right_Knee", "Left_Ankle", "Right_Ankle",
]


def main():
    env_cfg = BdxrEnvCfg()
    env_cfg.scene.robot.spawn.fix_base = True
    env_cfg.scene.robot.init_state.pos = (0.0, 0.0, 0.6)
    env_cfg.scene.num_envs = 1

    for attr in ["physics_material", "add_base_mass", "reset_base",
                 "reset_robot_joints", "push_robot", "base_external_force_torque"]:
        if hasattr(env_cfg.events, attr):
            setattr(env_cfg.events, attr, None)
    if hasattr(env_cfg.events, "randomize_gains"):
        env_cfg.events.randomize_gains = None
    if hasattr(env_cfg.events, "randomize_com"):
        env_cfg.events.randomize_com = None
    env_cfg.observations.policy.enable_corruption = False
    if hasattr(env_cfg.terminations, "time_out"):
        env_cfg.terminations.time_out = None

    env = gym.make("bdx-velocity-v0", cfg=env_cfg)
    action_scale = env_cfg.actions.joint_pos.scale  # 0.5

    # PD gains from DelayedPDActuatorCfg
    kp_hip = 80.0
    kp_ankle = 40.0
    kp_list = [kp_hip, kp_hip, kp_hip, kp_hip, kp_hip,
               kp_hip, kp_hip, kp_hip, kp_ankle, kp_ankle]

    print("=" * 70)
    print("  Isaac Sim BD-X — base fixed, ctrl=0 (all targets at default)")
    print(f"  base pos: (0,0,0.6)   fix_base: True   envs: 1")
    print("=" * 70)
    print(f"  {'Joint':<20s} {'q':>8s} {'target':>8s} {'err':>8s} {'tau_est':>8s}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    obs, _ = env.reset()
    step = 0
    dt = env.unwrapped.step_dt

    try:
        while simulation_app.is_running():
            # action=0 → target = default_joint_pos + 0*scale = default_joint_pos
            action = torch.zeros(1, 10, device=env.unwrapped.device)
            obs, _, _, _, _ = env.step(action)

            if step % 200 == 0:
                t = step * dt

                # Get current joint positions and target positions
                # In Isaac Lab, the action sets the target via:
                #   target = default_offset + action * scale
                # With action=0 and use_default_offset=True:
                #   target = default_joint_pos
                default_pos = env.unwrapped.scene["robot"].data.default_joint_pos[0, :10].cpu().numpy()
                joint_pos = env.unwrapped.scene["robot"].data.joint_pos[0, :10].cpu().numpy()
                joint_vel = env.unwrapped.scene["robot"].data.joint_vel[0, :10].cpu().numpy()

                # Estimate actuator torque:
                # Isaac Lab DelayedPDActuatorCfg: tau = kp*(target - q) + kv*(0 - qdot)
                kv_hip = 10.0
                kv_ankle = 2.0
                kv_list = [kv_hip, kv_hip, kv_hip, kv_hip, kv_hip,
                           kv_hip, kv_hip, kv_hip, kv_ankle, kv_ankle]

                print(f"\n  t={t:.2f}s")
                print(f"  {'Joint':<20s} {'q':>8s} {'target':>8s} {'err':>8s} {'tau_est':>8s}")
                print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
                for i, name in enumerate(JOINT_NAMES):
                    q = joint_pos[i]
                    target = default_pos[i]
                    err = target - q
                    tau = kp_list[i] * err - kv_list[i] * joint_vel[i]
                    print(f"  {name:<20s} {q:>8.4f} {target:>8.4f} {err:>8.4f} {tau:>8.4f}")

            step += 1

    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
