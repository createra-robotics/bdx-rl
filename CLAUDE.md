# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an **Isaac Lab** extension for training reinforcement learning (PPO via RSL-RL) to control the **Disney BD-X bipedal robot** for velocity-tracking locomotion over rough terrain. It runs on NVIDIA Isaac Sim.

## Commands

```bash
# Install the BDX extension (editable)
python -m pip install -e source/BDX

# Train the locomotion policy (headless)
cd scripts/rsl_rl
python train.py --task=bdx-velocity-v0 --headless

# Evaluate a trained checkpoint
python play.py --task=bdx-velocity-play-v0 --num_envs 100

# Test environment with random actions
python ../random_agent.py --task=bdx-velocity-v0 --headless

# List all registered BDX environments
python ../list_envs.py
```

Training CLI flags: `--num_envs`, `--max_iterations`, `--seed`, `--video`, `--resume`, `--load_run`, `--logger wandb|tensorboard|neptune`, `--distributed`.

## Architecture

```
source/BDX/
├── __init__.py              # Registers Gym envs + UI extensions
├── config/extension.toml    # Isaac Lab extension metadata, dependencies (isaaclab, isaaclab_tasks, etc.)
├── robots/
│   └── bdx.py               # BDX_CFG: ArticulationCfg (URDF, DelayedPDActuatorCfg with PD gains per joint)
├── tasks/
│   ├── __init__.py           # Auto-imports sub-packages via isaaclab_tasks.utils.import_packages
│   └── bdx_locomotion/
│       ├── __init__.py       # Registers gym.make() IDs: "bdx-velocity-v0" and "bdx-velocity-play-v0"
│       ├── bdx_env_cfg.py   # BdxrEnvCfg (train), BdxrEnvCfg_PLAY (eval): extends LocomotionVelocityRoughEnvCfg
│       │                     #   Defines CommandsCfg, ActionsCfg, ObservationsCfg (policy + critic groups),
│       │                     #   BDXRRewards, EventCfg (domain randomization), TerminationsCfg
│       ├── agents/
│       │   └── rsl_rl_ppo_cfg.py  # PPORunnerCfg: actor/critic [512,256,128] ELU, adaptive PPO
│       └── mdp/
│           ├── __init__.py   # Re-exports from isaaclab.envs.mdp + isaaclab_tasks...velocity.mdp + local rewards
│           ├── rewards.py    # Custom reward functions: bipedal_air_time_reward, foot_clearance_reward,
│           │                  #   foot_slip_penalty, joint_position_penalty, base_orientation_penalty, etc.
│           ├── curriculums.py # lin_vel_cmd_levels, ang_vel_cmd_levels adaptive curriculum
│           └── commands/
│               └── velocity_command.py  # UniformLevelVelocityCommandCfg (extends isaaclab's)
└── assets/BDX/
    ├── BDX.urdf             # Robot URDF model
    └── meshes/              # STL mesh files for all body parts
```

**Two registered Gym environments** (in `tasks/bdx_locomotion/__init__.py`):
- `bdx-velocity-v0` → `BdxrEnvCfg` (full training with domain randomization, noise, terrain curriculum)
- `bdx-velocity-play-v0` → `BdxrEnvCfg_PLAY` (simplified eval: smaller terrain, no noise, fixed forward commands)

**Key design patterns:**
- The task is a **ManagerBasedRLEnv** using Isaac Lab's manager system (command manager, reward manager, observation manager, event manager, termination manager).
- `BdxrEnvCfg` extends `LocomotionVelocityRoughEnvCfg` and overrides `__post_init__` to swap in the BD-X robot, configure IMU, terrain, randomization ranges, and reward scales.
- The MDP module imports all standard Isaac Lab MDP terms (`isaaclab.envs.mdp.*` and `isaaclab_tasks.manager_based.locomotion.velocity.mdp.*`) and adds custom reward functions for bipedal locomotion.
- The robot uses per-joint PD gains and effort/velocity limits defined in `DelayedPDActuatorCfg` (5 DOFs per leg × 2 legs: Hip_Yaw, Hip_Roll, Hip_Pitch, Knee, Ankle).

## Code Style

- Python >= 3.10, Isaac Sim 4.5.0
- **Pre-commit**: black (line-length 120), flake8 (complexity ≤ 30, google docstrings), isort (black profile), pyupgrade (py310+), codespell
- License header insertion via pre-commit (BSD-3-Clause for most files)
- Configuration classes use `@configclass` decorator from `isaaclab.utils`
- Flake8 per-file ignores for `__init__.py`: `F401` (unused imports are intentional for registration)
