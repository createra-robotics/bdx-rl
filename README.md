# BD-X Locomotion Isaac Lab

---

## 🚀 Installation

To install the necessary packages for this project, after cloning the repo, run the following command:

```bash
python -m pip install -e source/BDX
```

```bash
cd scripts/rsl_rl/
```

```bash
python train.py --task=bdx-velocity-v0 --headless

#or continue with:
python train.py --task=bdx-velocity-v0 --headless --resume --load_run <datetime> --checkpoint model_<0~9>.pt
```

```bash
python play.py --task=bdx-velocity-play-v0 --num_envs 100 --video --video_length 1000
```

```bash
tensorboard --logdir=/home/ubuntu/dev/bdx-rl/scripts/rsl_rl/logs/rsl_rl/bdxr_rough
```

## Sim2Sim: Isaac Checkpoint to MuJoCo

Install the runtime dependency for ONNX policy inference:

```bash
python -m pip install onnxruntime
```

Export a trained RSL-RL checkpoint to ONNX. By default, the ONNX file is saved to `sim2sim/onnx/<checkpoint_name>.onnx`.

```bash
cd /home/ubuntu/dev/bdx-rl && python sim2sim/scripts/torch2onnx.py --checkpoint scripts/rsl_rl/logs/rsl_rl/bdxr_rough/2026-06-15_16-52-12/model_17300.pt
```

Run the exported ONNX policy in MuJoCo with the BDX XML model:

```bash
cd /home/ubuntu/dev/bdx-rl && python sim2sim/scripts/sim2sim_bdx.py
```

For robot-side ROS deployment guidance, see `sim2sim/ROBOT_DEPLOYMENT.md`.

The sim2sim runner loads defaults from `sim2sim/configs/bdx.yaml`. Use this file to tune the MuJoCo XML path, ONNX
path, velocity command, control gains, action scaling, simulation timestep, and foot/floor friction.

Useful YAML fields:

```yaml
paths:
  xml: xmls/scene.xml
  onnx: onnx/model_17300.onnx

command: [0.3, 0.0, 0.0]  # lin_vel_x, lin_vel_y, ang_vel_z

simulation:
  duration: 0.0      # 0.0 means run until the viewer is closed or Ctrl+C is pressed
  sim_dt: 0.005
  policy_rate: 50.0

contact:
  foot_friction: 0.8
  floor_friction: 0.8
```

Command-line arguments override the YAML config when provided:

```bash
# Run without the viewer for a short smoke test.
python sim2sim/scripts/sim2sim_bdx.py --no-viewer --duration 2.0 --print-every 0.5

# Change the velocity command: --cmd <lin_vel_x> <lin_vel_y> <ang_vel_z>
python sim2sim/scripts/sim2sim_bdx.py --cmd 0.5 0.0 0.2

# Override the policy path without editing YAML.
python sim2sim/scripts/sim2sim_bdx.py --onnx sim2sim/onnx/model_17300.onnx

# Sweep contact friction.
python sim2sim/scripts/sim2sim_bdx.py --foot-friction 1.1 --floor-friction 1.0
```
