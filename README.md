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