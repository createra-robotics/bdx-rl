# Repository Guidelines

## Project Structure & Module Organization

This repository is an Isaac Lab extension for training BD-X biped locomotion policies with RSL-RL. Core extension code lives in `source/BDX/`: robot configuration is in `robots/`, task registration and environment configuration are in `tasks/bdx_locomotion/`, and custom MDP terms are in `tasks/bdx_locomotion/mdp/`. Robot assets are stored under `source/BDX/assets/BDX/`, including `BDX.urdf` and STL meshes. Training and utility entry points are in `scripts/`, especially `scripts/rsl_rl/train.py`, `scripts/rsl_rl/play.py`, `scripts/random_agent.py`, and `scripts/list_envs.py`. MuJoCo/Isaac comparison utilities and XML assets are in `sim2sim/`.

## Build, Test, and Development Commands

- `python -m pip install -e source/BDX`: install the BDX extension in editable mode.
- `cd scripts/rsl_rl && python train.py --task=bdx-velocity-v0 --headless`: start headless PPO training.
- `cd scripts/rsl_rl && python play.py --task=bdx-velocity-play-v0 --num_envs 100`: evaluate a trained policy.
- `python scripts/random_agent.py --task=bdx-velocity-v0 --headless`: smoke-test the environment with random actions.
- `python scripts/list_envs.py`: list registered BDX Gym environments.
- `pre-commit run --all-files`: run formatting and lint checks before submitting changes.

## Coding Style & Naming Conventions

Use Python 3.10+ and keep code compatible with Isaac Sim 4.5.0. Formatting is handled by Black with 120-character lines, imports by isort using the Black profile, and linting by flake8. Flake8 uses Google-style docstrings, ignores unused imports in `__init__.py` for registration side effects, and limits complexity to 30. Prefer Isaac Lab `@configclass` configuration patterns. Name environment IDs with the existing `bdx-...-v0` pattern and keep task-specific modules under `tasks/bdx_locomotion/`.

## Testing Guidelines

There is no dedicated test suite in the current tree. For behavior changes, run the smallest relevant smoke test first, then a short training or play run when Isaac Sim resources are available. Add future tests under a conventional `tests/` directory and name them `test_*.py`.

## Commit & Pull Request Guidelines

Recent commits use short, direct subjects such as `add sym loss` and `Update README.md`; keep messages concise and imperative where possible. Pull requests should describe the changed behavior, list validation commands run, link related issues or experiments, and include screenshots or videos when UI, simulation visuals, or robot motion changes are relevant.

## Security & Configuration Tips

Do not commit private keys, credentials, large generated logs, or checkpoint artifacts. Keep training outputs under ignored log directories, and prefer Git LFS or external storage for large meshes, recordings, and model checkpoints.
