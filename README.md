# Mario RAM RL

This project trains reinforcement learning agents to play **Super Mario Bros** from emulator RAM instead of pixels.

The local repo is the control room: build wrappers, run smoke tests, inspect checkpoints, and make videos. Colab is the training rig: run PPO/Recurrent PPO with an A100/H100 and many parallel emulator workers.

## Why RAM?

RAM observations are much smaller than frame observations and expose game state more directly. That lets us use an MLP policy instead of a CNN, run faster experiments, and still keep the full NES controller action space.

## Setup

Use Python 3.12 or 3.11. The default `python3` on this Mac is currently too new for the RL stack, so prefer `uv`:

```bash
uv sync
source .venv/bin/activate
```

On Apple Silicon, this project installs `stable-retro-apple-silicon` locally because the upstream `stable-retro==1.0.0` macOS wheel currently ships an incompatible binary in this environment. Colab/Linux uses upstream `stable-retro==1.0.0`.

Stable-Retro does not bundle commercial ROMs. Put your legally obtained ROM in `roms/`, then import it:

```bash
python -m stable_retro.import roms/
```

Check that the environment works:

```bash
mario-doctor
mario-smoke --steps 300
```

## Local Training Smoke Test

Local training is just for plumbing checks:

```bash
mario-train --timesteps 20000 --n-envs 2 --run-name local-smoke
```

## Colab Training

See [docs/COLAB.md](docs/COLAB.md). The short version is:

```bash
git clone https://github.com/hbofz/mario-rl-ram.git
cd mario-rl-ram
pip install -e .
python -m stable_retro.import /content/drive/MyDrive/mario_roms/
mario-train --timesteps 10000000 --n-envs 16 --action-mode mario --run-name ppo-ram-mario-actions-v4 --device cpu
```

## PC Training

See [docs/PC_SETUP.md](docs/PC_SETUP.md) for Windows, WSL2 Ubuntu, and Linux setup. WSL2 is recommended on Windows.

## Evaluation

Render a trained checkpoint:

```bash
mario-eval --model models/ppo-ram-mario-actions-v4/final_model.zip --episodes 3 --video-dir videos
```

## Current Baseline

- Environment: `SuperMarioBros-Nes-v0`
- Observation: RAM
- Action space: curated Mario actions by default for training
- Reward mode: `smart` by default for training
- Episode mode: single-life episodes by default for training
- Algorithm: PPO with `MlpPolicy`
- Stretch: RecurrentPPO with `MlpLstmPolicy`

Action details are in [docs/ACTIONS.md](docs/ACTIONS.md), and reward details are in [docs/REWARD.md](docs/REWARD.md).
