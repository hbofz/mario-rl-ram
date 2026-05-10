# Mario RL — RAM & CNN Agents for Super Mario Bros

This project trains reinforcement learning agents to play **Super Mario Bros (World 1-1)**
using two observation strategies: raw **RAM** (2048 bytes) and **pixel frames** (84×84 grayscale).
Both agents use [PPO](https://arxiv.org/abs/1707.06347) via Stable-Baselines3 and train on
Google Colab.

## Why Two Observation Modes?

| | RAM Agent | CNN Agent |
|---|---|---|
| **Observation** | 2048-byte RAM vector → float32 | 84×84 grayscale, 4-frame stack → uint8 |
| **Policy** | MLP (256×256) | NatureCNN |
| **Speed** | Very fast — no image ops | Slower — pixel preprocessing per step |
| **Training device** | CPU (no CNN workload) | GPU (benefits from CUDA) |
| **Insight** | Direct game-state access | Learns from visual appearance like humans |

## Setup

Requires Python 3.11–3.12. Use [`uv`](https://github.com/astral-sh/uv) for local dev:

```bash
uv sync
source .venv/bin/activate
```

Import your legally obtained ROM:

```bash
python -m stable_retro.import roms/
```

Verify the environment:

```bash
mario-doctor
mario-smoke --state Level1-1 --obs-mode ram --steps 300
mario-smoke --state Level1-1 --obs-mode pixel --steps 300
```

## Level Design

Training is locked to **World 1-1** (`--state Level1-1 --flag-lock`). Episodes end when:

- Mario **dies** (single-life mode, on by default), or
- Mario **completes the level** — the `FlagLockEpisode` wrapper detects the level
  transition and terminates, then resets to the start of 1-1.

Available level states: `Level1-1`, `Level2-1`, `Level3-1`, `Level4-1`, `Level5-1`,
`Level6-1`, `Level7-1`, `Level8-1`.

## Action Space

Uses `--action-mode mario` — a curated 11-action discrete set (no START/SELECT noise).
See [docs/ACTIONS.md](docs/ACTIONS.md) for the full table.

## Reward Design

Uses `--reward-mode smart` — a dense reward with components for forward progress,
checkpoints, coins, score, time pressure, stall penalties, and death penalties.
See [docs/REWARD.md](docs/REWARD.md) for details.

## Local Smoke Training

```bash
# RAM
mario-train --obs-mode ram --state Level1-1 --flag-lock --timesteps 20000 --n-envs 2 --run-name local-smoke-ram

# CNN
mario-train --obs-mode pixel --state Level1-1 --flag-lock --timesteps 20000 --n-envs 2 --run-name local-smoke-cnn
```

## Colab Training

See [docs/COLAB.md](docs/COLAB.md). Quick reference:

```bash
# RAM model (10M steps, CPU)
mario-train --obs-mode ram --state Level1-1 --flag-lock \
  --timesteps 10000000 --n-envs 16 --run-name ram-1-1 --device cpu

# CNN model (5M steps, GPU)
mario-train --obs-mode pixel --state Level1-1 --flag-lock \
  --timesteps 5000000 --n-envs 8 --n-steps 128 --batch-size 512 \
  --run-name cnn-1-1 --device auto
```

## Evaluation

```bash
mario-eval --model models/ram-1-1/final_model.zip --obs-mode ram --state Level1-1 --episodes 5 --video-dir videos/ram
mario-eval --model models/cnn-1-1/final_model.zip --obs-mode pixel --state Level1-1 --episodes 5 --video-dir videos/cnn
```

## Project Structure

```
src/mario_rl/
├── __init__.py        # Package exports
├── env.py             # make_mario_env() — unified RAM/pixel factory
├── levels.py          # Per-level reward zone configs
├── wrappers.py        # All Gym wrappers (action, obs, reward, episode)
└── scripts/
    ├── train_ppo.py   # mario-train CLI
    ├── evaluate.py    # mario-eval CLI
    ├── smoke.py       # mario-smoke CLI
    └── doctor.py      # mario-doctor CLI
```

## PC Setup

See [docs/PC_SETUP.md](docs/PC_SETUP.md) for Windows/WSL2/Linux setup.
