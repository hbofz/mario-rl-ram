# PC Setup Guide

This guide is for running training outside Colab on a Windows or Linux PC.

Recommended path on Windows: **WSL2 Ubuntu**. Native Windows may work, but RL/emulator packages are usually smoother under Linux.

## Option A: Windows With WSL2 Ubuntu

### 1. Install WSL2

Open PowerShell as Administrator:

```powershell
wsl --install -d Ubuntu
```

Restart if Windows asks, then open Ubuntu from the Start menu.

### 2. Install System Packages

Inside Ubuntu:

```bash
sudo apt update
sudo apt install -y git python3 python3-venv python3-pip build-essential ffmpeg
```

### 3. Clone The Repo

```bash
git clone https://github.com/hbofz/mario-rl-ram.git
cd mario-rl-ram
```

### 4. Create The Python Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

### 5. Import The ROM

Put your legally obtained `.nes` ROM in the repo's `roms/` folder.

If the ROM is in your Windows Downloads folder, copy it from WSL like this:

```bash
mkdir -p roms
cp /mnt/c/Users/<YOUR_WINDOWS_USERNAME>/Downloads/*.nes roms/
```

Then import it:

```bash
python -m stable_retro.import roms/
```

Check setup:

```bash
mario-doctor
mario-smoke --steps 300 --reward-mode smart --action-mode mario
```

## Option B: Native Windows

Native Windows is less tested for `stable-retro`, but this is the rough path.

Install:

- Git for Windows
- Python 3.10, 3.11, or 3.12

Then in PowerShell:

```powershell
git clone https://github.com/hbofz/mario-rl-ram.git
cd mario-rl-ram
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Import the ROM:

```powershell
mkdir roms
# Put your .nes file in roms/ first
python -m stable_retro.import roms/
```

If native Windows fails while installing or importing `stable-retro`, use WSL2.

## Overnight Training Command

Create or copy the 5-2 state first:

```bash
mario-smoke \
  --state Level5-2 \
  --custom-integration-path custom_integrations \
  --expected-stage 5-2 \
  --reward-mode stage-score \
  --action-mode mario-secrets \
  --single-stage
```

Then use this for the main World 5-2 run:

```bash
mario-train \
  --timesteps 20000000 \
  --model-dir models \
  --log-dir runs \
  --device cpu
```

If your PC struggles, reduce parallel environments:

```bash
--n-envs 8 --batch-size 1024
```

## Resume Training

Checkpoints save under:

```text
models/<run-name>/
```

Resume from the newest checkpoint:

```bash
mario-train \
  --timesteps 5000000 \
  --n-envs 16 \
  --n-steps 512 \
  --batch-size 2048 \
  --reward-mode smart \
  --action-mode mario \
  --action-repeat 4 \
  --run-name ppo-ram-mario-actions-v4 \
  --model-dir models \
  --log-dir runs \
  --resume-from models/ppo-ram-mario-actions-v4/ppo_10000000_steps.zip \
  --device cpu
```

The action mode must match the checkpoint. For `ppo-ram-mario-actions-v4`, use `--action-mode mario`.

## TensorBoard

```bash
tensorboard --logdir runs
```

Open the URL it prints, usually:

```text
http://localhost:6006
```

## Evaluate And Make Videos

Deterministic:

```bash
mario-eval \
  --model models/recurrent-ppo-ram-5-2-stage-score/final_model.zip \
  --vecnormalize models/recurrent-ppo-ram-5-2-stage-score/vecnormalize.pkl \
  --episodes 5 \
  --video-dir videos/final-deterministic \
  --device cpu
```

Stochastic, usually better for finding the strongest rollout:

```bash
mario-eval \
  --model models/recurrent-ppo-ram-5-2-stage-score/final_model.zip \
  --vecnormalize models/recurrent-ppo-ram-5-2-stage-score/vecnormalize.pkl \
  --episodes 30 \
  --video-dir videos/final-stochastic \
  --device cpu \
  --no-deterministic
```

## Quick Troubleshooting

- `Game not found`: run `python -m stable_retro.import roms/`.
- `Module not found`: activate the venv, then run `pip install -e .`.
- Training is slow: lower `--n-envs` to `8`.
- GPU is idle: expected for RAM + MLP PPO. Use CPU for this baseline.
- Videos overwrite: change `--video-dir` to a new folder.
