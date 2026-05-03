# Colab Training Guide

## 1. Runtime

Use a GPU runtime. A100 or H100 is ideal, but remember the emulator is still CPU-sensitive. The GPU helps more during PPO updates than during frame/RAM collection.

## 2. Clone The Repo

```bash
git clone https://github.com/hbofz/mario-rl-ram.git
cd mario-rl-ram
pip install -e .
```

The repo uses platform-specific dependencies: Colab/Linux installs upstream `stable-retro==1.0.0`; Apple Silicon Macs install `stable-retro-apple-silicon` for local smoke tests.

If Colab already has incompatible packages loaded, restart the runtime after installing.

## 3. Import The ROM

Upload your legally obtained ROM to Google Drive, for example:

```text
/content/drive/MyDrive/mario_roms/
```

Then:

```python
from google.colab import drive
drive.mount("/content/drive")
```

```bash
python -m stable_retro.import /content/drive/MyDrive/mario_roms/
```

## 4. Smoke Test

```bash
mario-smoke --steps 300
```

You should see the RAM observation shape, action space, reward total, and `info` keys.

## 5. Train PPO

Start with:

```bash
mario-train \
  --timesteps 5000000 \
  --n-envs 16 \
  --n-steps 512 \
  --batch-size 2048 \
  --reward-mode smart \
  --run-name ppo-ram-full-controller \
  --device auto
```

If CPU usage is high but GPU usage is low, that is normal for RAM PPO. Increase `--n-envs` until env stepping stops improving or Colab RAM gets tight.

## 6. Save To Drive

Use Drive paths when you want checkpoints to survive runtime resets:

```bash
mario-train \
  --timesteps 10000000 \
  --n-envs 16 \
  --reward-mode smart \
  --run-name ppo-ram-full-controller \
  --model-dir /content/drive/MyDrive/mario_rl/models \
  --log-dir /content/drive/MyDrive/mario_rl/runs
```

## 7. Recurrent Stretch Run

After PPO works:

```bash
mario-train \
  --algo recurrent-ppo \
  --timesteps 5000000 \
  --n-envs 8 \
  --n-steps 512 \
  --batch-size 1024 \
  --reward-mode smart \
  --run-name recurrent-ppo-ram-full-controller \
  --device auto
```

## 8. Make Videos

```bash
mario-eval \
  --model /content/drive/MyDrive/mario_rl/models/ppo-ram-full-controller/final_model.zip \
  --episodes 3 \
  --video-dir /content/drive/MyDrive/mario_rl/videos
```

For the presentation, save videos from multiple checkpoints: random, early training, middle training, and final.
