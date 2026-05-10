# Colab Training Guide

## 1. Runtime

Use a GPU runtime. A100 or H100 is ideal. For RAM + MLP training the GPU helps
mostly during PPO updates; for CNN + pixel training it helps significantly more.

## 2. Clone The Repo

```bash
git clone https://github.com/hbofz/mario-rl-ram.git
cd mario-rl-ram
pip install -e .
```

Restart the runtime after installing if Colab already has conflicting packages loaded.

## 3. Import The ROM

```python
from google.colab import drive
drive.mount("/content/drive")
```

```bash
python -m stable_retro.import /content/drive/MyDrive/mario_roms/
```

## 4. Smoke Test

```bash
# RAM pipeline
mario-smoke --state Level1-1 --obs-mode ram --steps 300

# Pixel pipeline
mario-smoke --state Level1-1 --obs-mode pixel --steps 300
```

Both should print the observation space, action space, and `info` keys without errors.

## 5. Train — RAM Model (MLP Policy)

Trains a 2048-byte RAM observation agent with an MLP policy on **World 1-1 only**.
Episodes restart when Mario dies **or** completes the level (flag lock on by default).

```bash
mario-train \
  --obs-mode ram \
  --state Level1-1 \
  --flag-lock \
  --timesteps 10000000 \
  --n-envs 16 \
  --n-steps 512 \
  --batch-size 2048 \
  --reward-mode smart \
  --action-mode mario \
  --run-name ram-1-1 \
  --model-dir /content/drive/MyDrive/mario_rl/models \
  --log-dir /content/drive/MyDrive/mario_rl/runs \
  --device cpu
```

> RAM + MLP trains faster on CPU than GPU because there is no CNN workload.

## 6. Train — CNN Model (Pixel Policy)

Trains an 84×84 grayscale 4-frame-stack agent with NatureCNN on **World 1-1 only**.

```bash
mario-train \
  --obs-mode pixel \
  --state Level1-1 \
  --flag-lock \
  --timesteps 5000000 \
  --n-envs 8 \
  --n-steps 128 \
  --batch-size 512 \
  --reward-mode smart \
  --action-mode mario \
  --run-name cnn-1-1 \
  --model-dir /content/drive/MyDrive/mario_rl/models \
  --log-dir /content/drive/MyDrive/mario_rl/runs \
  --device auto
```

> CNN training benefits from GPU. Use `--device auto` to let SB3 pick CUDA.
> Fewer parallel envs (`--n-envs 8`) are used because pixel preprocessing is heavier.

## 7. Auto-Resume After Disconnects

Auto-resume is **on by default**. If Colab disconnects, just re-run the exact same
`mario-train` command — it scans `--model-dir` for the latest `.zip` and continues.

To resume from a specific checkpoint:

```bash
mario-train \
  --obs-mode ram \
  --state Level1-1 \
  --resume-from /content/drive/MyDrive/mario_rl/models/ram-1-1/ppo_5000000_steps.zip \
  --timesteps 5000000 \
  --n-envs 16 \
  --run-name ram-1-1 \
  --model-dir /content/drive/MyDrive/mario_rl/models \
  --log-dir /content/drive/MyDrive/mario_rl/runs \
  --device cpu
```

> **Important:** `--obs-mode` must match the checkpoint you are resuming from.

## 8. Evaluate & Make Videos

```bash
# RAM model
mario-eval \
  --model /content/drive/MyDrive/mario_rl/models/ram-1-1/final_model.zip \
  --obs-mode ram \
  --state Level1-1 \
  --episodes 5 \
  --video-dir /content/drive/MyDrive/mario_rl/videos/ram

# CNN model
mario-eval \
  --model /content/drive/MyDrive/mario_rl/models/cnn-1-1/final_model.zip \
  --obs-mode pixel \
  --state Level1-1 \
  --episodes 5 \
  --video-dir /content/drive/MyDrive/mario_rl/videos/cnn
```

For your submission, record videos from: random baseline, an early checkpoint,
a mid-training checkpoint, and the final model — for both RAM and CNN.

## 9. Recurrent Stretch (Optional)

After PPO works, try RecurrentPPO for temporal memory:

```bash
mario-train \
  --algo recurrent-ppo \
  --obs-mode ram \
  --state Level1-1 \
  --flag-lock \
  --timesteps 5000000 \
  --n-envs 8 \
  --run-name recurrent-ram-1-1 \
  --device cpu
```
