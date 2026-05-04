# Reward Design

The Stable-Retro Mario integration has a very small default reward:

```json
{
  "reward": {
    "variables": {
      "xscrollLo": {
        "reward": 1
      }
    }
  }
}
```

That teaches "move right", but it does not directly value safer play, coins, score, or new territory beyond the low scroll byte.

This project therefore trains with `--reward-mode smart` by default.

## Smart Reward Components

Each agent decision is repeated for 4 emulator frames by default, then the smart reward is computed from Stable-Retro `info` variables.

| Component | Purpose |
|---|---|
| `progress` | Rewards forward movement using `xscrollLo + 256 * xscrollHi`; lightly penalizes moving backward. |
| `checkpoint` | Gives a bonus when Mario reaches each new 128-pixel progress band. |
| `score` | Rewards score increases from enemies, blocks, powerups, and flag scoring. |
| `coin` | Rewards coin collection, including coin counter wraparound. |
| `level` | Rewards a level variable change when the integration exposes one. |
| `life` | Penalizes losing lives. |
| `death` | Adds a larger penalty when an episode terminates after all lives are gone. |
| `time` | Adds a small per-decision time cost so standing still is not free. |
| `stall` | Adds increasing pressure when progress has been flat/backward for several decisions. |

The wrapper also adds an `info["smart_reward"]` dictionary so we can inspect component-level rewards during debugging.

Training uses single-life episodes by default. When Mario loses a life, the episode ends immediately, which gives PPO cleaner feedback than waiting until every life is gone.

The recommended `mario` action mode also adds a small action-quality term. It lightly penalizes jump spam, neutral jumps, left movement, and impossible/bad buttons. Forward progress is still worth much more than the action penalty, so useful jumps remain profitable.

## Modes

```bash
--reward-mode base
```

Use only the Stable-Retro scenario reward.

```bash
--reward-mode shaped
```

Use the Stable-Retro reward plus light full-x progress shaping.

```bash
--reward-mode smart
```

Use the project reward above. This is the default for training.
