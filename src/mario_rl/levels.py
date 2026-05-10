"""Level-specific configuration registry.

Each entry maps a Stable-Retro state name to keyword overrides for
:class:`~mario_rl.wrappers.SmartMarioReward`.  The most important values
are the approximate x-scroll positions where the finish zone and flagpole
begin for that level (used for one-time bonus rewards).

Only World 1-1 is the current training target.  Other entries are rough
estimates kept here for future multi-level work.
"""

from __future__ import annotations

# Approximate x-scroll positions (xscrollLo + 256*xscrollHi) for each level.
# Level 1-1 flagpole is around x~3313; finish zone slightly before that.
LEVEL_CONFIGS: dict[str, dict] = {
    "Level1-1": {
        # ── Geometry ─────────────────────────────────────────────────────────
        "finish_zone_x": 3100.0,
        "flag_zone_x": 3300.0,
        # ── Score & kill tuning ───────────────────────────────────────────────
        # score_scale doubled: Goomba stomp (100 pts) → +5 reward instead of +2.5,
        # making enemy kills competitive with forward progress.
        "score_scale": 0.05,
        # kill_bonus: flat +3 when score jumps ≥100 pts without a coin being
        # collected — nearly guarantees an enemy stomp/shell hit.
        "kill_bonus": 3.0,
        # coin_bonus reduced: coins are now +0.5 direct + score via score_scale,
        # keeping them valuable but no longer 2.4× more rewarding than a kill.
        "coin_bonus": 0.5,
        # ── Jump penalty relaxation ───────────────────────────────────────────
        # Stomping enemies requires jumping.  Ease the penalties so a stomp
        # clearly pays off (net +8 per Goomba kill vs old net +2.5).
        "jump_penalty": 0.01,           # was 0.03
        "neutral_jump_penalty": 0.04,   # was 0.08
        # ── Redundancy cleanup ────────────────────────────────────────────────
        # level_bonus fires at level transition which is already covered by the
        # 200-point finish zone bonuses; disable to avoid double-counting.
        "level_bonus": 0.0,
        # life_loss_penalty stacks with death_penalty in single-life mode (−75
        # total on death); drop the life component to keep signal clean.
        "life_loss_penalty": 0.0,
    },
    "Level1-1-99lives": {
        "finish_zone_x": 3100.0,
        "flag_zone_x": 3300.0,
    },
    "Level2-1": {
        "finish_zone_x": 3400.0,
        "flag_zone_x": 3580.0,
    },
    "Level3-1": {
        "finish_zone_x": 2500.0,
        "flag_zone_x": 2700.0,
    },
    "Level4-1": {
        "finish_zone_x": 2700.0,
        "flag_zone_x": 2900.0,
    },
    "Level5-1": {
        "finish_zone_x": 3000.0,
        "flag_zone_x": 3200.0,
    },
    "Level6-1": {
        "finish_zone_x": 2700.0,
        "flag_zone_x": 2900.0,
    },
    "Level7-1": {
        "finish_zone_x": 2800.0,
        "flag_zone_x": 3000.0,
    },
    "Level8-1": {
        "finish_zone_x": 3000.0,
        "flag_zone_x": 3200.0,
    },
}


def get_level_reward_kwargs(state: str) -> dict:
    """Return SmartMarioReward kwargs for *state*, falling back to defaults."""
    return dict(LEVEL_CONFIGS.get(state, {}))
