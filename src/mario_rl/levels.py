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
        "finish_zone_x": 3100.0,
        "flag_zone_x": 3300.0,
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
