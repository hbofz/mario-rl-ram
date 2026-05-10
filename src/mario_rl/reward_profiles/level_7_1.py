"""World 7-1 reward profile.

World 7-1 is a cannon and Hammer Bros. gauntlet.  Bullet Bills can be farmed
forever, so this profile rewards section clears, survival, coins, and real
score events while capping score/kill reward so the optimal policy is still to
finish the level.
"""

from __future__ import annotations


LEVEL7_1: dict[str, object] = {
    # Geometry: the full map image is 3328 px wide; the flag is near the final
    # 10% after the springboard and Buzzy Beetle staircase.
    "finish_zone_x": 2860.0,
    "flag_zone_x": 3040.0,
    # Movement: this level needs decisive forward motion, but a little less
    # raw progress reward than 1-1 so tactical jumps and coin pickups can win.
    "progress_scale": 0.17,
    "backtrack_scale": 0.04,
    "checkpoint_bonus": 2.5,
    "checkpoint_width": 128,
    # Section bonuses are one-time rewards for clearing the major hazard beats:
    # early cannon stack, first cannon gauntlet, Hammer Bros. bridge/pipe area,
    # late cannon/Hammer Bros. stretch, springboard, and final stairs.
    "zone_bonuses": (
        (520.0, 12.0),
        (1040.0, 18.0),
        (1500.0, 22.0),
        (2020.0, 24.0),
        (2460.0, 18.0),
        (2780.0, 24.0),
    ),
    # Events: reward coins and enemy clears, but cap farmable score/kill reward
    # because Bullet Bills spawn indefinitely from the cannons.
    "score_scale": 0.03,
    "coin_bonus": 2.5,
    "kill_bonus": 5.0,
    "max_score_reward": 170.0,
    "max_kill_reward": 70.0,
    # Finish: beating this map matters more than perfect collection.
    "finish_zone_bonus": 120.0,
    "flag_bonus": 150.0,
    # Survival and pacing: waiting is sometimes necessary around Hammer Bros.,
    # but long stalls near cannons should become expensive.
    "death_penalty": 100.0,
    "life_loss_penalty": 0.0,
    "level_bonus": 0.0,
    "time_penalty": 0.008,
    "stall_penalty": 0.025,
    "stall_window": 34,
    "max_stall_penalty": 0.65,
    # Action quality: keep jumps cheap because this map requires them, while
    # still discouraging panic hopping and aimless left movement.
    "jump_penalty": 0.006,
    "neutral_jump_penalty": 0.018,
    "repeated_jump_penalty": 0.02,
    "repeated_jump_window": 6,
    "max_repeated_jump_penalty": 0.3,
    "left_penalty": 0.012,
    "bad_button_penalty": 0.25,
}
