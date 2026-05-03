from __future__ import annotations

from typing import Literal

import gymnasium as gym
import stable_retro as retro
from stable_baselines3.common.monitor import Monitor

from mario_rl.wrappers import ActionRepeat, InfoRewardShaping, RamFloat32, SingleLifeEpisode, SmartMarioReward


ActionMode = Literal["all", "discrete", "multidiscrete"]
RewardMode = Literal["base", "shaped", "smart"]


def make_mario_env(
    game: str = "SuperMarioBros-Nes-v0",
    action_mode: ActionMode = "all",
    reward_mode: RewardMode = "base",
    action_repeat: int = 4,
    render_mode: str | None = None,
    record: str | bool = False,
    monitor: bool = True,
    single_life: bool = False,
) -> gym.Env:
    """Create a RAM-observation Mario environment."""

    env = retro.make(
        game=game,
        obs_type=retro.Observations.RAM,
        use_restricted_actions=_action_mode(action_mode),
        render_mode=render_mode,
        record=record,
    )
    env = RamFloat32(env)
    env = ActionRepeat(env, repeat=action_repeat)

    if reward_mode == "shaped":
        env = InfoRewardShaping(env)
    elif reward_mode == "smart":
        env = SmartMarioReward(env)

    if single_life:
        env = SingleLifeEpisode(env)

    if monitor:
        env = Monitor(env)

    return env


def _action_mode(action_mode: ActionMode):
    if action_mode == "all":
        return retro.Actions.ALL
    if action_mode == "discrete":
        return retro.Actions.DISCRETE
    if action_mode == "multidiscrete":
        return retro.Actions.MULTI_DISCRETE
    raise ValueError(f"Unknown action mode: {action_mode}")
