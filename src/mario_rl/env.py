from __future__ import annotations

from pathlib import Path
from typing import Literal

import gymnasium as gym
import stable_retro as retro
from stable_baselines3.common.monitor import Monitor

from mario_rl.wrappers import (
    ActionRepeat,
    InfoRewardShaping,
    MarioActionSpace,
    RamFloat32,
    SingleLifeEpisode,
    SingleStageEpisode,
    StageScoreReward,
    SmartMarioReward,
    ValidateInitialStage,
)


ActionMode = Literal["all", "discrete", "multidiscrete", "mario", "mario-secrets"]
RewardMode = Literal["base", "shaped", "smart", "stage-score"]


def make_mario_env(
    game: str = "SuperMarioBros-Nes-v0",
    action_mode: ActionMode = "all",
    reward_mode: RewardMode = "base",
    action_repeat: int = 4,
    render_mode: str | None = None,
    record: str | bool = False,
    monitor: bool = True,
    single_life: bool = False,
    single_stage: bool = False,
    state: str | None = None,
    custom_integration_path: str | Path | None = None,
    expected_stage: tuple[int, int] | None = None,
) -> gym.Env:
    """Create a RAM-observation Mario environment."""

    inttype = _retro_inttype(custom_integration_path)
    if state is not None:
        state_file = state if state.endswith(".state") else f"{state}.state"
        if retro.data.get_file_path(game, state_file, inttype) is None:
            raise FileNotFoundError(
                f"Could not find savestate {state_file} for {game}. "
                "For World 5-2, create custom_integrations/SuperMarioBros-Nes-v0/Level5-2.state first."
            )
    env = retro.make(
        game=game,
        state=state or retro.State.DEFAULT,
        obs_type=retro.Observations.RAM,
        use_restricted_actions=_retro_action_mode(action_mode),
        render_mode=render_mode,
        record=record,
        inttype=inttype,
    )
    if action_mode in {"mario", "mario-secrets"}:
        env = MarioActionSpace(env, profile=action_mode)
    env = RamFloat32(env)
    env = ActionRepeat(env, repeat=action_repeat)

    if expected_stage is not None:
        env = ValidateInitialStage(env, expected_stage=expected_stage)
    if single_life:
        env = SingleLifeEpisode(env)
    if single_stage:
        env = SingleStageEpisode(env)

    if reward_mode == "shaped":
        env = InfoRewardShaping(env)
    elif reward_mode == "smart":
        env = SmartMarioReward(env)
    elif reward_mode == "stage-score":
        env = StageScoreReward(env)

    if monitor:
        env = Monitor(env)

    return env


def _retro_action_mode(action_mode: ActionMode):
    if action_mode in {"mario", "mario-secrets"}:
        return retro.Actions.ALL
    if action_mode == "all":
        return retro.Actions.ALL
    if action_mode == "discrete":
        return retro.Actions.DISCRETE
    if action_mode == "multidiscrete":
        return retro.Actions.MULTI_DISCRETE
    raise ValueError(f"Unknown action mode: {action_mode}")


def _retro_inttype(custom_integration_path: str | Path | None):
    if custom_integration_path is None:
        return retro.data.Integrations.DEFAULT
    path = Path(custom_integration_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Custom integration path does not exist: {path}")
    existing = getattr(retro.data.Integrations, "CUSTOM_PATHS", [])
    if str(path) not in existing:
        retro.data.Integrations.add_custom_path(str(path))
    return retro.data.Integrations.CUSTOM
