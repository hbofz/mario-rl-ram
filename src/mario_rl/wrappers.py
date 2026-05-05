from __future__ import annotations

from collections.abc import Hashable
from typing import Any, Literal

import gymnasium as gym
import numpy as np
from gymnasium import spaces


MARIO_ACTIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("NOOP", ()),
    ("RIGHT", ("RIGHT",)),
    ("RIGHT_RUN", ("RIGHT", "B")),
    ("RIGHT_JUMP", ("RIGHT", "A")),
    ("RIGHT_RUN_JUMP", ("RIGHT", "B", "A")),
    ("JUMP", ("A",)),
    ("RUN_JUMP", ("B", "A")),
    ("LEFT", ("LEFT",)),
    ("LEFT_JUMP", ("LEFT", "A")),
    ("DOWN", ("DOWN",)),
    ("RIGHT_DOWN", ("RIGHT", "DOWN")),
)

MARIO_SECRET_ACTIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    *MARIO_ACTIONS,
    ("UP", ("UP",)),
    ("RIGHT_UP", ("RIGHT", "UP")),
    ("LEFT_UP", ("LEFT", "UP")),
)

MarioActionProfile = Literal["mario", "mario-secrets"]


class MarioActionSpace(gym.Wrapper):
    """Curated discrete Mario actions over the full NES button vector."""

    def __init__(self, env: gym.Env, profile: MarioActionProfile = "mario"):
        super().__init__(env)
        buttons = getattr(env.unwrapped, "buttons", None)
        if not buttons:
            raise ValueError("MarioActionSpace requires an environment with NES button metadata.")
        if profile == "mario":
            actions = MARIO_ACTIONS
        elif profile == "mario-secrets":
            actions = MARIO_SECRET_ACTIONS
        else:
            raise ValueError(f"Unknown Mario action profile: {profile}")
        self.button_names = list(buttons)
        self._button_index = {button: index for index, button in enumerate(self.button_names) if button}
        self._actions = tuple((name, self._button_vector(buttons)) for name, buttons in actions)
        self.action_space = spaces.Discrete(len(self._actions))

    @property
    def action_meanings(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self._actions)

    def step(self, action: Any):
        action_index = int(action)
        name, vector = self._actions[action_index]
        obs, reward, terminated, truncated, info = self.env.step(vector)
        info = dict(info)
        info["action_index"] = action_index
        info["action_name"] = name
        info["buttons_pressed"] = tuple(self.button_names[i] for i, pressed in enumerate(vector) if pressed)
        return obs, reward, terminated, truncated, info

    def _button_vector(self, buttons: tuple[str, ...]) -> np.ndarray:
        vector = np.zeros(len(self.button_names), dtype=np.int8)
        for button in buttons:
            vector[self._button_index[button]] = 1
        return vector


class RamFloat32(gym.ObservationWrapper):
    """Scale uint8 RAM observations to float32 in [0, 1]."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=env.observation_space.shape,
            dtype=np.float32,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return np.asarray(observation, dtype=np.float32) / 255.0


class ActionRepeat(gym.Wrapper):
    """Repeat a full controller action for a few emulator frames."""

    def __init__(self, env: gym.Env, repeat: int = 4):
        super().__init__(env)
        if repeat < 1:
            raise ValueError("repeat must be >= 1")
        self.repeat = repeat

    def step(self, action: Any):
        total_reward = 0.0
        last_obs = None
        last_info: dict[str, Any] = {}
        terminated = False
        truncated = False

        for _ in range(self.repeat):
            last_obs, reward, terminated, truncated, last_info = self.env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break

        return last_obs, total_reward, terminated, truncated, last_info


class SingleLifeEpisode(gym.Wrapper):
    """End an episode immediately when Mario loses a life."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._last_lives: int | None = None

    def reset(self, **kwargs):
        self._last_lives = None
        return self.env.reset(**kwargs)

    def step(self, action: Any):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if "lives" in info:
            lives = int(info["lives"])
            if self._last_lives is not None and lives < self._last_lives:
                terminated = True
                info = dict(info)
                info["single_life_done"] = True
            self._last_lives = lives
        return obs, reward, terminated, truncated, info


class SingleStageEpisode(gym.Wrapper):
    """End an episode when the level variables move away from the reset stage."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._start_stage: tuple[int, int] | None = None

    def reset(self, **kwargs):
        self._start_stage = None
        return self.env.reset(**kwargs)

    def step(self, action: Any):
        obs, reward, terminated, truncated, info = self.env.step(action)
        stage = self._stage_from_info(info)
        if stage is not None:
            if self._start_stage is None:
                self._start_stage = stage
            elif stage != self._start_stage:
                terminated = True
                info = dict(info)
                info["stage_clear_done"] = True
                info["start_stage"] = self._start_stage
                info["final_stage"] = stage
        return obs, reward, terminated, truncated, info

    @staticmethod
    def _stage_from_info(info: dict[str, Any]) -> tuple[int, int] | None:
        if "levelHi" not in info or "levelLo" not in info:
            return None
        try:
            return int(info["levelHi"]), int(info["levelLo"])
        except (TypeError, ValueError):
            return None


class ValidateInitialStage(gym.Wrapper):
    """Fail early when a named savestate does not start on the expected world-stage."""

    LEVEL_HI_ADDR = 1887
    LEVEL_LO_ADDR = 1884

    def __init__(self, env: gym.Env, expected_stage: tuple[int, int]):
        super().__init__(env)
        self.expected_stage = expected_stage

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        stage = self._stage_from_ram()
        if stage != self.expected_stage:
            expected = self._display_stage(self.expected_stage)
            actual = self._display_stage(stage) if stage is not None else "unknown"
            raise RuntimeError(
                f"Expected initial stage {expected}, but savestate starts at {actual}. "
                "Check --state and --custom-integration-path."
            )
        return obs, info

    def _stage_from_ram(self) -> tuple[int, int] | None:
        get_ram = getattr(self.env.unwrapped, "get_ram", None)
        if get_ram is None:
            return None
        ram = get_ram()
        try:
            return int(ram[self.LEVEL_HI_ADDR]), int(ram[self.LEVEL_LO_ADDR])
        except (IndexError, TypeError, ValueError):
            return None

    @staticmethod
    def _display_stage(stage: tuple[int, int]) -> str:
        return f"{stage[0] + 1}-{stage[1] + 1}"


class InfoRewardShaping(gym.Wrapper):
    """Light shaping from Stable-Retro info variables when they are present."""

    X_KEYS: tuple[Hashable, ...] = (
        "x",
        "x_pos",
        "x_position",
        "screen_x",
        "scroll_x",
        "xscroll",
    )

    def __init__(
        self,
        env: gym.Env,
        progress_scale: float = 0.05,
        death_penalty: float = 25.0,
        flag_bonus: float = 100.0,
    ):
        super().__init__(env)
        self.progress_scale = progress_scale
        self.death_penalty = death_penalty
        self.flag_bonus = flag_bonus
        self._last_x: float | None = None

    def reset(self, **kwargs):
        self._last_x = None
        return self.env.reset(**kwargs)

    def step(self, action: Any):
        obs, reward, terminated, truncated, info = self.env.step(action)
        shaped = float(reward)

        x_pos = self._extract_x(info)
        if x_pos is not None:
            if self._last_x is not None:
                delta = max(-5.0, min(5.0, x_pos - self._last_x))
                shaped += self.progress_scale * delta
            self._last_x = x_pos

        if bool(info.get("flag_get", False)):
            shaped += self.flag_bonus

        dead_like = any(bool(info.get(key, False)) for key in ("dead", "death", "gameover"))
        if dead_like or (terminated and not bool(info.get("flag_get", False))):
            shaped -= self.death_penalty

        return obs, shaped, terminated, truncated, info

    def _extract_x(self, info: dict[str, Any]) -> float | None:
        if "xscrollLo" in info and "xscrollHi" in info:
            try:
                return float(info["xscrollLo"]) + 256.0 * float(info["xscrollHi"])
            except (TypeError, ValueError):
                return None
        for key in self.X_KEYS:
            if key in info:
                try:
                    return float(info[key])
                except (TypeError, ValueError):
                    return None
        return None


class SmartMarioReward(gym.Wrapper):
    """Dense Mario reward that balances progress, skill events, and anti-stall pressure."""
    X_KEYS = InfoRewardShaping.X_KEYS

    def __init__(
        self,
        env: gym.Env,
        progress_scale: float = 0.25,
        backtrack_scale: float = 0.05,
        score_scale: float = 0.025,
        coin_bonus: float = 1.0,
        checkpoint_bonus: float = 5.0,
        checkpoint_width: int = 128,
        level_bonus: float = 50.0,
        finish_zone_x: float = 3100.0,
        finish_zone_bonus: float = 100.0,
        flag_zone_x: float = 3300.0,
        flag_bonus: float = 100.0,
        death_penalty: float = 50.0,
        life_loss_penalty: float = 25.0,
        time_penalty: float = 0.01,
        stall_penalty: float = 0.02,
        stall_window: int = 30,
        max_stall_penalty: float = 0.5,
        jump_penalty: float = 0.03,
        neutral_jump_penalty: float = 0.08,
        repeated_jump_penalty: float = 0.04,
        repeated_jump_window: int = 4,
        max_repeated_jump_penalty: float = 0.5,
        left_penalty: float = 0.02,
        bad_button_penalty: float = 0.25,
    ):
        super().__init__(env)
        self.progress_scale = progress_scale
        self.backtrack_scale = backtrack_scale
        self.score_scale = score_scale
        self.coin_bonus = coin_bonus
        self.checkpoint_bonus = checkpoint_bonus
        self.checkpoint_width = checkpoint_width
        self.level_bonus = level_bonus
        self.finish_zone_x = finish_zone_x
        self.finish_zone_bonus = finish_zone_bonus
        self.flag_zone_x = flag_zone_x
        self.flag_bonus = flag_bonus
        self.death_penalty = death_penalty
        self.life_loss_penalty = life_loss_penalty
        self.time_penalty = time_penalty
        self.stall_penalty = stall_penalty
        self.stall_window = stall_window
        self.max_stall_penalty = max_stall_penalty
        self.jump_penalty = jump_penalty
        self.neutral_jump_penalty = neutral_jump_penalty
        self.repeated_jump_penalty = repeated_jump_penalty
        self.repeated_jump_window = repeated_jump_window
        self.max_repeated_jump_penalty = max_repeated_jump_penalty
        self.left_penalty = left_penalty
        self.bad_button_penalty = bad_button_penalty
        self._last_x: float | None = None
        self._max_x = 0.0
        self._next_checkpoint = float(checkpoint_width)
        self._last_score: float | None = None
        self._last_coins: int | None = None
        self._last_lives: int | None = None
        self._last_level: tuple[int, int] | None = None
        self._stall_steps = 0
        self._jump_streak = 0
        self._finish_zone_awarded = False
        self._flag_zone_awarded = False

    def reset(self, **kwargs):
        self._last_x = None
        self._max_x = 0.0
        self._next_checkpoint = float(self.checkpoint_width)
        self._last_score = None
        self._last_coins = None
        self._last_lives = None
        self._last_level = None
        self._stall_steps = 0
        self._jump_streak = 0
        self._finish_zone_awarded = False
        self._flag_zone_awarded = False
        return self.env.reset(**kwargs)

    def step(self, action: Any):
        obs, reward, terminated, truncated, info = self.env.step(action)
        shaped = 0.0
        components = {
            "progress": 0.0,
            "checkpoint": 0.0,
            "score": 0.0,
            "coin": 0.0,
            "level": 0.0,
            "finish": 0.0,
            "life": 0.0,
            "time": -self.time_penalty,
            "stall": 0.0,
            "action": 0.0,
            "death": 0.0,
        }

        x_pos = self._extract_x(info)
        if x_pos is not None:
            if self._last_x is not None:
                delta_x = x_pos - self._last_x
                if delta_x > 0:
                    components["progress"] += self.progress_scale * min(delta_x, 16.0)
                    self._stall_steps = 0
                else:
                    components["progress"] += self.backtrack_scale * max(delta_x, -16.0)
                    self._stall_steps += 1

            self._max_x = max(self._max_x, x_pos)
            while self._max_x >= self._next_checkpoint:
                components["checkpoint"] += self.checkpoint_bonus
                self._next_checkpoint += self.checkpoint_width
            components["finish"] = self._finish_reward()
            self._last_x = x_pos

        components["score"] = self._score_reward(info)
        components["coin"] = self._coin_reward(info)
        components["level"] = self._level_reward(info)
        components["life"] = self._life_reward(info)
        components["action"] = self._action_reward(info)

        if self._stall_steps >= self.stall_window:
            components["stall"] = -min(
                self.max_stall_penalty,
                self.stall_penalty * (self._stall_steps - self.stall_window + 1),
            )

        if terminated and (self._life_like(info) <= -1 or bool(info.get("single_life_done", False))):
            components["death"] = -self.death_penalty

        shaped = sum(components.values())
        info = dict(info)
        info["smart_reward"] = components

        return obs, shaped, terminated, truncated, info

    def _extract_x(self, info: dict[str, Any]) -> float | None:
        return InfoRewardShaping._extract_x(self, info)

    def _score_reward(self, info: dict[str, Any]) -> float:
        if "score" not in info:
            return 0.0
        score = float(info["score"])
        reward = 0.0
        if self._last_score is not None:
            reward = self.score_scale * max(0.0, score - self._last_score)
        self._last_score = score
        return reward

    def _coin_reward(self, info: dict[str, Any]) -> float:
        if "coins" not in info:
            return 0.0
        coins = int(info["coins"])
        reward = 0.0
        if self._last_coins is not None:
            delta = coins - self._last_coins
            if delta < 0:
                delta += 100
            reward = self.coin_bonus * max(0, delta)
        self._last_coins = coins
        return reward

    def _level_reward(self, info: dict[str, Any]) -> float:
        if "levelLo" not in info or "levelHi" not in info:
            return 0.0
        level = (int(info["levelHi"]), int(info["levelLo"]))
        reward = 0.0
        if self._last_level is not None and level != self._last_level:
            reward = self.level_bonus
        self._last_level = level
        return reward

    def _life_reward(self, info: dict[str, Any]) -> float:
        if "lives" not in info:
            return 0.0
        lives = int(info["lives"])
        reward = 0.0
        if self._last_lives is not None and lives < self._last_lives:
            reward = -self.life_loss_penalty * (self._last_lives - lives)
        self._last_lives = lives
        return reward

    def _life_like(self, info: dict[str, Any]) -> int:
        try:
            return int(info.get("lives", 0))
        except (TypeError, ValueError):
            return 0

    def _finish_reward(self) -> float:
        reward = 0.0
        if not self._finish_zone_awarded and self._max_x >= self.finish_zone_x:
            reward += self.finish_zone_bonus
            self._finish_zone_awarded = True
        if not self._flag_zone_awarded and self._max_x >= self.flag_zone_x:
            reward += self.flag_bonus
            self._flag_zone_awarded = True
        return reward

    def _action_reward(self, info: dict[str, Any]) -> float:
        buttons = set(info.get("buttons_pressed", ()))
        reward = 0.0

        if "A" in buttons:
            self._jump_streak += 1
            reward -= self.jump_penalty
            if "RIGHT" not in buttons:
                reward -= self.neutral_jump_penalty
            if self._jump_streak > self.repeated_jump_window:
                reward -= min(
                    self.max_repeated_jump_penalty,
                    self.repeated_jump_penalty * (self._jump_streak - self.repeated_jump_window),
                )
        else:
            self._jump_streak = 0

        if "LEFT" in buttons:
            reward -= self.left_penalty
        if "LEFT" in buttons and "RIGHT" in buttons:
            reward -= self.bad_button_penalty
        if "START" in buttons or "SELECT" in buttons:
            reward -= self.bad_button_penalty

        return reward


class StageScoreReward(gym.Wrapper):
    """World-stage reward profile that prioritizes clearing, then high score."""

    X_KEYS = InfoRewardShaping.X_KEYS

    def __init__(
        self,
        env: gym.Env,
        progress_scale: float = 0.35,
        score_scale: float = 0.02,
        max_score_reward: float = 8.0,
        coin_bonus: float = 1.5,
        clear_bonus: float = 750.0,
        route_transition_bonus: float = 35.0,
        checkpoint_bonus: float = 8.0,
        checkpoint_width: int = 128,
        life_loss_penalty: float = 75.0,
        death_penalty: float = 150.0,
        time_penalty: float = 0.015,
        stall_penalty: float = 0.08,
        stall_window: int = 45,
        max_stall_penalty: float = 2.0,
        backtrack_penalty: float = 0.03,
        route_reset_threshold: float = 192.0,
        left_penalty: float = 0.01,
        bad_button_penalty: float = 0.25,
    ):
        super().__init__(env)
        self.progress_scale = progress_scale
        self.score_scale = score_scale
        self.max_score_reward = max_score_reward
        self.coin_bonus = coin_bonus
        self.clear_bonus = clear_bonus
        self.route_transition_bonus = route_transition_bonus
        self.checkpoint_bonus = checkpoint_bonus
        self.checkpoint_width = checkpoint_width
        self.life_loss_penalty = life_loss_penalty
        self.death_penalty = death_penalty
        self.time_penalty = time_penalty
        self.stall_penalty = stall_penalty
        self.stall_window = stall_window
        self.max_stall_penalty = max_stall_penalty
        self.backtrack_penalty = backtrack_penalty
        self.route_reset_threshold = route_reset_threshold
        self.left_penalty = left_penalty
        self.bad_button_penalty = bad_button_penalty
        self._area_index = 0
        self._last_x: float | None = None
        self._area_max_x = 0.0
        self._next_checkpoint = float(checkpoint_width)
        self._last_score: float | None = None
        self._last_coins: int | None = None
        self._last_lives: int | None = None
        self._stall_steps = 0

    def reset(self, **kwargs):
        self._area_index = 0
        self._last_x = None
        self._area_max_x = 0.0
        self._next_checkpoint = float(self.checkpoint_width)
        self._last_score = None
        self._last_coins = None
        self._last_lives = None
        self._stall_steps = 0
        return self.env.reset(**kwargs)

    def step(self, action: Any):
        obs, reward, terminated, truncated, info = self.env.step(action)
        components = {
            "progress": 0.0,
            "checkpoint": 0.0,
            "route": 0.0,
            "score": 0.0,
            "coin": 0.0,
            "clear": 0.0,
            "life": 0.0,
            "death": 0.0,
            "time": -self.time_penalty,
            "stall": 0.0,
            "action": 0.0,
        }

        x_pos = self._extract_x(info)
        if x_pos is not None:
            if self._last_x is not None and x_pos + self.route_reset_threshold < self._last_x:
                self._area_index += 1
                self._area_max_x = 0.0
                self._next_checkpoint = float(self.checkpoint_width)
                self._stall_steps = 0
                components["route"] += self.route_transition_bonus

            if x_pos > self._area_max_x:
                delta = x_pos - self._area_max_x
                components["progress"] += self.progress_scale * min(delta, 24.0)
                self._area_max_x = x_pos
                self._stall_steps = 0
            else:
                backtrack = self._area_max_x - x_pos
                components["progress"] -= self.backtrack_penalty * min(backtrack, 24.0)
                self._stall_steps += 1

            while self._area_max_x >= self._next_checkpoint:
                components["checkpoint"] += self.checkpoint_bonus
                self._next_checkpoint += self.checkpoint_width
            self._last_x = x_pos

        components["score"] = self._score_reward(info)
        components["coin"] = self._coin_reward(info)
        components["life"] = self._life_reward(info)
        components["action"] = self._action_reward(info)

        if bool(info.get("stage_clear_done", False)):
            components["clear"] = self.clear_bonus
        if terminated and self._is_failure(info):
            components["death"] = -self.death_penalty

        if self._stall_steps >= self.stall_window:
            components["stall"] = -min(
                self.max_stall_penalty,
                self.stall_penalty * (self._stall_steps - self.stall_window + 1),
            )

        shaped = sum(components.values())
        info = dict(info)
        info["stage_score_reward"] = components
        info["stage_score_area"] = self._area_index
        info["stage_score_area_max_x"] = self._area_max_x
        return obs, shaped, terminated, truncated, info

    def _extract_x(self, info: dict[str, Any]) -> float | None:
        return InfoRewardShaping._extract_x(self, info)

    def _score_reward(self, info: dict[str, Any]) -> float:
        if "score" not in info:
            return 0.0
        score = float(info["score"])
        reward = 0.0
        if self._last_score is not None:
            delta = max(0.0, score - self._last_score)
            reward = min(self.max_score_reward, self.score_scale * delta)
        self._last_score = score
        return reward

    def _coin_reward(self, info: dict[str, Any]) -> float:
        if "coins" not in info:
            return 0.0
        coins = int(info["coins"])
        reward = 0.0
        if self._last_coins is not None:
            delta = coins - self._last_coins
            if delta < 0:
                delta += 100
            reward = self.coin_bonus * max(0, delta)
        self._last_coins = coins
        return reward

    def _life_reward(self, info: dict[str, Any]) -> float:
        if "lives" not in info:
            return 0.0
        lives = int(info["lives"])
        reward = 0.0
        if self._last_lives is not None and lives < self._last_lives:
            reward = -self.life_loss_penalty * (self._last_lives - lives)
        self._last_lives = lives
        return reward

    def _action_reward(self, info: dict[str, Any]) -> float:
        buttons = set(info.get("buttons_pressed", ()))
        reward = 0.0
        if "LEFT" in buttons:
            reward -= self.left_penalty
        if "LEFT" in buttons and "RIGHT" in buttons:
            reward -= self.bad_button_penalty
        if "START" in buttons or "SELECT" in buttons:
            reward -= self.bad_button_penalty
        return reward

    def _is_failure(self, info: dict[str, Any]) -> bool:
        if bool(info.get("stage_clear_done", False)):
            return False
        if bool(info.get("single_life_done", False)):
            return True
        try:
            return int(info.get("lives", 0)) <= -1
        except (TypeError, ValueError):
            return False
