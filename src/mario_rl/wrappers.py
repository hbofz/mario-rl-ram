from __future__ import annotations

from collections.abc import Hashable
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


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
        progress_scale: float = 1.0,
        backtrack_scale: float = 0.25,
        score_scale: float = 0.02,
        coin_bonus: float = 5.0,
        checkpoint_bonus: float = 25.0,
        checkpoint_width: int = 128,
        level_bonus: float = 250.0,
        death_penalty: float = 200.0,
        life_loss_penalty: float = 100.0,
        time_penalty: float = 0.01,
        stall_penalty: float = 0.05,
        stall_window: int = 30,
    ):
        super().__init__(env)
        self.progress_scale = progress_scale
        self.backtrack_scale = backtrack_scale
        self.score_scale = score_scale
        self.coin_bonus = coin_bonus
        self.checkpoint_bonus = checkpoint_bonus
        self.checkpoint_width = checkpoint_width
        self.level_bonus = level_bonus
        self.death_penalty = death_penalty
        self.life_loss_penalty = life_loss_penalty
        self.time_penalty = time_penalty
        self.stall_penalty = stall_penalty
        self.stall_window = stall_window
        self._last_x: float | None = None
        self._max_x = 0.0
        self._next_checkpoint = float(checkpoint_width)
        self._last_score: float | None = None
        self._last_coins: int | None = None
        self._last_lives: int | None = None
        self._last_level: tuple[int, int] | None = None
        self._stall_steps = 0

    def reset(self, **kwargs):
        self._last_x = None
        self._max_x = 0.0
        self._next_checkpoint = float(self.checkpoint_width)
        self._last_score = None
        self._last_coins = None
        self._last_lives = None
        self._last_level = None
        self._stall_steps = 0
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
            "life": 0.0,
            "time": -self.time_penalty,
            "stall": 0.0,
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
            self._last_x = x_pos

        components["score"] = self._score_reward(info)
        components["coin"] = self._coin_reward(info)
        components["level"] = self._level_reward(info)
        components["life"] = self._life_reward(info)

        if self._stall_steps >= self.stall_window:
            components["stall"] = -self.stall_penalty * (self._stall_steps - self.stall_window + 1)

        if terminated and self._life_like(info) <= -1:
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
