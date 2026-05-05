from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv, sync_envs_normalization


class MarioMetricsCallback(BaseCallback):
    """Log Mario-specific terminal info from vectorized training."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._stage_clears = 0
        self._life_losses = 0
        self._episodes = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if bool(info.get("stage_clear_done", False)):
                self._stage_clears += 1
            if bool(info.get("single_life_done", False)):
                self._life_losses += 1
            if "episode" in info:
                self._episodes += 1
                self._record_terminal_info(info)

        if self._episodes:
            self.logger.record("mario/stage_clear_rate", self._stage_clears / self._episodes)
            self.logger.record("mario/life_loss_rate", self._life_losses / self._episodes)
        return True

    def _record_terminal_info(self, info: dict[str, Any]) -> None:
        self.logger.record("mario/terminal_score", _float_info(info, "score"))
        self.logger.record("mario/terminal_coins", _float_info(info, "coins"))
        self.logger.record("mario/terminal_time", _float_info(info, "time"))
        self.logger.record("mario/terminal_level_hi", _float_info(info, "levelHi"))
        self.logger.record("mario/terminal_level_lo", _float_info(info, "levelLo"))
        self.logger.record("mario/terminal_max_x", _x_position(info))


class StageScoreEvalCallback(BaseCallback):
    """Evaluate by completion first and score second, then save the best model."""

    def __init__(
        self,
        eval_env: VecEnv,
        best_model_save_path: str | Path,
        eval_freq: int = 50_000,
        n_eval_episodes: int = 5,
        deterministic: bool = True,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.best_model_save_path = Path(best_model_save_path)
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        self.best_score = -float("inf")

    def _init_callback(self) -> None:
        self.best_model_save_path.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        if self.eval_freq <= 0 or self.n_calls % self.eval_freq != 0:
            return True

        if self.model.get_vec_normalize_env() is not None:
            sync_envs_normalization(self.training_env, self.eval_env)

        metrics = self._evaluate()
        completion_rate = metrics["clears"] / max(metrics["episodes"], 1)
        completion_score = completion_rate * 1_000_000.0 + metrics["mean_score"]

        self.logger.record("eval_stage_score/completion_rate", completion_rate)
        self.logger.record("eval_stage_score/mean_score", metrics["mean_score"])
        self.logger.record("eval_stage_score/mean_reward", metrics["mean_reward"])
        self.logger.record("eval_stage_score/mean_max_x", metrics["mean_max_x"])
        self.logger.record("eval_stage_score/mean_time", metrics["mean_time"])
        self.logger.record("eval_stage_score/completion_score", completion_score)

        if completion_score > self.best_score:
            self.best_score = completion_score
            self.model.save(self.best_model_save_path / "best_model")
            vecnormalize = self.model.get_vec_normalize_env()
            if vecnormalize is not None:
                vecnormalize.save(self.best_model_save_path / "best_vecnormalize.pkl")
            if self.verbose:
                print(
                    "New best 5-2 model: "
                    f"completion_rate={completion_rate:.2f} "
                    f"mean_score={metrics['mean_score']:.1f}"
                )
        return True

    def _evaluate(self) -> dict[str, float]:
        obs = self.eval_env.reset()
        episode_rewards = np.zeros(self.eval_env.num_envs, dtype=np.float64)
        episode_max_x = np.zeros(self.eval_env.num_envs, dtype=np.float64)
        episode_scores = []
        episode_times = []
        episode_reward_totals = []
        episode_max_xs = []
        clears = 0
        episodes = 0
        lstm_states = None
        episode_starts = np.ones((self.eval_env.num_envs,), dtype=bool)

        while episodes < self.n_eval_episodes:
            action, lstm_states = self.model.predict(
                obs,
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=self.deterministic,
            )
            obs, rewards, dones, infos = self.eval_env.step(action)
            episode_rewards += rewards

            for index, info in enumerate(infos):
                episode_max_x[index] = max(episode_max_x[index], _x_position(info))
                if not dones[index]:
                    continue
                episodes += 1
                if bool(info.get("stage_clear_done", False)):
                    clears += 1
                episode_scores.append(_float_info(info, "score"))
                episode_times.append(_float_info(info, "time"))
                episode_reward_totals.append(float(episode_rewards[index]))
                episode_max_xs.append(float(episode_max_x[index]))
                episode_rewards[index] = 0.0
                episode_max_x[index] = 0.0
                if episodes >= self.n_eval_episodes:
                    break

            episode_starts = dones

        return {
            "episodes": float(episodes),
            "clears": float(clears),
            "mean_score": _mean(episode_scores),
            "mean_time": _mean(episode_times),
            "mean_reward": _mean(episode_reward_totals),
            "mean_max_x": _mean(episode_max_xs),
        }


def _float_info(info: dict[str, Any], key: str) -> float:
    try:
        return float(info.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0


def _x_position(info: dict[str, Any]) -> float:
    if "xscrollLo" in info and "xscrollHi" in info:
        try:
            return float(info["xscrollLo"]) + 256.0 * float(info["xscrollHi"])
        except (TypeError, ValueError):
            return 0.0
    try:
        return float(info.get("stage_score_area_max_x", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(values))
