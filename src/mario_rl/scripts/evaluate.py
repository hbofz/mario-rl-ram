from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

from mario_rl.env import make_mario_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained Mario policy.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--algo", default="recurrent-ppo", choices=["ppo", "recurrent-ppo"])
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--game", default="SuperMarioBros-Nes-v0")
    parser.add_argument("--state", default="Level5-2")
    parser.add_argument("--custom-integration-path", default="custom_integrations")
    parser.add_argument("--expected-stage", default="5-2")
    parser.add_argument("--video-dir", default=None)
    parser.add_argument("--action-mode", default="mario-secrets", choices=["all", "discrete", "multidiscrete", "mario", "mario-secrets"])
    parser.add_argument("--reward-mode", default="stage-score", choices=["base", "shaped", "smart", "stage-score"])
    parser.add_argument("--action-repeat", type=int, default=4)
    parser.add_argument("--single-life", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--single-stage", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vecnormalize", default=None, help="Path to saved VecNormalize .pkl stats.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render_mode = "rgb_array" if args.video_dir else None

    def make_eval_env():
        env = make_mario_env(
            game=args.game,
            action_mode=args.action_mode,
            reward_mode=args.reward_mode,
            action_repeat=args.action_repeat,
            render_mode=render_mode,
            monitor=False,
            single_life=args.single_life,
            single_stage=args.single_stage,
            state=args.state,
            custom_integration_path=args.custom_integration_path,
            expected_stage=_parse_stage(args.expected_stage),
        )
        if args.video_dir:
            video_dir = Path(args.video_dir)
            video_dir.mkdir(parents=True, exist_ok=True)
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=str(video_dir),
                name_prefix=Path(args.model).stem,
                episode_trigger=lambda episode_id: True,
            )
        return env

    env = DummyVecEnv([make_eval_env])
    env = VecMonitor(env)
    if args.vecnormalize:
        env = VecNormalize.load(args.vecnormalize, env)
        env.training = False
        env.norm_reward = False

    model_cls = PPO if args.algo == "ppo" else RecurrentPPO
    model = model_cls.load(args.model, env=env, device=args.device)

    clears = 0
    total_score = 0.0
    for episode in range(args.episodes):
        obs = env.reset()
        done = False
        episode_reward = 0.0
        lstm_states = None
        episode_start = [True]
        max_x = 0.0
        final_info = {}

        while not done:
            action, lstm_states = model.predict(
                obs,
                state=lstm_states,
                episode_start=episode_start,
                deterministic=args.deterministic,
            )
            obs, reward, dones, infos = env.step(action)
            done = bool(dones[0])
            episode_start = [done]
            info = infos[0]
            episode_reward += float(reward[0])
            final_info = info
            max_x = max(max_x, _x_position(info) or 0.0)

        if bool(final_info.get("stage_clear_done", False)):
            clears += 1
        total_score += float(final_info.get("score") or 0.0)
        print(
            f"episode={episode + 1} "
            f"reward={episode_reward:.2f} "
            f"clear={bool(final_info.get('stage_clear_done', False))} "
            f"max_x={max_x:.0f} "
            f"score={final_info.get('score')} "
            f"coins={final_info.get('coins')} "
            f"time={final_info.get('time')} "
            f"lives={final_info.get('lives')} "
            f"final_stage=({final_info.get('levelHi')},{final_info.get('levelLo')}) "
            f"info={final_info}"
        )

    print(
        f"summary episodes={args.episodes} "
        f"clear_rate={clears / max(args.episodes, 1):.2f} "
        f"mean_score={total_score / max(args.episodes, 1):.1f}"
    )
    env.close()


def _x_position(info: dict) -> float | None:
    if "xscrollLo" in info and "xscrollHi" in info:
        return float(info["xscrollLo"]) + 256.0 * float(info["xscrollHi"])
    return None


def _parse_stage(value: str | None) -> tuple[int, int] | None:
    if not value:
        return None
    world_text, stage_text = value.split("-", maxsplit=1)
    return int(world_text) - 1, int(stage_text) - 1


if __name__ == "__main__":
    main()
