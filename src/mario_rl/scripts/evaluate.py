from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO

from mario_rl.env import make_mario_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained Mario policy.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--algo", default="ppo", choices=["ppo", "recurrent-ppo"])
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--game", default="SuperMarioBros-Nes-v0")
    parser.add_argument("--video-dir", default=None)
    parser.add_argument("--action-mode", default="all", choices=["all", "discrete", "multidiscrete"])
    parser.add_argument("--reward-mode", default="smart", choices=["base", "shaped", "smart"])
    parser.add_argument("--action-repeat", type=int, default=4)
    parser.add_argument("--single-life", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--deterministic", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render_mode = "rgb_array" if args.video_dir else None
    env = make_mario_env(
        game=args.game,
        action_mode=args.action_mode,
        reward_mode=args.reward_mode,
        action_repeat=args.action_repeat,
        render_mode=render_mode,
        monitor=False,
        single_life=args.single_life,
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

    model_cls = PPO if args.algo == "ppo" else RecurrentPPO
    model = model_cls.load(args.model, env=env)

    for episode in range(args.episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0
        lstm_states = None
        episode_start = True

        while not done:
            if args.algo == "recurrent-ppo":
                action, lstm_states = model.predict(
                    obs,
                    state=lstm_states,
                    episode_start=[episode_start],
                    deterministic=args.deterministic,
                )
            else:
                action, _ = model.predict(obs, deterministic=args.deterministic)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_start = done
            episode_reward += float(reward)

        print(f"episode={episode + 1} reward={episode_reward:.2f} info={info}")

    env.close()


if __name__ == "__main__":
    main()
