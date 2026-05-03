from __future__ import annotations

import argparse

from mario_rl.env import make_mario_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test the Mario RAM environment.")
    parser.add_argument("--game", default="SuperMarioBros-Nes-v0")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--action-mode", default="all", choices=["all", "discrete", "multidiscrete"])
    parser.add_argument("--reward-mode", default="base", choices=["base", "shaped", "smart"])
    parser.add_argument("--action-repeat", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        env = make_mario_env(
            game=args.game,
            action_mode=args.action_mode,
            reward_mode=args.reward_mode,
            action_repeat=args.action_repeat,
            render_mode=None,
        )
    except FileNotFoundError as exc:
        raise SystemExit(
            f"{exc}\n\nImport your legally obtained ROM first:\n"
            "  python -m stable_retro.import roms/"
        ) from exc

    obs, info = env.reset()
    total_reward = 0.0
    last_info = info

    for _ in range(args.steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, last_info = env.step(action)
        total_reward += float(reward)
        if terminated or truncated:
            obs, last_info = env.reset()

    print(f"observation_space={env.observation_space}")
    print(f"action_space={env.action_space}")
    print(f"last_obs_shape={getattr(obs, 'shape', None)}")
    print(f"total_reward={total_reward:.2f}")
    print(f"info_keys={sorted(last_info.keys())}")
    env.close()


if __name__ == "__main__":
    main()
