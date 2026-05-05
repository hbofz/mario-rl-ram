from __future__ import annotations

import argparse
import gzip
from pathlib import Path

from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO

from mario_rl.env import make_mario_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture a Stable-Retro savestate when Mario reaches a target stage.")
    parser.add_argument("--model", default=None, help="Optional SB3 checkpoint used to drive from the start state.")
    parser.add_argument("--algo", default="ppo", choices=["ppo", "recurrent-ppo"])
    parser.add_argument("--game", default="SuperMarioBros-Nes-v0")
    parser.add_argument("--start-state", default="Level5-1")
    parser.add_argument("--target-stage", default="5-2")
    parser.add_argument("--output", default="custom_integrations/SuperMarioBros-Nes-v0/Level5-2.state")
    parser.add_argument("--action-mode", default="mario", choices=["all", "discrete", "multidiscrete", "mario", "mario-secrets"])
    parser.add_argument("--action-repeat", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_stage = _parse_stage(args.target_stage)
    env = make_mario_env(
        game=args.game,
        state=args.start_state,
        action_mode=args.action_mode,
        reward_mode="base",
        action_repeat=args.action_repeat,
        monitor=False,
    )

    model = None
    if args.model:
        model_cls = PPO if args.algo == "ppo" else RecurrentPPO
        model = model_cls.load(args.model, env=env, device=args.device)

    obs, _ = env.reset(seed=args.seed)
    lstm_states = None
    episode_start = True

    for step in range(1, args.max_steps + 1):
        if model is None:
            action = env.action_space.sample()
        elif args.algo == "recurrent-ppo":
            action, lstm_states = model.predict(
                obs,
                state=lstm_states,
                episode_start=[episode_start],
                deterministic=args.deterministic,
            )
        else:
            action, _ = model.predict(obs, deterministic=args.deterministic)

        obs, _, terminated, truncated, info = env.step(action)
        episode_start = terminated or truncated
        current_stage = _stage_from_info(info)
        if current_stage == target_stage:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(output, "wb") as fh:
                fh.write(env.unwrapped.em.get_state())
            print(f"saved_state={output} step={step} stage={args.target_stage} info={info}")
            env.close()
            return

        if terminated or truncated:
            obs, _ = env.reset()
            lstm_states = None

    env.close()
    raise SystemExit(
        f"Did not reach {args.target_stage} within {args.max_steps} steps. "
        "Fallback: create the same output file with the Stable-Retro Integration UI."
    )


def _parse_stage(value: str) -> tuple[int, int]:
    world_text, stage_text = value.split("-", maxsplit=1)
    return int(world_text) - 1, int(stage_text) - 1


def _stage_from_info(info: dict) -> tuple[int, int] | None:
    if "levelHi" not in info or "levelLo" not in info:
        return None
    try:
        return int(info["levelHi"]), int(info["levelLo"])
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    main()
