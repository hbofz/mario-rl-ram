from __future__ import annotations

import argparse

from mario_rl.env import make_mario_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test the Mario RAM environment.")
    parser.add_argument("--game", default="SuperMarioBros-Nes-v0")
    parser.add_argument("--state", default=None)
    parser.add_argument("--custom-integration-path", default=None)
    parser.add_argument("--expected-stage", default=None)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--action-mode", default="mario", choices=["all", "discrete", "multidiscrete", "mario", "mario-secrets"])
    parser.add_argument("--reward-mode", default="base", choices=["base", "shaped", "smart", "stage-score"])
    parser.add_argument("--action-repeat", type=int, default=4)
    parser.add_argument("--single-life", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--single-stage", action=argparse.BooleanOptionalAction, default=False)
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
            single_life=args.single_life,
            single_stage=args.single_stage,
            state=args.state,
            custom_integration_path=args.custom_integration_path,
            expected_stage=_parse_stage(args.expected_stage),
        )
    except FileNotFoundError as exc:
        message = str(exc)
        if "savestate" in message:
            raise SystemExit(message) from exc
        raise SystemExit(
            f"{message}\n\nImport your legally obtained ROM first:\n"
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
    action_meanings = _find_action_meanings(env)
    if action_meanings is not None:
        print(f"action_meanings={action_meanings}")
    print(f"last_obs_shape={getattr(obs, 'shape', None)}")
    print(f"total_reward={total_reward:.2f}")
    print(f"info_keys={sorted(last_info.keys())}")
    if "stage_score_reward" in last_info:
        print(f"stage_score_reward_keys={sorted(last_info['stage_score_reward'].keys())}")
    env.close()


def _parse_stage(value: str | None) -> tuple[int, int] | None:
    if not value:
        return None
    world_text, stage_text = value.split("-", maxsplit=1)
    return int(world_text) - 1, int(stage_text) - 1


def _find_action_meanings(env) -> tuple[str, ...] | None:
    current = env
    while current is not None:
        meanings = getattr(current, "action_meanings", None)
        if meanings is not None:
            return tuple(meanings)
        current = getattr(current, "env", None)
    return None


if __name__ == "__main__":
    main()
