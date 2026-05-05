from __future__ import annotations

import argparse
import gzip
from pathlib import Path

import numpy as np
import pygame
import stable_retro as retro


KEY_TO_BUTTON = {
    pygame.K_RIGHT: "RIGHT",
    pygame.K_LEFT: "LEFT",
    pygame.K_UP: "UP",
    pygame.K_DOWN: "DOWN",
    pygame.K_z: "A",
    pygame.K_x: "B",
    pygame.K_RETURN: "START",
    pygame.K_RSHIFT: "SELECT",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play manually and save a Stable-Retro state at a target Mario stage.")
    parser.add_argument("--game", default="SuperMarioBros-Nes-v0")
    parser.add_argument("--start-state", default="Level5-1")
    parser.add_argument("--target-stage", default="5-2")
    parser.add_argument("--output", default="custom_integrations/SuperMarioBros-Nes-v0/Level5-2.state")
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--target-frames", type=int, default=30, help="Consecutive target-stage frames before auto-save.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_stage = _parse_stage(args.target_stage)
    env = retro.make(
        args.game,
        state=args.start_state,
        obs_type=retro.Observations.IMAGE,
        use_restricted_actions=retro.Actions.ALL,
        render_mode=None,
    )
    obs, _ = env.reset()
    buttons = list(env.buttons)
    button_index = {button: index for index, button in enumerate(buttons)}

    pygame.init()
    pygame.display.set_caption(f"Mario manual capture: {args.start_state} -> {args.target_stage}")
    height, width = obs.shape[:2]
    screen = pygame.display.set_mode((width * args.scale, height * args.scale))
    clock = pygame.time.Clock()
    target_count = 0

    print("Controls: arrows move, Z=A/jump, X=B/run, Enter=START, Right Shift=SELECT.")
    print("Press S to save immediately. Press Esc or close the window to quit.")

    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return
                    if event.key == pygame.K_s:
                        _save_state(env, args.output, "manual")
                        return

            action = np.zeros(len(buttons), dtype=np.int8)
            pressed = pygame.key.get_pressed()
            for key, button in KEY_TO_BUTTON.items():
                if pressed[key] and button in button_index:
                    action[button_index[button]] = 1

            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()
                target_count = 0
                print("Episode reset before target stage.")
                continue

            stage = _stage_from_info(info)
            if stage == target_stage:
                target_count += 1
                if target_count >= args.target_frames:
                    _save_state(env, args.output, f"auto target={args.target_stage}")
                    return
            else:
                target_count = 0

            _draw(screen, obs, args.scale)
            pygame.display.flip()
            clock.tick(args.fps)
    finally:
        env.close()
        pygame.quit()


def _draw(screen: pygame.Surface, obs: np.ndarray, scale: int) -> None:
    surface = pygame.surfarray.make_surface(np.swapaxes(obs, 0, 1))
    if scale != 1:
        surface = pygame.transform.scale(surface, (obs.shape[1] * scale, obs.shape[0] * scale))
    screen.blit(surface, (0, 0))


def _save_state(env, output: str, reason: str) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as fh:
        fh.write(env.em.get_state())
    print(f"saved_state={path} reason={reason}")


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
