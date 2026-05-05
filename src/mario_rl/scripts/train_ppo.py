from __future__ import annotations

import argparse
import re
from pathlib import Path

from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize

from mario_rl.callbacks import MarioMetricsCallback, StageScoreEvalCallback
from mario_rl.env import make_mario_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Mario from RAM with PPO.")
    parser.add_argument("--algo", default="recurrent-ppo", choices=["ppo", "recurrent-ppo"])
    parser.add_argument("--game", default="SuperMarioBros-Nes-v0")
    parser.add_argument("--state", default="Level5-2")
    parser.add_argument("--custom-integration-path", default="custom_integrations")
    parser.add_argument("--expected-stage", default="5-2", help="Displayed world-stage expected after reset, e.g. 5-2.")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-name", default="recurrent-ppo-ram-5-2-stage-score")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--log-dir", default="runs")
    parser.add_argument("--action-mode", default="mario-secrets", choices=["all", "discrete", "multidiscrete", "mario", "mario-secrets"])
    parser.add_argument("--reward-mode", default="stage-score", choices=["base", "shaped", "smart", "stage-score"])
    parser.add_argument("--action-repeat", type=int, default=4)
    parser.add_argument("--single-life", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--single-stage", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vecnormalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--checkpoint-freq", type=int, default=250_000)
    parser.add_argument("--eval-freq", type=int, default=100_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--resume-from", default=None, help="Path to a saved SB3 checkpoint to continue training.")
    parser.add_argument("--auto-resume", action=argparse.BooleanOptionalAction, default=True, help="Automatically resume from the latest checkpoint in model-dir.")
    return parser.parse_args()


def make_env_factory(args: argparse.Namespace, rank: int):
    def _init():
        env = make_mario_env(
            game=args.game,
            action_mode=args.action_mode,
            reward_mode=args.reward_mode,
            action_repeat=args.action_repeat,
            render_mode=None,
            monitor=False,
            single_life=args.single_life,
            single_stage=args.single_stage,
            state=args.state,
            custom_integration_path=args.custom_integration_path,
            expected_stage=parse_stage(args.expected_stage),
        )
        env.reset(seed=args.seed + rank)
        return env

    return _init


def main() -> None:
    args = parse_args()
    run_model_dir = Path(args.model_dir) / args.run_name
    run_log_dir = Path(args.log_dir) / args.run_name
    run_model_dir.mkdir(parents=True, exist_ok=True)
    run_log_dir.mkdir(parents=True, exist_ok=True)

    if args.auto_resume and not args.resume_from:
        checkpoints = list(run_model_dir.glob("*.zip"))
        if checkpoints:
            def get_step(path: Path) -> int:
                match = re.search(r"(\d+)_steps", path.name)
                return int(match.group(1)) if match else -1
            latest_checkpoint = max(checkpoints, key=get_step)
            if get_step(latest_checkpoint) >= 0:
                args.resume_from = str(latest_checkpoint)
                print(f"Auto-resume found latest checkpoint: {args.resume_from}")

    env = SubprocVecEnv(
        [make_env_factory(args, rank) for rank in range(args.n_envs)],
        start_method="spawn",
    )
    env = VecMonitor(env)
    vecnormalize_path = run_model_dir / "vecnormalize.pkl"
    if args.vecnormalize:
        resume_vecnormalize = _latest_vecnormalize_path(run_model_dir, args.resume_from)
        if resume_vecnormalize is not None:
            print(f"Loading VecNormalize stats: {resume_vecnormalize}")
            env = VecNormalize.load(str(resume_vecnormalize), env)
            env.training = True
            env.norm_reward = True
        else:
            env = VecNormalize(env, norm_obs=False, norm_reward=True)

    eval_env = DummyVecEnv([make_env_factory(args, 10_000)])
    eval_env = VecMonitor(eval_env)
    if args.vecnormalize:
        eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=False, training=False)

    common_kwargs = dict(
        env=env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef,
        clip_range=args.clip_range,
        tensorboard_log=str(args.log_dir),
        verbose=1,
        seed=args.seed,
        device=args.device,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
    )

    model_cls = PPO if args.algo == "ppo" else RecurrentPPO
    if args.resume_from:
        print(f"Resuming from checkpoint: {args.resume_from}")
        model = model_cls.load(
            args.resume_from,
            env=env,
            device=args.device,
            tensorboard_log=str(args.log_dir),
            print_system_info=True,
        )
        print(f"Loaded checkpoint with num_timesteps={model.num_timesteps}")
    else:
        policy = "MlpPolicy" if args.algo == "ppo" else "MlpLstmPolicy"
        model = model_cls(policy, **common_kwargs)

    checkpoint_freq = max(args.checkpoint_freq // args.n_envs, 1)
    checkpoint = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=str(run_model_dir),
        name_prefix=args.algo,
        save_replay_buffer=False,
        save_vecnormalize=args.vecnormalize,
    )
    eval_freq = max(args.eval_freq // args.n_envs, 1)
    callbacks = [
        checkpoint,
        MarioMetricsCallback(),
    ]
    if args.eval_freq > 0:
        callbacks.extend(
            [
                EvalCallback(
                    eval_env,
                    best_model_save_path=str(run_model_dir / "best_reward"),
                    log_path=str(run_log_dir / "eval_reward"),
                    eval_freq=eval_freq,
                    n_eval_episodes=args.eval_episodes,
                    deterministic=True,
                    render=False,
                ),
                StageScoreEvalCallback(
                    eval_env=eval_env,
                    best_model_save_path=run_model_dir / "best_stage_score",
                    eval_freq=eval_freq,
                    n_eval_episodes=args.eval_episodes,
                ),
            ],
        )

    model.learn(
        total_timesteps=args.timesteps,
        callback=CallbackList(callbacks),
        tb_log_name=args.run_name,
        reset_num_timesteps=not bool(args.resume_from),
        progress_bar=True,
    )
    model.save(run_model_dir / "final_model")
    if args.vecnormalize:
        env.save(str(vecnormalize_path))
    env.close()
    eval_env.close()


def parse_stage(value: str | None) -> tuple[int, int] | None:
    if not value:
        return None
    match = re.fullmatch(r"\s*(\d+)\s*-\s*(\d+)\s*", value)
    if not match:
        raise argparse.ArgumentTypeError(f"Stage must look like 5-2, got: {value}")
    world, stage = int(match.group(1)), int(match.group(2))
    if world < 1 or stage < 1:
        raise argparse.ArgumentTypeError(f"Stage numbers must be 1-indexed, got: {value}")
    return world - 1, stage - 1


def _latest_vecnormalize_path(run_model_dir: Path, resume_from: str | None) -> Path | None:
    candidates: list[Path] = []
    if resume_from:
        resume_path = Path(resume_from)
        stem = resume_path.name.removesuffix(".zip")
        candidates.extend(
            [
                resume_path.with_name(f"{stem}_vecnormalize.pkl"),
                resume_path.with_name("vecnormalize.pkl"),
                resume_path.with_name("final_vecnormalize.pkl"),
            ],
        )
    candidates.extend(run_model_dir.glob("*vecnormalize*.pkl"))
    existing = [path for path in candidates if path.exists()]
    if not existing:
        return None

    def get_step(path: Path) -> int:
        match = re.search(r"(\d+)_steps", path.name)
        return int(match.group(1)) if match else -1

    return max(existing, key=lambda path: (get_step(path), path.stat().st_mtime))


if __name__ == "__main__":
    main()
