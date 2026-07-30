from __future__ import annotations

import argparse
import json

from stickman_rl.training import train_ppo


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the articulated stickman with PPO.")
    parser.add_argument("--stage", type=int, default=1, choices=range(0, 6))
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--train-config", type=str, default=None)
    parser.add_argument("--env-config", type=str, default=None)
    parser.add_argument("--anneal-from-stage", type=int, default=None)
    parser.add_argument("--anneal-timesteps", type=int, default=None)
    args = parser.parse_args()
    checkpoint, summary = train_ppo(
        stage=args.stage,
        total_timesteps=args.timesteps,
        resume=args.resume,
        seed=args.seed,
        run_name=args.run_name,
        train_config_path=args.train_config,
        env_config_path=args.env_config,
        anneal_from_stage=args.anneal_from_stage,
        anneal_timesteps=args.anneal_timesteps,
    )
    print(f"Saved checkpoint: {checkpoint}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
