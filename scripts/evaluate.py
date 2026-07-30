from __future__ import annotations

import argparse
import json
from pathlib import Path

from stickman_rl.evaluation import evaluate_policy_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO checkpoint.")
    parser.add_argument("model")
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--env-config", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    result = evaluate_policy_path(
        args.model,
        stage=args.stage,
        episodes=args.episodes,
        deterministic=not args.stochastic,
        render=args.render,
        seed=args.seed,
        env_config_path=args.env_config,
    )
    payload = result.to_dict()
    rendered = json.dumps(payload, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(f"saved: {args.output}")
    print(rendered)


if __name__ == "__main__":
    main()
