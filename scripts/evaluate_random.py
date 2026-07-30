from __future__ import annotations

import argparse
import json

from stickman_rl.evaluation import evaluate_random_policy


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a random-action baseline.")
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--env-config", type=str, default=None)
    args = parser.parse_args()
    print(json.dumps(evaluate_random_policy(args.stage, args.episodes, args.seed, args.env_config).to_dict(), indent=2))


if __name__ == "__main__":
    main()
