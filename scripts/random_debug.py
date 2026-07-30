from __future__ import annotations

import argparse

from stickman_rl.config import load_env_config
from stickman_rl.env import StickmanReachEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Run random actions with real-time rendering.")
    parser.add_argument("--stage", type=int, default=0)
    parser.add_argument("--env-config", type=str, default=None)
    parser.add_argument("--steps", type=int, default=3000)
    args = parser.parse_args()
    env = StickmanReachEnv(config=load_env_config(stage=args.stage, config_path=args.env_config), render_mode="human")
    observation, _ = env.reset(seed=42)
    try:
        for _ in range(args.steps):
            observation, _, terminated, truncated, _ = env.step(env.action_space.sample())
            if terminated or truncated:
                observation, _ = env.reset()
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
