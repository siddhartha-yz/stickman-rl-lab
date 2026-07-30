from __future__ import annotations

import argparse

import numpy as np
from gymnasium.utils.env_checker import check_env

from stickman_rl.config import load_env_config
from stickman_rl.env import StickmanReachEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the Gymnasium API and run a random-action stability check.")
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--env-config", type=str, default=None)
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()
    env = StickmanReachEnv(config=load_env_config(stage=args.stage, config_path=args.env_config))
    check_env(env, skip_render_check=True)
    observation, _ = env.reset(seed=123)
    episodes = 0
    min_height = float("inf")
    for _ in range(args.steps):
        action = env.action_space.sample() * np.float32(1.5)
        observation, _, terminated, truncated, info = env.step(action)
        assert observation.shape == env.observation_space.shape
        assert np.isfinite(observation).all()
        min_height = min(min_height, float(info["torso_height"]))
        if terminated or truncated:
            episodes += 1
            observation, _ = env.reset()
    env.close()
    print(f"Gymnasium check passed; random steps={args.steps}, completed episodes={episodes}, min torso height={min_height:.3f}")


if __name__ == "__main__":
    main()
