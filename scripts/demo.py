from __future__ import annotations

import argparse

from stable_baselines3 import PPO

from stickman_rl.config import load_env_config
from stickman_rl.env import StickmanReachEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch a random or trained stickman policy.")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--env-config", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()
    env = StickmanReachEnv(config=load_env_config(stage=args.stage, config_path=args.env_config), render_mode="human")
    model = PPO.load(args.model) if args.model else None
    try:
        for episode in range(args.episodes):
            observation, _ = env.reset(seed=episode)
            done = False
            total = 0.0
            while not done:
                if model is None:
                    action = env.action_space.sample()
                else:
                    action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = env.step(action)
                total += float(reward)
                done = terminated or truncated
            print(f"episode={episode + 1} reward={total:.2f} success={info['is_success']} distance={info['final_distance']:.3f}")
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
