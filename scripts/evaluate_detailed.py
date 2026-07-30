from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from stable_baselines3 import PPO

from stickman_rl.config import load_env_config
from stickman_rl.env import StickmanReachEnv


def evaluate_episodes(
    model_path: str | Path,
    *,
    stage: int,
    episodes: int,
    seed: int,
    env_config_path: str | Path | None,
    deterministic: bool,
) -> list[dict[str, Any]]:
    model = PPO.load(str(model_path))
    env = StickmanReachEnv(config=load_env_config(stage=stage, config_path=env_config_path))
    records: list[dict[str, Any]] = []
    try:
        for episode in range(episodes):
            episode_seed = seed + episode
            observation, _ = env.reset(seed=episode_seed)
            target = env.target_position.copy()
            total_reward = 0.0
            min_distance = float("inf")
            max_torso_x = float(env.stickman.torso.position.x)
            max_goal_hold = 0
            steps = 0
            info: dict[str, Any] = {}
            while True:
                action, _ = model.predict(observation, deterministic=deterministic)
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                steps += 1
                min_distance = min(min_distance, float(info["distance"]))
                max_torso_x = max(max_torso_x, float(env.stickman.torso.position.x))
                max_goal_hold = max(max_goal_hold, int(info["goal_hold_count"]))
                if terminated or truncated:
                    break
            torso = env.stickman.torso.position
            records.append(
                {
                    "episode": episode,
                    "seed": episode_seed,
                    "success": bool(info.get("is_success", False)),
                    "steps": steps,
                    "reward": total_reward,
                    "target_x": float(target[0]),
                    "target_y": float(target[1]),
                    "final_torso_x": float(torso.x),
                    "final_torso_y": float(torso.y),
                    "final_dx": float(torso.x - target[0]),
                    "final_dy": float(torso.y - target[1]),
                    "final_distance": float(info.get("final_distance", 0.0)),
                    "min_distance": min_distance,
                    "max_torso_x": max_torso_x,
                    "waypoints_completed": int(info.get("waypoints_completed", 0)),
                    "route_complete": bool(info.get("route_complete", False)),
                    "max_goal_hold_count": max_goal_hold,
                }
            )
    finally:
        env.close()
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Write per-episode deterministic policy diagnostics.")
    parser.add_argument("model")
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--env-config", default=None)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args()

    records = evaluate_episodes(
        args.model,
        stage=args.stage,
        episodes=args.episodes,
        seed=args.seed,
        env_config_path=args.env_config,
        deterministic=not args.stochastic,
    )
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(records, indent=2), encoding="utf-8")
    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    successes = sum(int(record["success"]) for record in records)
    print(f"saved {len(records)} episodes; successes={successes}; json={args.json}")
    if args.csv is not None:
        print(f"csv={args.csv}")


if __name__ == "__main__":
    main()
