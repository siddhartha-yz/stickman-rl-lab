from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO

from stickman_rl.config import load_env_config
from stickman_rl.env import StickmanReachEnv
from stickman_rl.evaluation import EvaluationResult


def use_far_expert(target_x: float, threshold: float) -> bool:
    """Select the far-target expert once at episode reset."""
    return float(target_x) >= float(threshold)


def evaluate_routed(
    near_model_path: str | Path,
    far_model_path: str | Path,
    *,
    threshold: float,
    stage: int,
    episodes: int,
    seed: int,
    env_config_path: str | Path | None,
) -> tuple[EvaluationResult, dict[str, int]]:
    near_model = PPO.load(str(near_model_path))
    far_model = PPO.load(str(far_model_path))
    if near_model.observation_space != far_model.observation_space:
        raise ValueError("Near and far experts have different observation spaces")
    if near_model.action_space != far_model.action_space:
        raise ValueError("Near and far experts have different action spaces")

    env = StickmanReachEnv(config=load_env_config(stage=stage, config_path=env_config_path))
    rewards: list[float] = []
    lengths: list[int] = []
    successes: list[float] = []
    distances: list[float] = []
    energies: list[float] = []
    heights: list[float] = []
    route_completions: list[float] = []
    waypoints_completed: list[float] = []
    max_torso_positions: list[float] = []
    counts = {"near_episodes": 0, "far_episodes": 0}
    try:
        for episode in range(episodes):
            observation, _ = env.reset(seed=seed + episode)
            is_far = use_far_expert(float(env.target_position[0]), threshold)
            model = far_model if is_far else near_model
            counts["far_episodes" if is_far else "near_episodes"] += 1
            total_reward = 0.0
            length = 0
            height_samples: list[float] = []
            max_torso_x = float(env.stickman.torso.position.x)
            info: dict[str, Any] = {}
            while True:
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                length += 1
                height_samples.append(float(info["torso_height"]))
                max_torso_x = max(max_torso_x, float(info.get("torso_x", 0.0)))
                if terminated or truncated:
                    break
            rewards.append(total_reward)
            lengths.append(length)
            successes.append(float(info.get("is_success", False)))
            distances.append(float(info.get("final_distance", 0.0)))
            energies.append(float(info.get("mean_energy", 0.0)))
            heights.append(float(np.mean(height_samples)) if height_samples else 0.0)
            completed = float(info.get("active_waypoint_index", 0))
            waypoints_completed.append(completed)
            route_completions.append(
                float(bool(env.navigation_waypoints) and completed >= len(env.navigation_waypoints))
            )
            max_torso_positions.append(max_torso_x)
    finally:
        env.close()

    result = EvaluationResult(
        episodes=episodes,
        success_rate=float(np.mean(successes)),
        mean_reward=float(np.mean(rewards)),
        std_reward=float(np.std(rewards)),
        mean_length=float(np.mean(lengths)),
        mean_final_distance=float(np.mean(distances)),
        mean_energy=float(np.mean(energies)),
        mean_torso_height=float(np.mean(heights)),
        route_completion_rate=float(np.mean(route_completions)),
        mean_waypoints_completed=float(np.mean(waypoints_completed)),
        mean_max_torso_x=float(np.mean(max_torso_positions)),
    )
    return result, counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate target-position routed PPO experts.")
    parser.add_argument("near_model", type=Path)
    parser.add_argument("far_model", type=Path)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--stage", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--env-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result, counts = evaluate_routed(
        args.near_model,
        args.far_model,
        threshold=args.threshold,
        stage=args.stage,
        episodes=args.episodes,
        seed=args.seed,
        env_config_path=args.env_config,
    )
    payload = {
        "near_model": str(args.near_model),
        "far_model": str(args.far_model),
        "threshold": args.threshold,
        **counts,
        **result.to_dict(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, indent=2)
    args.output.write_text(rendered, encoding="utf-8")
    print(f"saved: {args.output}")
    print(rendered)


if __name__ == "__main__":
    main()
