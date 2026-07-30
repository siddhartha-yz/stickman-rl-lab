from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from stickman_rl.config import load_env_config
from stickman_rl.env import StickmanReachEnv
from stickman_rl.evaluation import EvaluationResult


def use_final_expert(route_complete: bool) -> bool:
    """Switch permanently to the final-target expert after the route is complete."""
    return bool(route_complete)


def evaluate_phase_routed(
    route_model_path: str | Path,
    final_model_path: str | Path,
    *,
    stage: int,
    episodes: int,
    seed: int,
    env_config_path: str | Path | None,
) -> tuple[EvaluationResult, dict[str, float]]:
    route_model = PPO.load(str(route_model_path))
    final_model = PPO.load(str(final_model_path))
    if route_model.observation_space != final_model.observation_space:
        raise ValueError("Route and final experts have different observation spaces")
    if route_model.action_space != final_model.action_space:
        raise ValueError("Route and final experts have different action spaces")

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
    switch_steps: list[float] = []
    try:
        for episode in range(episodes):
            observation, info = env.reset(seed=seed + episode)
            switched = False
            switch_step = float("nan")
            total_reward = 0.0
            length = 0
            height_samples: list[float] = []
            max_torso_x = float(env.stickman.torso.position.x)
            while True:
                if not switched and use_final_expert(bool(info.get("route_complete", False))):
                    switched = True
                    switch_step = float(length)
                model = final_model if switched else route_model
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
            switch_steps.append(switch_step)
    finally:
        env.close()

    finite_switch_steps = np.asarray([step for step in switch_steps if np.isfinite(step)], dtype=float)
    diagnostics = {
        "switch_rate": float(len(finite_switch_steps) / episodes),
        "mean_switch_step": (
            float(np.mean(finite_switch_steps)) if len(finite_switch_steps) else float("nan")
        ),
    }
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
    return result, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate route-to-final phase-routed PPO experts.")
    parser.add_argument("route_model", type=Path)
    parser.add_argument("final_model", type=Path)
    parser.add_argument("--stage", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--env-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result, diagnostics = evaluate_phase_routed(
        args.route_model,
        args.final_model,
        stage=args.stage,
        episodes=args.episodes,
        seed=args.seed,
        env_config_path=args.env_config,
    )
    payload = {
        "route_model": str(args.route_model),
        "final_model": str(args.final_model),
        **diagnostics,
        **result.to_dict(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, indent=2)
    args.output.write_text(rendered, encoding="utf-8")
    print(f"saved: {args.output}")
    print(rendered)


if __name__ == "__main__":
    main()
