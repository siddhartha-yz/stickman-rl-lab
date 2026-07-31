"""Policy evaluation utilities independent of the training CLI."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO

from stickman_rl.config import load_env_config
from stickman_rl.env import StickmanReachEnv


def _positive_episode_count(episodes: int) -> int:
    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    return episodes


@dataclass(slots=True)
class EvaluationResult:
    episodes: int
    success_rate: float
    mean_reward: float
    std_reward: float
    mean_length: float
    mean_final_distance: float
    mean_energy: float
    mean_torso_height: float
    route_completion_rate: float
    mean_waypoints_completed: float
    mean_max_torso_x: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_policy_path(
    model_path: str | Path,
    stage: int = 1,
    episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    seed: int = 1000,
    env_config_path: str | Path | None = None,
) -> EvaluationResult:
    """Load a PPO checkpoint and evaluate episode-level task metrics."""
    episodes = _positive_episode_count(episodes)
    model = PPO.load(str(model_path))
    env_config = load_env_config(stage=stage, config_path=env_config_path)
    env = StickmanReachEnv(config=env_config, render_mode="human" if render else None)
    rewards: list[float] = []
    lengths: list[int] = []
    successes: list[float] = []
    distances: list[float] = []
    energies: list[float] = []
    heights: list[float] = []
    route_completions: list[float] = []
    waypoints_completed: list[float] = []
    max_torso_positions: list[float] = []
    try:
        for episode in range(episodes):
            observation, _ = env.reset(seed=seed + episode)
            done = False
            total_reward = 0.0
            length = 0
            height_samples: list[float] = []
            max_torso_x = float(env.stickman.torso.position.x)
            info: dict[str, Any] = {}
            while not done:
                action, _ = model.predict(observation, deterministic=deterministic)
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                length += 1
                height_samples.append(float(info["torso_height"]))
                max_torso_x = max(max_torso_x, float(info.get("torso_x", 0.0)))
                done = terminated or truncated
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
    return EvaluationResult(
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



def evaluate_random_policy(
    stage: int = 1,
    episodes: int = 10,
    seed: int = 2000,
    env_config_path: str | Path | None = None,
) -> EvaluationResult:
    """Evaluate uniformly random joint commands as a reproducible baseline."""
    episodes = _positive_episode_count(episodes)
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
    try:
        for episode in range(episodes):
            observation, _ = env.reset(seed=seed + episode)
            done = False
            total_reward = 0.0
            length = 0
            height_samples: list[float] = []
            max_torso_x = float(env.stickman.torso.position.x)
            info: dict[str, Any] = {}
            while not done:
                action = env.action_space.sample()
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                length += 1
                height_samples.append(float(info["torso_height"]))
                max_torso_x = max(max_torso_x, float(info.get("torso_x", 0.0)))
                done = terminated or truncated
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
    return EvaluationResult(
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
