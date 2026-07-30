"""Stable-Baselines3 callbacks for environment diagnostics."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class EnvironmentMetricsCallback(BaseCallback):
    """Record success, distance, energy, posture, and decomposed rewards to TensorBoard."""

    def __init__(self, window: int = 100, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        self.window = window
        self.history: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=window))

    def _on_step(self) -> bool:
        infos: list[dict[str, Any]] = self.locals.get("infos", [])
        dones = np.asarray(self.locals.get("dones", []), dtype=bool)
        for index, info in enumerate(infos):
            for key, value in info.items():
                if key.startswith("reward_") and isinstance(value, (int, float)):
                    self.logger.record(f"reward_components/{key.removeprefix('reward_')}", float(value))
            self.logger.record("environment/distance", float(info.get("distance", 0.0)))
            self.logger.record("environment/torso_height", float(info.get("torso_height", 0.0)))
            self.logger.record("environment/mean_energy", float(info.get("mean_energy", 0.0)))
            if index < len(dones) and dones[index]:
                for key in ("is_success", "final_distance", "mean_energy", "torso_height"):
                    value = float(info.get(key, 0.0))
                    self.history[key].append(value)
                    self.logger.record(f"episode_metrics/{key}", float(np.mean(self.history[key])))
        return True


class RewardAnnealingCallback(BaseCallback):
    """Linearly interpolate reward weights during a curriculum transition."""

    def __init__(
        self,
        start_weights: dict[str, float],
        end_weights: dict[str, float],
        anneal_timesteps: int,
        update_freq: int = 256,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.start_weights = {key: float(value) for key, value in start_weights.items()}
        self.end_weights = {key: float(value) for key, value in end_weights.items()}
        self.anneal_timesteps = max(1, int(anneal_timesteps))
        self.update_freq = max(1, int(update_freq))
        self.keys = sorted(set(self.start_weights) | set(self.end_weights))

    def _weights_at(self, alpha: float) -> dict[str, float]:
        return {
            key: (1.0 - alpha) * self.start_weights.get(key, 0.0)
            + alpha * self.end_weights.get(key, 0.0)
            for key in self.keys
        }

    def _on_training_start(self) -> None:
        self.training_env.env_method("set_reward_weights", self._weights_at(0.0))

    def _on_step(self) -> bool:
        if self.num_timesteps % self.update_freq != 0 and self.num_timesteps < self.anneal_timesteps:
            return True
        alpha = min(1.0, self.num_timesteps / self.anneal_timesteps)
        weights = self._weights_at(alpha)
        self.training_env.env_method("set_reward_weights", weights)
        self.logger.record("curriculum/reward_anneal_alpha", alpha)
        for key, value in weights.items():
            self.logger.record(f"curriculum_weights/{key}", value)
        return True
