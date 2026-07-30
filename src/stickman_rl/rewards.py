"""Composable reward calculation with per-component diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class RewardInputs:
    previous_distance: float
    distance: float
    success: bool
    action: np.ndarray
    previous_action: np.ndarray
    torso_height: float
    torso_angle: float
    feet_contact: float
    hands_contact: float
    joint_limit_fraction: float
    out_of_bounds: bool
    progress_scale: float = 1.0


class RewardCalculator:
    """Compute independently logged reward terms from configurable weights."""

    def __init__(self, weights: dict[str, Any], desired_height: float = 1.7) -> None:
        self.weights = {key: float(value) for key, value in weights.items()}
        self.desired_height = desired_height

    def calculate(self, data: RewardInputs) -> tuple[float, dict[str, float]]:
        progress = (data.previous_distance - data.distance) * data.progress_scale
        upright_score = float(np.cos(data.torso_angle))
        height_score = float(np.clip(data.torso_height / self.desired_height, 0.0, 1.25))
        terms = {
            "progress": self.weights.get("progress", 0.0) * progress,
            "goal": self.weights.get("goal", 0.0) if data.success else 0.0,
            "time": self.weights.get("time", 0.0),
            "energy": self.weights.get("energy", 0.0) * float(np.mean(np.square(data.action))),
            "alive": self.weights.get("alive", 0.0),
            "height": self.weights.get("height", 0.0) * height_score,
            "upright": self.weights.get("upright", 0.0) * upright_score,
            "feet_contact": self.weights.get("feet_contact", 0.0) * data.feet_contact,
            "hand_contact": self.weights.get("hand_contact", 0.0) * data.hands_contact,
            "action_smoothness": self.weights.get("action_smoothness", 0.0)
            * float(np.mean(np.square(data.action - data.previous_action))),
            "joint_limit": self.weights.get("joint_limit", 0.0) * data.joint_limit_fraction,
            "out_of_bounds": self.weights.get("out_of_bounds", 0.0) if data.out_of_bounds else 0.0,
        }
        total = float(sum(terms.values()))
        terms["total"] = total
        return total, terms
