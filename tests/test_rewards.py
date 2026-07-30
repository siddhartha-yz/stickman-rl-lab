from __future__ import annotations

import numpy as np

from stickman_rl.rewards import RewardCalculator, RewardInputs


def test_progress_scale_multiplies_potential_difference() -> None:
    calculator = RewardCalculator({"progress": 2.0})
    base = RewardInputs(
        previous_distance=2.0,
        distance=1.5,
        success=False,
        action=np.zeros(8, dtype=np.float32),
        previous_action=np.zeros(8, dtype=np.float32),
        torso_height=1.0,
        torso_angle=0.0,
        feet_contact=0.0,
        hands_contact=0.0,
        joint_limit_fraction=0.0,
        out_of_bounds=False,
    )
    scaled = RewardInputs(
        previous_distance=base.previous_distance,
        distance=base.distance,
        success=base.success,
        action=base.action,
        previous_action=base.previous_action,
        torso_height=base.torso_height,
        torso_angle=base.torso_angle,
        feet_contact=base.feet_contact,
        hands_contact=base.hands_contact,
        joint_limit_fraction=base.joint_limit_fraction,
        out_of_bounds=base.out_of_bounds,
        progress_scale=4.0,
    )
    _, base_terms = calculator.calculate(base)
    _, scaled_terms = calculator.calculate(scaled)
    assert base_terms["progress"] == 1.0
    assert scaled_terms["progress"] == 4.0
