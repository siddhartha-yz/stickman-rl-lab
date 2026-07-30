from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.lab_server import (
    PROJECT_ROOT,
    _manifest,
    _project_path,
    experiments,
    health,
    load_trajectory_payload,
    trajectory,
)


def test_lab_manifest_has_unique_reproducible_experiments() -> None:
    manifest = _manifest()
    experiments = manifest["experiments"]
    ids = [experiment["id"] for experiment in experiments]
    assert len(ids) == len(set(ids))
    assert len(experiments) >= 5
    for experiment in experiments:
        assert experiment["seed"] >= 0
        assert experiment["stage"] in {0, 1, 2, 3, 4, 5}
        assert _project_path(experiment["trajectory"]).is_relative_to(PROJECT_ROOT)


def test_load_trajectory_payload_serializes_physics_frames(tmp_path: Path) -> None:
    metadata = {
        "format_version": 2,
        "body_names": ["torso"],
        "body_geometry": {"torso": {"kind": "polygon", "vertices": [[-0.2, -0.4], [0.2, -0.4], [0.2, 0.4]]}},
        "room": {"width": 12.0, "height": 7.0},
        "target": {"position": [8.0, 0.55], "size": [0.8, 0.9]},
        "waypoints": [],
        "action_names": ["joint"],
    }
    path = tmp_path / "trajectory.npz"
    np.savez_compressed(
        path,
        metadata_json=np.asarray(json.dumps(metadata)),
        body_positions=np.asarray([[[1.0, 1.0]], [[2.0, 1.1]]], dtype=np.float32),
        body_angles=np.asarray([[0.0], [0.1]], dtype=np.float32),
        actions=np.asarray([[0.2], [0.4]], dtype=np.float32),
        rewards=np.asarray([1.0, 2.5], dtype=np.float32),
        cumulative_rewards=np.asarray([1.0, 3.5], dtype=np.float32),
        distances=np.asarray([7.0, 6.0], dtype=np.float32),
        torso_heights=np.asarray([1.0, 1.1], dtype=np.float32),
        torso_x_positions=np.asarray([1.0, 2.0], dtype=np.float32),
        waypoint_indices=np.asarray([0, 0], dtype=np.int16),
        navigation_distances=np.asarray([7.0, 6.0], dtype=np.float32),
        goal_hold_counts=np.asarray([0, 0], dtype=np.int16),
        successes=np.asarray([False, True], dtype=np.bool_),
    )

    payload = load_trajectory_payload(path)

    assert payload["frame_count"] == 2
    assert payload["summary"]["success"] is True
    assert payload["summary"]["total_reward"] == 3.5
    assert payload["summary"]["max_torso_x"] == 2.0
    assert payload["body_positions"][1][0] == [2.0, 1.1]


def test_lab_api_contract_serves_manifest_and_verified_trajectory() -> None:
    health_payload = health()
    assert health_payload["ok"] is True

    manifest_payload = experiments()
    experiment_items = manifest_payload["experiments"]
    assert len(experiment_items) == 8
    assert all(experiment["available"] for experiment in experiment_items)

    trajectory_payload = trajectory("full-recommended")
    assert trajectory_payload["frame_count"] == 787
    assert trajectory_payload["summary"]["success"] is True
    assert trajectory_payload["metadata"]["format_version"] == 2
