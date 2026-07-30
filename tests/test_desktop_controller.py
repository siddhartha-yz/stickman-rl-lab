from __future__ import annotations

import time
from pathlib import Path

from stickman_rl.desktop.controller import DesktopTrainingController, TrainingRequest


def test_desktop_controller_streams_real_training_events(tmp_path: Path) -> None:
    controller = DesktopTrainingController(runs_root=tmp_path)
    run_id = controller.start(
        TrainingRequest(
            stage=1,
            timesteps=64,
            seed=123,
            train_config="configs/train_smoke.yaml",
        )
    )

    event_types: set[str] = set()
    deadline = time.monotonic() + 45.0
    while time.monotonic() < deadline and controller.is_active:
        event_types.update(event["type"] for event in controller.drain_events())
        time.sleep(0.05)
    exit_code = controller.wait(timeout=10.0)
    event_types.update(event["type"] for event in controller.drain_events())

    snapshot = controller.snapshot()
    assert exit_code == 0
    assert run_id.startswith("desktop-")
    assert {"metadata", "frame", "status", "metrics"}.issubset(event_types)
    assert snapshot["status"]["state"] == "completed"
    assert snapshot["status"]["num_timesteps"] == 64
    assert Path(snapshot["status"]["final_checkpoint"]).is_file()
    assert len(snapshot["frame"]["metadata"]["body_names"]) == 10
    assert len(snapshot["frame"]["frame"]["body_positions"]) == 10


def test_training_request_rejects_non_training_config() -> None:
    request = TrainingRequest(train_config="configs/stage1.yaml")
    try:
        request.validated()
    except ValueError as exc:
        assert "must start with train" in str(exc)
    else:
        raise AssertionError("Expected a non-training config to be rejected")
