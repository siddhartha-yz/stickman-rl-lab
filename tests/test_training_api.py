from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import HTTPException

from scripts import training_api


def test_training_options_expose_live_from_scratch_defaults() -> None:
    payload = training_api.training_options()
    assert payload["defaults"]["stage"] == 1
    assert payload["defaults"]["train_config"] == "configs/train_live.yaml"
    assert "configs/train_live.yaml" in payload["train_configs"]
    assert any(stage["value"] == 3 for stage in payload["stages"])


def test_training_control_updates_active_run_files(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(training_api, "RUNS_ROOT", tmp_path)
    run_id = "ui-test-run"
    run_dir = tmp_path / run_id
    run_dir.mkdir()
    (run_dir / "status.json").write_text(
        json.dumps({"state": "running", "num_timesteps": 12}), encoding="utf-8"
    )
    (run_dir / "control.json").write_text(
        json.dumps({"paused": False, "stop": False, "save_request": None}),
        encoding="utf-8",
    )

    training_api._control(run_id, "pause")
    paused = json.loads((run_dir / "control.json").read_text(encoding="utf-8"))
    assert paused["paused"] is True

    training_api._control(run_id, "resume")
    resumed = json.loads((run_dir / "control.json").read_text(encoding="utf-8"))
    assert resumed["paused"] is False

    training_api._control(run_id, "save")
    saved = json.loads((run_dir / "control.json").read_text(encoding="utf-8"))
    assert saved["save_request"]

    training_api._control(run_id, "stop")
    stopped = json.loads((run_dir / "control.json").read_text(encoding="utf-8"))
    assert stopped["stop"] is True

    (run_dir / "status.json").write_text(json.dumps({"state": "completed"}), encoding="utf-8")
    with pytest.raises(HTTPException) as error:
        training_api._control(run_id, "save")
    assert error.value.status_code == 409
