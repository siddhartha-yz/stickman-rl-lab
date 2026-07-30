from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.live_train_worker import LiveTrainingCallback, build_failure_status


def test_control_read_retries_transient_permission_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    callback = LiveTrainingCallback(run_dir=tmp_path, total_timesteps=64)
    callback.control_path.write_text(
        '{"paused": true, "stop": false, "save_request": "save-1"}',
        encoding="utf-8",
    )
    original_read_text = Path.read_text
    attempts = 0

    def flaky_read_text(path: Path, *args: object, **kwargs: object) -> str:
        nonlocal attempts
        if path == callback.control_path and attempts < 2:
            attempts += 1
            raise PermissionError("temporary sharing lock")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", flaky_read_text)

    control = callback._control()

    assert attempts == 2
    assert control == {"paused": True, "stop": False, "save_request": "save-1"}


def test_control_read_retains_last_valid_state_after_retry_exhaustion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    callback = LiveTrainingCallback(run_dir=tmp_path, total_timesteps=64)
    callback.last_control = {"paused": True, "stop": False, "save_request": "save-2"}
    original_read_text = Path.read_text

    def locked_read_text(path: Path, *args: object, **kwargs: object) -> str:
        if path == callback.control_path:
            raise PermissionError("persistent sharing lock")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", locked_read_text)

    assert callback._control() == {
        "paused": True,
        "stop": False,
        "save_request": "save-2",
    }


def test_failure_status_preserves_last_durable_progress(tmp_path: Path) -> None:
    (tmp_path / "status.json").write_text(
        json.dumps(
            {
                "state": "running",
                "num_timesteps": 384,
                "episode": 3,
                "rolling_reward": 12.5,
                "total_timesteps": 5000,
            }
        ),
        encoding="utf-8",
    )

    payload = build_failure_status(tmp_path, RuntimeError("boom"), "traceback text")

    assert payload["state"] == "failed"
    assert payload["num_timesteps"] == 384
    assert payload["episode"] == 3
    assert payload["rolling_reward"] == 12.5
    assert payload["total_timesteps"] == 5000
    assert payload["error"] == "boom"
    assert payload["traceback"] == "traceback text"


def test_failure_status_defaults_to_zero_without_valid_status(tmp_path: Path) -> None:
    (tmp_path / "status.json").write_text("not json", encoding="utf-8")

    payload = build_failure_status(tmp_path, RuntimeError("boom"), "traceback text")

    assert payload["state"] == "failed"
    assert payload["num_timesteps"] == 0
