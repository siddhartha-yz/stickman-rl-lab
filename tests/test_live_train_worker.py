from __future__ import annotations

from pathlib import Path

import pytest

from scripts.live_train_worker import LiveTrainingCallback


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
