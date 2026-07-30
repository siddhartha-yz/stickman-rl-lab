from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

import stickman_rl.desktop.controller as controller_module
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


def test_desktop_controller_pause_save_resume_and_stop(tmp_path: Path) -> None:
    controller = DesktopTrainingController(runs_root=tmp_path)
    controller.start(
        TrainingRequest(
            stage=1,
            timesteps=5_000,
            seed=321,
            train_config="configs/train_smoke.yaml",
        )
    )
    controller.wait_for_state({"running"}, timeout=30.0)

    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        before_pause = int(controller.snapshot().get("status", {}).get("num_timesteps", 0))
        if before_pause >= 16:
            break
        time.sleep(0.05)
    else:
        raise AssertionError("Training did not advance before pause")

    controller.control("pause")
    paused = controller.wait_for_state({"paused"}, timeout=10.0)
    paused_steps = int(paused["num_timesteps"])
    time.sleep(0.4)
    assert int(controller.snapshot()["status"]["num_timesteps"]) == paused_steps

    controller.control("save")
    save_deadline = time.monotonic() + 15.0
    while time.monotonic() < save_deadline:
        last_save = controller.snapshot().get("last_save")
        if last_save:
            break
        time.sleep(0.05)
    else:
        raise AssertionError("Manual checkpoint was not created while paused")
    assert Path(last_save["path"]).is_file()

    controller.control("resume")
    controller.wait_for_state({"running"}, timeout=10.0)
    resume_deadline = time.monotonic() + 10.0
    while time.monotonic() < resume_deadline:
        resumed_steps = int(controller.snapshot()["status"]["num_timesteps"])
        if resumed_steps > paused_steps:
            break
        time.sleep(0.05)
    else:
        raise AssertionError("Training did not resume")

    controller.control("stop")
    assert controller.wait(timeout=15.0) == 0
    stopped = controller.snapshot()["status"]
    assert stopped["state"] == "stopped"
    assert Path(stopped["final_checkpoint"]).is_file()


def test_read_json_retries_transient_permission_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status_path = tmp_path / "status.json"
    status_path.write_text('{"state": "running"}', encoding="utf-8")
    original_read_text = Path.read_text
    attempts = 0

    def flaky_read_text(path: Path, *args: object, **kwargs: object) -> str:
        nonlocal attempts
        if path == status_path and attempts < 2:
            attempts += 1
            raise PermissionError("temporary sharing lock")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", flaky_read_text)

    assert controller_module._read_json(status_path, {}) == {"state": "running"}
    assert attempts == 2


def test_desktop_controller_persists_non_event_worker_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "project"
    (project_root / "configs").mkdir(parents=True)
    (project_root / "configs" / "train_fake.yaml").write_text("placeholder: true\n", encoding="utf-8")
    scripts_dir = project_root / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "live_train_worker.py").write_text(
        "import json\n"
        "import sys\n"
        "print('plain stdout diagnostic', flush=True)\n"
        "print('STICKMAN_EVENT\\t' + json.dumps({'type': 'status', 'payload': {'state': 'completed'}}), flush=True)\n"
        "print('plain stderr diagnostic', file=sys.stderr, flush=True)\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(controller_module, "PROJECT_ROOT", project_root)

    controller = DesktopTrainingController(
        runs_root=tmp_path / "runs",
        python_executable=sys.executable,
    )
    controller.start(
        TrainingRequest(
            stage=1,
            timesteps=64,
            seed=1,
            train_config="configs/train_fake.yaml",
        )
    )
    assert controller.wait(timeout=10.0) == 0

    deadline = time.monotonic() + 5.0
    log_path = controller.run_dir / "worker.log"  # type: ignore[operator]
    while time.monotonic() < deadline:
        text = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
        if "plain stdout diagnostic" in text and "plain stderr diagnostic" in text:
            break
        time.sleep(0.05)
    else:
        raise AssertionError("Worker diagnostics were not persisted")

    assert "controller: started pid=" in text
    assert "controller: process exited with code 0" in text
    assert "STICKMAN_EVENT" not in text
    assert any(event.get("type") == "status" for event in controller.drain_events())


def test_desktop_controller_records_spawn_failure(tmp_path: Path) -> None:
    controller = DesktopTrainingController(
        runs_root=tmp_path,
        python_executable=tmp_path / "missing-python.exe",
    )

    with pytest.raises(RuntimeError, match="Unable to start desktop trainer"):
        controller.start(
            TrainingRequest(
                stage=1,
                timesteps=64,
                seed=2,
                train_config="configs/train_smoke.yaml",
            )
        )

    snapshot = controller.snapshot()
    assert snapshot["status"]["state"] == "failed"
    assert snapshot["status"]["num_timesteps"] == 0
    assert controller.process is None
    assert "Unable to start desktop trainer" in (controller.run_dir / "worker.log").read_text(  # type: ignore[operator]
        encoding="utf-8"
    )


def test_training_request_rejects_non_training_config() -> None:
    request = TrainingRequest(train_config="configs/stage1.yaml")
    try:
        request.validated()
    except ValueError as exc:
        assert "must start with train" in str(exc)
    else:
        raise AssertionError("Expected a non-training config to be rejected")
