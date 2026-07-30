from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

import scripts.live_train_worker as worker_module
from scripts.live_train_worker import LiveTrainingCallback, build_failure_status, emit_stdout_event


def test_emit_stdout_event_returns_false_for_broken_pipe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenStream:
        def write(self, _text: str) -> int:
            raise OSError(22, "simulated closed pipe")

        def flush(self) -> None:
            raise OSError(22, "simulated closed pipe")

    monkeypatch.setattr(sys, "stdout", BrokenStream())

    assert not emit_stdout_event("status", {"state": "running"}, enabled=True)


def test_callback_disables_stdout_after_emit_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    callback = LiveTrainingCallback(run_dir=tmp_path, total_timesteps=64, stream_stdout=True)
    monkeypatch.setattr(worker_module, "atomic_json", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(worker_module, "emit_stdout_event", lambda *_args, **_kwargs: False)

    callback._write_metrics()

    assert not callback.stream_stdout


def test_frame_snapshot_write_failure_backs_off_without_stopping_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeVecEnv:
        def env_method(self, name: str, include_metadata: bool = False) -> list[dict[str, object]]:
            assert name == "live_snapshot"
            snapshot: dict[str, object] = {
                "frame": {
                    "body_positions": [[1.0, 1.0]],
                    "body_angles": [0.0],
                    "info": {},
                }
            }
            if include_metadata:
                snapshot["metadata"] = {"body_names": ["torso"]}
            return [snapshot]

    class FakeModel:
        def __init__(self) -> None:
            self.env = FakeVecEnv()

        def get_env(self) -> FakeVecEnv:
            return self.env

    (tmp_path / "metadata.json").write_text("{}", encoding="utf-8")
    callback = LiveTrainingCallback(
        run_dir=tmp_path,
        total_timesteps=64,
        stream_stdout=True,
    )
    callback.model = FakeModel()  # type: ignore[assignment]
    callback.locals = {"actions": [[0.0] * 8]}
    callback.num_timesteps = 16
    writes = 0
    emitted: list[str] = []

    def flaky_frame_write(_path: Path, _payload: object) -> None:
        nonlocal writes
        writes += 1
        if writes == 1:
            raise PermissionError("simulated frame lock")

    def capture_event(event_type: str, _payload: object, *, enabled: bool) -> bool:
        if enabled:
            emitted.append(event_type)
        return enabled

    monkeypatch.setattr(worker_module, "atomic_json_compact", flaky_frame_write)
    monkeypatch.setattr(worker_module, "emit_stdout_event", capture_event)

    callback._write_frame(force_disk=True)

    assert writes == 1
    assert callback.frame_snapshot_error == "PermissionError: simulated frame lock"
    assert callback.frame_snapshot_retry_after > 0.0
    assert emitted[-1] == "log"

    callback._write_frame(force_disk=True)

    assert writes == 2
    assert callback.frame_snapshot_error is None
    assert callback.frame_snapshot_retry_after == 0.0
    assert callback.last_disk_frame_write > 0.0


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


def test_control_read_retains_last_valid_state_for_non_utf8_bytes(tmp_path: Path) -> None:
    callback = LiveTrainingCallback(run_dir=tmp_path, total_timesteps=64)
    callback.last_control = {"paused": True, "stop": False, "save_request": "save-utf8"}
    callback.control_path.write_bytes(b"\xff\xfe\xfa")

    assert callback._control() == callback.last_control


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


def test_manual_save_failure_is_reported_without_stopping_training(tmp_path: Path) -> None:
    class FailingModel:
        def save(self, _path: str) -> None:
            raise PermissionError("simulated checkpoint directory lock")

    callback = LiveTrainingCallback(run_dir=tmp_path, total_timesteps=5000)
    callback.model = FailingModel()  # type: ignore[assignment]
    callback.num_timesteps = 320
    callback.control_path.write_text(
        json.dumps({"paused": False, "stop": False, "save_request": "save-request-1"}),
        encoding="utf-8",
    )

    assert callback._handle_control()

    payload = json.loads((tmp_path / "last_save.json").read_text(encoding="utf-8"))
    assert callback.state == "running"
    assert callback.last_save_request == "save-request-1"
    assert payload["state"] == "failed"
    assert payload["num_timesteps"] == 320
    assert payload["request_id"] == "save-request-1"
    assert "simulated checkpoint directory lock" in payload["error"]


def test_metric_history_is_bounded_in_memory_and_on_disk(tmp_path: Path) -> None:
    callback = LiveTrainingCallback(run_dir=tmp_path, total_timesteps=1_000_000)
    for index in range(20_000):
        callback.episodes.append({"episode": index})
        callback.updates.append({"update": index})

    callback._write_metrics()

    payload = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert len(callback.episodes) == 500
    assert len(callback.updates) == 500
    assert payload["episodes"][0]["episode"] == 19_500
    assert payload["episodes"][-1]["episode"] == 19_999
    assert payload["updates"][0]["update"] == 19_500
    assert payload["updates"][-1]["update"] == 19_999


def test_run_closes_environment_when_ppo_construction_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeEnv:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    fake_env = FakeEnv()
    monkeypatch.setattr(worker_module, "make_vec_env", lambda *args, **kwargs: fake_env)
    monkeypatch.setattr(
        worker_module,
        "load_train_config",
        lambda _path: {
            "policy_layers": [8, 8],
            "learning_rate": 0.0003,
            "n_steps": 8,
            "batch_size": 4,
            "n_epochs": 1,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.0,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
        },
    )

    def fail_ppo(*args: object, **kwargs: object) -> None:
        raise RuntimeError("constructor failure")

    monkeypatch.setattr(worker_module, "PPO", fail_ppo)
    args = argparse.Namespace(
        run_id="resource-failure",
        run_dir=str(tmp_path / "run"),
        stage=1,
        timesteps=64,
        seed=1,
        train_config="configs/train_smoke.yaml",
        env_config=None,
        stream_stdout=False,
    )

    with pytest.raises(RuntimeError, match="constructor failure"):
        worker_module.run(args)

    assert fake_env.closed


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


def test_failure_status_retries_transient_status_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status_path = tmp_path / "status.json"
    status_path.write_text(
        json.dumps({"state": "running", "num_timesteps": 512, "episode": 4}),
        encoding="utf-8",
    )
    original_read_text = Path.read_text
    attempts = 0

    def flaky_read_text(path: Path, *args: object, **kwargs: object) -> str:
        nonlocal attempts
        if path == status_path and attempts < 2:
            attempts += 1
            raise PermissionError("temporary sharing lock")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", flaky_read_text)

    payload = build_failure_status(tmp_path, RuntimeError("boom"), "traceback text")

    assert attempts == 2
    assert payload["num_timesteps"] == 512
    assert payload["episode"] == 4


def test_failure_status_defaults_to_zero_without_valid_status(tmp_path: Path) -> None:
    (tmp_path / "status.json").write_text("not json", encoding="utf-8")

    payload = build_failure_status(tmp_path, RuntimeError("boom"), "traceback text")

    assert payload["state"] == "failed"
    assert payload["num_timesteps"] == 0
