from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

import scripts.live_train_worker as worker_module
from scripts.live_train_worker import (
    LiveTrainingCallback,
    build_failure_status,
    emit_stdout_event,
    json_safe,
)


def test_atomic_json_retries_transient_temporary_write_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "status.json"
    original_write_text = Path.write_text
    locked_attempts = 0

    def flaky_write_text(path: Path, *args: object, **kwargs: object) -> int:
        nonlocal locked_attempts
        if path.name == "status.json.tmp" and locked_attempts < 2:
            locked_attempts += 1
            raise PermissionError("simulated temporary file lock")
        return original_write_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", flaky_write_text)

    worker_module.atomic_json(destination, {"state": "running"})

    assert locked_attempts == 2
    assert json.loads(destination.read_text(encoding="utf-8")) == {"state": "running"}


def test_json_safe_sanitizes_numpy_nonfinite_values() -> None:
    assert json_safe(worker_module.np.float32(worker_module.np.inf)) is None
    assert json_safe(
        worker_module.np.array([1.0, worker_module.np.nan, worker_module.np.inf])
    ) == [1.0, None, None]


def test_json_safe_handles_recursive_containers() -> None:
    recursive_dict: dict[str, object] = {}
    recursive_dict["self"] = recursive_dict
    recursive_list: list[object] = []
    recursive_list.append(recursive_list)
    shared = {"value": 1}

    assert json_safe(recursive_dict) == {"self": "<recursive-reference>"}
    assert json_safe(recursive_list) == ["<recursive-reference>"]
    assert json_safe([shared, shared]) == [{"value": 1}, {"value": 1}]


def test_json_safe_handles_unprintable_keys_and_excessive_depth(tmp_path: Path) -> None:
    class BadKey:
        def __str__(self) -> str:
            raise RuntimeError("simulated key stringification failure")

    nested: list[object] = []
    current = nested
    for _ in range(worker_module.JSON_SAFE_MAX_DEPTH + 10):
        child: list[object] = []
        current.append(child)
        current = child

    safe = json_safe({BadKey(): 1, "nested": nested})
    encoded = json.dumps(safe, ensure_ascii=False, allow_nan=False)

    assert any(key.startswith("<unprintable-key:") for key in safe)
    assert "<max-depth>" in encoded

    class FakeLogger:
        name_to_value: dict[str, object] = {}

    class FakeModel:
        logger = FakeLogger()

    callback = LiveTrainingCallback(run_dir=tmp_path, total_timesteps=64)
    callback.model = FakeModel()  # type: ignore[assignment]
    callback.num_timesteps = 1
    callback.last_info = {BadKey(): nested}

    payload = callback._status_payload()
    json.dumps(payload, ensure_ascii=False, allow_nan=False)
    assert "<max-depth>" in json.dumps(payload["last_info"])


def test_json_safe_converts_unsupported_objects_to_stable_text() -> None:
    class CustomInfo:
        pass

    custom_marker = f"<{CustomInfo.__module__}.{CustomInfo.__qualname__}>"
    payload = json_safe(
        {
            "path": Path("artifact.bin"),
            "bytes": b"abc\xff",
            "custom": CustomInfo(),
        }
    )

    assert payload == {
        "path": "artifact.bin",
        "bytes": "abc�",
        "custom": custom_marker,
    }
    json.dumps(payload, ensure_ascii=False, allow_nan=False)


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


def test_emit_stdout_event_sanitizes_malformed_payloads(
    capsys: pytest.CaptureFixture[str],
) -> None:
    class BadKey:
        def __str__(self) -> str:
            raise RuntimeError("simulated key stringification failure")

    nested: list[object] = []
    current = nested
    for _ in range(2_500):
        child: list[object] = []
        current.append(child)
        current = child

    assert emit_stdout_event("status", {BadKey(): 1}, enabled=True)
    assert emit_stdout_event("frame", nested, enabled=True)
    output = capsys.readouterr().out
    assert "<unprintable-key:" in output
    assert "<max-depth>" in output


def test_logger_metrics_ignore_non_scalar_and_invalid_values(tmp_path: Path) -> None:
    class FakeLogger:
        name_to_value = {
            "train/policy_gradient_loss": worker_module.np.array([1.25]),
            "train/value_loss": worker_module.np.array([1.0, 2.0]),
            "train/entropy_loss": {"bad": 1},
            "train/approx_kl": float("inf"),
        }

    class FakeModel:
        logger = FakeLogger()

    callback = LiveTrainingCallback(run_dir=tmp_path, total_timesteps=64)
    callback.model = FakeModel()  # type: ignore[assignment]

    metrics = callback._logger_metrics()

    assert metrics["policy_loss"] == 1.25
    assert metrics["value_loss"] is None
    assert metrics["entropy_loss"] is None
    assert metrics["approx_kl"] is None


def test_callback_disables_stdout_after_emit_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    callback = LiveTrainingCallback(run_dir=tmp_path, total_timesteps=64, stream_stdout=True)
    monkeypatch.setattr(worker_module, "atomic_json", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(worker_module, "emit_stdout_event", lambda *_args, **_kwargs: False)

    callback._write_metrics()

    assert not callback.stream_stdout


def test_metadata_snapshot_write_failure_caches_and_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    include_metadata_calls: list[bool] = []

    class FakeVecEnv:
        def env_method(self, name: str, include_metadata: bool = False) -> list[dict[str, object]]:
            assert name == "live_snapshot"
            include_metadata_calls.append(include_metadata)
            snapshot: dict[str, object] = {
                "frame": {
                    "body_positions": [[1.0, 1.0]],
                    "body_angles": [0.0],
                    "info": {},
                }
            }
            if include_metadata:
                snapshot["metadata"] = {
                    "body_names": ["torso"],
                    "room": {"width": 12.0, "height": 7.0},
                }
            return [snapshot]

    class FakeModel:
        def __init__(self) -> None:
            self.env = FakeVecEnv()

        def get_env(self) -> FakeVecEnv:
            return self.env

    callback = LiveTrainingCallback(
        run_dir=tmp_path,
        total_timesteps=64,
        stream_stdout=True,
    )
    callback.model = FakeModel()  # type: ignore[assignment]
    callback.locals = {"actions": [[0.0] * 8]}
    callback.num_timesteps = 1
    writes = 0
    emitted: list[str] = []

    def flaky_metadata_write(path: Path, payload: object) -> None:
        nonlocal writes
        assert path == callback.metadata_path
        writes += 1
        if writes == 1:
            raise PermissionError("simulated metadata lock")
        path.write_text(json.dumps(payload), encoding="utf-8")

    def capture_event(event_type: str, _payload: object, *, enabled: bool) -> bool:
        if enabled:
            emitted.append(event_type)
        return enabled

    monkeypatch.setattr(worker_module, "atomic_json", flaky_metadata_write)
    monkeypatch.setattr(worker_module, "atomic_json_compact", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(worker_module, "emit_stdout_event", capture_event)

    callback._write_frame(force_disk=True)

    assert writes == 1
    assert include_metadata_calls == [True]
    assert callback.cached_metadata is not None
    assert callback.metadata_snapshot_error == "PermissionError: simulated metadata lock"
    assert callback.metadata_snapshot_retry_after > 0.0
    assert emitted[:3] == ["metadata", "frame", "log"]

    callback._write_frame(force_disk=True)

    assert writes == 2
    assert include_metadata_calls == [True, False]
    assert callback.metadata_path.is_file()
    assert callback.metadata_snapshot_error is None
    assert callback.metadata_snapshot_retry_after == 0.0
    assert emitted[-1] == "frame"


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


def test_metrics_snapshot_write_failure_backs_off_without_stopping_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    callback = LiveTrainingCallback(
        run_dir=tmp_path,
        total_timesteps=64,
        stream_stdout=True,
    )
    callback.episodes.append({"episode": 1, "reward": 1.0})
    writes = 0
    emitted: list[str] = []

    def flaky_metrics_write(_path: Path, _payload: object) -> None:
        nonlocal writes
        writes += 1
        if writes == 1:
            raise PermissionError("simulated metrics lock")

    def capture_event(event_type: str, _payload: object, *, enabled: bool) -> bool:
        if enabled:
            emitted.append(event_type)
        return enabled

    monkeypatch.setattr(worker_module, "atomic_json", flaky_metrics_write)
    monkeypatch.setattr(worker_module, "emit_stdout_event", capture_event)

    callback._write_metrics(force_disk=True)

    assert writes == 1
    assert emitted[:2] == ["metrics", "log"]
    assert callback.metrics_snapshot_error == "PermissionError: simulated metrics lock"
    assert callback.metrics_snapshot_retry_after > 0.0

    callback._write_metrics(force_disk=True)

    assert writes == 2
    assert emitted[-1] == "metrics"
    assert callback.metrics_snapshot_error is None
    assert callback.metrics_snapshot_retry_after == 0.0


def test_status_snapshot_write_failure_emits_live_state_and_recovers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeLogger:
        name_to_value: dict[str, float] = {}

    class FakeModel:
        logger = FakeLogger()

    callback = LiveTrainingCallback(
        run_dir=tmp_path,
        total_timesteps=64,
        stream_stdout=True,
    )
    callback.model = FakeModel()  # type: ignore[assignment]
    callback.num_timesteps = 8
    writes = 0
    emitted: list[tuple[str, object]] = []

    def flaky_status_write(path: Path, payload: object) -> None:
        nonlocal writes
        assert path == callback.status_path
        writes += 1
        if writes == 1:
            raise PermissionError("simulated persistent status lock")
        path.write_text(json.dumps(payload), encoding="utf-8")

    def capture_event(event_type: str, payload: object, *, enabled: bool) -> bool:
        if enabled:
            emitted.append((event_type, payload))
        return enabled

    monkeypatch.setattr(worker_module, "atomic_json", flaky_status_write)
    monkeypatch.setattr(worker_module, "emit_stdout_event", capture_event)

    callback._write_status(force_disk=True)

    assert writes == 1
    assert callback.status_snapshot_error == (
        "PermissionError: simulated persistent status lock"
    )
    assert callback.status_snapshot_retry_after > 0.0
    assert [event_type for event_type, _ in emitted] == ["status", "log"]
    assert emitted[0][1]["state"] == "starting"  # type: ignore[index]
    assert emitted[0][1]["status_snapshot_error"] == callback.status_snapshot_error  # type: ignore[index]

    callback._write_status(force_disk=True)

    assert writes == 2
    assert callback.status_path.is_file()
    assert callback.status_snapshot_error is None
    assert callback.status_snapshot_retry_after == 0.0
    assert emitted[-1][0] == "status"
    assert emitted[-1][1]["status_snapshot_error"] is None  # type: ignore[index]


def test_on_step_clears_stale_last_info(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    callback = LiveTrainingCallback(run_dir=tmp_path, total_timesteps=64)
    callback.num_timesteps = 2
    callback.last_info = {"is_success": True, "final_distance": 0.1}
    callback.locals = {"rewards": [0.0], "infos": [{}], "dones": [False]}
    monkeypatch.setattr(callback, "_write_metrics", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(callback, "_write_frame", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(callback, "_write_status", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(callback, "_handle_control", lambda: True)

    assert callback._on_step()
    assert callback.last_info == {}


@pytest.mark.parametrize(
    ("rewards", "dones", "expected_reward"),
    [
        ([{"bad": 1}], [False], 0.0),
        ([1.0], {"bad": 1}, 1.0),
    ],
)
def test_on_step_tolerates_malformed_reward_and_done_vectors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rewards: object,
    dones: object,
    expected_reward: float,
) -> None:
    callback = LiveTrainingCallback(run_dir=tmp_path, total_timesteps=64)
    callback.num_timesteps = 1
    callback.locals = {"rewards": rewards, "infos": [{}], "dones": dones}
    monkeypatch.setattr(callback, "_write_metrics", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(callback, "_write_frame", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(callback, "_write_status", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(callback, "_handle_control", lambda: True)

    assert callback._on_step()

    assert callback.current_episode_reward == expected_reward
    assert callback.episode_step == 1
    assert not callback.episodes


@pytest.mark.parametrize(
    ("success_value", "expected_success"),
    [
        ("false", False),
        ("true", True),
        ("unknown", False),
        ({"bad": 1}, False),
    ],
)
def test_on_step_parses_success_flags_strictly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    success_value: object,
    expected_success: bool,
) -> None:
    callback = LiveTrainingCallback(run_dir=tmp_path, total_timesteps=64)
    callback.num_timesteps = 1
    callback.locals = {
        "rewards": [1.0],
        "infos": [
            {
                "episode": {"r": 1.0, "l": 1},
                "is_success": success_value,
                "final_distance": 2.0,
            }
        ],
        "dones": [True],
    }
    monkeypatch.setattr(callback, "_write_metrics", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(callback, "_write_frame", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(callback, "_write_status", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(callback, "_handle_control", lambda: True)

    assert callback._on_step()
    assert callback.episodes[0]["success"] is expected_success


@pytest.mark.parametrize(
    "info_value",
    [
        {"episode": None, "is_success": False, "final_distance": None},
        "legacy-info-row",
    ],
)
def test_on_step_tolerates_malformed_episode_info(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    info_value: object,
) -> None:
    callback = LiveTrainingCallback(run_dir=tmp_path, total_timesteps=64)
    callback.num_timesteps = 1
    callback.locals = {
        "rewards": [1.0],
        "infos": [info_value],
        "dones": [True],
    }
    monkeypatch.setattr(callback, "_write_metrics", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(callback, "_write_frame", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(callback, "_write_status", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(callback, "_handle_control", lambda: True)

    assert callback._on_step()

    assert len(callback.episodes) == 1
    episode = callback.episodes[0]
    assert episode["reward"] == 1.0
    assert episode["length"] == 1
    assert episode["success"] is False
    assert episode["final_distance"] == 0.0


def test_control_parses_boolean_strings_without_python_truthiness(tmp_path: Path) -> None:
    callback = LiveTrainingCallback(run_dir=tmp_path, total_timesteps=64)
    callback.control_path.write_text(
        json.dumps({"paused": "false", "stop": "0", "save_request": None}),
        encoding="utf-8",
    )

    assert callback._control() == {
        "paused": False,
        "stop": False,
        "save_request": None,
    }

    callback.control_path.write_text(
        json.dumps({"paused": "yes", "stop": "on", "save_request": None}),
        encoding="utf-8",
    )
    assert callback._control()["paused"] is True
    assert callback.last_control["stop"] is True

    callback.control_path.write_text(
        json.dumps({"paused": "unknown", "stop": {}, "save_request": None}),
        encoding="utf-8",
    )
    assert callback._control()["paused"] is True
    assert callback.last_control["stop"] is True


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
