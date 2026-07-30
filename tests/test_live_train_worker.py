from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

import scripts.live_train_worker as worker_module
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


def test_failure_status_defaults_to_zero_without_valid_status(tmp_path: Path) -> None:
    (tmp_path / "status.json").write_text("not json", encoding="utf-8")

    payload = build_failure_status(tmp_path, RuntimeError("boom"), "traceback text")

    assert payload["state"] == "failed"
    assert payload["num_timesteps"] == 0
