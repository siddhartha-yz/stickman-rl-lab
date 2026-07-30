from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from stickman_rl.config import PROJECT_ROOT, load_env_config, load_train_config
from stickman_rl.env import StickmanReachEnv

EVENT_PREFIX = "STICKMAN_EVENT\t"
METRIC_HISTORY_LIMIT = 500


def emit_stdout_event(event_type: str, payload: Any, *, enabled: bool) -> bool:
    if not enabled:
        return False
    event = {"type": event_type, "payload": json_safe(payload)}
    try:
        print(
            f"{EVENT_PREFIX}{json.dumps(event, ensure_ascii=False, separators=(',', ':'))}",
            flush=True,
        )
    except (OSError, ValueError):
        return False
    return True


def replace_with_retry(temporary: Path, destination: Path) -> None:
    for attempt in range(20):
        try:
            temporary.replace(destination)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.005)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    replace_with_retry(temporary, path)


def atomic_json_compact(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    replace_with_retry(temporary, path)


def read_json_with_retry(path: Path, default: Any = None) -> Any:
    for attempt in range(20):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, PermissionError, UnicodeDecodeError, json.JSONDecodeError):
            if attempt == 19:
                return default
            time.sleep(0.005)
    return default


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (OverflowError, TypeError, ValueError):
        return default
    return parsed if np.isfinite(parsed) else default


def nonnegative_int(value: Any, default: int = 0) -> int:
    try:
        parsed = int(value)
    except (OverflowError, TypeError, ValueError):
        return default
    return max(0, parsed)


def first_sequence_item(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.reshape(-1)[0] if value.size else None
    if isinstance(value, list | tuple):
        return value[0] if value else None
    return None


def finite_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool | np.bool_):
        return bool(value)
    if isinstance(value, int | float | np.number):
        parsed = finite_float(value, float("nan"))
        return bool(parsed) if np.isfinite(parsed) else default
    return default


def optional_finite_float(value: Any) -> float | None:
    if isinstance(value, np.ndarray):
        if value.size != 1:
            return None
        value = value.reshape(-1)[0]
    parsed = finite_float(value, float("nan"))
    return parsed if np.isfinite(parsed) else None


def json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def monitored_env(stage: int, seed: int, env_config: str | None):
    def factory() -> Monitor:
        env = StickmanReachEnv(config=load_env_config(stage=stage, config_path=env_config))
        return Monitor(
            env,
            info_keywords=("is_success", "final_distance", "mean_energy", "torso_height"),
        )

    return factory


class LiveTrainingCallback(BaseCallback):
    def __init__(
        self,
        run_dir: Path,
        total_timesteps: int,
        stream_stdout: bool = False,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.run_dir = run_dir
        self.total_timesteps = total_timesteps
        self.control_path = run_dir / "control.json"
        self.status_path = run_dir / "status.json"
        self.frame_path = run_dir / "frame.json"
        self.metadata_path = run_dir / "metadata.json"
        self.metrics_path = run_dir / "metrics.json"
        self.stream_stdout = stream_stdout
        self.last_disk_frame_write = 0.0
        self.frame_snapshot_retry_after = 0.0
        self.frame_snapshot_error: str | None = None
        self.metrics_snapshot_retry_after = 0.0
        self.metrics_snapshot_error: str | None = None
        self.metadata_snapshot_retry_after = 0.0
        self.metadata_snapshot_error: str | None = None
        self.status_snapshot_retry_after = 0.0
        self.status_snapshot_error: str | None = None
        self.cached_metadata: Any | None = None
        self.started_monotonic = time.monotonic()
        self.started_at = datetime.now().isoformat(timespec="seconds")
        self.last_frame_write = 0.0
        self.last_status_write = 0.0
        self.last_save_request: str | None = None
        self.last_control: dict[str, Any] = {
            "paused": False,
            "stop": False,
            "save_request": None,
        }
        self.episode_index = 0
        self.episode_step = 0
        self.current_episode_reward = 0.0
        self.recent_rewards: deque[float] = deque(maxlen=50)
        self.recent_successes: deque[float] = deque(maxlen=50)
        self.recent_distances: deque[float] = deque(maxlen=50)
        self.episodes: deque[dict[str, Any]] = deque(maxlen=METRIC_HISTORY_LIMIT)
        self.updates: deque[dict[str, Any]] = deque(maxlen=METRIC_HISTORY_LIMIT)
        self.last_info: dict[str, Any] = {}
        self.state = "starting"
        self.stop_requested = False

    def _control(self) -> dict[str, Any]:
        payload = read_json_with_retry(self.control_path, self.last_control)
        if not isinstance(payload, dict):
            return dict(self.last_control)
        self.last_control = {
            "paused": bool(payload.get("paused", False)),
            "stop": bool(payload.get("stop", False)),
            "save_request": payload.get("save_request"),
        }
        return dict(self.last_control)

    def _logger_metrics(self) -> dict[str, float | None]:
        values = getattr(self.model.logger, "name_to_value", {})
        names = {
            "policy_loss": "train/policy_gradient_loss",
            "value_loss": "train/value_loss",
            "entropy_loss": "train/entropy_loss",
            "approx_kl": "train/approx_kl",
            "clip_fraction": "train/clip_fraction",
            "explained_variance": "train/explained_variance",
            "learning_rate": "train/learning_rate",
        }
        result: dict[str, float | None] = {}
        for output_name, logger_name in names.items():
            result[output_name] = optional_finite_float(values.get(logger_name))
        return result

    def _status_payload(self, state: str | None = None) -> dict[str, Any]:
        elapsed = max(time.monotonic() - self.started_monotonic, 1e-6)
        rolling_reward = float(np.mean(self.recent_rewards)) if self.recent_rewards else None
        rolling_success = float(np.mean(self.recent_successes)) if self.recent_successes else None
        rolling_distance = float(np.mean(self.recent_distances)) if self.recent_distances else None
        return {
            "state": state or self.state,
            "pid": os.getpid(),
            "started_at": self.started_at,
            "updated_at": datetime.now().isoformat(timespec="milliseconds"),
            "elapsed_seconds": elapsed,
            "total_timesteps": self.total_timesteps,
            "num_timesteps": int(self.num_timesteps),
            "progress": min(1.0, self.num_timesteps / max(1, self.total_timesteps)),
            "fps": float(self.num_timesteps / elapsed),
            "episode": self.episode_index + 1,
            "episode_step": self.episode_step,
            "current_episode_reward": self.current_episode_reward,
            "completed_episodes": self.episode_index,
            "rolling_reward": rolling_reward,
            "rolling_success_rate": rolling_success,
            "rolling_final_distance": rolling_distance,
            "last_info": json_safe(self.last_info),
            "losses": self._logger_metrics(),
            "from_scratch": True,
            "event_transport": "stdout" if self.stream_stdout else "disk-only",
            "frame_snapshot_error": self.frame_snapshot_error,
            "metrics_snapshot_error": self.metrics_snapshot_error,
            "metadata_snapshot_error": self.metadata_snapshot_error,
            "status_snapshot_error": self.status_snapshot_error,
        }

    def _write_status(self, state: str | None = None, force_disk: bool = False) -> None:
        now = time.monotonic()
        new_error: str | None = None
        if force_disk or now >= self.status_snapshot_retry_after:
            try:
                atomic_json(self.status_path, self._status_payload(state))
            except OSError as exc:
                new_error = f"{type(exc).__name__}: {exc}"
                self.status_snapshot_error = new_error
                self.status_snapshot_retry_after = now + 1.0
            else:
                self.status_snapshot_error = None
                self.status_snapshot_retry_after = 0.0
        payload = self._status_payload(state)
        self.stream_stdout = emit_stdout_event("status", payload, enabled=self.stream_stdout)
        if new_error is not None:
            self.stream_stdout = emit_stdout_event(
                "log",
                {
                    "stream": "worker",
                    "text": f"status snapshot write failed; retrying later: {new_error}",
                },
                enabled=self.stream_stdout,
            )
        self.last_status_write = now

    def _write_metrics(self, force_disk: bool = False) -> None:
        payload = {
            "episodes": list(self.episodes),
            "updates": list(self.updates),
        }
        self.stream_stdout = emit_stdout_event("metrics", payload, enabled=self.stream_stdout)
        now = time.monotonic()
        if not force_disk and now < self.metrics_snapshot_retry_after:
            return
        try:
            atomic_json(self.metrics_path, payload)
        except OSError as exc:
            self.metrics_snapshot_error = f"{type(exc).__name__}: {exc}"
            self.metrics_snapshot_retry_after = now + 5.0
            self.stream_stdout = emit_stdout_event(
                "log",
                {
                    "stream": "worker",
                    "text": f"metrics snapshot write failed; retrying later: {self.metrics_snapshot_error}",
                },
                enabled=self.stream_stdout,
            )
        else:
            self.metrics_snapshot_retry_after = 0.0
            self.metrics_snapshot_error = None

    def _write_frame(self, force_disk: bool = False) -> None:
        needs_metadata = self.cached_metadata is None and not self.metadata_path.exists()
        snapshot = self.training_env.env_method(
            "live_snapshot", include_metadata=needs_metadata
        )[0]
        if needs_metadata:
            self.cached_metadata = json_safe(snapshot.pop("metadata"))
            self.stream_stdout = emit_stdout_event(
                "metadata", self.cached_metadata, enabled=self.stream_stdout
            )
        actions = np.asarray(self.locals.get("actions", np.zeros((1, 8), dtype=np.float32)))
        action = actions[0].astype(float).tolist() if actions.ndim > 1 else actions.astype(float).tolist()
        payload = json_safe(
            {
                **snapshot,
                "training": {
                    "num_timesteps": int(self.num_timesteps),
                    "episode": self.episode_index + 1,
                    "episode_step": self.episode_step,
                    "current_episode_reward": self.current_episode_reward,
                    "action": action,
                    "updated_at": datetime.now().isoformat(timespec="milliseconds"),
                },
            }
        )
        self.stream_stdout = emit_stdout_event("frame", payload, enabled=self.stream_stdout)
        now = time.monotonic()
        metadata_write_due = (
            self.cached_metadata is not None
            and not self.metadata_path.exists()
            and (force_disk or now >= self.metadata_snapshot_retry_after)
        )
        if metadata_write_due:
            try:
                atomic_json(self.metadata_path, self.cached_metadata)
            except OSError as exc:
                self.metadata_snapshot_error = f"{type(exc).__name__}: {exc}"
                self.metadata_snapshot_retry_after = now + 5.0
                self.stream_stdout = emit_stdout_event(
                    "log",
                    {
                        "stream": "worker",
                        "text": f"metadata snapshot write failed; retrying later: {self.metadata_snapshot_error}",
                    },
                    enabled=self.stream_stdout,
                )
            else:
                self.metadata_snapshot_retry_after = 0.0
                self.metadata_snapshot_error = None
        disk_write_due = force_disk or (
            now >= self.frame_snapshot_retry_after
            and now - self.last_disk_frame_write >= 0.2
        )
        if disk_write_due:
            try:
                atomic_json_compact(self.frame_path, payload)
            except OSError as exc:
                self.frame_snapshot_error = f"{type(exc).__name__}: {exc}"
                self.frame_snapshot_retry_after = now + 5.0
                self.stream_stdout = emit_stdout_event(
                    "log",
                    {
                        "stream": "worker",
                        "text": f"frame snapshot write failed; retrying later: {self.frame_snapshot_error}",
                    },
                    enabled=self.stream_stdout,
                )
            else:
                self.last_disk_frame_write = now
                self.frame_snapshot_retry_after = 0.0
                self.frame_snapshot_error = None
        self.last_frame_write = now

    def _manual_save(self, request_id: str) -> dict[str, Any]:
        save_path = self.run_dir / "checkpoints" / f"manual-{self.num_timesteps}"
        save_payload: dict[str, Any] = {
            "request_id": request_id,
            "num_timesteps": int(self.num_timesteps),
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        }
        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(str(save_path))
        except Exception as exc:  # noqa: BLE001 - a manual save must not terminate training
            save_payload.update(
                {
                    "state": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        else:
            save_payload.update(
                {
                    "state": "saved",
                    "path": str(save_path.with_suffix(".zip")),
                }
            )
        try:
            atomic_json(self.run_dir / "last_save.json", save_payload)
        except OSError as exc:
            save_payload["metadata_error"] = f"{type(exc).__name__}: {exc}"
        self.stream_stdout = emit_stdout_event(
            "checkpoint", save_payload, enabled=self.stream_stdout
        )
        return save_payload

    def _handle_control(self) -> bool:
        control = self._control()
        request_id = control.get("save_request")
        if request_id and request_id != self.last_save_request:
            self._manual_save(str(request_id))
            self.last_save_request = str(request_id)
        if control.get("stop"):
            self.state = "stopping"
            self.stop_requested = True
            self._write_status(force_disk=True)
            return False
        while control.get("paused"):
            self.state = "paused"
            self._write_status()
            time.sleep(0.15)
            control = self._control()
            request_id = control.get("save_request")
            if request_id and request_id != self.last_save_request:
                self._manual_save(str(request_id))
                self.last_save_request = str(request_id)
            if control.get("stop"):
                self.state = "stopping"
                self.stop_requested = True
                self._write_status(force_disk=True)
                return False
        self.state = "running"
        return True

    def _on_training_start(self) -> None:
        self.state = "running"
        self._write_frame(force_disk=True)
        self._write_status(force_disk=True)
        self._write_metrics(force_disk=True)

    def _on_step(self) -> bool:
        reward = finite_float(first_sequence_item(self.locals.get("rewards", [])))
        raw_infos = self.locals.get("infos", [])
        infos = raw_infos if isinstance(raw_infos, list | tuple) else []
        info = dict(infos[0]) if infos and isinstance(infos[0], dict) else {}
        done = finite_bool(first_sequence_item(self.locals.get("dones", [])))
        self.current_episode_reward += reward
        self.episode_step += 1
        if info:
            self.last_info = info
        if done:
            raw_episode_info = info.get("episode", {})
            episode_info = raw_episode_info if isinstance(raw_episode_info, dict) else {}
            episode_reward = finite_float(
                episode_info.get("r"), self.current_episode_reward
            )
            episode_length = nonnegative_int(
                episode_info.get("l"), self.episode_step
            )
            success = bool(info.get("is_success", False))
            final_distance = finite_float(
                info.get("final_distance", info.get("distance")), 0.0
            )
            self.recent_rewards.append(episode_reward)
            self.recent_successes.append(float(success))
            self.recent_distances.append(final_distance)
            self.episodes.append(
                {
                    "episode": self.episode_index + 1,
                    "timesteps": int(self.num_timesteps),
                    "reward": episode_reward,
                    "length": episode_length,
                    "success": success,
                    "final_distance": final_distance,
                }
            )
            self.episode_index += 1
            self.episode_step = 0
            self.current_episode_reward = 0.0
            self._write_metrics()

        now = time.monotonic()
        if now - self.last_frame_write >= 1.0 / 60.0:
            self._write_frame()
        if now - self.last_status_write >= 0.25:
            self._write_status()
        return self._handle_control()

    def _on_rollout_end(self) -> None:
        self.updates.append(
            {
                "timesteps": int(self.num_timesteps),
                **self._logger_metrics(),
            }
        )
        self._write_metrics()
        self._write_status()

    def _on_training_end(self) -> None:
        self._write_frame(force_disk=True)
        self._write_metrics(force_disk=True)
        self._write_status(
            "stopped" if self.stop_requested else "saving", force_disk=True
        )


def run(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir).resolve() if args.run_dir else PROJECT_ROOT / "lab" / "runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "status.json"
    request = {
        "run_id": args.run_id,
        "stage": args.stage,
        "timesteps": args.timesteps,
        "seed": args.seed,
        "train_config": args.train_config,
        "env_config": args.env_config,
        "from_scratch": True,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    atomic_json(run_dir / "request.json", request)
    if not (run_dir / "control.json").exists():
        atomic_json(run_dir / "control.json", {"paused": False, "stop": False, "save_request": None})
    atomic_json(status_path, {"state": "starting", **request, "pid": os.getpid(), "num_timesteps": 0})

    train_cfg = load_train_config(args.train_config)
    actual_seed = int(args.seed)
    env = make_vec_env(monitored_env(args.stage, actual_seed, args.env_config), n_envs=1, seed=actual_seed)
    try:
        policy_kwargs = {
            "net_arch": list(train_cfg.get("policy_layers", [256, 256])),
            "log_std_init": float(train_cfg.get("log_std_init", 0.0)),
        }
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=float(train_cfg["learning_rate"]),
            n_steps=int(train_cfg["n_steps"]),
            batch_size=int(train_cfg["batch_size"]),
            n_epochs=int(train_cfg["n_epochs"]),
            gamma=float(train_cfg["gamma"]),
            gae_lambda=float(train_cfg["gae_lambda"]),
            clip_range=float(train_cfg["clip_range"]),
            ent_coef=float(train_cfg["ent_coef"]),
            vf_coef=float(train_cfg["vf_coef"]),
            max_grad_norm=float(train_cfg["max_grad_norm"]),
            target_kl=float(train_cfg["target_kl"]) if "target_kl" in train_cfg else None,
            policy_kwargs=policy_kwargs,
            verbose=0,
            tensorboard_log=str(run_dir / "tensorboard"),
            seed=actual_seed,
            device="auto",
        )
        live_callback = LiveTrainingCallback(
            run_dir=run_dir,
            total_timesteps=args.timesteps,
            stream_stdout=args.stream_stdout,
        )
        checkpoint_callback = CheckpointCallback(
            save_freq=max(1, int(train_cfg.get("checkpoint_freq", 5000))),
            save_path=str(run_dir / "checkpoints"),
            name_prefix="ppo_live",
        )
        model.learn(
            total_timesteps=args.timesteps,
            callback=CallbackList([live_callback, checkpoint_callback]),
            reset_num_timesteps=True,
            progress_bar=False,
        )
        final_name = "stopped_model" if live_callback.stop_requested else "final_model"
        final_path = run_dir / "checkpoints" / final_name
        model.save(str(final_path))
        final_status = live_callback._status_payload(
            "stopped" if live_callback.stop_requested else "completed"
        )
        final_status["final_checkpoint"] = str(final_path.with_suffix(".zip"))
        atomic_json(status_path, final_status)
        emit_stdout_event("status", final_status, enabled=args.stream_stdout)
    finally:
        env.close()


def build_failure_status(
    run_dir: Path,
    exc: Exception,
    traceback_text: str,
) -> dict[str, Any]:
    loaded = read_json_with_retry(run_dir / "status.json", {})
    previous: dict[str, Any] = loaded if isinstance(loaded, dict) else {}
    try:
        num_timesteps = int(previous.get("num_timesteps", 0) or 0)
    except (TypeError, ValueError):
        num_timesteps = 0
    return {
        **previous,
        "state": "failed",
        "error": str(exc),
        "traceback": traceback_text,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "pid": os.getpid(),
        "num_timesteps": num_timesteps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one UI-controlled PPO training process.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stage", type=int, required=True, choices=range(0, 6))
    parser.add_argument("--timesteps", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--train-config", type=str, default=None)
    parser.add_argument("--env-config", type=str, default=None)
    parser.add_argument("--stream-stdout", action="store_true")
    parser.add_argument("--run-dir", type=str, default=None)
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve() if args.run_dir else PROJECT_ROOT / "lab" / "runs" / args.run_id
    try:
        run(args)
    except Exception as exc:  # noqa: BLE001 - worker must persist failures for the UI
        failure_payload = build_failure_status(run_dir, exc, traceback.format_exc())
        atomic_json(run_dir / "status.json", failure_payload)
        emit_stdout_event("status", failure_payload, enabled=args.stream_stdout)
        raise


if __name__ == "__main__":
    main()
