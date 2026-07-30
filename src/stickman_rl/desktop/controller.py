from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from stickman_rl.config import PROJECT_ROOT

EVENT_PREFIX = "STICKMAN_EVENT\t"
ACTIVE_STATES = {"starting", "running", "paused", "saving", "stopping"}
ControlAction = Literal["pause", "resume", "stop", "save"]
LATEST_EVENT_ORDER = ("metadata", "status", "metrics", "frame", "checkpoint")


class DesktopEventBuffer:
    """Coalesce live state and bound diagnostic events while the UI is blocked."""

    def __init__(self, max_pending: int = 500) -> None:
        self._latest: dict[str, dict[str, Any]] = {}
        self._pending: deque[dict[str, Any]] = deque(maxlen=max_pending)
        self._lock = threading.Lock()

    @staticmethod
    def _normalized(event: Any) -> dict[str, Any]:
        if isinstance(event, dict):
            event_type = event.get("type")
            payload = event.get("payload")
            if isinstance(event_type, str) and event_type and isinstance(payload, dict):
                return event
        preview = repr(event)
        if len(preview) > 500:
            preview = preview[:497] + "..."
        return {
            "type": "log",
            "payload": {
                "stream": "controller",
                "text": f"Ignored malformed structured event: {preview}",
            },
        }

    def put(self, event: Any) -> None:
        normalized = self._normalized(event)
        event_type = normalized["type"]
        with self._lock:
            if event_type in LATEST_EVENT_ORDER:
                self._latest[event_type] = normalized
            else:
                self._pending.append(normalized)

    def drain(self, max_items: int = 500) -> list[dict[str, Any]]:
        if max_items <= 0:
            return []
        with self._lock:
            events: list[dict[str, Any]] = []
            for event_type in LATEST_EVENT_ORDER:
                if len(events) >= max_items:
                    break
                event = self._latest.pop(event_type, None)
                if event is not None:
                    events.append(event)
            while self._pending and len(events) < max_items:
                events.append(self._pending.popleft())
            return events

    def __len__(self) -> int:
        with self._lock:
            return len(self._latest) + len(self._pending)


def _replace_with_retry(temporary: Path, destination: Path) -> None:
    for attempt in range(20):
        try:
            temporary.replace(destination)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.005)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _replace_with_retry(temporary, path)


def read_json_file(path: Path, default: Any = None) -> Any:
    for attempt in range(20):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return default
        except (PermissionError, json.JSONDecodeError):
            if attempt == 19:
                return default
            time.sleep(0.005)
    return default


def _validated_config(value: str | None, *, training: bool) -> str | None:
    if value is None or not value.strip():
        return None
    candidate = (PROJECT_ROOT / value).resolve()
    config_root = (PROJECT_ROOT / "configs").resolve()
    try:
        candidate.relative_to(config_root)
    except ValueError as exc:
        raise ValueError("Config must be inside configs/") from exc
    if not candidate.is_file() or candidate.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError(f"Config does not exist: {value}")
    if training and not candidate.name.startswith("train"):
        raise ValueError("Training config must start with train")
    return str(candidate.relative_to(PROJECT_ROOT)).replace("\\", "/")


def training_options() -> dict[str, Any]:
    config_root = PROJECT_ROOT / "configs"
    return {
        "stages": [
            (0, "Stage 0 · 随机物理调试"),
            (1, "Stage 1 · 固定终点，从零学习"),
            (2, "Stage 2 · 随机终点"),
            (3, "Stage 3 · 障碍环境"),
            (4, "Stage 4 · 直立奖励"),
            (5, "Stage 5 · 行走奖励"),
        ],
        "train_configs": sorted(
            str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
            for path in config_root.glob("train*.yaml")
        ),
        "env_configs": sorted(
            str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
            for path in config_root.glob("stage*.yaml")
        ),
    }


@dataclass(frozen=True)
class TrainingRequest:
    stage: int = 1
    timesteps: int = 100_000
    seed: int = 0
    train_config: str = "configs/train_live.yaml"
    env_config: str | None = None

    def validated(self) -> TrainingRequest:
        if not 0 <= self.stage <= 5:
            raise ValueError("stage must be between 0 and 5")
        if not 64 <= self.timesteps <= 10_000_000:
            raise ValueError("timesteps must be between 64 and 10,000,000")
        if not 0 <= self.seed <= 2_147_483_647:
            raise ValueError("seed is outside the supported range")
        return TrainingRequest(
            stage=self.stage,
            timesteps=self.timesteps,
            seed=self.seed,
            train_config=_validated_config(self.train_config, training=True) or "configs/train_live.yaml",
            env_config=_validated_config(self.env_config, training=False),
        )


class DesktopTrainingController:
    """Launch and control one trainer without a web server."""

    def __init__(
        self,
        *,
        runs_root: Path | None = None,
        python_executable: str | Path | None = None,
    ) -> None:
        self.runs_root = (runs_root or PROJECT_ROOT / "lab" / "runs").resolve()
        self.python_executable = str(python_executable or sys.executable)
        self.process: subprocess.Popen[str] | None = None
        self.run_id: str | None = None
        self.run_dir: Path | None = None
        self._events = DesktopEventBuffer()
        self._stderr: deque[str] = deque(maxlen=200)
        self._threads: list[threading.Thread] = []
        self._log_lock = threading.RLock()
        self._handled_exit_processes: set[int] = set()

    @property
    def is_active(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def start(self, request: TrainingRequest) -> str:
        if self.is_active:
            raise RuntimeError("A desktop training process is already active")
        validated = request.validated()
        run_id = f"desktop-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        run_dir = self.runs_root / run_id
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
        except OSError as exc:
            raise RuntimeError(f"Unable to initialize desktop training run: {exc}") from exc
        self.run_id = run_id
        self.run_dir = run_dir
        self.process = None
        event_queue = DesktopEventBuffer()
        stderr_buffer: deque[str] = deque(maxlen=200)
        self._events = event_queue
        self._stderr = stderr_buffer
        self._threads = []
        request_payload = {"run_id": run_id, **asdict(validated), "from_scratch": True}
        try:
            _atomic_json(run_dir / "request.json", request_payload)
            _atomic_json(
                run_dir / "control.json",
                {"paused": False, "stop": False, "save_request": None},
            )
            _atomic_json(
                run_dir / "status.json",
                {"state": "starting", "num_timesteps": 0, **request_payload},
            )
        except OSError as exc:
            message = f"Unable to initialize desktop training run: {exc}"
            self._persist_start_failure(run_dir, request_payload, message)
            raise RuntimeError(message) from exc

        command = [
            self.python_executable,
            str(PROJECT_ROOT / "scripts" / "live_train_worker.py"),
            "--run-id",
            run_id,
            "--run-dir",
            str(run_dir),
            "--stage",
            str(validated.stage),
            "--timesteps",
            str(validated.timesteps),
            "--seed",
            str(validated.seed),
            "--train-config",
            validated.train_config,
            "--stream-stdout",
        ]
        if validated.env_config:
            command.extend(["--env-config", validated.env_config])
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        self._append_worker_log(
            "controller",
            f"launching: {subprocess.list2cmdline(command)}",
            run_dir=run_dir,
        )
        try:
            self.process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=creationflags,
                env=environment,
            )
        except OSError as exc:
            message = f"Unable to start desktop trainer: {exc}"
            self.process = None
            self._persist_start_failure(run_dir, request_payload, message)
            raise RuntimeError(message) from exc
        process = self.process
        self._append_worker_log("controller", f"started pid={process.pid}", run_dir=run_dir)
        stdout_thread = threading.Thread(
            target=self._read_stdout,
            args=(process, run_dir, event_queue),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=self._read_stderr,
            args=(process, run_dir, event_queue, stderr_buffer),
            daemon=True,
        )
        watcher_thread = threading.Thread(
            target=self._watch_process,
            args=(process, run_dir, event_queue, stderr_buffer, (stdout_thread, stderr_thread)),
            daemon=True,
        )
        self._threads = [stdout_thread, stderr_thread, watcher_thread]
        for thread in self._threads:
            thread.start()
        return run_id

    def _persist_start_failure(
        self,
        run_dir: Path,
        request_payload: dict[str, Any],
        message: str,
    ) -> None:
        failure = {
            "state": "failed",
            "error": message,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "num_timesteps": 0,
            **request_payload,
        }
        with suppress(OSError):
            _atomic_json(run_dir / "status.json", failure)
        with suppress(OSError):
            self._append_worker_log("controller", message, run_dir=run_dir)

    def _append_worker_log(
        self,
        stream: str,
        text: str,
        *,
        run_dir: Path | None = None,
    ) -> bool:
        destination = run_dir or self.run_dir
        if destination is None:
            return False
        timestamp = datetime.now().isoformat(timespec="milliseconds")
        try:
            with self._log_lock, (destination / "worker.log").open(
                "a",
                encoding="utf-8",
            ) as handle:
                handle.write(f"[{timestamp}] {stream}: {text}\n")
        except OSError:
            return False
        return True

    def _handle_process_exit(
        self,
        process: subprocess.Popen[str],
        run_dir: Path,
        event_queue: DesktopEventBuffer,
        stderr_buffer: deque[str],
        returncode: int,
    ) -> None:
        process_key = id(process)
        with self._log_lock:
            if process_key in self._handled_exit_processes:
                return
            self._handled_exit_processes.add(process_key)
        self._append_worker_log(
            "controller",
            f"process exited with code {returncode}",
            run_dir=run_dir,
        )
        status_path = run_dir / "status.json"
        status = read_json_file(status_path, {})
        if status.get("state") in {"completed", "stopped", "failed"}:
            return
        failure = {
            **status,
            "state": "failed",
            "error": (
                f"Trainer process exited with code {returncode} "
                "before publishing a terminal status"
            ),
            "process_exit_code": returncode,
            "stderr_tail": list(stderr_buffer)[-20:],
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        _atomic_json(status_path, failure)
        event_queue.put({"type": "status", "payload": failure})

    def _watch_process(
        self,
        process: subprocess.Popen[str],
        run_dir: Path,
        event_queue: DesktopEventBuffer,
        stderr_buffer: deque[str],
        reader_threads: tuple[threading.Thread, threading.Thread],
    ) -> None:
        returncode = process.wait()
        for thread in reader_threads:
            thread.join(timeout=1.0)
        self._handle_process_exit(process, run_dir, event_queue, stderr_buffer, returncode)

    def _read_stdout(
        self,
        process: subprocess.Popen[str],
        run_dir: Path,
        event_queue: DesktopEventBuffer,
    ) -> None:
        assert process.stdout is not None
        for raw_line in process.stdout:
            line = raw_line.rstrip("\r\n")
            if line.startswith(EVENT_PREFIX):
                try:
                    event = json.loads(line[len(EVENT_PREFIX) :])
                except json.JSONDecodeError:
                    self._append_worker_log("stdout", line, run_dir=run_dir)
                    event_queue.put({"type": "log", "payload": {"stream": "stdout", "text": line}})
                else:
                    event_queue.put(event)
            elif line:
                self._append_worker_log("stdout", line, run_dir=run_dir)
                event_queue.put({"type": "log", "payload": {"stream": "stdout", "text": line}})

    def _read_stderr(
        self,
        process: subprocess.Popen[str],
        run_dir: Path,
        event_queue: DesktopEventBuffer,
        stderr_buffer: deque[str],
    ) -> None:
        assert process.stderr is not None
        for raw_line in process.stderr:
            line = raw_line.rstrip("\r\n")
            if line:
                stderr_buffer.append(line)
                self._append_worker_log("stderr", line, run_dir=run_dir)
                event_queue.put({"type": "log", "payload": {"stream": "stderr", "text": line}})

    def drain_events(self, max_items: int = 500) -> list[dict[str, Any]]:
        return self._events.drain(max_items)

    def control(self, action: ControlAction) -> dict[str, Any]:
        if self.run_dir is None:
            raise RuntimeError("No desktop training run has been started")
        status = read_json_file(self.run_dir / "status.json", {})
        if status.get("state") not in ACTIVE_STATES:
            raise RuntimeError(f"Run is already {status.get('state', 'inactive')}")
        control_path = self.run_dir / "control.json"
        control = read_json_file(control_path, {"paused": False, "stop": False, "save_request": None})
        if action == "pause":
            control["paused"] = True
        elif action == "resume":
            control["paused"] = False
        elif action == "stop":
            control["paused"] = False
            control["stop"] = True
        elif action == "save":
            control["save_request"] = uuid.uuid4().hex
        else:
            raise ValueError(f"Unknown control action: {action}")
        _atomic_json(control_path, control)
        return control

    def snapshot(self) -> dict[str, Any]:
        if self.run_dir is None:
            return {}
        frame = read_json_file(self.run_dir / "frame.json")
        metadata = read_json_file(self.run_dir / "metadata.json")
        if frame is not None and metadata is not None and "metadata" not in frame:
            frame = {"metadata": metadata, **frame}
        return {
            "run_id": self.run_id,
            "request": read_json_file(self.run_dir / "request.json", {}),
            "status": read_json_file(self.run_dir / "status.json", {}),
            "metrics": read_json_file(self.run_dir / "metrics.json", {"episodes": [], "updates": []}),
            "frame": frame,
            "last_save": read_json_file(self.run_dir / "last_save.json"),
            "stderr": list(self._stderr),
        }

    def wait(self, timeout: float | None = None) -> int:
        if self.process is None:
            raise RuntimeError("No desktop training process has been started")
        process = self.process
        run_dir = self.run_dir
        if run_dir is None:
            raise RuntimeError("No desktop training run directory is available")
        returncode = process.wait(timeout=timeout)
        for thread in self._threads:
            if thread is not threading.current_thread():
                thread.join(timeout=1.0)
        self._handle_process_exit(process, run_dir, self._events, self._stderr, returncode)
        return returncode

    def stop_and_wait(self, timeout: float = 10.0) -> int:
        if not self.is_active:
            return self.process.returncode if self.process is not None else 0
        try:
            self.control("stop")
        except RuntimeError as exc:
            self._append_worker_log(
                "controller",
                f"stop command skipped while process exits: {exc}",
                run_dir=self.run_dir,
            )
        try:
            return self.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            assert self.process is not None
            self.process.terminate()
            try:
                return self.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                return self.wait(timeout=3.0)

    def wait_for_state(self, states: set[str], timeout: float = 15.0) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status = self.snapshot().get("status", {})
            if status.get("state") in states:
                return status
            time.sleep(0.05)
        raise TimeoutError(f"Timed out waiting for states: {sorted(states)}")
