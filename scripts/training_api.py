from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from stickman_rl.config import PROJECT_ROOT

RUNS_ROOT = PROJECT_ROOT / "lab" / "runs"
CONFIG_ROOT = PROJECT_ROOT / "configs"
ACTIVE_STATES = {"starting", "running", "paused", "saving", "stopping"}
PROCESSES: dict[str, subprocess.Popen[bytes]] = {}
LIVE_FRAMES: dict[str, dict[str, Any]] = {}
LIVE_FRAME_SEQUENCES: dict[str, int] = {}

router = APIRouter(prefix="/api/training", tags=["training"])


class TrainingStartRequest(BaseModel):
    stage: int = Field(default=1, ge=0, le=5)
    timesteps: int = Field(default=100_000, ge=64, le=10_000_000)
    seed: int = Field(default=0, ge=0, le=2_147_483_647)
    train_config: str = "configs/train_live.yaml"
    env_config: str | None = None


class ControlRequest(BaseModel):
    action: Literal["pause", "resume", "stop", "save"]


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _safe_config(value: str | None, *, training: bool) -> str | None:
    if value is None or not value.strip():
        return None
    candidate = (PROJECT_ROOT / value).resolve()
    try:
        candidate.relative_to(CONFIG_ROOT.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Config must be inside configs/") from exc
    if not candidate.exists() or candidate.suffix.lower() not in {".yaml", ".yml"}:
        raise HTTPException(status_code=400, detail=f"Config does not exist: {value}")
    if training and not candidate.name.startswith("train"):
        raise HTTPException(status_code=400, detail="Training config must start with train")
    return str(candidate.relative_to(PROJECT_ROOT)).replace("\\", "/")


def _run_dir(run_id: str) -> Path:
    if not run_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for char in run_id):
        raise HTTPException(status_code=400, detail="Invalid run id")
    return RUNS_ROOT / run_id


def _payload(run_id: str) -> dict[str, Any]:
    run_dir = _run_dir(run_id)
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail=f"Unknown training run: {run_id}")
    process = PROCESSES.get(run_id)
    request = _read_json(run_dir / "request.json", {})
    status = _read_json(run_dir / "status.json", {"state": "starting", "num_timesteps": 0})
    if process is not None:
        status["process_alive"] = process.poll() is None
        status["process_exit_code"] = process.poll()
    else:
        status["process_alive"] = status.get("state") in ACTIVE_STATES
        status["process_exit_code"] = None
    frame = LIVE_FRAMES.get(run_id) or _read_json(run_dir / "frame.json")
    metadata = _read_json(run_dir / "metadata.json")
    if frame is not None and metadata is not None and "metadata" not in frame:
        frame = {"metadata": metadata, **frame}
    return {
        "run_id": run_id,
        "request": request,
        "status": status,
        "metrics": _read_json(run_dir / "metrics.json", {"episodes": [], "updates": []}),
        "frame": frame,
        "last_save": _read_json(run_dir / "last_save.json"),
    }


def _runs() -> list[dict[str, Any]]:
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    items: list[dict[str, Any]] = []
    for run_dir in RUNS_ROOT.iterdir():
        if not run_dir.is_dir():
            continue
        request = _read_json(run_dir / "request.json", {})
        status = _read_json(run_dir / "status.json", {})
        items.append(
            {
                "run_id": run_dir.name,
                "request": request,
                "status": status,
                "updated_at": status.get("updated_at") or request.get("created_at") or "",
            }
        )
    return sorted(items, key=lambda item: item["updated_at"], reverse=True)


def _control(run_id: str, action: str) -> dict[str, Any]:
    run_dir = _run_dir(run_id)
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail=f"Unknown training run: {run_id}")
    control_path = run_dir / "control.json"
    control = _read_json(control_path, {"paused": False, "stop": False, "save_request": None})
    status = _read_json(run_dir / "status.json", {})
    if status.get("state") in {"completed", "stopped", "failed"}:
        raise HTTPException(status_code=409, detail=f"Run is already {status.get('state')}")
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
        raise HTTPException(status_code=400, detail=f"Unknown action: {action}")
    _atomic_json(control_path, control)
    return {"ok": True, "action": action, "control": control}


@router.get("/options")
def training_options() -> dict[str, Any]:
    train_configs = sorted(
        str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
        for path in CONFIG_ROOT.glob("train*.yaml")
    )
    env_configs = sorted(
        str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
        for path in CONFIG_ROOT.glob("stage*.yaml")
    )
    return {
        "stages": [
            {"value": 0, "label": "Stage 0 · 随机物理调试"},
            {"value": 1, "label": "Stage 1 · 固定终点，从零学习"},
            {"value": 2, "label": "Stage 2 · 随机终点"},
            {"value": 3, "label": "Stage 3 · 障碍环境"},
            {"value": 4, "label": "Stage 4 · 直立奖励"},
            {"value": 5, "label": "Stage 5 · 行走奖励"},
        ],
        "train_configs": train_configs,
        "env_configs": env_configs,
        "defaults": {
            "stage": 1,
            "timesteps": 100_000,
            "seed": 0,
            "train_config": "configs/train_live.yaml",
            "env_config": None,
        },
    }


@router.get("/runs")
def list_training_runs() -> dict[str, Any]:
    return {"runs": _runs()}


@router.get("/runs/current")
def current_training_run() -> dict[str, Any] | None:
    runs = _runs()
    active = next((item for item in runs if item.get("status", {}).get("state") in ACTIVE_STATES), None)
    selected = active or (runs[0] if runs else None)
    return _payload(selected["run_id"]) if selected else None


@router.post("/runs", status_code=201)
def start_training(request: TrainingStartRequest, http_request: Request) -> dict[str, Any]:
    for item in _runs():
        if item.get("status", {}).get("state") in ACTIVE_STATES:
            raise HTTPException(
                status_code=409,
                detail=f"已有训练任务正在运行：{item['run_id']}。请先停止它。",
            )
    train_config = _safe_config(request.train_config, training=True)
    env_config = _safe_config(request.env_config, training=False)
    run_id = f"ui-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    run_dir = RUNS_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    payload = {
        "run_id": run_id,
        "stage": request.stage,
        "timesteps": request.timesteps,
        "seed": request.seed,
        "train_config": train_config,
        "env_config": env_config,
        "from_scratch": True,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    _atomic_json(run_dir / "request.json", payload)
    _atomic_json(run_dir / "control.json", {"paused": False, "stop": False, "save_request": None})
    _atomic_json(
        run_dir / "status.json",
        {"state": "starting", "num_timesteps": 0, "progress": 0.0, **payload},
    )
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "live_train_worker.py"),
        "--run-id",
        run_id,
        "--stage",
        str(request.stage),
        "--timesteps",
        str(request.timesteps),
        "--seed",
        str(request.seed),
        "--train-config",
        str(train_config),
    ]
    if env_config:
        command.extend(["--env-config", env_config])
    websocket_scheme = "wss" if http_request.url.scheme == "https" else "ws"
    stream_url = (
        f"{websocket_scheme}://{http_request.url.netloc}"
        f"/api/training/ingest/{run_id}"
    )
    command.extend(["--stream-url", stream_url])
    log_handle = (run_dir / "worker.log").open("ab")
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        creationflags=creationflags,
    )
    log_handle.close()
    PROCESSES[run_id] = process
    return _payload(run_id)


@router.get("/runs/{run_id}")
def get_training_run(run_id: str) -> dict[str, Any]:
    return _payload(run_id)


@router.post("/runs/{run_id}/control")
def control_training_run(run_id: str, request: ControlRequest) -> dict[str, Any]:
    return _control(run_id, request.action)


@router.websocket("/ingest/{run_id}")
async def ingest_training_frames(websocket: WebSocket, run_id: str) -> None:
    """Receive high-frequency dynamic frames from the local trainer process."""
    await websocket.accept()
    run_dir = _run_dir(run_id)
    if not run_dir.exists():
        await websocket.close(code=4404)
        return
    try:
        while True:
            payload = await websocket.receive_json()
            if not isinstance(payload, dict):
                continue
            LIVE_FRAMES[run_id] = payload
            LIVE_FRAME_SEQUENCES[run_id] = LIVE_FRAME_SEQUENCES.get(run_id, 0) + 1
    except WebSocketDisconnect:
        return


@router.websocket("/runs/{run_id}/stream")
async def stream_training_run(websocket: WebSocket, run_id: str) -> None:
    """Push changed training files without high-frequency HTTP polling."""
    await websocket.accept()
    try:
        run_dir = _run_dir(run_id)
        if not run_dir.exists():
            await websocket.send_json({"type": "error", "detail": f"Unknown training run: {run_id}"})
            await websocket.close(code=4404)
            return

        await websocket.send_json({"type": "snapshot", "payload": _payload(run_id)})
        watched = {
            "frame": run_dir / "frame.json",
            "status": run_dir / "status.json",
            "metrics": run_dir / "metrics.json",
            "last_save": run_dir / "last_save.json",
        }
        modified: dict[str, int] = {}
        last_live_sequence = -1
        while True:
            sent = False
            live_sequence = LIVE_FRAME_SEQUENCES.get(run_id, -1)
            if live_sequence != last_live_sequence:
                live_payload = LIVE_FRAMES.get(run_id)
                if live_payload is not None:
                    await websocket.send_json({"type": "frame", "payload": live_payload})
                    sent = True
                last_live_sequence = live_sequence
            for message_type, path in watched.items():
                if message_type == "frame" and live_sequence >= 0:
                    continue
                try:
                    modified_ns = path.stat().st_mtime_ns
                except FileNotFoundError:
                    continue
                if modified.get(message_type) == modified_ns:
                    continue
                modified[message_type] = modified_ns
                payload = _read_json(path)
                if payload is not None:
                    await websocket.send_json({"type": message_type, "payload": payload})
                    sent = True
            await asyncio.sleep(0.003 if sent else 0.006)
    except WebSocketDisconnect:
        return
