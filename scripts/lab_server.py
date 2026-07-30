from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from scripts.training_api import router as training_router

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "lab" / "experiments.json"
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"

app = FastAPI(title="Stickman RL Lab Observer", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
app.include_router(training_router)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _project_path(relative_path: str) -> Path:
    resolved = (PROJECT_ROOT / relative_path).resolve()
    try:
        resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Path escapes project root") from exc
    return resolved


@lru_cache(maxsize=1)
def _manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        raise RuntimeError(f"Missing experiment manifest: {MANIFEST_PATH}")
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _experiment(experiment_id: str) -> dict[str, Any]:
    for experiment in _manifest()["experiments"]:
        if experiment["id"] == experiment_id:
            return experiment
    raise HTTPException(status_code=404, detail=f"Unknown experiment: {experiment_id}")


def _round_array(array: np.ndarray, digits: int = 5) -> list[Any]:
    if np.issubdtype(array.dtype, np.floating):
        array = np.round(array.astype(np.float64), digits)
    return array.tolist()


@lru_cache(maxsize=32)
def _load_trajectory_cached(path_text: str, modified_ns: int) -> dict[str, Any]:
    del modified_ns
    path = Path(path_text)
    with np.load(path, allow_pickle=False) as data:
        required = {"body_positions", "body_angles", "metadata_json"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(
                f"Trajectory uses the legacy format and cannot be rendered: missing {', '.join(missing)}"
            )
        metadata = json.loads(str(data["metadata_json"].item()))
        rewards = np.asarray(data["rewards"], dtype=np.float64)
        frame_count = int(rewards.shape[0])
        return {
            "metadata": metadata,
            "frame_count": frame_count,
            "body_positions": _round_array(data["body_positions"]),
            "body_angles": _round_array(data["body_angles"]),
            "actions": _round_array(data["actions"]),
            "rewards": _round_array(rewards),
            "cumulative_rewards": _round_array(data["cumulative_rewards"]),
            "distances": _round_array(data["distances"]),
            "torso_heights": _round_array(data["torso_heights"]),
            "torso_x_positions": _round_array(data["torso_x_positions"]),
            "waypoint_indices": _round_array(data["waypoint_indices"]),
            "navigation_distances": _round_array(data["navigation_distances"]),
            "goal_hold_counts": _round_array(data["goal_hold_counts"]),
            "successes": _round_array(data["successes"]),
            "summary": {
                "total_reward": float(np.sum(rewards)),
                "final_distance": float(data["distances"][-1]) if frame_count else None,
                "max_torso_x": float(np.max(data["torso_x_positions"])) if frame_count else None,
                "success": bool(np.any(data["successes"])) if frame_count else False,
                "steps": frame_count,
            },
        }


def load_trajectory_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return _load_trajectory_cached(str(path), path.stat().st_mtime_ns)


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "project_root": str(PROJECT_ROOT),
        "frontend_built": FRONTEND_DIST.exists(),
        "experiments": len(_manifest()["experiments"]),
    }


@app.get("/api/experiments")
def experiments() -> dict[str, Any]:
    manifest = _manifest()
    enriched: list[dict[str, Any]] = []
    for experiment in manifest["experiments"]:
        item = dict(experiment)
        trajectory_path = _project_path(item["trajectory"])
        item["available"] = trajectory_path.exists()
        if trajectory_path.exists():
            item["trajectory_sha256"] = _sha256(trajectory_path)
            try:
                payload = load_trajectory_payload(trajectory_path)
                item["trajectory_summary"] = payload["summary"]
                item["format_version"] = payload["metadata"].get("format_version")
            except ValueError as exc:
                item["trajectory_error"] = str(exc)
        enriched.append(item)
    return {"project": manifest["project"], "experiments": enriched}


@app.get("/api/experiments/{experiment_id}/trajectory")
def trajectory(experiment_id: str) -> dict[str, Any]:
    experiment = _experiment(experiment_id)
    path = _project_path(experiment["trajectory"])
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Trajectory has not been exported: {experiment['trajectory']}")
    try:
        payload = load_trajectory_payload(path)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"experiment": experiment, "trajectory_sha256": _sha256(path), **payload}


@app.get("/")
def frontend_index() -> FileResponse:
    index = FRONTEND_DIST / "index.html"
    if not index.exists():
        raise HTTPException(
            status_code=503,
            detail="Frontend has not been built. Run: cd frontend && npm install && npm run build",
        )
    return FileResponse(index)


if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")


@app.get("/{path:path}")
def frontend_fallback(path: str) -> FileResponse:
    candidate = FRONTEND_DIST / path
    if candidate.exists() and candidate.is_file():
        return FileResponse(candidate)
    index = FRONTEND_DIST / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="Frontend is not built")
    return FileResponse(index)
