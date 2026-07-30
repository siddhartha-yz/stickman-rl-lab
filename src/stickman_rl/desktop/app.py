from __future__ import annotations

import ctypes
import json
import math
import os
import time
import tkinter as tk
from collections import deque
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any

from stickman_rl.desktop.controller import (
    ACTIVE_STATES,
    DesktopTrainingController,
    TrainingRequest,
    read_json_file,
    state_name,
    training_options,
)

COLORS = {
    "background": "#06111f",
    "sidebar": "#071522",
    "panel": "#0b1d2f",
    "panel_alt": "#0e2439",
    "border": "#1b3b54",
    "text": "#e8f1fa",
    "muted": "#8ba2b8",
    "blue": "#38bdf8",
    "green": "#34d399",
    "orange": "#f59e0b",
    "red": "#fb7185",
    "purple": "#a78bfa",
}

STATUS_LABELS = {
    "starting": "正在启动",
    "running": "训练中",
    "paused": "已暂停",
    "saving": "正在保存",
    "stopping": "正在停止",
    "stopped": "已停止",
    "completed": "训练完成",
    "failed": "训练失败",
    "idle": "等待训练",
}


def rotate_point(point: list[float], angle: float, position: list[float]) -> tuple[float, float]:
    x, y = point
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return (
        position[0] + x * cosine - y * sine,
        position[1] + x * sine + y * cosine,
    )


def finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def metric_records(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def nonnegative_int(value: Any, default: int = 0) -> int:
    try:
        parsed = int(value)
    except (OverflowError, TypeError, ValueError):
        return default
    return max(0, parsed)


def format_number(value: Any, digits: int = 2) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def format_percent(value: Any) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "—"


def _json_object(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _json_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def finite_point(value: Any, *, fill_missing: bool = False) -> list[float] | None:
    coordinates = value if isinstance(value, list) else []
    x = finite_float(coordinates[0]) if len(coordinates) >= 1 else None
    y = finite_float(coordinates[1]) if len(coordinates) >= 2 else None
    if fill_missing:
        return [x if x is not None else 0.0, y if y is not None else 0.0]
    if x is None or y is None:
        return None
    return [x, y]


def normalize_metadata_payload(value: Any) -> dict[str, Any]:
    payload = dict(_json_object(value))
    for field, prefix in (("action_names", "action"), ("body_names", "body")):
        names: list[str] = []
        for index, name in enumerate(_json_list(payload.get(field))):
            if isinstance(name, str) and name.strip():
                names.append(name.strip())
            else:
                names.append(f"{prefix}_{index}")
        payload[field] = names
    payload["body_geometry"] = _json_object(payload.get("body_geometry"))
    return payload


def normalize_frame_payload(value: Any) -> dict[str, Any]:
    payload = dict(_json_object(value))
    training = dict(_json_object(payload.get("training")))
    actions: list[float] = []
    for value in _json_list(training.get("action")):
        parsed = finite_float(value)
        actions.append(parsed if parsed is not None else 0.0)
    training["action"] = actions
    frame = dict(_json_object(payload.get("frame")))
    frame["info"] = _json_object(frame.get("info"))
    positions: list[list[float]] = []
    for value in _json_list(frame.get("body_positions")):
        positions.append(finite_point(value, fill_missing=True) or [0.0, 0.0])
    frame["body_positions"] = positions
    angles: list[float] = []
    for value in _json_list(frame.get("body_angles")):
        parsed = finite_float(value)
        angles.append(parsed if parsed is not None else 0.0)
    frame["body_angles"] = angles
    if "target_position" in frame:
        target_position = finite_point(frame.get("target_position"))
        if target_position is None:
            frame.pop("target_position", None)
        else:
            frame["target_position"] = target_position
    payload["training"] = training
    payload["frame"] = frame
    return payload


def _history_timestamp(value: Any, fallback: float) -> float:
    if isinstance(value, int | float) and not isinstance(value, bool):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else fallback
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except (OSError, OverflowError, ValueError):
            return fallback
    return fallback


def run_summaries(runs_root: Path) -> list[dict[str, Any]]:
    if not runs_root.exists():
        return []
    items: list[tuple[float, dict[str, Any]]] = []
    for run_dir in runs_root.iterdir():
        if not run_dir.is_dir():
            continue
        request = _json_object(read_json_file(run_dir / "request.json", {}))
        status = _json_object(read_json_file(run_dir / "status.json", {}))
        updated_at = status.get("updated_at") or request.get("created_at") or ""
        try:
            fallback = run_dir.stat().st_mtime
        except OSError:
            fallback = 0.0
        items.append(
            (
                _history_timestamp(updated_at, fallback),
                {
                    "run_id": run_dir.name,
                    "request": request,
                    "status": status,
                    "updated_at": updated_at,
                },
            )
        )
    return [item for _, item in sorted(items, key=lambda entry: entry[0], reverse=True)]


class Sparkline(tk.Canvas):
    def __init__(self, master: tk.Misc, title: str, **kwargs: Any) -> None:
        super().__init__(master, background=COLORS["panel"], highlightthickness=0, **kwargs)
        self.title = title
        self.values: list[float] = []
        self.bind("<Configure>", lambda _event: self.redraw())

    def set_values(self, values: list[Any]) -> None:
        self.values = [parsed for value in values if (parsed := finite_float(value)) is not None]
        self.redraw()

    def redraw(self) -> None:
        self.delete("all")
        width = max(self.winfo_width(), 20)
        height = max(self.winfo_height(), 20)
        self.create_text(12, 11, text=self.title, anchor="nw", fill=COLORS["muted"], font=("Segoe UI", 9))
        if len(self.values) < 2:
            self.create_text(width / 2, height / 2, text="等待数据", fill="#547087", font=("Segoe UI", 10))
            return
        values = self.values[-120:]
        low = min(values)
        high = max(values)
        if abs(high - low) < 1e-9:
            low -= 1.0
            high += 1.0
        left, right, top, bottom = 12.0, width - 12.0, 31.0, height - 12.0
        points: list[float] = []
        for index, value in enumerate(values):
            x = left + index / max(1, len(values) - 1) * (right - left)
            y = bottom - (value - low) / (high - low) * (bottom - top)
            points.extend([x, y])
        self.create_line(*points, fill=COLORS["blue"], width=2, smooth=True)
        self.create_text(right, 11, text=format_number(values[-1], 3), anchor="ne", fill=COLORS["text"], font=("Consolas", 9, "bold"))


class MetricTile(ttk.Frame):
    def __init__(self, master: tk.Misc, title: str, accent: str) -> None:
        super().__init__(master, style="Panel.TFrame", padding=(14, 10))
        self.columnconfigure(0, weight=1)
        ttk.Label(self, text=title, style="Muted.TLabel").grid(row=0, column=0, sticky="w")
        self.value = ttk.Label(self, text="—", style="Metric.TLabel")
        self.value.grid(row=1, column=0, sticky="w", pady=(4, 1))
        self.detail = ttk.Label(self, text="", style="Tiny.TLabel")
        self.detail.grid(row=2, column=0, sticky="w")
        stripe = tk.Frame(self, background=accent, width=4)
        stripe.grid(row=0, column=1, rowspan=3, sticky="ns", padx=(12, 0))

    def set(self, value: str, detail: str = "") -> None:
        self.value.configure(text=value)
        self.detail.configure(text=detail)


class PhysicsCanvas(tk.Canvas):
    def __init__(self, master: tk.Misc, **kwargs: Any) -> None:
        super().__init__(master, background="#081624", highlightthickness=0, **kwargs)
        self.metadata: dict[str, Any] | None = None
        self.frame: dict[str, Any] | None = None
        self.trail: deque[list[float]] = deque(maxlen=140)
        self.last_episode: int | None = None
        self.bind("<Configure>", lambda _event: self.redraw())

    def set_metadata(self, metadata: dict[str, Any]) -> None:
        self.metadata = normalize_metadata_payload(metadata)
        self.redraw()

    def set_frame(self, frame: dict[str, Any]) -> None:
        frame = normalize_frame_payload(frame)
        self.frame = frame
        episode = nonnegative_int(frame["training"].get("episode", 0))
        if self.last_episode is not None and episode != self.last_episode:
            self.trail.clear()
        self.last_episode = episode
        if self.metadata:
            names = self.metadata.get("body_names", [])
            if "torso" in names:
                index = names.index("torso")
                positions = frame.get("frame", {}).get("body_positions", [])
                if index < len(positions):
                    self.trail.append(positions[index])
        self.redraw()

    def _point_transform(self) -> tuple[Any, float, float, float]:
        assert self.metadata is not None
        width = max(self.winfo_width(), 1)
        height = max(self.winfo_height(), 1)
        room = self.metadata["room"]
        padding = 24.0
        scale = min((width - padding * 2) / room["width"], (height - padding * 2) / room["height"])
        offset_x = (width - room["width"] * scale) / 2
        offset_y = (height - room["height"] * scale) / 2

        def point(value: list[float] | tuple[float, float]) -> tuple[float, float]:
            return offset_x + value[0] * scale, height - offset_y - value[1] * scale

        return point, scale, offset_x, offset_y

    def redraw(self) -> None:
        self.delete("all")
        width = max(self.winfo_width(), 1)
        height = max(self.winfo_height(), 1)
        if not self.metadata or not self.frame:
            self.create_text(width / 2, height / 2, text="启动训练后显示当前真实物理状态", fill="#5f7890", font=("Segoe UI", 13))
            return
        point, scale, offset_x, offset_y = self._point_transform()
        room = self.metadata["room"]
        room_right = offset_x + room["width"] * scale
        room_bottom = height - offset_y

        for x in range(int(room["width"]) + 1):
            px, _ = point((x, 0))
            self.create_line(px, offset_y, px, room_bottom, fill="#10283b")
        for y in range(int(room["height"]) + 1):
            _, py = point((0, y))
            self.create_line(offset_x, py, room_right, py, fill="#10283b")
        self.create_rectangle(offset_x, offset_y, room_right, room_bottom, outline="#5b7085", width=3)

        for obstacle in self.metadata.get("obstacles", []):
            if str(obstacle.get("type", "box")).lower() not in {"box", "platform", "wall"}:
                continue
            cx, cy = obstacle["position"]
            obstacle_width, obstacle_height = obstacle["size"]
            left, top = point((cx - obstacle_width / 2, cy + obstacle_height / 2))
            self.create_rectangle(
                left,
                top,
                left + obstacle_width * scale,
                top + obstacle_height * scale,
                fill="#35485b",
                outline="#8193a6",
                width=2,
            )

        frame = self.frame["frame"]
        target_position = frame.get("target_position", self.metadata["target"]["position"])
        target_width, target_height = self.metadata["target"]["size"]
        target_left, target_top = point((target_position[0] - target_width / 2, target_position[1] + target_height / 2))
        self.create_rectangle(
            target_left - 4,
            target_top - 4,
            target_left + target_width * scale + 4,
            target_top + target_height * scale + 4,
            outline="#7f1d2d",
            width=4,
        )
        self.create_rectangle(
            target_left,
            target_top,
            target_left + target_width * scale,
            target_top + target_height * scale,
            fill="#ef4454",
            outline="#fecdd3",
            width=2,
        )

        active_waypoint = nonnegative_int(frame.get("active_waypoint_index", 0))
        for index, waypoint in enumerate(self.metadata.get("waypoints", [])):
            x, y = point(waypoint)
            color = COLORS["green"] if index < active_waypoint else COLORS["orange"] if index == active_waypoint else "#64748b"
            radius = max(7, scale * 0.17)
            self.create_oval(x - radius, y - radius, x + radius, y + radius, outline=color, width=2, dash=(5, 4))
            self.create_text(x + 13, y - 12, text=f"W{index + 1}", fill=COLORS["text"], anchor="w", font=("Segoe UI", 9, "bold"))

        if len(self.trail) > 1:
            trail_points: list[float] = []
            for position in self.trail:
                trail_points.extend(point(position))
            self.create_line(*trail_points, fill="#16799e", width=2, smooth=True)

        positions = frame.get("body_positions", [])
        angles = frame.get("body_angles", [])
        for index, name in enumerate(self.metadata.get("body_names", [])):
            if index >= len(positions) or index >= len(angles):
                continue
            geometry = self.metadata["body_geometry"].get(name)
            if not geometry:
                continue
            position = positions[index]
            angle = float(angles[index])
            fill = "#3b82f6" if name == "torso" else "#dce7f3"
            outline = "#93c5fd" if name == "torso" else "#172033"
            if geometry["kind"] == "circle":
                center = point(rotate_point(geometry.get("offset", [0, 0]), angle, position))
                radius = max(2, geometry["radius"] * scale)
                self.create_oval(center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius, fill=fill, outline=outline, width=2)
            elif geometry["kind"] == "segment":
                start = point(rotate_point(geometry["a"], angle, position))
                end = point(rotate_point(geometry["b"], angle, position))
                self.create_line(*start, *end, fill="#172033", width=max(5, geometry["radius"] * scale * 2 + 3), capstyle=tk.ROUND)
                self.create_line(*start, *end, fill="#dce7f3", width=max(3, geometry["radius"] * scale * 2), capstyle=tk.ROUND)
            elif geometry["kind"] == "polygon":
                points: list[float] = []
                for vertex in geometry["vertices"]:
                    points.extend(point(rotate_point(vertex, angle, position)))
                self.create_polygon(*points, fill=fill, outline=outline, width=2)

        if frame.get("info", {}).get("is_success"):
            self.create_text(offset_x + 20, offset_y + 20, text="GOAL REACHED", fill="#a7f3d0", anchor="nw", font=("Segoe UI", 16, "bold"))


class DesktopLabApp:
    def __init__(
        self,
        root: tk.Tk,
        *,
        controller: DesktopTrainingController | None = None,
        smoke_output: Path | None = None,
        smoke_screenshot: Path | None = None,
    ) -> None:
        self.root = root
        self.controller = controller or DesktopTrainingController()
        self.smoke_output = smoke_output
        self.smoke_screenshot = smoke_screenshot
        self.options = training_options()
        self.metadata: dict[str, Any] | None = None
        self.frame: dict[str, Any] | None = None
        self.status: dict[str, Any] = {"state": "idle"}
        self.metrics: dict[str, list[dict[str, Any]]] = {"episodes": [], "updates": []}
        self.frames_received = 0
        self.frame_times: deque[float] = deque(maxlen=120)
        self.logs: deque[str] = deque(maxlen=80)
        self.last_snapshot_at = 0.0
        self.smoke_started = False
        self.smoke_finished = False

        self._configure_root()
        self._build_styles()
        self._build_layout()
        self._refresh_history()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(16, self._tick)

    def _configure_root(self) -> None:
        self.root.title("Stickman RL Lab · Desktop Training Console")
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        width = max(1120, min(1480, screen_width - 40))
        height = max(720, min(940, screen_height - 80))
        x = max(0, (screen_width - width) // 2)
        y = max(0, (screen_height - height) // 2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")
        self.root.minsize(1120, 720)
        self.root.configure(background=COLORS["background"])

    def _build_styles(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("Root.TFrame", background=COLORS["background"])
        style.configure("Sidebar.TFrame", background=COLORS["sidebar"])
        style.configure("Panel.TFrame", background=COLORS["panel"], relief="flat")
        style.configure("PanelAlt.TFrame", background=COLORS["panel_alt"])
        style.configure("TLabel", background=COLORS["background"], foreground=COLORS["text"], font=("Segoe UI", 10))
        style.configure("Panel.TLabel", background=COLORS["panel"], foreground=COLORS["text"])
        style.configure("Muted.TLabel", background=COLORS["panel"], foreground=COLORS["muted"], font=("Segoe UI", 9))
        style.configure("Tiny.TLabel", background=COLORS["panel"], foreground="#58748b", font=("Consolas", 8))
        style.configure("Metric.TLabel", background=COLORS["panel"], foreground=COLORS["text"], font=("Segoe UI", 22, "bold"))
        style.configure("Header.TLabel", background=COLORS["background"], foreground=COLORS["text"], font=("Segoe UI", 25, "bold"))
        style.configure("Eyebrow.TLabel", background=COLORS["background"], foreground=COLORS["blue"], font=("Consolas", 9, "bold"))
        style.configure("Sidebar.TLabel", background=COLORS["sidebar"], foreground=COLORS["text"])
        style.configure("SidebarMuted.TLabel", background=COLORS["sidebar"], foreground=COLORS["muted"], font=("Segoe UI", 9))
        style.configure("TButton", padding=(10, 7), font=("Segoe UI", 9, "bold"))
        style.configure("Primary.TButton", background="#1179aa", foreground="white")
        style.map("Primary.TButton", background=[("active", "#159bd4"), ("disabled", "#15364a")])
        style.configure("Danger.TButton", background="#582334", foreground="#fecdd3")
        style.map("Danger.TButton", background=[("active", "#7f1d35")])
        style.configure("TCombobox", fieldbackground="#0c2032", background="#0c2032", foreground=COLORS["text"], arrowcolor=COLORS["text"])
        style.configure("TEntry", fieldbackground="#0c2032", foreground=COLORS["text"], insertcolor=COLORS["text"])
        style.configure("Horizontal.TProgressbar", background=COLORS["blue"], troughcolor="#10283b")

    def _build_layout(self) -> None:
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)
        sidebar = ttk.Frame(self.root, style="Sidebar.TFrame", padding=16, width=300)
        sidebar.grid(row=0, column=0, sticky="nsw")
        sidebar.grid_propagate(False)
        sidebar.columnconfigure(0, weight=1)
        sidebar.rowconfigure(5, weight=1)

        brand = ttk.Label(sidebar, text="RL  Stickman RL Lab", style="Sidebar.TLabel", font=("Segoe UI", 14, "bold"))
        brand.grid(row=0, column=0, sticky="w")
        ttk.Label(sidebar, text="NATIVE DESKTOP TRAINING CONSOLE", style="SidebarMuted.TLabel").grid(row=1, column=0, sticky="w", pady=(1, 16))

        form = tk.Frame(sidebar, background="#0a1d2e", highlightbackground=COLORS["border"], highlightthickness=1, padx=12, pady=12)
        form.grid(row=2, column=0, sticky="ew", pady=(0, 16))
        form.columnconfigure(0, weight=1)
        tk.Label(form, text="从随机权重开始训练", background="#0a1d2e", foreground=COLORS["text"], font=("Segoe UI", 12, "bold")).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 9))

        self.stage_var = tk.StringVar(value=self.options["stages"][1][1])
        self.timesteps_var = tk.StringVar(value="100000")
        self.seed_var = tk.StringVar(value="0")
        self.train_config_var = tk.StringVar(value="configs/train_live.yaml")
        self.env_config_var = tk.StringVar(value="使用 Stage 默认配置")

        self._form_label(form, "课程阶段", 1)
        self.stage_combo = ttk.Combobox(form, textvariable=self.stage_var, values=[label for _, label in self.options["stages"]], state="readonly")
        self.stage_combo.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        self._form_label(form, "总训练步数", 3, column=0)
        self._form_label(form, "随机种子", 3, column=1)
        ttk.Entry(form, textvariable=self.timesteps_var, width=13).grid(row=4, column=0, sticky="ew", padx=(0, 4), pady=(0, 8))
        ttk.Entry(form, textvariable=self.seed_var, width=10).grid(row=4, column=1, sticky="ew", padx=(4, 0), pady=(0, 8))
        self._form_label(form, "PPO 配置", 5)
        ttk.Combobox(form, textvariable=self.train_config_var, values=self.options["train_configs"], state="readonly").grid(row=6, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        self._form_label(form, "环境覆盖配置", 7)
        env_values = ["使用 Stage 默认配置", *self.options["env_configs"]]
        ttk.Combobox(form, textvariable=self.env_config_var, values=env_values, state="readonly").grid(row=8, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        self.start_button = ttk.Button(form, text="▶ 创建并启动真实训练", style="Primary.TButton", command=self._start_training)
        self.start_button.grid(row=9, column=0, columnspan=2, sticky="ew")

        ttk.Label(sidebar, text="训练记录", style="Sidebar.TLabel", font=("Segoe UI", 10, "bold")).grid(row=3, column=0, sticky="w", pady=(0, 8))
        self.history = tk.Listbox(
            sidebar,
            background=COLORS["sidebar"],
            foreground=COLORS["muted"],
            selectbackground="#0c3650",
            selectforeground=COLORS["text"],
            borderwidth=0,
            highlightthickness=0,
            font=("Consolas", 8),
        )
        self.history.grid(row=5, column=0, sticky="nsew")
        self.history.bind("<<ListboxSelect>>", self._select_history)
        self.history_items: list[dict[str, Any]] = []

        main = ttk.Frame(self.root, style="Root.TFrame", padding=(24, 18))
        main.grid(row=0, column=1, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.rowconfigure(1, weight=7, minsize=390)
        main.rowconfigure(3, weight=2, minsize=210)

        header = ttk.Frame(main, style="Root.TFrame")
        header.grid(row=0, column=0, sticky="ew", pady=(0, 14))
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="LIVE REINFORCEMENT LEARNING", style="Eyebrow.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(header, text="桌面训练控制台", style="Header.TLabel").grid(row=1, column=0, sticky="w")
        self.status_label = ttk.Label(header, text="● 等待训练", style="TLabel", font=("Segoe UI", 10, "bold"))
        self.status_label.grid(row=0, column=1, rowspan=2, sticky="e")

        simulation = ttk.Frame(main, style="Panel.TFrame", padding=10)
        simulation.grid(row=1, column=0, sticky="nsew")
        simulation.columnconfigure(0, weight=1)
        simulation.rowconfigure(1, weight=1)
        toolbar = ttk.Frame(simulation, style="Panel.TFrame")
        toolbar.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        toolbar.columnconfigure(0, weight=1)
        self.stream_label = ttk.Label(toolbar, text="LIVE PHYSICS · native pipe", style="Muted.TLabel")
        self.stream_label.grid(row=0, column=0, sticky="w")
        self.pause_button = ttk.Button(toolbar, text="Ⅱ 暂停", command=lambda: self._control("pause"))
        self.pause_button.grid(row=0, column=1, padx=3)
        self.save_button = ttk.Button(toolbar, text="保存 checkpoint", command=lambda: self._control("save"))
        self.save_button.grid(row=0, column=2, padx=3)
        self.stop_button = ttk.Button(toolbar, text="停止训练", style="Danger.TButton", command=lambda: self._control("stop"))
        self.stop_button.grid(row=0, column=3, padx=(3, 0))
        self.physics = PhysicsCanvas(simulation, height=360)
        self.physics.grid(row=1, column=0, sticky="nsew")
        progress_row = ttk.Frame(simulation, style="Panel.TFrame")
        progress_row.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        progress_row.columnconfigure(0, weight=1)
        self.progress = ttk.Progressbar(progress_row, maximum=100)
        self.progress.grid(row=0, column=0, sticky="ew")
        self.progress_text = ttk.Label(progress_row, text="0 / 0", style="Muted.TLabel")
        self.progress_text.grid(row=0, column=1, padx=(10, 0))

        metrics = ttk.Frame(main, style="Root.TFrame")
        metrics.grid(row=2, column=0, sticky="ew", pady=12)
        for column in range(4):
            metrics.columnconfigure(column, weight=1)
        self.metric_steps = MetricTile(metrics, "当前训练步数", COLORS["blue"])
        self.metric_reward = MetricTile(metrics, "当前 Episode 奖励", COLORS["orange"])
        self.metric_success = MetricTile(metrics, "最近成功率", COLORS["green"])
        self.metric_distance = MetricTile(metrics, "最近终点距离", COLORS["purple"])
        for column, tile in enumerate((self.metric_steps, self.metric_reward, self.metric_success, self.metric_distance)):
            tile.grid(row=0, column=column, sticky="ew", padx=(0 if column == 0 else 5, 0 if column == 3 else 5))

        analysis = ttk.Frame(main, style="Root.TFrame")
        analysis.grid(row=3, column=0, sticky="nsew")
        for column in range(3):
            analysis.columnconfigure(column, weight=1)
        analysis.rowconfigure(0, weight=1)
        analysis.rowconfigure(1, weight=1)
        self.reward_chart = Sparkline(analysis, "Episode reward", height=105)
        self.success_chart = Sparkline(analysis, "Rolling success rate", height=105)
        self.loss_chart = Sparkline(analysis, "PPO value loss", height=105)
        self.reward_chart.grid(row=0, column=0, sticky="nsew", padx=(0, 5), pady=(0, 5))
        self.success_chart.grid(row=0, column=1, sticky="nsew", padx=5, pady=(0, 5))
        self.loss_chart.grid(row=0, column=2, sticky="nsew", padx=(5, 0), pady=(0, 5))

        self.session_text = tk.Text(
            analysis,
            height=6,
            background=COLORS["panel"],
            foreground=COLORS["muted"],
            insertbackground=COLORS["text"],
            borderwidth=0,
            font=("Consolas", 8),
            state="disabled",
            wrap="word",
        )
        self.session_text.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=(0, 5), pady=(5, 0))
        self.action_text = tk.Text(
            analysis,
            width=34,
            background=COLORS["panel"],
            foreground=COLORS["muted"],
            borderwidth=0,
            font=("Consolas", 8),
            state="disabled",
        )
        self.action_text.grid(row=1, column=2, sticky="nsew", padx=(5, 0), pady=(5, 0))
        self._set_controls_enabled(False)

    @staticmethod
    def _form_label(master: tk.Misc, text: str, row: int, column: int = 0) -> None:
        tk.Label(master, text=text, background="#0a1d2e", foreground=COLORS["muted"], font=("Segoe UI", 8)).grid(row=row, column=column, sticky="w", pady=(0, 3))

    def _set_controls_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        self.pause_button.configure(state=state)
        self.save_button.configure(state=state)
        self.stop_button.configure(state=state)

    def _start_training(self) -> None:
        try:
            stage_label = self.stage_var.get()
            stage = next(value for value, label in self.options["stages"] if label == stage_label)
            request = TrainingRequest(
                stage=stage,
                timesteps=int(self.timesteps_var.get()),
                seed=int(self.seed_var.get()),
                train_config=self.train_config_var.get(),
                env_config=None if self.env_config_var.get() == "使用 Stage 默认配置" else self.env_config_var.get(),
            )
            run_id = self.controller.start(request)
        except (ValueError, RuntimeError) as exc:
            messagebox.showerror("无法启动训练", str(exc), parent=self.root)
            return
        self.metadata = None
        self.frame = None
        self.status = {"state": "starting", "num_timesteps": 0, "total_timesteps": request.timesteps}
        self.metrics = {"episodes": [], "updates": []}
        self.frames_received = 0
        self.frame_times.clear()
        self.start_button.configure(state="disabled")
        self._set_controls_enabled(True)
        self.status_label.configure(text=f"● 正在启动 · {run_id}", foreground=COLORS["orange"])
        self._refresh_history()

    def _control(self, action: str) -> None:
        try:
            self.controller.control(action)  # type: ignore[arg-type]
        except (RuntimeError, ValueError) as exc:
            messagebox.showerror("控制失败", str(exc), parent=self.root)
            return
        if action == "pause":
            self.pause_button.configure(text="▶ 继续", command=lambda: self._control("resume"))
        elif action == "resume":
            self.pause_button.configure(text="Ⅱ 暂停", command=lambda: self._control("pause"))

    def _handle_event(self, event: dict[str, Any]) -> None:
        event_type = event.get("type")
        payload = event.get("payload", {})
        if event_type == "metadata":
            normalized = normalize_metadata_payload(payload)
            self.metadata = normalized
            self.physics.set_metadata(normalized)
        elif event_type == "frame":
            normalized = normalize_frame_payload(payload)
            self.frame = normalized
            self.frames_received += 1
            self.frame_times.append(time.monotonic())
            self.physics.set_frame(normalized)
        elif event_type == "status":
            self.status = payload
        elif event_type == "metrics":
            self.metrics = payload
        elif event_type == "checkpoint":
            self.logs.append(f"checkpoint: {payload.get('path')}")
        elif event_type == "log":
            text = payload.get("text")
            if text:
                self.logs.append(str(text))

    def _tick(self) -> None:
        for event in self.controller.drain_events(1000):
            self._handle_event(event)
        now = time.monotonic()
        if self.controller.run_dir and now - self.last_snapshot_at >= 0.5:
            snapshot = self.controller.snapshot()
            self.last_snapshot_at = now
            if snapshot.get("status"):
                self.status = snapshot["status"]
            if snapshot.get("metrics"):
                self.metrics = snapshot["metrics"]
            snapshot_frame = snapshot.get("frame")
            if self.metadata is None and snapshot_frame and snapshot_frame.get("metadata"):
                self.metadata = normalize_metadata_payload(snapshot_frame["metadata"])
                self.physics.set_metadata(self.metadata)
            if self.frame is None and snapshot_frame:
                compact = dict(snapshot_frame)
                compact.pop("metadata", None)
                normalized = normalize_frame_payload(compact)
                self.frame = normalized
                self.physics.set_frame(normalized)
        self._refresh_ui()
        self._maybe_finish_smoke()
        self.root.after(16, self._tick)

    def _refresh_ui(self) -> None:
        state = state_name(self.status.get("state"), "idle")
        color = COLORS["green"] if state == "running" else COLORS["orange"] if state in {"starting", "paused", "saving", "stopping"} else COLORS["red"] if state == "failed" else COLORS["muted"]
        self.status_label.configure(text=f"● {STATUS_LABELS.get(state, state)}", foreground=color)
        total = nonnegative_int(
            self.status.get("total_timesteps") or self.status.get("timesteps") or 0
        )
        current = nonnegative_int(self.status.get("num_timesteps") or 0)
        percent = current / max(1, total) * 100
        self.progress.configure(value=min(100, percent))
        self.progress_text.configure(text=f"{current:,} / {total:,} · {format_number(self.status.get('fps'), 1)} steps/s")
        self.metric_steps.set(f"{current:,}", f"目标 {total:,}")
        self.metric_reward.set(format_number(self.status.get("current_episode_reward")), f"episode {self.status.get('episode', 0)}")
        self.metric_success.set(format_percent(self.status.get("rolling_success_rate")), f"完成 {self.status.get('completed_episodes', 0)} episodes")
        self.metric_distance.set(format_number(self.status.get("rolling_final_distance"), 3), "rolling final distance")

        episodes = metric_records(self.metrics.get("episodes", []))
        updates = metric_records(self.metrics.get("updates", []))
        self.reward_chart.set_values([item.get("reward") for item in episodes])
        rolling: list[float] = []
        successes: list[float] = []
        for item in episodes:
            successes.append(float(bool(item.get("success"))))
            window = successes[-20:]
            rolling.append(sum(window) / len(window))
        self.success_chart.set_values(rolling)
        self.loss_chart.set_values([item.get("value_loss") for item in updates])

        if len(self.frame_times) >= 2:
            duration = self.frame_times[-1] - self.frame_times[0]
            source_fps = (len(self.frame_times) - 1) / max(duration, 1e-6)
        else:
            source_fps = 0.0
        self.stream_label.configure(text=f"LIVE PHYSICS · native pipe · {source_fps:.0f} source FPS")
        active = state in ACTIVE_STATES
        self.start_button.configure(state="disabled" if active else "normal")
        self._set_controls_enabled(active)
        if state == "paused":
            self.pause_button.configure(text="▶ 继续", command=lambda: self._control("resume"))
        else:
            self.pause_button.configure(text="Ⅱ 暂停", command=lambda: self._control("pause"))

        session_lines = [
            f"RUN ID       {self.controller.run_id or '—'}",
            f"STATE        {state}",
            f"PROCESS      {'alive' if self.controller.is_active else 'ended'}",
            f"FRAME COUNT  {self.frames_received}",
            f"CHECKPOINT   {self.status.get('final_checkpoint') or 'not saved yet'}",
        ]
        if self.logs:
            session_lines.extend(["", "RECENT LOG", *list(self.logs)[-5:]])
        self._replace_text(self.session_text, "\n".join(session_lines))

        if self.metadata and self.frame:
            names = self.metadata.get("action_names", [])
            actions = self.frame.get("training", {}).get("action", [])
            action_lines = ["POLICY OUTPUT"]
            for name, value in zip(names, actions, strict=False):
                normalized = max(-1.0, min(1.0, float(value)))
                left = int((normalized + 1) / 2 * 14)
                bar = "·" * left + "●" + "·" * (28 - left)
                action_lines.append(f"{name[:14]:14} {bar} {normalized:+.2f}")
            self._replace_text(self.action_text, "\n".join(action_lines))

    @staticmethod
    def _replace_text(widget: tk.Text, content: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", tk.END)
        widget.insert("1.0", content)
        widget.configure(state="disabled")

    def _refresh_history(self) -> None:
        self.history_items = run_summaries(self.controller.runs_root)[:80]
        self.history.delete(0, tk.END)
        for item in self.history_items:
            status = item["status"]
            history_state = state_name(status.get("state"), "?")
            self.history.insert(tk.END, f"{item['run_id']}\n  Stage {item['request'].get('stage', '?')} · {status.get('num_timesteps', 0)} · {STATUS_LABELS.get(history_state, history_state)}")

    def _select_history(self, _event: tk.Event[tk.Misc]) -> None:
        selection = self.history.curselection()
        if not selection:
            return
        item = self.history_items[selection[0]]
        if item["run_id"] == self.controller.run_id:
            return
        run_dir = self.controller.runs_root / item["run_id"]
        status = _json_object(read_json_file(run_dir / "status.json"))
        metrics = _json_object(read_json_file(run_dir / "metrics.json"))
        if not status or not metrics:
            return
        self.status = status
        self.metrics = metrics

    def _maybe_finish_smoke(self) -> None:
        if not self.smoke_output or self.smoke_finished:
            return
        state = state_name(self.status.get("state"), "idle")
        if state not in {"completed", "stopped", "failed"} or self.controller.is_active:
            return
        self.smoke_finished = True
        self.root.deiconify()
        self.root.lift()
        self.root.attributes("-topmost", True)
        self.root.focus_force()
        self.root.update_idletasks()
        self.root.update()
        time.sleep(0.2)
        screenshot_saved = False
        if self.smoke_screenshot:
            try:
                from PIL import ImageGrab

                self.smoke_screenshot.parent.mkdir(parents=True, exist_ok=True)
                x = self.root.winfo_rootx()
                y = self.root.winfo_rooty()
                width = self.root.winfo_width()
                height = self.root.winfo_height()
                ImageGrab.grab(bbox=(x, y, x + width, y + height)).save(self.smoke_screenshot)
                screenshot_saved = True
            except Exception as exc:  # noqa: BLE001 - smoke report must preserve capture failures
                self.logs.append(f"screenshot failed: {exc}")
        self.root.attributes("-topmost", False)
        report = {
            "state": state,
            "num_timesteps": self.status.get("num_timesteps"),
            "frames_received": self.frames_received,
            "body_count": len(self.metadata.get("body_names", [])) if self.metadata else 0,
            "process_exit_code": self.controller.process.poll() if self.controller.process else None,
            "screenshot_saved": screenshot_saved,
        }
        self.smoke_output.parent.mkdir(parents=True, exist_ok=True)
        self.smoke_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        self.root.after(250, self.root.destroy)

    def start_smoke(self) -> None:
        if self.smoke_started:
            return
        self.smoke_started = True
        self.timesteps_var.set("64")
        self.seed_var.set("4242")
        self.train_config_var.set("configs/train_smoke.yaml")
        self._start_training()

    def _on_close(self) -> None:
        if self.controller.is_active:
            if not messagebox.askyesno("停止训练并退出", "当前训练仍在运行。关闭桌面端会先保存并停止训练，是否继续？", parent=self.root):
                return
            self.controller.stop_and_wait(timeout=8.0)
        self.root.destroy()


def enable_dpi_awareness() -> None:
    if os.name != "nt":
        return
    with suppress(AttributeError, OSError):
        ctypes.windll.shcore.SetProcessDpiAwareness(1)


def launch_desktop(
    *,
    smoke_output: Path | None = None,
    smoke_screenshot: Path | None = None,
) -> None:
    enable_dpi_awareness()
    root = tk.Tk()
    app = DesktopLabApp(root, smoke_output=smoke_output, smoke_screenshot=smoke_screenshot)
    if smoke_output:
        root.after(600, app.start_smoke)
    root.mainloop()


def main() -> None:
    launch_desktop()


if __name__ == "__main__":
    main()
