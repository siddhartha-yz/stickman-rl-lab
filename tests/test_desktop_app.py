from __future__ import annotations

import json
import math
from collections import deque
from pathlib import Path

import pytest

from stickman_rl.desktop.app import (
    DesktopLabApp,
    PhysicsCanvas,
    finite_float,
    finite_point,
    format_number,
    format_percent,
    metric_records,
    nonnegative_int,
    normalize_frame_payload,
    normalize_metadata_payload,
    rotate_point,
    run_summaries,
)


def test_rotate_point_matches_quarter_turn() -> None:
    x, y = rotate_point([1.0, 0.0], math.pi / 2, [2.0, 3.0])
    assert math.isclose(x, 2.0, abs_tol=1e-8)
    assert math.isclose(y, 4.0, abs_tol=1e-8)


def test_metadata_payload_normalizes_name_arrays() -> None:
    normalized = normalize_metadata_payload(
        {
            "action_names": [{"bad": 1}, " left_hip ", ""],
            "body_names": [{"bad": 1}, " torso "],
            "body_geometry": [],
        }
    )
    assert normalized["action_names"] == ["action_0", "left_hip", "action_2"]
    assert normalized["body_names"] == ["body_0", "torso"]
    assert normalized["body_geometry"] == {}

    canvas = PhysicsCanvas.__new__(PhysicsCanvas)
    canvas.redraw = lambda: None
    PhysicsCanvas.set_metadata(
        canvas,
        {"body_names": [{"bad": 1}], "action_names": [{"bad": 1}]},
    )
    assert canvas.metadata["body_names"] == ["body_0"]
    assert canvas.metadata["action_names"] == ["action_0"]


def test_metadata_payload_normalizes_room_dimensions() -> None:
    normalized = normalize_metadata_payload({"room": {"width": 0, "height": "bad"}})
    assert normalized["room"] == {"width": 12.0, "height": 7.0}

    valid = normalize_metadata_payload({"room": {"width": "15.5", "height": 8}})
    assert valid["room"] == {"width": 15.5, "height": 8.0}

    canvas = PhysicsCanvas.__new__(PhysicsCanvas)
    canvas.metadata = normalized
    canvas.winfo_width = lambda: 800
    canvas.winfo_height = lambda: 500
    _point, scale, offset_x, offset_y = PhysicsCanvas._point_transform(canvas)
    assert math.isfinite(scale) and scale > 0.0
    assert math.isfinite(offset_x)
    assert math.isfinite(offset_y)


@pytest.mark.parametrize(
    "target",
    [
        None,
        [],
        {"position": [8], "size": [1, 1]},
        {"position": [8, 1], "size": ["bad", 1]},
        {"position": [8, 1], "size": None},
    ],
)
def test_metadata_payload_normalizes_target_geometry(target: object) -> None:
    normalized = normalize_metadata_payload(
        {
            "room": {"width": 12, "height": 7},
            "target": target,
            "obstacles": [],
            "waypoints": [],
            "body_names": [],
            "body_geometry": {},
        }
    )
    assert len(normalized["target"]["position"]) == 2
    assert all(math.isfinite(value) for value in normalized["target"]["position"])
    assert all(value > 0.0 for value in normalized["target"]["size"])

    canvas = PhysicsCanvas.__new__(PhysicsCanvas)
    canvas.metadata = normalized
    canvas.frame = normalize_frame_payload(
        {"frame": {"body_positions": [], "body_angles": [], "info": {}}}
    )
    canvas.trail = deque(maxlen=140)
    canvas.winfo_width = lambda: 800
    canvas.winfo_height = lambda: 500
    for name in (
        "delete",
        "create_text",
        "create_line",
        "create_rectangle",
        "create_oval",
        "create_polygon",
    ):
        setattr(canvas, name, lambda *args, **kwargs: None)
    PhysicsCanvas.redraw(canvas)


def test_metadata_payload_normalizes_obstacles_before_redraw() -> None:
    normalized = normalize_metadata_payload(
        {
            "room": {"width": 12, "height": 7},
            "target": {"position": [9.5, 0.55], "size": [0.8, 0.9]},
            "obstacles": [
                None,
                {"type": "box", "position": [2], "size": [1, 1]},
                {"type": "platform", "position": [3, 1], "size": ["bad", 1]},
                {"type": " WALL ", "position": [4, "2"], "size": ["1.5", 2]},
                {"type": "slope", "custom": True},
            ],
            "waypoints": [],
            "body_names": [],
            "body_geometry": {},
        }
    )
    assert normalized["obstacles"] == [
        {"type": "wall", "position": [4.0, 2.0], "size": [1.5, 2.0]},
        {"type": "slope", "custom": True},
    ]

    canvas = PhysicsCanvas.__new__(PhysicsCanvas)
    canvas.metadata = normalized
    canvas.frame = normalize_frame_payload(
        {"frame": {"body_positions": [], "body_angles": [], "info": {}}}
    )
    canvas.trail = deque(maxlen=140)
    canvas.winfo_width = lambda: 800
    canvas.winfo_height = lambda: 500
    for name in (
        "delete",
        "create_text",
        "create_line",
        "create_rectangle",
        "create_oval",
        "create_polygon",
    ):
        setattr(canvas, name, lambda *args, **kwargs: None)
    PhysicsCanvas.redraw(canvas)


def test_finite_point_distinguishes_strict_and_fill_missing_modes() -> None:
    assert finite_point([1.0, "2.0"]) == [1.0, 2.0]
    assert finite_point([1.0]) is None
    assert finite_point([1.0], fill_missing=True) == [1.0, 0.0]
    assert finite_point({"bad": 1}) is None
    assert finite_point([float("nan"), 2.0]) is None


def test_frame_payload_normalizes_nested_wrong_shapes() -> None:
    malformed = {"training": "bad", "frame": []}
    normalized = normalize_frame_payload(malformed)

    assert normalized["training"] == {"action": []}
    assert normalized["frame"] == {
        "info": {},
        "body_positions": [],
        "body_angles": [],
    }

    canvas = PhysicsCanvas.__new__(PhysicsCanvas)
    canvas.trail = deque(maxlen=140)
    canvas.last_episode = None
    canvas.metadata = {"body_names": []}
    canvas.redraw = lambda: None
    PhysicsCanvas.set_frame(canvas, malformed)
    assert canvas.frame == normalized

    valid = normalize_frame_payload(
        {
            "training": {"episode": 3, "action": [0.25]},
            "frame": {
                "body_positions": [[1.0, 2.0]],
                "body_angles": [0.5],
                "info": {"is_success": True},
                "target_position": [8.0, 1.0],
            },
        }
    )
    assert valid["training"] == {"episode": 3, "action": [0.25]}
    assert valid["frame"]["body_positions"] == [[1.0, 2.0]]
    assert valid["frame"]["body_angles"] == [0.5]
    assert valid["frame"]["info"] == {"is_success": True}
    assert valid["frame"]["target_position"] == [8.0, 1.0]

    sanitized = normalize_frame_payload(
        {
            "training": {
                "action": [{"bad": 1}, "0.5", float("nan"), float("inf")]
            },
            "frame": {
                "body_positions": [
                    {"bad": 1},
                    [1.0, "2.0"],
                    [float("nan"), 4.0],
                    [5.0],
                ],
                "body_angles": [{"bad": 1}, "0.5", float("nan"), float("inf")],
            },
        }
    )
    assert sanitized["training"]["action"] == [0.0, 0.5, 0.0, 0.0]
    assert sanitized["frame"]["body_positions"] == [
        [0.0, 0.0],
        [1.0, 2.0],
        [0.0, 4.0],
        [5.0, 0.0],
    ]
    assert sanitized["frame"]["body_angles"] == [0.0, 0.5, 0.0, 0.0]

    invalid_target = normalize_frame_payload(
        {"training": {}, "frame": {"target_position": {"bad": 1}}}
    )
    assert "target_position" not in invalid_target["frame"]
    valid_target = normalize_frame_payload(
        {"training": {}, "frame": {"target_position": ["8.0", 1.5]}}
    )
    assert valid_target["frame"]["target_position"] == [8.0, 1.5]


def test_metric_input_helpers_filter_invalid_records_and_values() -> None:
    records = metric_records(
        ["legacy-row", {"reward": 1.5}, 7, {"value_loss": 2.5}]
    )
    assert records == [{"reward": 1.5}, {"value_loss": 2.5}]
    assert metric_records({"episodes": []}) == []
    assert finite_float(1.25) == 1.25
    assert finite_float("2.5") == 2.5
    assert finite_float("not-a-number") is None
    assert finite_float({"bad": 1}) is None
    assert finite_float(float("inf")) is None


def test_nonnegative_int_handles_persisted_status_values() -> None:
    assert nonnegative_int(128) == 128
    assert nonnegative_int("256") == 256
    assert nonnegative_int(-5) == 0
    assert nonnegative_int("not-an-int") == 0
    assert nonnegative_int({"bad": 1}) == 0
    assert nonnegative_int(float("inf")) == 0


def test_desktop_formatters_handle_missing_values() -> None:
    assert format_number(None) == "—"
    assert format_number(1.234, 2) == "1.23"
    assert format_percent(None) == "—"
    assert format_percent(0.625) == "62.5%"
    assert format_percent("not-a-number") == "—"
    assert format_percent({"bad": 1}) == "—"


def test_refresh_ui_tolerates_non_string_state() -> None:
    class Widget:
        def __init__(self) -> None:
            self.options: dict[str, object] = {}

        def configure(self, **kwargs: object) -> None:
            self.options.update(kwargs)

        def delete(self, *_args: object) -> None:
            return None

        def insert(self, *_args: object) -> None:
            return None

    class Metric:
        def set(self, *_args: object) -> None:
            return None

    class Chart:
        def set_values(self, _values: object) -> None:
            return None

    class Controller:
        run_id = "repro"
        is_active = False

    app = DesktopLabApp.__new__(DesktopLabApp)
    app.status = {"state": {"bad": 1}, "total_timesteps": 64, "num_timesteps": 1}
    app.metrics = {"episodes": [], "updates": []}
    app.metadata = None
    app.frame = None
    app.frame_times = deque(maxlen=120)
    app.logs = deque(maxlen=80)
    app.frames_received = 0
    app.controller = Controller()
    for name in (
        "status_label",
        "progress",
        "progress_text",
        "stream_label",
        "start_button",
        "pause_button",
        "save_button",
        "stop_button",
        "session_text",
        "action_text",
    ):
        setattr(app, name, Widget())
    for name in ("metric_steps", "metric_reward", "metric_success", "metric_distance"):
        setattr(app, name, Metric())
    for name in ("reward_chart", "success_chart", "loss_chart"):
        setattr(app, name, Chart())

    DesktopLabApp._refresh_ui(app)

    assert "等待训练" in str(app.status_label.options["text"])
    assert app.start_button.options["state"] == "normal"


def test_run_summaries_sorts_latest_first(tmp_path: Path) -> None:
    older = tmp_path / "desktop-older"
    newer = tmp_path / "desktop-newer"
    older.mkdir()
    newer.mkdir()
    (older / "request.json").write_text(json.dumps({"stage": 1}), encoding="utf-8")
    (older / "status.json").write_text(
        json.dumps({"state": "completed", "updated_at": "2026-07-30T10:00:00"}),
        encoding="utf-8",
    )
    (newer / "request.json").write_text(json.dumps({"stage": 3}), encoding="utf-8")
    (newer / "status.json").write_text(
        json.dumps({"state": "running", "updated_at": "2026-07-30T11:00:00"}),
        encoding="utf-8",
    )

    items = run_summaries(tmp_path)

    assert [item["run_id"] for item in items] == ["desktop-newer", "desktop-older"]
    assert items[0]["request"]["stage"] == 3


def test_run_summaries_normalizes_mixed_timestamp_types(tmp_path: Path) -> None:
    numeric = tmp_path / "desktop-numeric"
    text = tmp_path / "desktop-text"
    invalid = tmp_path / "desktop-invalid"
    for run_dir in (numeric, text, invalid):
        run_dir.mkdir()
        (run_dir / "request.json").write_text(json.dumps({"stage": 1}), encoding="utf-8")
    (numeric / "status.json").write_text(
        json.dumps({"state": "completed", "updated_at": 4_102_444_800}),
        encoding="utf-8",
    )
    (text / "status.json").write_text(
        json.dumps({"state": "completed", "updated_at": "2026-07-31T00:00:00"}),
        encoding="utf-8",
    )
    (invalid / "status.json").write_text(
        json.dumps({"state": "failed", "updated_at": {"unexpected": True}}),
        encoding="utf-8",
    )

    items = run_summaries(tmp_path)

    assert items[0]["run_id"] == "desktop-numeric"
    assert {item["run_id"] for item in items} == {
        "desktop-numeric",
        "desktop-text",
        "desktop-invalid",
    }


def test_run_summaries_rejects_non_object_json_payloads(tmp_path: Path) -> None:
    run_dir = tmp_path / "desktop-invalid-shape"
    run_dir.mkdir()
    (run_dir / "request.json").write_text('["legacy", 1]', encoding="utf-8")
    (run_dir / "status.json").write_text('"completed"', encoding="utf-8")

    items = run_summaries(tmp_path)

    assert len(items) == 1
    assert items[0]["run_id"] == "desktop-invalid-shape"
    assert items[0]["request"] == {}
    assert items[0]["status"] == {}
    assert items[0]["updated_at"] == ""


def test_run_summaries_retries_locked_status_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "desktop-locked"
    run_dir.mkdir()
    (run_dir / "request.json").write_text(json.dumps({"stage": 2}), encoding="utf-8")
    status_path = run_dir / "status.json"
    status_path.write_text(
        json.dumps({"state": "running", "updated_at": "2026-07-30T12:00:00"}),
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

    items = run_summaries(tmp_path)

    assert attempts == 2
    assert items[0]["status"]["state"] == "running"
