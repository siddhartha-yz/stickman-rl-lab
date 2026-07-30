from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from stickman_rl.desktop.app import (
    format_number,
    format_percent,
    nonnegative_int,
    rotate_point,
    run_summaries,
)


def test_rotate_point_matches_quarter_turn() -> None:
    x, y = rotate_point([1.0, 0.0], math.pi / 2, [2.0, 3.0])
    assert math.isclose(x, 2.0, abs_tol=1e-8)
    assert math.isclose(y, 4.0, abs_tol=1e-8)


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
