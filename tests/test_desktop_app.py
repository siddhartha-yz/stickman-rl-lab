from __future__ import annotations

import json
import math
from pathlib import Path

from stickman_rl.desktop.app import format_number, format_percent, rotate_point, run_summaries


def test_rotate_point_matches_quarter_turn() -> None:
    x, y = rotate_point([1.0, 0.0], math.pi / 2, [2.0, 3.0])
    assert math.isclose(x, 2.0, abs_tol=1e-8)
    assert math.isclose(y, 4.0, abs_tol=1e-8)


def test_desktop_formatters_handle_missing_values() -> None:
    assert format_number(None) == "—"
    assert format_number(1.234, 2) == "1.23"
    assert format_percent(None) == "—"
    assert format_percent(0.625) == "62.5%"


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
