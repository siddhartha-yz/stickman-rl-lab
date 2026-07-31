from __future__ import annotations

import argparse
from pathlib import Path

import pytest

import scripts.autonomous_review_round as review_module


def test_main_stops_before_training_when_preflight_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels: list[str] = []

    def fake_run_command(command: list[str], log_handle: object, label: str) -> int:
        del command, log_handle
        labels.append(label)
        return 7 if label == "pytest" else 0

    monkeypatch.setattr(review_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        review_module,
        "parse_args",
        lambda: argparse.Namespace(min_minutes=0.0, tag="preflight-failure"),
    )
    monkeypatch.setattr(review_module, "run_command", fake_run_command)
    monkeypatch.setattr(review_module, "load_summary", lambda _name: None)
    monkeypatch.setattr(review_module, "append_progress", lambda *_args: None)

    assert review_module.main() == 7
    assert labels == ["ruff", "pytest"]
    assert not any("medium-deterministic" in label for label in labels)
