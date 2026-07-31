from __future__ import annotations

import argparse
from pathlib import Path

import pytest

import scripts.autonomous_review_round as review_module


def test_result_key_uses_valid_candidate_when_other_metrics_are_malformed() -> None:
    summary = {
        "final_evaluation": {
            "success_rate": "not-a-number",
            "mean_reward": {"bad": 1},
            "mean_final_distance": [],
        },
        "best_evaluation": {
            "success_rate": 0.5,
            "mean_reward": 12.0,
            "mean_final_distance": 1.25,
        },
    }

    assert review_module.result_key(summary) == pytest.approx((0.5, 12.0, -1.25))


def test_append_progress_uses_valid_candidate_when_other_metrics_are_malformed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    progress_path = tmp_path / "PROGRESS.md"
    progress_path.write_text("# Progress\n", encoding="utf-8")
    monkeypatch.setattr(review_module, "PROJECT_ROOT", tmp_path)
    records = [
        {
            "name": "branch-a",
            "exit_code": 0,
            "summary": {
                "final_evaluation": {
                    "success_rate": "not-a-number",
                    "mean_reward": {"bad": 1},
                    "mean_final_distance": [],
                },
                "best_evaluation": {
                    "success_rate": 0.5,
                    "mean_reward": 12.0,
                    "mean_final_distance": 1.25,
                },
                "recommended_checkpoint": "checkpoints/branch-a/model.zip",
            },
        }
    ]

    review_module.append_progress("round-a", 1.5, records)

    text = progress_path.read_text(encoding="utf-8")
    assert "| branch-a | 0 | 0.500 | 12.000 | 1.250 |" in text
    assert "`checkpoints/branch-a/model.zip`" in text


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


def test_main_returns_failed_final_validation_after_preserving_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels: list[str] = []
    progress_rounds: list[str] = []

    def fake_run_command(command: list[str], log_handle: object, label: str) -> int:
        del command, log_handle
        labels.append(label)
        return 9 if label == "final-pytest" else 0

    monkeypatch.setattr(review_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        review_module,
        "parse_args",
        lambda: argparse.Namespace(min_minutes=0.0, tag="final-failure"),
    )
    monkeypatch.setattr(review_module, "run_command", fake_run_command)
    monkeypatch.setattr(review_module, "load_summary", lambda _name: None)
    monkeypatch.setattr(
        review_module,
        "append_progress",
        lambda round_id, *_args: progress_rounds.append(round_id),
    )

    assert review_module.main() == 9
    assert labels[-3:] == ["final-ruff", "final-pytest", "final-stage3-check"]
    assert len(progress_rounds) == 1
    assert len(list((tmp_path / "logs").glob("*/round_summary.json"))) == 1
