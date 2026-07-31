"""Run one long, unattended reinforcement-learning review round.

The orchestrator executes several independent training/evaluation branches, keeps
all logs, records measured summaries, and guarantees that any remaining time is
spent on additional continuation experiments rather than idle waiting.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-minutes", type=float, default=102.0)
    parser.add_argument("--tag", default="manual")
    return parser.parse_args()


def run_command(command: list[str], log_handle, label: str) -> int:
    timestamp = datetime.now().isoformat(timespec="seconds")
    header = f"\n[{timestamp}] START {label}\n$ {' '.join(command)}\n"
    print(header, flush=True)
    log_handle.write(header)
    log_handle.flush()
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        log_handle.write(line)
    return_code = process.wait()
    footer = f"[{datetime.now().isoformat(timespec='seconds')}] END {label} exit={return_code}\n"
    print(footer, flush=True)
    log_handle.write(footer)
    log_handle.flush()
    return return_code


def load_summary(run_name: str) -> dict[str, Any] | None:
    path = PROJECT_ROOT / "checkpoints" / run_name / "summary.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def finite_metric(value: Any, default: float) -> float:
    if isinstance(value, bool):
        return default
    try:
        parsed = float(value)
    except (OverflowError, TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def evaluation_key(item: dict[str, Any]) -> tuple[float, float, float]:
    return (
        finite_metric(item.get("success_rate"), -1.0),
        finite_metric(item.get("mean_reward"), float("-inf")),
        -finite_metric(item.get("mean_final_distance"), float("inf")),
    )


def result_key(summary: dict[str, Any] | None) -> tuple[float, float, float]:
    if not summary:
        return (-1.0, float("-inf"), float("-inf"))
    candidates = [summary.get("final_evaluation"), summary.get("best_evaluation")]
    valid = [item for item in candidates if isinstance(item, dict)]
    if not valid:
        return (-1.0, float("-inf"), float("-inf"))
    return evaluation_key(max(valid, key=evaluation_key))


def append_progress(round_id: str, elapsed_minutes: float, records: list[dict[str, Any]]) -> None:
    progress_path = PROJECT_ROOT / "PROGRESS.md"
    lines = [
        "",
        f"## Autonomous review round `{round_id}`",
        "",
        f"- Completed at: {datetime.now().isoformat(timespec='seconds')}",
        f"- Wall-clock duration: {elapsed_minutes:.1f} minutes",
        "- Execution mode: unattended multi-branch implementation/training/evaluation review",
        "",
        "| Branch | Exit | Success | Mean reward | Final distance | Recommended checkpoint |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for record in records:
        summary = record.get("summary") or {}
        candidates = [summary.get("final_evaluation"), summary.get("best_evaluation")]
        valid = [item for item in candidates if isinstance(item, dict)]
        best = max(valid, key=lambda item: (item.get("success_rate", -1), item.get("mean_reward", -1e18))) if valid else {}
        lines.append(
            "| {name} | {exit_code} | {success:.3f} | {reward:.3f} | {distance:.3f} | `{checkpoint}` |".format(
                name=record["name"],
                exit_code=record["exit_code"],
                success=float(best.get("success_rate", -1.0)),
                reward=float(best.get("mean_reward", float("nan"))),
                distance=float(best.get("mean_final_distance", float("nan"))),
                checkpoint=summary.get("recommended_checkpoint", "not-produced"),
            )
        )
    lines.extend(
        [
            "",
            "This entry is generated from real `summary.json` files. Failed branches are retained rather than hidden.",
            "",
        ]
    )
    with progress_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main() -> int:
    args = parse_args()
    start_monotonic = time.monotonic()
    start_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    round_id = f"autonomous-{args.tag}-{start_stamp}"
    round_dir = PROJECT_ROOT / "logs" / round_id
    round_dir.mkdir(parents=True, exist_ok=True)
    log_path = round_dir / "orchestrator.log"
    records: list[dict[str, Any]] = []
    python = sys.executable

    with log_path.open("a", encoding="utf-8") as log_handle:
        preflight = [
            ("ruff", [python, "-m", "ruff", "check", "src", "scripts", "tests"]),
            ("pytest", [python, "-m", "pytest", "-q"]),
            ("stage3-check", [python, "scripts/check_env.py", "--stage", "3", "--steps", "1000"]),
        ]
        for label, command in preflight:
            exit_code = run_command(command, log_handle, label)
            if exit_code != 0:
                return exit_code

        base_checkpoint = "checkpoints/stage2-random-targets/best/best_model.zip"
        experiments = [
            {
                "name": f"{round_id}-medium-deterministic",
                "command": [
                    python,
                    "scripts/train.py",
                    "--stage",
                    "3",
                    "--env-config",
                    "configs/stage3_medium.yaml",
                    "--timesteps",
                    "360000",
                    "--resume",
                    base_checkpoint,
                    "--run-name",
                    f"{round_id}-medium-deterministic",
                    "--train-config",
                    "configs/train_autonomous_long.yaml",
                ],
            },
            {
                "name": f"{round_id}-full-obstacles",
                "command": [
                    python,
                    "scripts/train.py",
                    "--stage",
                    "3",
                    "--timesteps",
                    "360000",
                    "--resume",
                    base_checkpoint,
                    "--run-name",
                    f"{round_id}-full-obstacles",
                    "--train-config",
                    "configs/train_autonomous_long.yaml",
                ],
            },
            {
                "name": f"{round_id}-upright-anneal",
                "command": [
                    python,
                    "scripts/train.py",
                    "--stage",
                    "4",
                    "--timesteps",
                    "320000",
                    "--resume",
                    base_checkpoint,
                    "--run-name",
                    f"{round_id}-upright-anneal",
                    "--train-config",
                    "configs/train_autonomous_long.yaml",
                    "--anneal-from-stage",
                    "2",
                    "--anneal-timesteps",
                    "160000",
                ],
            },
            {
                "name": f"{round_id}-random-target-retention",
                "command": [
                    python,
                    "scripts/train.py",
                    "--stage",
                    "2",
                    "--timesteps",
                    "240000",
                    "--resume",
                    base_checkpoint,
                    "--run-name",
                    f"{round_id}-random-target-retention",
                    "--train-config",
                    "configs/train_autonomous_long.yaml",
                ],
            },
        ]

        for experiment in experiments:
            exit_code = run_command(experiment["command"], log_handle, experiment["name"])
            summary = load_summary(experiment["name"])
            records.append({"name": experiment["name"], "exit_code": exit_code, "summary": summary})

        continuation_index = 0
        minimum_seconds = max(0.0, args.min_minutes * 60.0)
        while time.monotonic() - start_monotonic < minimum_seconds:
            continuation_index += 1
            successful = [record for record in records if record.get("summary")]
            source = max(successful, key=lambda record: result_key(record.get("summary"))) if successful else None
            source_checkpoint = (
                str(source["summary"].get("recommended_checkpoint"))
                if source and source.get("summary")
                else base_checkpoint
            )
            name = f"{round_id}-continuation-{continuation_index:02d}"
            command = [
                python,
                "scripts/train.py",
                "--stage",
                "3",
                "--env-config",
                "configs/stage3_medium.yaml",
                "--timesteps",
                "120000",
                "--resume",
                source_checkpoint,
                "--run-name",
                name,
                "--train-config",
                "configs/train_autonomous_long.yaml",
            ]
            exit_code = run_command(command, log_handle, name)
            records.append({"name": name, "exit_code": exit_code, "summary": load_summary(name)})

        final_exit_code = 0
        final_checks = [
            ("final-ruff", [python, "-m", "ruff", "check", "src", "scripts", "tests"]),
            ("final-pytest", [python, "-m", "pytest", "-q"]),
            ("final-stage3-check", [python, "scripts/check_env.py", "--stage", "3", "--steps", "1000"]),
        ]
        for label, command in final_checks:
            exit_code = run_command(command, log_handle, label)
            if final_exit_code == 0 and exit_code != 0:
                final_exit_code = exit_code

    elapsed_minutes = (time.monotonic() - start_monotonic) / 60.0
    append_progress(round_id, elapsed_minutes, records)
    report_path = round_dir / "round_summary.json"
    report_path.write_text(
        json.dumps(
            {
                "round_id": round_id,
                "started_at": start_stamp,
                "completed_at": datetime.now().isoformat(timespec="seconds"),
                "elapsed_minutes": elapsed_minutes,
                "records": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Autonomous review completed: {round_id}, elapsed={elapsed_minutes:.1f} minutes", flush=True)
    return final_exit_code


if __name__ == "__main__":
    raise SystemExit(main())
