from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "lab" / "experiments.json"


def _command(experiment: dict[str, Any]) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "record_trajectory.py"),
        "--stage",
        str(experiment["stage"]),
        "--seed",
        str(experiment["seed"]),
        "--max-steps",
        "900",
        "--output",
        str(PROJECT_ROOT / experiment["trajectory"]),
    ]
    if experiment.get("model"):
        command.extend(["--model", str(PROJECT_ROOT / experiment["model"])])
    if experiment.get("env_config"):
        command.extend(["--env-config", str(PROJECT_ROOT / experiment["env_config"])])
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description="Export all browser-replay trajectories from lab/experiments.json.")
    parser.add_argument("--force", action="store_true", help="Regenerate trajectories that already exist.")
    parser.add_argument("--only", action="append", default=[], help="Export only the selected experiment id.")
    args = parser.parse_args()

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    selected = set(args.only)
    failures: list[str] = []
    for experiment in manifest["experiments"]:
        if selected and experiment["id"] not in selected:
            continue
        output = PROJECT_ROOT / experiment["trajectory"]
        if output.exists() and not args.force:
            print(f"skip {experiment['id']}: {output.relative_to(PROJECT_ROOT)}")
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        print(f"export {experiment['id']} -> {output.relative_to(PROJECT_ROOT)}", flush=True)
        completed = subprocess.run(_command(experiment), cwd=PROJECT_ROOT, check=False)
        if completed.returncode != 0:
            failures.append(experiment["id"])
    if failures:
        raise SystemExit(f"Failed exports: {', '.join(failures)}")
    print("all requested lab trajectories exported")


if __name__ == "__main__":
    main()
