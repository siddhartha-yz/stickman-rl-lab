from __future__ import annotations

import argparse
from pathlib import Path

from stickman_rl.config import PROJECT_ROOT
from stickman_rl.desktop.app import launch_desktop


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the native Stickman RL Lab desktop console.")
    parser.add_argument("--smoke", action="store_true", help="Run a 64-step native UI smoke test and exit.")
    parser.add_argument(
        "--smoke-output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "desktop-smoke.json",
    )
    parser.add_argument(
        "--smoke-screenshot",
        type=Path,
        default=PROJECT_ROOT / "reports" / "desktop-training-console.png",
    )
    args = parser.parse_args()
    launch_desktop(
        smoke_output=args.smoke_output.resolve() if args.smoke else None,
        smoke_screenshot=args.smoke_screenshot.resolve() if args.smoke else None,
    )


if __name__ == "__main__":
    main()
