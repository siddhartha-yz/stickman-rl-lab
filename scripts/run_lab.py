from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_INDEX = PROJECT_ROOT / "frontend" / "dist" / "index.html"


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the Stickman RL Lab live training console.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    if not FRONTEND_INDEX.exists():
        raise SystemExit(
            "Frontend build is missing. Run `cd frontend`, `npm install`, and `npm run build` first."
        )
    print(f"Stickman RL Lab live training console: http://{args.host}:{args.port}")
    uvicorn.run(
        "scripts.lab_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        app_dir=str(PROJECT_ROOT),
    )


if __name__ == "__main__":
    main()
