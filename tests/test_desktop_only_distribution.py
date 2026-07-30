from __future__ import annotations

from stickman_rl.config import PROJECT_ROOT

REMOVED_WEB_PATHS = [
    "frontend",
    "scripts/training_api.py",
    "scripts/lab_server.py",
    "scripts/run_lab.py",
    "scripts/export_lab_assets.py",
    "tests/test_training_api.py",
    "tests/test_lab_observer.py",
    "lab/experiments.json",
]


def test_webui_files_are_absent_from_desktop_distribution() -> None:
    for relative_path in REMOVED_WEB_PATHS:
        assert not (PROJECT_ROOT / relative_path).exists(), relative_path


def test_runtime_requirements_exclude_web_stack() -> None:
    requirements = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    for package in ("fastapi", "uvicorn", "websockets"):
        assert package not in requirements


def test_live_worker_has_no_websocket_or_fastapi_transport() -> None:
    source = (PROJECT_ROOT / "scripts" / "live_train_worker.py").read_text(encoding="utf-8").lower()
    assert "--stream-url" not in source
    assert "websockets" not in source
    assert "fastapi" not in source
    assert "--stream-stdout" in source


def test_native_desktop_launcher_remains_available() -> None:
    launcher = PROJECT_ROOT / "scripts" / "run_desktop.py"
    assert launcher.is_file()
    assert "launch_desktop" in launcher.read_text(encoding="utf-8")
