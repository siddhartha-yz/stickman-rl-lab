"""YAML-backed configuration loading with recursive overrides."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge dictionaries without mutating either input."""
    result: dict[str, Any] = deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_yaml(path: str | Path) -> dict[str, Any]:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    if not resolved.exists():
        raise FileNotFoundError(f"Configuration file not found: {resolved}")
    with resolved.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected a mapping in {resolved}, got {type(data).__name__}")
    return data


def load_env_config(stage: int = 1, config_path: str | Path | None = None) -> dict[str, Any]:
    config = load_yaml(CONFIG_DIR / "base.yaml")
    stage_path = CONFIG_DIR / f"stage{stage}.yaml"
    if stage_path.exists():
        config = deep_merge(config, load_yaml(stage_path))
    if config_path is not None:
        config = deep_merge(config, load_yaml(config_path))
    reward_cfg = load_yaml(CONFIG_DIR / "rewards.yaml")
    reward_cfg.update(config.pop("reward_overrides", {}))
    config["rewards"] = reward_cfg
    config["stage"] = int(config.get("stage", stage))
    return config


def load_train_config(path: str | Path | None = None) -> dict[str, Any]:
    return load_yaml(path or CONFIG_DIR / "train.yaml")
