from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO


def actor_parameter_names(model: PPO) -> tuple[str, ...]:
    return tuple(
        name
        for name, _ in model.policy.named_parameters()
        if "mlp_extractor.policy_net" in name or name.startswith("action_net")
    )


def interpolate_actor(source: PPO, target: PPO, alpha: float) -> dict[str, float]:
    """Move only the deterministic actor parameters from source toward target."""
    source_parameters = dict(source.policy.named_parameters())
    target_parameters = dict(target.policy.named_parameters())
    names = actor_parameter_names(source)
    if not names:
        raise RuntimeError("No actor parameters were found")
    if names != actor_parameter_names(target):
        raise ValueError("Source and target actor parameter layouts differ")

    absolute_deltas: list[np.ndarray] = []
    with torch.no_grad():
        for name in names:
            source_parameter = source_parameters[name]
            target_parameter = target_parameters[name]
            if source_parameter.shape != target_parameter.shape:
                raise ValueError(f"Actor parameter shape mismatch for {name}")
            delta = target_parameter.detach() - source_parameter.detach()
            source_parameter.add_(float(alpha) * delta)
            absolute_deltas.append(torch.abs(float(alpha) * delta).cpu().numpy().ravel())
    concatenated = np.concatenate(absolute_deltas)
    return {
        "actor_parameter_count": float(concatenated.size),
        "mean_absolute_parameter_delta": float(np.mean(concatenated)),
        "max_absolute_parameter_delta": float(np.max(concatenated)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interpolate only PPO actor parameters between two compatible checkpoints."
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    source = PPO.load(str(args.source), device="auto")
    target = PPO.load(str(args.target), device="auto")
    if source.observation_space != target.observation_space:
        raise ValueError("Source and target observation spaces differ")
    if source.action_space != target.action_space:
        raise ValueError("Source and target action spaces differ")

    stats = interpolate_actor(source, target, args.alpha)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    source.save(str(args.output))
    output_zip = args.output.with_suffix(".zip")
    report = {
        "source_model": str(args.source),
        "target_model": str(args.target),
        "output_model": str(output_zip),
        "alpha": float(args.alpha),
        **stats,
    }
    report_path = args.report or args.output.with_suffix(".json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
