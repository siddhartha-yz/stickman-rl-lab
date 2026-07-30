from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Stable-Baselines3 evaluation rewards and success rates.")
    parser.add_argument("evaluations", type=Path, help="Path to evaluations.npz")
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--prefix", type=str, default="evaluation")
    args = parser.parse_args()

    data = np.load(args.evaluations)
    timesteps = np.asarray(data["timesteps"], dtype=np.int64)
    results = np.asarray(data["results"], dtype=np.float64)
    successes = np.asarray(data["successes"], dtype=np.float64) if "successes" in data.files else None
    args.output_dir.mkdir(parents=True, exist_ok=True)

    reward_mean = results.mean(axis=1)
    reward_std = results.std(axis=1)
    figure = plt.figure(figsize=(8, 4.5))
    axis = figure.add_subplot(111)
    axis.plot(timesteps, reward_mean, marker="o")
    axis.fill_between(timesteps, reward_mean - reward_std, reward_mean + reward_std, alpha=0.2)
    axis.set_xlabel("Training timesteps")
    axis.set_ylabel("Episode reward")
    axis.set_title("Deterministic evaluation reward")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    reward_path = args.output_dir / f"{args.prefix}-reward.png"
    figure.savefig(reward_path, dpi=160)
    plt.close(figure)
    print(f"saved: {reward_path}")

    if successes is not None:
        success_rate = successes.mean(axis=1)
        figure = plt.figure(figsize=(8, 4.5))
        axis = figure.add_subplot(111)
        axis.plot(timesteps, success_rate, marker="o")
        axis.set_ylim(-0.05, 1.05)
        axis.set_xlabel("Training timesteps")
        axis.set_ylabel("Success rate")
        axis.set_title("Deterministic evaluation success")
        axis.grid(True, alpha=0.3)
        figure.tight_layout()
        success_path = args.output_dir / f"{args.prefix}-success.png"
        figure.savefig(success_path, dpi=160)
        plt.close(figure)
        print(f"saved: {success_path}")


if __name__ == "__main__":
    main()
