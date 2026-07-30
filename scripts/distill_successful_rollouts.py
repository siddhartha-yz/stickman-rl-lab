from __future__ import annotations

import argparse
import json
from math import ceil
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from stickman_rl.config import load_env_config
from stickman_rl.env import StickmanReachEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Distill successful stochastic PPO rollouts into the deterministic actor mean."
    )
    parser.add_argument("model", type=Path)
    parser.add_argument("--stage", type=int, default=3)
    parser.add_argument("--env-config", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--max-successes", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--anchor-coef", type=float, default=1.0)
    parser.add_argument("--final-phase-weight", type=float, default=4.0)
    parser.add_argument(
        "--final-batch-fraction",
        type=float,
        default=None,
        help="Optional fraction of every batch sampled from the post-waypoint phase.",
    )
    parser.add_argument("--input-dataset", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def collect_successful_rollouts(
    model: PPO,
    env: StickmanReachEnv,
    episodes: int,
    max_successes: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    successful_observations: list[np.ndarray] = []
    successful_actions: list[np.ndarray] = []
    successful_phases: list[int] = []
    attempted = 0
    successes = 0
    total_success_steps = 0

    for episode in range(episodes):
        attempted += 1
        observation, _ = env.reset(seed=seed + episode)
        episode_observations: list[np.ndarray] = []
        episode_actions: list[np.ndarray] = []
        episode_phases: list[int] = []
        done = False
        info: dict[str, object] = {}
        while not done:
            phase = env.active_waypoint_index
            action, _ = model.predict(observation, deterministic=False)
            episode_observations.append(observation.copy())
            episode_actions.append(np.asarray(action, dtype=np.float32).copy())
            episode_phases.append(phase)
            observation, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        if bool(info.get("is_success", False)):
            successes += 1
            total_success_steps += len(episode_observations)
            successful_observations.extend(episode_observations)
            successful_actions.extend(episode_actions)
            successful_phases.extend(episode_phases)
            print(
                f"success {successes}/{attempted}: steps={len(episode_observations)}, "
                f"final_distance={float(info.get('final_distance', 0.0)):.3f}",
                flush=True,
            )
            if successes >= max_successes:
                break
        elif attempted % 5 == 0:
            print(f"collection progress: successes={successes}/{attempted}", flush=True)

    if successes == 0:
        raise RuntimeError("No successful stochastic rollouts were collected")
    observations = np.asarray(successful_observations, dtype=np.float32)
    actions = np.asarray(successful_actions, dtype=np.float32)
    phases = np.asarray(successful_phases, dtype=np.int64)
    stats = {
        "attempted_episodes": float(attempted),
        "successful_episodes": float(successes),
        "collection_success_rate": float(successes / attempted),
        "successful_steps": float(total_success_steps),
    }
    return observations, actions, phases, stats


def actor_parameters(model: PPO) -> list[torch.nn.Parameter]:
    return [
        parameter
        for name, parameter in model.policy.named_parameters()
        if "mlp_extractor.policy_net" in name or name.startswith("action_net")
    ]


def distribution_mean(model: PPO, observations: np.ndarray) -> torch.Tensor:
    observation_tensor, _ = model.policy.obs_to_tensor(observations)
    return model.policy.get_distribution(observation_tensor).distribution.mean


def epoch_batch_indices(
    rng: np.random.Generator,
    sample_count: int,
    batch_size: int,
    final_mask: np.ndarray,
    final_batch_fraction: float | None,
) -> list[np.ndarray]:
    """Build natural or phase-balanced batches for one distillation epoch."""
    all_indices = np.arange(sample_count)
    if final_batch_fraction is None:
        rng.shuffle(all_indices)
        return [all_indices[start : start + batch_size] for start in range(0, sample_count, batch_size)]
    if not 0.0 < final_batch_fraction < 1.0:
        raise ValueError("final-batch-fraction must be strictly between 0 and 1")
    final_indices = np.flatnonzero(final_mask)
    anchor_indices = np.flatnonzero(~final_mask)
    if len(final_indices) == 0 or len(anchor_indices) == 0:
        raise ValueError("Balanced batches require both final and pre-final samples")
    final_count = max(1, min(batch_size - 1, int(round(batch_size * final_batch_fraction))))
    anchor_count = batch_size - final_count
    batches: list[np.ndarray] = []
    for _ in range(ceil(sample_count / batch_size)):
        final_batch = rng.choice(final_indices, size=final_count, replace=len(final_indices) < final_count)
        anchor_batch = rng.choice(anchor_indices, size=anchor_count, replace=len(anchor_indices) < anchor_count)
        batch = np.concatenate([final_batch, anchor_batch])
        rng.shuffle(batch)
        batches.append(batch)
    return batches


def distill_actor(
    model: PPO,
    observations: np.ndarray,
    actions: np.ndarray,
    phases: np.ndarray,
    waypoint_count: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    anchor_coef: float,
    final_phase_weight: float,
    final_batch_fraction: float | None,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model.policy.set_training_mode(True)
    with torch.no_grad():
        anchor_means = distribution_mean(model, observations).detach().cpu().numpy()

    final_mask = phases >= waypoint_count
    if not np.any(final_mask):
        raise RuntimeError("Successful trajectories contain no post-waypoint samples")
    targets = anchor_means.copy()
    targets[final_mask] = actions[final_mask]
    sample_weights = np.ones(len(observations), dtype=np.float32)
    sample_weights[final_mask] = float(final_phase_weight)

    parameters = actor_parameters(model)
    if not parameters:
        raise RuntimeError("No actor parameters were found")
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    last_loss = 0.0
    last_target_loss = 0.0
    last_anchor_loss = 0.0

    for epoch in range(epochs):
        epoch_losses: list[float] = []
        epoch_targets: list[float] = []
        epoch_anchors: list[float] = []
        batches = epoch_batch_indices(
            rng,
            sample_count=len(observations),
            batch_size=batch_size,
            final_mask=final_mask,
            final_batch_fraction=final_batch_fraction,
        )
        for batch_indices in batches:
            batch_observations = observations[batch_indices]
            prediction = distribution_mean(model, batch_observations)
            device = prediction.device
            target = torch.as_tensor(targets[batch_indices], device=device)
            anchor = torch.as_tensor(anchor_means[batch_indices], device=device)
            weights = torch.as_tensor(sample_weights[batch_indices], device=device)
            target_error = torch.mean(torch.square(prediction - target), dim=1)
            anchor_error = torch.mean(torch.square(prediction - anchor), dim=1)
            target_loss = torch.mean(weights * target_error)
            anchor_loss = torch.mean(anchor_error)
            loss = target_loss + anchor_coef * anchor_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=0.5)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
            epoch_targets.append(float(target_loss.detach().cpu()))
            epoch_anchors.append(float(anchor_loss.detach().cpu()))
        last_loss = float(np.mean(epoch_losses))
        last_target_loss = float(np.mean(epoch_targets))
        last_anchor_loss = float(np.mean(epoch_anchors))
        print(
            f"epoch={epoch + 1}/{epochs} loss={last_loss:.6f} "
            f"target={last_target_loss:.6f} anchor={last_anchor_loss:.6f}",
            flush=True,
        )

    model.policy.set_training_mode(False)
    return {
        "samples": float(len(observations)),
        "final_phase_samples": float(np.count_nonzero(final_mask)),
        "final_phase_fraction": float(np.mean(final_mask)),
        "final_loss": last_loss,
        "final_target_loss": last_target_loss,
        "final_anchor_loss": last_anchor_loss,
    }


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 1:
        raise ValueError("epochs must be positive and batch-size must be greater than one")
    if args.input_dataset is None and (args.episodes <= 0 or args.max_successes <= 0):
        raise ValueError("episodes and max-successes must be positive when collecting rollouts")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = PPO.load(str(args.model), device="auto")
    config = load_env_config(stage=args.stage, config_path=args.env_config)
    waypoint_count = len(config.get("navigation", {}).get("waypoints", []))
    if args.input_dataset is not None:
        with np.load(args.input_dataset) as data:
            required = {"observations", "actions", "phases"}
            missing = required.difference(data.files)
            if missing:
                raise ValueError(f"Input dataset is missing arrays: {sorted(missing)}")
            observations = np.asarray(data["observations"], dtype=np.float32)
            actions = np.asarray(data["actions"], dtype=np.float32)
            phases = np.asarray(data["phases"], dtype=np.int64)
        if not (len(observations) == len(actions) == len(phases)):
            raise ValueError("Input dataset arrays have different sample counts")
        collection_stats = {
            "input_dataset_samples": float(len(observations)),
            "input_dataset_final_samples": float(np.count_nonzero(phases >= waypoint_count)),
        }
    else:
        env = StickmanReachEnv(config=config)
        try:
            observations, actions, phases, collection_stats = collect_successful_rollouts(
                model=model,
                env=env,
                episodes=args.episodes,
                max_successes=args.max_successes,
                seed=args.seed,
            )
        finally:
            env.close()

    args.dataset.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.dataset,
        observations=observations,
        actions=actions,
        phases=phases,
    )
    distillation_stats = distill_actor(
        model=model,
        observations=observations,
        actions=actions,
        phases=phases,
        waypoint_count=waypoint_count,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        anchor_coef=args.anchor_coef,
        final_phase_weight=args.final_phase_weight,
        final_batch_fraction=args.final_batch_fraction,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(args.output))
    output_zip = args.output.with_suffix(".zip")
    report = {
        "source_model": str(args.model),
        "output_model": str(output_zip),
        "env_config": str(args.env_config),
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "anchor_coef": args.anchor_coef,
        "final_phase_weight": args.final_phase_weight,
        "final_batch_fraction": args.final_batch_fraction,
        "input_dataset": str(args.input_dataset) if args.input_dataset is not None else None,
        **collection_stats,
        **distillation_stats,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
