"""PPO training orchestration and checkpoint management."""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from numbers import Integral
from pathlib import Path
from typing import Any

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
    StopTrainingOnNoModelImprovement,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import FloatSchedule

from stickman_rl.callbacks import EnvironmentMetricsCallback, RewardAnnealingCallback
from stickman_rl.config import PROJECT_ROOT, load_env_config, load_train_config
from stickman_rl.env import StickmanReachEnv
from stickman_rl.evaluation import evaluate_policy_path


def _make_monitored_env(stage: int, seed: int, env_config_path: str | Path | None = None):
    def factory() -> Monitor:
        return Monitor(StickmanReachEnv(config=load_env_config(stage=stage, config_path=env_config_path)), info_keywords=("is_success", "final_distance", "mean_energy", "torso_height"))

    return factory


def train_ppo(
    stage: int = 1,
    total_timesteps: int | None = None,
    resume: str | Path | None = None,
    seed: int | None = None,
    run_name: str | None = None,
    train_config_path: str | Path | None = None,
    anneal_from_stage: int | None = None,
    anneal_timesteps: int | None = None,
    env_config_path: str | Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Train or resume PPO and return the final checkpoint plus evaluation summary."""
    train_cfg = load_train_config(train_config_path)
    env_cfg = load_env_config(stage=stage, config_path=env_config_path)
    actual_seed = int(env_cfg.get("seed", 0) if seed is None else seed)
    timestep_value = train_cfg["total_timesteps"] if total_timesteps is None else total_timesteps
    if isinstance(timestep_value, bool) or not isinstance(timestep_value, Integral):
        raise ValueError("total_timesteps must be an integer")
    timesteps = int(timestep_value)
    if timesteps < 1:
        raise ValueError("total_timesteps must be at least 1")
    run_id = run_name or f"stage{stage}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    checkpoint_dir = PROJECT_ROOT / "checkpoints" / run_id
    log_dir = PROJECT_ROOT / "logs" / run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "env_config.yaml").write_text(
        yaml.safe_dump(env_cfg, sort_keys=False),
        encoding="utf-8",
    )
    (checkpoint_dir / "train_config.yaml").write_text(
        yaml.safe_dump(train_cfg, sort_keys=False),
        encoding="utf-8",
    )
    package_versions: dict[str, str] = {}
    for package in ("gymnasium", "pymunk", "pygame", "stable-baselines3", "torch", "numpy"):
        try:
            package_versions[package] = version(package)
        except PackageNotFoundError:
            package_versions[package] = "not-installed"
    run_metadata = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version,
        "platform": platform.platform(),
        "stage": stage,
        "seed": actual_seed,
        "resume": str(resume) if resume else None,
        "anneal_from_stage": anneal_from_stage,
        "anneal_timesteps": anneal_timesteps,
        "env_config_path": str(env_config_path) if env_config_path else None,
        "packages": package_versions,
    }
    (checkpoint_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2),
        encoding="utf-8",
    )

    n_envs = int(train_cfg.get("n_envs", 1))
    env = make_vec_env(_make_monitored_env(stage, actual_seed, env_config_path), n_envs=n_envs, seed=actual_seed)
    try:
        eval_env = make_vec_env(
            _make_monitored_env(stage, actual_seed + 10000, env_config_path),
            n_envs=1,
            seed=actual_seed + 10000,
        )
    except BaseException:  # noqa: BLE001 - release the allocated training env before propagating
        env.close()
        raise
    try:
        policy_kwargs = {
            "net_arch": list(train_cfg.get("policy_layers", [256, 256])),
            "log_std_init": float(train_cfg.get("log_std_init", 0.0)),
        }
        if resume:
            model = PPO.load(str(resume), env=env, tensorboard_log=str(log_dir), device="auto")
            model.set_random_seed(actual_seed)
            learning_rate = float(train_cfg["learning_rate"])
            model.learning_rate = learning_rate
            model.lr_schedule = FloatSchedule(learning_rate)
            for parameter_group in model.policy.optimizer.param_groups:
                parameter_group["lr"] = learning_rate
            model.clip_range = FloatSchedule(float(train_cfg["clip_range"]))
            model.gamma = float(train_cfg["gamma"])
            model.gae_lambda = float(train_cfg["gae_lambda"])
            model.ent_coef = float(train_cfg["ent_coef"])
            model.vf_coef = float(train_cfg["vf_coef"])
            model.max_grad_norm = float(train_cfg["max_grad_norm"])
            if "target_kl" in train_cfg:
                model.target_kl = float(train_cfg["target_kl"])
            model.n_epochs = int(train_cfg["n_epochs"])
            model.batch_size = int(train_cfg["batch_size"])
            if "resume_log_std" in train_cfg:
                model.policy.log_std.data.fill_(float(train_cfg["resume_log_std"]))
        else:
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=float(train_cfg["learning_rate"]),
                n_steps=int(train_cfg["n_steps"]),
                batch_size=int(train_cfg["batch_size"]),
                n_epochs=int(train_cfg["n_epochs"]),
                gamma=float(train_cfg["gamma"]),
                gae_lambda=float(train_cfg["gae_lambda"]),
                clip_range=float(train_cfg["clip_range"]),
                ent_coef=float(train_cfg["ent_coef"]),
                vf_coef=float(train_cfg["vf_coef"]),
                max_grad_norm=float(train_cfg["max_grad_norm"]),
                target_kl=float(train_cfg["target_kl"]) if "target_kl" in train_cfg else None,
                policy_kwargs=policy_kwargs,
                verbose=1,
                tensorboard_log=str(log_dir),
                seed=actual_seed,
                device="auto",
            )
    except BaseException:  # noqa: BLE001 - close both allocated envs before propagating
        env.close()
        eval_env.close()
        raise
    try:
        callback_items: list[Any] = [EnvironmentMetricsCallback()]
        if anneal_from_stage is not None:
            start_rewards = load_env_config(stage=anneal_from_stage)["rewards"]
            callback_items.append(
                RewardAnnealingCallback(
                    start_weights=start_rewards,
                    end_weights=env_cfg["rewards"],
                    anneal_timesteps=int(anneal_timesteps or max(1, timesteps // 2)),
                )
            )
        early_stop_callback = None
        patience = int(train_cfg.get("early_stopping_patience_evals", 0))
        if patience > 0:
            early_stop_callback = StopTrainingOnNoModelImprovement(
                max_no_improvement_evals=patience,
                min_evals=int(train_cfg.get("early_stopping_min_evals", 2)),
                verbose=1,
            )
        callback_items.extend(
            [
                CheckpointCallback(
                    save_freq=max(1, int(train_cfg["checkpoint_freq"]) // n_envs),
                    save_path=str(checkpoint_dir),
                    name_prefix="ppo_stickman",
                ),
                EvalCallback(
                    eval_env,
                    best_model_save_path=str(checkpoint_dir / "best"),
                    log_path=str(log_dir / "eval"),
                    eval_freq=max(1, int(train_cfg["eval_freq"]) // n_envs),
                    n_eval_episodes=int(train_cfg["eval_episodes"]),
                    deterministic=True,
                    render=False,
                    callback_after_eval=early_stop_callback,
                ),
            ]
        )
        callbacks = CallbackList(callback_items)
    except BaseException:  # noqa: BLE001 - close both allocated envs before propagating
        env.close()
        eval_env.close()
        raise
    try:
        model.learn(total_timesteps=timesteps, callback=callbacks, reset_num_timesteps=not bool(resume), progress_bar=False)
        final_path = checkpoint_dir / "final_model"
        model.save(str(final_path))
    finally:
        env.close()
        eval_env.close()
    evaluation_episodes = min(5, int(train_cfg["eval_episodes"]))
    final_zip = final_path.with_suffix(".zip")
    final_result = evaluate_policy_path(
        final_zip,
        stage=stage,
        episodes=evaluation_episodes,
        seed=actual_seed + 20000,
        env_config_path=env_config_path,
    )
    best_zip = checkpoint_dir / "best" / "best_model.zip"
    best_result = None
    if best_zip.exists():
        best_result = evaluate_policy_path(
            best_zip,
            stage=stage,
            episodes=evaluation_episodes,
            seed=actual_seed + 30000,
            env_config_path=env_config_path,
        )
    recommended = final_zip
    if best_result is not None:
        best_key = (best_result.success_rate, best_result.mean_reward)
        final_key = (final_result.success_rate, final_result.mean_reward)
        if best_key > final_key:
            recommended = best_zip
    summary = {
        "run_id": run_id,
        "stage": stage,
        "timesteps_requested": timesteps,
        "seed": actual_seed,
        "checkpoint": str(final_zip),
        "best_checkpoint": str(best_zip) if best_zip.exists() else None,
        "recommended_checkpoint": str(recommended),
        "final_evaluation": final_result.to_dict(),
        "best_evaluation": best_result.to_dict() if best_result is not None else None,
        **final_result.to_dict(),
    }
    (checkpoint_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return final_zip, summary
