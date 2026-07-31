from __future__ import annotations

import numpy as np
import pytest
from stable_baselines3 import PPO

import stickman_rl.training as training_module
from stickman_rl.env import StickmanReachEnv


def test_train_ppo_closes_training_env_when_eval_env_creation_fails(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeVecEnv:
        closed = False

        def close(self) -> None:
            self.closed = True

    train_env = FakeVecEnv()
    creation_count = 0

    def fake_make_vec_env(*args: object, **kwargs: object) -> FakeVecEnv:
        nonlocal creation_count
        del args, kwargs
        creation_count += 1
        if creation_count == 1:
            return train_env
        raise RuntimeError("eval environment creation failure")

    monkeypatch.setattr(training_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        training_module,
        "load_train_config",
        lambda _path: {"total_timesteps": 64},
    )
    monkeypatch.setattr(
        training_module,
        "load_env_config",
        lambda **_kwargs: {"seed": 0},
    )
    monkeypatch.setattr(training_module, "make_vec_env", fake_make_vec_env)

    with pytest.raises(RuntimeError, match="eval environment creation failure"):
        training_module.train_ppo(run_name="eval-env-creation-failure")

    assert train_env.closed


def test_train_ppo_closes_both_envs_when_ppo_construction_fails(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeVecEnv:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    train_env = FakeVecEnv()
    eval_env = FakeVecEnv()
    environments = iter([train_env, eval_env])

    def fail_ppo(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise RuntimeError("PPO construction failure")

    monkeypatch.setattr(training_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        training_module,
        "load_train_config",
        lambda _path: {
            "total_timesteps": 64,
            "learning_rate": 0.0003,
            "n_steps": 32,
            "batch_size": 16,
            "n_epochs": 1,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.0,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
        },
    )
    monkeypatch.setattr(
        training_module,
        "load_env_config",
        lambda **_kwargs: {"seed": 0},
    )
    monkeypatch.setattr(
        training_module,
        "make_vec_env",
        lambda *_args, **_kwargs: next(environments),
    )
    monkeypatch.setattr(training_module, "PPO", fail_ppo)

    with pytest.raises(RuntimeError, match="PPO construction failure"):
        training_module.train_ppo(run_name="ppo-construction-failure")

    assert train_env.closed
    assert eval_env.closed


def test_ppo_can_train_save_and_reload(tmp_path) -> None:
    env = StickmanReachEnv(stage=0)
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=32,
        batch_size=16,
        n_epochs=1,
        policy_kwargs={"net_arch": [32, 32]},
        verbose=0,
        seed=11,
        device="cpu",
    )
    model.learn(total_timesteps=64)
    path = tmp_path / "smoke_model"
    model.save(path)
    env.close()

    loaded = PPO.load(path)
    eval_env = StickmanReachEnv(stage=0)
    observation, _ = eval_env.reset(seed=12)
    action, _ = loaded.predict(observation, deterministic=True)
    assert action.shape == (8,)
    assert np.isfinite(action).all()
    next_observation, _, _, _, _ = eval_env.step(action)
    assert np.isfinite(next_observation).all()
    eval_env.close()
