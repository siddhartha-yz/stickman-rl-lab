from __future__ import annotations

import numpy as np
import pytest
from stable_baselines3 import PPO

import stickman_rl.training as training_module
from stickman_rl.env import StickmanReachEnv


@pytest.mark.parametrize(
    ("stage", "message"),
    [
        (True, "stage must be an integer"),
        (1.5, "stage must be an integer"),
        (-1, "stage must be between 0 and 5"),
        (6, "stage must be between 0 and 5"),
    ],
)
def test_train_ppo_rejects_invalid_stage_before_config_loading(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    stage: object,
    message: str,
) -> None:
    def unexpected_load_train_config(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("config must not load for an invalid stage")

    monkeypatch.setattr(training_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(training_module, "load_train_config", unexpected_load_train_config)

    with pytest.raises(ValueError, match=message):
        training_module.train_ppo(
            stage=stage,  # type: ignore[arg-type]
            run_name="invalid-stage",
        )

    assert not (tmp_path / "checkpoints").exists()
    assert not (tmp_path / "logs").exists()


@pytest.mark.parametrize(
    ("anneal_from_stage", "message"),
    [
        (True, "anneal_from_stage must be an integer"),
        (1.5, "anneal_from_stage must be an integer"),
        (-1, "anneal_from_stage must be between 0 and 5"),
        (6, "anneal_from_stage must be between 0 and 5"),
    ],
)
def test_train_ppo_rejects_invalid_anneal_stage_before_config_loading(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    anneal_from_stage: object,
    message: str,
) -> None:
    def unexpected_load_train_config(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("config must not load for an invalid anneal stage")

    monkeypatch.setattr(training_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(training_module, "load_train_config", unexpected_load_train_config)

    with pytest.raises(ValueError, match=message):
        training_module.train_ppo(
            anneal_from_stage=anneal_from_stage,  # type: ignore[arg-type]
            run_name="invalid-anneal-stage",
        )

    assert not (tmp_path / "checkpoints").exists()
    assert not (tmp_path / "logs").exists()


@pytest.mark.parametrize(
    ("anneal_timesteps", "message"),
    [
        (True, "anneal_timesteps must be an integer"),
        (1.5, "anneal_timesteps must be an integer"),
        (0, "anneal_timesteps must be at least 1"),
        (-1, "anneal_timesteps must be at least 1"),
    ],
)
def test_train_ppo_rejects_invalid_anneal_timesteps_before_config_loading(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    anneal_timesteps: object,
    message: str,
) -> None:
    def unexpected_load_train_config(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("config must not load for invalid anneal timesteps")

    monkeypatch.setattr(training_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(training_module, "load_train_config", unexpected_load_train_config)

    with pytest.raises(ValueError, match=message):
        training_module.train_ppo(
            anneal_from_stage=1,
            anneal_timesteps=anneal_timesteps,  # type: ignore[arg-type]
            run_name="invalid-anneal-timesteps",
        )

    assert not (tmp_path / "checkpoints").exists()
    assert not (tmp_path / "logs").exists()


@pytest.mark.parametrize("seed", [True, 1.5])
def test_train_ppo_rejects_noninteger_seed_before_run_creation(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    seed: object,
) -> None:
    def unexpected_make_vec_env(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("environment must not be created for an invalid seed")

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
    monkeypatch.setattr(training_module, "make_vec_env", unexpected_make_vec_env)

    with pytest.raises(ValueError, match="seed must be an integer"):
        training_module.train_ppo(
            seed=seed,  # type: ignore[arg-type]
            run_name="invalid-seed",
        )

    assert not (tmp_path / "checkpoints").exists()
    assert not (tmp_path / "logs").exists()


def test_train_ppo_rejects_negative_seed_before_run_creation(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_make_vec_env(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("environment must not be created for an invalid seed")

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
    monkeypatch.setattr(training_module, "make_vec_env", unexpected_make_vec_env)

    with pytest.raises(ValueError, match="seed must be non-negative"):
        training_module.train_ppo(seed=-1, run_name="invalid-seed")

    assert not (tmp_path / "checkpoints").exists()
    assert not (tmp_path / "logs").exists()


@pytest.mark.parametrize("timesteps", [True, 1.5])
def test_train_ppo_rejects_noninteger_timesteps_before_run_creation(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    timesteps: object,
) -> None:
    def unexpected_make_vec_env(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("environment must not be created for invalid timesteps")

    monkeypatch.setattr(training_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        training_module,
        "load_train_config",
        lambda _path: {"total_timesteps": 1_000_000},
    )
    monkeypatch.setattr(
        training_module,
        "load_env_config",
        lambda **_kwargs: {"seed": 0},
    )
    monkeypatch.setattr(training_module, "make_vec_env", unexpected_make_vec_env)

    with pytest.raises(ValueError, match="total_timesteps must be an integer"):
        training_module.train_ppo(
            total_timesteps=timesteps,  # type: ignore[arg-type]
            run_name="invalid-timesteps",
        )

    assert not (tmp_path / "checkpoints").exists()
    assert not (tmp_path / "logs").exists()


@pytest.mark.parametrize("timesteps", [0, -1])
def test_train_ppo_rejects_nonpositive_timesteps_before_run_creation(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    timesteps: int,
) -> None:
    def unexpected_make_vec_env(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("environment must not be created for invalid timesteps")

    monkeypatch.setattr(training_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        training_module,
        "load_train_config",
        lambda _path: {"total_timesteps": 1_000_000},
    )
    monkeypatch.setattr(
        training_module,
        "load_env_config",
        lambda **_kwargs: {"seed": 0},
    )
    monkeypatch.setattr(training_module, "make_vec_env", unexpected_make_vec_env)

    with pytest.raises(ValueError, match="total_timesteps must be at least 1"):
        training_module.train_ppo(
            total_timesteps=timesteps,
            run_name="invalid-timesteps",
        )

    assert not (tmp_path / "checkpoints").exists()
    assert not (tmp_path / "logs").exists()


@pytest.mark.parametrize("n_envs", [True, 1.5])
def test_train_ppo_rejects_noninteger_n_envs_before_run_creation(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    n_envs: object,
) -> None:
    def unexpected_make_vec_env(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("environment must not be created for invalid n_envs")

    monkeypatch.setattr(training_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        training_module,
        "load_train_config",
        lambda _path: {"total_timesteps": 64, "n_envs": n_envs},
    )
    monkeypatch.setattr(
        training_module,
        "load_env_config",
        lambda **_kwargs: {"seed": 0},
    )
    monkeypatch.setattr(training_module, "make_vec_env", unexpected_make_vec_env)

    with pytest.raises(ValueError, match="n_envs must be an integer"):
        training_module.train_ppo(run_name="invalid-n-envs")

    assert not (tmp_path / "checkpoints").exists()
    assert not (tmp_path / "logs").exists()


@pytest.mark.parametrize("n_envs", [0, -1])
def test_train_ppo_rejects_nonpositive_n_envs_before_run_creation(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    n_envs: int,
) -> None:
    def unexpected_make_vec_env(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("environment must not be created for invalid n_envs")

    monkeypatch.setattr(training_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        training_module,
        "load_train_config",
        lambda _path: {"total_timesteps": 64, "n_envs": n_envs},
    )
    monkeypatch.setattr(
        training_module,
        "load_env_config",
        lambda **_kwargs: {"seed": 0},
    )
    monkeypatch.setattr(training_module, "make_vec_env", unexpected_make_vec_env)

    with pytest.raises(ValueError, match="n_envs must be at least 1"):
        training_module.train_ppo(run_name="invalid-n-envs")

    assert not (tmp_path / "checkpoints").exists()
    assert not (tmp_path / "logs").exists()


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


def test_train_ppo_closes_both_envs_when_callback_construction_fails(
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

    def fail_callback(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise RuntimeError("callback construction failure")

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
    monkeypatch.setattr(training_module, "PPO", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(training_module, "EnvironmentMetricsCallback", fail_callback)

    with pytest.raises(RuntimeError, match="callback construction failure"):
        training_module.train_ppo(run_name="callback-construction-failure")

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
