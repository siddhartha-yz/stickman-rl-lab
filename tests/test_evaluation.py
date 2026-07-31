from __future__ import annotations

import pytest

import stickman_rl.evaluation as evaluation_module


@pytest.mark.parametrize("episodes", [True, 1.5])
def test_policy_evaluation_rejects_noninteger_episode_count_before_model_load(
    monkeypatch: pytest.MonkeyPatch,
    episodes: object,
) -> None:
    class UnexpectedPPO:
        @staticmethod
        def load(*args: object, **kwargs: object) -> None:
            del args, kwargs
            raise AssertionError("model must not load for an invalid episode count")

    monkeypatch.setattr(evaluation_module, "PPO", UnexpectedPPO)

    with pytest.raises(ValueError, match="episodes must be an integer"):
        evaluation_module.evaluate_policy_path("missing-model.zip", episodes=episodes)  # type: ignore[arg-type]


@pytest.mark.parametrize("episodes", [0, -1])
def test_policy_evaluation_rejects_nonpositive_episode_count_before_model_load(
    monkeypatch: pytest.MonkeyPatch,
    episodes: int,
) -> None:
    class UnexpectedPPO:
        @staticmethod
        def load(*args: object, **kwargs: object) -> None:
            del args, kwargs
            raise AssertionError("model must not load for an invalid episode count")

    monkeypatch.setattr(evaluation_module, "PPO", UnexpectedPPO)

    with pytest.raises(ValueError, match="episodes must be at least 1"):
        evaluation_module.evaluate_policy_path("missing-model.zip", episodes=episodes)


@pytest.mark.parametrize("episodes", [True, 1.5])
def test_random_evaluation_rejects_noninteger_episode_count_before_env_creation(
    monkeypatch: pytest.MonkeyPatch,
    episodes: object,
) -> None:
    class UnexpectedEnvironment:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs
            raise AssertionError("environment must not be created for an invalid episode count")

    monkeypatch.setattr(evaluation_module, "StickmanReachEnv", UnexpectedEnvironment)

    with pytest.raises(ValueError, match="episodes must be an integer"):
        evaluation_module.evaluate_random_policy(episodes=episodes)  # type: ignore[arg-type]


@pytest.mark.parametrize("episodes", [0, -1])
def test_random_evaluation_rejects_nonpositive_episode_count_before_env_creation(
    monkeypatch: pytest.MonkeyPatch,
    episodes: int,
) -> None:
    class UnexpectedEnvironment:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs
            raise AssertionError("environment must not be created for an invalid episode count")

    monkeypatch.setattr(evaluation_module, "StickmanReachEnv", UnexpectedEnvironment)

    with pytest.raises(ValueError, match="episodes must be at least 1"):
        evaluation_module.evaluate_random_policy(episodes=episodes)
