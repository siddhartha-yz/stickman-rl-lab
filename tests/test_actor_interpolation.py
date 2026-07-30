from __future__ import annotations

from copy import deepcopy

import torch
from stable_baselines3 import PPO

from scripts.interpolate_actor import actor_parameter_names, interpolate_actor
from stickman_rl.env import StickmanReachEnv


def test_interpolate_actor_changes_only_actor_parameters() -> None:
    env = StickmanReachEnv(stage=0)
    source = PPO(
        "MlpPolicy",
        env,
        n_steps=32,
        batch_size=16,
        n_epochs=1,
        policy_kwargs={"net_arch": [32, 32]},
        verbose=0,
        seed=19,
        device="cpu",
    )
    target = deepcopy(source)
    actor_names = set(actor_parameter_names(source))
    with torch.no_grad():
        for name, parameter in target.policy.named_parameters():
            if name in actor_names:
                parameter.add_(2.0)

    before = {name: parameter.detach().clone() for name, parameter in source.policy.named_parameters()}
    stats = interpolate_actor(source, target, alpha=0.25)

    for name, parameter in source.policy.named_parameters():
        expected = before[name] + 0.5 if name in actor_names else before[name]
        assert torch.allclose(parameter, expected)
    assert stats["actor_parameter_count"] > 0
    assert stats["mean_absolute_parameter_delta"] > 0.0
    env.close()
