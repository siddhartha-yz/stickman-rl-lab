from __future__ import annotations

import numpy as np
from stable_baselines3 import PPO

from stickman_rl.env import StickmanReachEnv


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
