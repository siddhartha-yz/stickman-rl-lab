from __future__ import annotations

import json

import numpy as np
from gymnasium.utils.env_checker import check_env

from stickman_rl.config import load_env_config
from stickman_rl.env import StickmanReachEnv


def test_gymnasium_api_check() -> None:
    env = StickmanReachEnv(stage=0)
    check_env(env, skip_render_check=True)
    env.close()


def test_reset_and_step_shapes() -> None:
    env = StickmanReachEnv(stage=0)
    observation, info = env.reset(seed=1)
    assert observation.shape == (49,)
    assert observation.dtype == np.float32
    assert np.isfinite(observation).all()
    assert "distance" in info
    result = env.step(np.zeros(8, dtype=np.float32))
    next_observation, reward, terminated, truncated, step_info = result
    assert next_observation.shape == (49,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "reward_progress" in step_info
    env.close()


def test_live_snapshot_is_json_safe_and_tracks_physics() -> None:
    env = StickmanReachEnv(stage=1)
    env.reset(seed=17)
    initial = env.live_snapshot()
    env.step(np.zeros(8, dtype=np.float32))
    updated = env.live_snapshot()

    assert len(initial["metadata"]["body_names"]) == 10
    assert len(initial["frame"]["body_positions"]) == 10
    assert initial["frame"]["step"] == 0
    assert updated["frame"]["step"] == 1
    assert updated["metadata"]["action_names"] == list(env.stickman.actuated_joint_names)
    json.dumps(updated)
    env.close()


def test_action_is_clipped_before_motor_command() -> None:
    env = StickmanReachEnv(stage=0)
    env.reset(seed=2)
    env.step(np.full(8, 5.0, dtype=np.float32))
    expected = float(env.config["stickman"]["max_joint_speed"])
    rates = [env.stickman.joints[name].motor.rate for name in env.stickman.actuated_joint_names]
    assert np.allclose(rates, expected)
    env.close()


def test_target_overlap_detection() -> None:
    env = StickmanReachEnv(stage=0)
    env.reset(seed=3)
    assert env.world is not None
    torso = env.stickman.torso
    delta = env.target_position - np.asarray(torso.position, dtype=np.float32)
    for body in env.stickman.bodies.values():
        body.position = (body.position.x + float(delta[0]), body.position.y + float(delta[1]))
        env.world.space.reindex_shapes_for_body(body)
    assert env._target_overlap()
    env.close()


def test_random_rollout_stays_finite() -> None:
    env = StickmanReachEnv(stage=0)
    observation, _ = env.reset(seed=4)
    for _ in range(600):
        observation, _, terminated, truncated, _ = env.step(env.action_space.sample())
        assert np.isfinite(observation).all()
        if terminated or truncated:
            observation, _ = env.reset()
    env.close()


def test_obstacle_rays_are_versioned_and_detect_medium_box() -> None:
    config = load_env_config(stage=3, config_path="configs/stage3_medium_lidar.yaml")
    env = StickmanReachEnv(config=config)
    observation, _ = env.reset(seed=5)
    ray_count = len(config["observation"]["obstacle_rays"]["angles_deg"])
    rays = observation[-ray_count:]
    assert observation.shape == (49 + ray_count,)
    assert np.all((rays >= 0.0) & (rays <= 1.0))
    assert np.count_nonzero(rays) > 0
    env.close()


def test_obstacle_rays_return_zero_without_obstacles() -> None:
    config = load_env_config(stage=0)
    config["observation"]["obstacle_rays"]["enabled"] = True
    env = StickmanReachEnv(config=config)
    observation, _ = env.reset(seed=6)
    ray_count = len(config["observation"]["obstacle_rays"]["angles_deg"])
    assert observation.shape == (49 + ray_count,)
    assert np.allclose(observation[-ray_count:], 0.0)
    env.close()


def test_navigation_waypoint_drives_goal_then_advances_to_target() -> None:
    config = load_env_config(stage=3, config_path="configs/stage3_medium_lidar_waypoint.yaml")
    env = StickmanReachEnv(config=config)
    observation, info = env.reset(seed=7)
    waypoint = np.asarray(config["navigation"]["waypoints"][0]["position"], dtype=np.float32)
    torso = np.asarray(env.stickman.torso.position, dtype=np.float32)
    width = float(config["physics"]["width"])
    height = float(config["physics"]["height"])
    assert env.active_waypoint_index == 0
    assert np.allclose(env._navigation_goal_position(), waypoint)
    assert np.allclose(observation[42:44], (waypoint - torso) / np.array([width, height]))
    assert info["navigation_distance"] < info["final_distance"]

    env.stickman.torso.position = (5.5, float(waypoint[1]))
    assert not env._advance_navigation_waypoint()
    assert env.active_waypoint_index == 0

    env.stickman.torso.position = (6.0, env.stickman.torso.position.y)
    assert env._advance_navigation_waypoint()
    assert env.active_waypoint_index == 1
    assert np.allclose(env._navigation_goal_position(), env.target_position)
    env.close()


def test_full_course_advances_through_two_strict_waypoints() -> None:
    config = load_env_config(stage=3, config_path="configs/stage3_lidar_waypoints.yaml")
    env = StickmanReachEnv(config=config)
    env.reset(seed=8)
    assert env.world is not None
    assert len(env.world.obstacle_shapes) == 2
    assert len(env.navigation_waypoints) == 2
    assert np.allclose(env._navigation_goal_position(), [6.0, 1.25])

    env.stickman.torso.position = (6.0, env.stickman.torso.position.y)
    assert env._advance_navigation_waypoint()
    assert env.active_waypoint_index == 1
    assert np.allclose(env._navigation_goal_position(), [8.15, 1.2])

    env.stickman.torso.position = (8.1, env.stickman.torso.position.y)
    assert env._advance_navigation_waypoint()
    assert env.active_waypoint_index == 2
    assert np.allclose(env._navigation_goal_position(), env.target_position)
    env.close()


def test_target_relative_waypoint_tracks_sampled_target() -> None:
    config = load_env_config(stage=3, config_path="configs/stage3_lidar_target_route.yaml")
    env = StickmanReachEnv(config=config)
    env.reset(seed=9)
    assert len(env.navigation_waypoints) == 3

    env.stickman.torso.position = (6.0, env.stickman.torso.position.y)
    assert env._advance_navigation_waypoint()
    env.stickman.torso.position = (8.1, env.stickman.torso.position.y)
    assert env._advance_navigation_waypoint()
    assert env.active_waypoint_index == 2

    offset = np.asarray(config["navigation"]["waypoints"][2]["target_offset"], dtype=np.float32)
    assert np.allclose(env._navigation_goal_position(), env.target_position + offset)

    threshold = float(env.target_position[0]) - 0.2
    env.stickman.torso.position = (threshold - 0.01, env.stickman.torso.position.y)
    assert not env._advance_navigation_waypoint()
    env.stickman.torso.position = (threshold + 0.001, env.stickman.torso.position.y)
    assert env._advance_navigation_waypoint()
    assert env.active_waypoint_index == 3
    assert np.allclose(env._navigation_goal_position(), env.target_position)
    env.close()


def test_calibrated_platform_gate_remains_beyond_obstacle_edge() -> None:
    config = load_env_config(stage=3, config_path="configs/stage3_lidar_target_route_v2.yaml")
    platform = config["obstacles"][1]
    platform_right = float(platform["position"][0]) + float(platform["size"][0]) * 0.5
    gate = float(config["navigation"]["waypoints"][1]["advance_x"])
    assert gate > platform_right

    env = StickmanReachEnv(config=config)
    env.reset(seed=10)
    env.stickman.torso.position = (6.0, env.stickman.torso.position.y)
    assert env._advance_navigation_waypoint()
    env.stickman.torso.position = (gate - 0.01, env.stickman.torso.position.y)
    assert not env._advance_navigation_waypoint()
    env.stickman.torso.position = (gate + 0.001, env.stickman.torso.position.y)
    assert env._advance_navigation_waypoint()
    assert env.active_waypoint_index == 2
    env.close()


def test_final_progress_scale_activates_after_all_waypoints() -> None:
    config = load_env_config(stage=3, config_path="configs/stage3_lidar_waypoints_goalboost.yaml")
    env = StickmanReachEnv(config=config)
    _, info = env.reset(seed=11)
    assert info["progress_scale"] == 1.0

    env.stickman.torso.position = (6.0, env.stickman.torso.position.y)
    assert env._advance_navigation_waypoint()
    assert env._build_info({}, False)["progress_scale"] == 1.0
    env.stickman.torso.position = (8.0, env.stickman.torso.position.y)
    assert env._advance_navigation_waypoint()
    assert env._build_info({}, False)["progress_scale"] == 4.0
    env.close()


def test_stage2_randomizes_target_reproducibly() -> None:
    env = StickmanReachEnv(stage=2)
    env.reset(seed=101)
    first = env.target_position.copy()
    env.reset()
    second = env.target_position.copy()
    assert not np.allclose(first, second)
    assert float(env.config["target"]["min_x"]) <= float(first[0]) <= float(env.config["target"]["max_x"])
    env.close()


def test_stage3_builds_configured_obstacles() -> None:
    env = StickmanReachEnv(stage=3)
    env.reset(seed=102)
    assert env.world is not None
    assert len(env.world.obstacle_shapes) == 2
    env.close()


def test_rgb_array_render_shape(monkeypatch) -> None:
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    env = StickmanReachEnv(stage=0, render_mode="rgb_array")
    env.reset(seed=103)
    frame = env.render()
    assert frame is not None
    assert frame.shape == (
        int(env.config["render"]["height"]),
        int(env.config["render"]["width"]),
        3,
    )
    assert frame.dtype == np.uint8
    env.close()



def test_stage3_targets_are_clear_of_obstacles() -> None:
    env = StickmanReachEnv(stage=3)
    for seed in range(30):
        env.reset(seed=1000 + seed)
        x, y = map(float, env.target_position)
        assert env._target_position_is_clear(x, y)
        assert x >= float(env.config["target"]["min_x"])
    env.close()
