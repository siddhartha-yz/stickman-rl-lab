from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pymunk
from PIL import Image
from stable_baselines3 import PPO

from stickman_rl.config import load_env_config
from stickman_rl.env import StickmanReachEnv


def _sha256(path: str | Path | None) -> str | None:
    if path is None:
        return None
    file_path = Path(path)
    if not file_path.exists():
        return None
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _shape_geometry(shape: pymunk.Shape) -> dict[str, Any]:
    if isinstance(shape, pymunk.Circle):
        return {
            "kind": "circle",
            "radius": float(shape.radius),
            "offset": [float(shape.offset.x), float(shape.offset.y)],
        }
    if isinstance(shape, pymunk.Segment):
        return {
            "kind": "segment",
            "a": [float(shape.a.x), float(shape.a.y)],
            "b": [float(shape.b.x), float(shape.b.y)],
            "radius": float(shape.radius),
        }
    if isinstance(shape, pymunk.Poly):
        return {
            "kind": "polygon",
            "vertices": [[float(vertex.x), float(vertex.y)] for vertex in shape.get_vertices()],
        }
    raise TypeError(f"Unsupported trajectory shape: {type(shape)!r}")


def _metadata(
    env: StickmanReachEnv,
    *,
    model_path: str | None,
    stage: int,
    env_config: str | None,
    seed: int,
) -> dict[str, Any]:
    body_names = list(env.stickman.bodies)
    waypoints = [env._waypoint_position(waypoint).astype(float).tolist() for waypoint in env.navigation_waypoints]
    return {
        "format_version": 2,
        "stage": stage,
        "seed": seed,
        "model_path": model_path,
        "model_sha256": _sha256(model_path),
        "env_config_path": env_config,
        "env_config_sha256": _sha256(env_config),
        "room": {
            "width": float(env.config["physics"]["width"]),
            "height": float(env.config["physics"]["height"]),
        },
        "target": {
            "position": env.target_position.astype(float).tolist(),
            "size": [float(value) for value in env.config["target"]["size"]],
            "hold_steps": int(env.config["target"]["hold_steps"]),
        },
        "obstacles": env.config.get("obstacles", []),
        "waypoints": waypoints,
        "waypoint_specs": env.navigation_waypoints,
        "body_names": body_names,
        "body_geometry": {name: _shape_geometry(env.stickman.shapes[name]) for name in body_names},
        "action_names": list(env.stickman.actuated_joint_names),
        "max_steps": int(env.config["episode"]["max_steps"]),
        "control_repeat": int(env.config["physics"].get("control_repeat", 1)),
    }


def _body_state(env: StickmanReachEnv) -> tuple[np.ndarray, np.ndarray]:
    body_names = list(env.stickman.bodies)
    positions = np.asarray(
        [[env.stickman.bodies[name].position.x, env.stickman.bodies[name].position.y] for name in body_names],
        dtype=np.float32,
    )
    angles = np.asarray([env.stickman.bodies[name].angle for name in body_names], dtype=np.float32)
    return positions, angles


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a policy trajectory and optionally an animated GIF.")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--env-config", type=str, default=None)
    parser.add_argument("--output", type=Path, default=Path("trajectories/trajectory.npz"))
    parser.add_argument("--gif", type=Path, default=None)
    parser.add_argument("--max-steps", type=int, default=900)
    parser.add_argument("--seed", type=int, default=77)
    args = parser.parse_args()
    model = PPO.load(args.model) if args.model else None
    env = StickmanReachEnv(
        config=load_env_config(stage=args.stage, config_path=args.env_config),
        render_mode="rgb_array" if args.gif else None,
    )
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    cumulative_rewards: list[float] = []
    body_positions: list[np.ndarray] = []
    body_angles: list[np.ndarray] = []
    distances: list[float] = []
    torso_heights: list[float] = []
    torso_x_positions: list[float] = []
    waypoint_indices: list[int] = []
    navigation_distances: list[float] = []
    goal_hold_counts: list[int] = []
    successes: list[bool] = []
    frames: list[Image.Image] = []
    observation, _ = env.reset(seed=args.seed)
    metadata = _metadata(
        env,
        model_path=args.model,
        stage=args.stage,
        env_config=args.env_config,
        seed=args.seed,
    )
    cumulative_reward = 0.0
    try:
        for _ in range(args.max_steps):
            action = env.action_space.sample() if model is None else model.predict(observation, deterministic=True)[0]
            observations.append(observation.copy())
            actions.append(np.asarray(action).copy())
            observation, reward, terminated, truncated, info = env.step(action)
            rewards.append(float(reward))
            cumulative_reward += float(reward)
            cumulative_rewards.append(cumulative_reward)
            positions, angles = _body_state(env)
            body_positions.append(positions)
            body_angles.append(angles)
            distances.append(float(info["distance"]))
            torso_heights.append(float(info["torso_height"]))
            torso_x_positions.append(float(info["torso_x"]))
            waypoint_indices.append(int(info["active_waypoint_index"]))
            navigation_distances.append(float(info["navigation_distance"]))
            goal_hold_counts.append(int(info["goal_hold_count"]))
            successes.append(bool(info["is_success"]))
            if args.gif:
                frame = env.render()
                if frame is not None:
                    frames.append(Image.fromarray(frame))
            if terminated or truncated:
                break
    finally:
        env.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        observations=np.asarray(observations),
        actions=np.asarray(actions),
        rewards=np.asarray(rewards),
        cumulative_rewards=np.asarray(cumulative_rewards, dtype=np.float32),
        body_positions=np.asarray(body_positions, dtype=np.float32),
        body_angles=np.asarray(body_angles, dtype=np.float32),
        distances=np.asarray(distances, dtype=np.float32),
        torso_heights=np.asarray(torso_heights, dtype=np.float32),
        torso_x_positions=np.asarray(torso_x_positions, dtype=np.float32),
        waypoint_indices=np.asarray(waypoint_indices, dtype=np.int16),
        navigation_distances=np.asarray(navigation_distances, dtype=np.float32),
        goal_hold_counts=np.asarray(goal_hold_counts, dtype=np.int16),
        successes=np.asarray(successes, dtype=np.bool_),
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )
    print(f"saved trajectory: {args.output} ({len(rewards)} steps)")
    if args.gif and frames:
        args.gif.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(args.gif, save_all=True, append_images=frames[1:], duration=50, loop=0)
        print(f"saved GIF: {args.gif}")


if __name__ == "__main__":
    main()
