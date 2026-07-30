"""Gymnasium environment for articulated 2D target reaching."""

from __future__ import annotations

from math import pi
from typing import Any

import gymnasium as gym
import numpy as np
import pymunk
from gymnasium import spaces

from stickman_rl.config import load_env_config
from stickman_rl.physics.world import PhysicsWorld
from stickman_rl.renderer import PygameRenderer
from stickman_rl.rewards import RewardCalculator, RewardInputs


class StickmanReachEnv(gym.Env[np.ndarray, np.ndarray]):
    """Continuous-control environment where a stickman learns to reach a red target."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        stage: int = 1,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        if render_mode not in {None, "human", "rgb_array"}:
            raise ValueError(f"Unsupported render_mode: {render_mode}")
        self.config = config or load_env_config(stage=stage)
        self.render_mode = render_mode
        self.action_space = spaces.Box(-1.0, 1.0, shape=(8,), dtype=np.float32)
        ray_cfg = self.config.get("observation", {}).get("obstacle_rays", {})
        ray_enabled = bool(ray_cfg.get("enabled", False))
        self.obstacle_ray_angles_deg = (
            tuple(float(angle) for angle in ray_cfg.get("angles_deg", [])) if ray_enabled else ()
        )
        self.obstacle_ray_max_distance = float(ray_cfg.get("max_distance", 6.0))
        if ray_enabled and not self.obstacle_ray_angles_deg:
            raise ValueError("Obstacle rays are enabled but no angles_deg were configured")
        if self.obstacle_ray_angles_deg and self.obstacle_ray_max_distance <= 0.0:
            raise ValueError("Obstacle ray max_distance must be positive")
        # Base: 8 torso + 16 joints + 18 relative body positions + 3 target + 4 contact/posture = 49.
        observation_size = 49 + len(self.obstacle_ray_angles_deg)
        self.observation_space = spaces.Box(-5.0, 5.0, shape=(observation_size,), dtype=np.float32)
        self.world: PhysicsWorld | None = None
        self.renderer: PygameRenderer | None = None
        self.reward_calculator = RewardCalculator(self.config["rewards"])
        self.target_position = np.zeros(2, dtype=np.float32)
        self.previous_distance = 0.0
        raw_waypoints = self.config.get("navigation", {}).get("waypoints", [])
        self.navigation_waypoints = tuple(dict(waypoint) for waypoint in raw_waypoints)
        for waypoint in self.navigation_waypoints:
            has_position = "position" in waypoint
            has_target_offset = "target_offset" in waypoint
            if has_position == has_target_offset:
                raise ValueError(
                    "Each navigation waypoint requires exactly one of position or target_offset"
                )
            coordinates = waypoint["position"] if has_position else waypoint["target_offset"]
            if not isinstance(coordinates, list | tuple) or len(coordinates) != 2:
                raise ValueError("Waypoint position or target_offset must contain two values")
        self.active_waypoint_index = 0
        self.previous_navigation_distance = 0.0
        self.previous_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self.step_count = 0
        self.goal_hold_count = 0
        self.episode_energy = 0.0
        self._last_info: dict[str, Any] = {}
        self._has_reset = False

    @property
    def stickman(self):
        if self.world is None:
            raise RuntimeError("Environment has not been reset")
        return self.world.stickman

    def _target_position_is_clear(self, x: float, y: float) -> bool:
        """Return whether a target AABB avoids configured obstacles and trench gaps."""
        target_cfg = self.config["target"]
        target_width, target_height = map(float, target_cfg["size"])
        clearance = float(target_cfg.get("clearance", 0.0))
        left = x - target_width * 0.5 - clearance
        right = x + target_width * 0.5 + clearance
        bottom = y - target_height * 0.5 - clearance
        top = y + target_height * 0.5 + clearance
        for spec in self.config.get("obstacles", []):
            kind = str(spec.get("type", "box")).lower()
            if kind == "trench":
                center_x, _ = map(float, spec.get("position", [6.0, 0.0]))
                width = float(spec.get("width", spec.get("size", [1.0, 0.0])[0]))
                if right >= center_x - width * 0.5 and left <= center_x + width * 0.5:
                    return False
                continue
            if kind in {"box", "platform", "wall"}:
                center_x, center_y = map(float, spec.get("position", [6.0, 0.5]))
                width, height = map(float, spec.get("size", [1.0, 0.5]))
                obstacle_left = center_x - width * 0.5
                obstacle_right = center_x + width * 0.5
                obstacle_bottom = center_y - height * 0.5
                obstacle_top = center_y + height * 0.5
            elif kind == "slope":
                start = tuple(map(float, spec.get("start", [x - 1.0, y])))
                end = tuple(map(float, spec.get("end", [x + 1.0, y + 0.8])))
                radius = float(spec.get("radius", 0.05))
                obstacle_left = min(start[0], end[0]) - radius
                obstacle_right = max(start[0], end[0]) + radius
                obstacle_bottom = min(start[1], end[1]) - radius
                obstacle_top = max(start[1], end[1]) + radius
            else:
                continue
            overlaps = not (
                right < obstacle_left
                or left > obstacle_right
                or top < obstacle_bottom
                or bottom > obstacle_top
            )
            if overlaps:
                return False
        return True

    def _sample_target(self) -> np.ndarray:
        target_cfg = self.config["target"]
        if target_cfg.get("randomize", False):
            min_x = float(target_cfg["min_x"])
            max_x = float(target_cfg["max_x"])
            y = float(target_cfg.get("position", [0.0, 0.55])[1])
            for _ in range(int(target_cfg.get("max_sample_attempts", 128))):
                x = float(self.np_random.uniform(min_x, max_x))
                if self._target_position_is_clear(x, y):
                    return np.array([x, y], dtype=np.float32)
            candidates = np.linspace(min_x, max_x, 257)
            valid = [float(x) for x in candidates if self._target_position_is_clear(float(x), y)]
            if not valid:
                raise ValueError("No collision-free target position exists for the configured obstacles")
            x = valid[int(self.np_random.integers(0, len(valid)))]
            return np.array([x, y], dtype=np.float32)
        position = np.asarray(target_cfg["position"], dtype=np.float32)
        if not self._target_position_is_clear(float(position[0]), float(position[1])):
            raise ValueError("Configured fixed target overlaps an obstacle")
        return position

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        effective_seed = seed if seed is not None else (int(self.config.get("seed", 0)) if not self._has_reset else None)
        super().reset(seed=effective_seed)
        self._has_reset = True
        self.target_position = self._sample_target()
        self.world = PhysicsWorld(self.config, tuple(map(float, self.target_position)))
        self.step_count = 0
        self.goal_hold_count = 0
        self.episode_energy = 0.0
        self.previous_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self.previous_distance = self._distance_to_target()
        self.active_waypoint_index = 0
        self.previous_navigation_distance = self._distance_to_navigation_goal()
        observation = self._observation()
        info = self._build_info({}, False)
        self._last_info = info
        if self.render_mode == "human":
            self.render()
        return observation, info

    def _distance_to_target(self) -> float:
        torso = self.stickman.torso.position
        return float(np.hypot(torso.x - self.target_position[0], torso.y - self.target_position[1]))

    def _waypoint_position(self, waypoint: dict[str, Any]) -> np.ndarray:
        if "target_offset" in waypoint:
            return self.target_position + np.asarray(waypoint["target_offset"], dtype=np.float32)
        return np.asarray(waypoint["position"], dtype=np.float32)

    def _navigation_goal_position(self) -> np.ndarray:
        if self.active_waypoint_index < len(self.navigation_waypoints):
            return self._waypoint_position(self.navigation_waypoints[self.active_waypoint_index])
        return self.target_position

    def _distance_to_navigation_goal(self) -> float:
        torso = self.stickman.torso.position
        goal = self._navigation_goal_position()
        return float(np.hypot(torso.x - goal[0], torso.y - goal[1]))

    def _advance_navigation_waypoint(self) -> bool:
        if self.active_waypoint_index >= len(self.navigation_waypoints):
            return False
        waypoint = self.navigation_waypoints[self.active_waypoint_index]
        torso = self.stickman.torso.position
        position = self._waypoint_position(waypoint)
        radius = float(waypoint.get("radius", 0.45))
        if "advance_target_x_offset" in waypoint:
            reached = torso.x >= float(
                self.target_position[0] + float(waypoint["advance_target_x_offset"])
            )
        elif "advance_x" in waypoint:
            reached = torso.x >= float(waypoint["advance_x"])
        else:
            reached = float(np.hypot(torso.x - position[0], torso.y - position[1])) <= radius
        if reached:
            self.active_waypoint_index += 1
            return True
        return False

    def _target_overlap(self) -> bool:
        assert self.world is not None
        target_bb = self.world.target_shape.cache_bb()
        torso_position = self.stickman.torso.position
        return bool(
            target_bb.left <= torso_position.x <= target_bb.right
            and target_bb.bottom <= torso_position.y <= target_bb.top
        )

    def _contact_flags(self) -> tuple[float, float, float]:
        threshold = 0.14
        foot = float(sum(self.stickman.shapes[name].bb.bottom <= threshold for name in ("left_shin", "right_shin"))) / 2.0
        hands = float(sum(self.stickman.shapes[name].bb.bottom <= threshold for name in ("left_forearm", "right_forearm"))) / 2.0
        body = float(any(shape.bb.bottom <= threshold for shape in self.stickman.shapes.values()))
        return foot, hands, body

    def _joint_limit_fraction(self) -> float:
        near = 0
        for name in self.stickman.actuated_joint_names:
            joint = self.stickman.joints[name]
            span = max(joint.limit.max - joint.limit.min, 1e-6)
            normalized = (joint.angle - joint.limit.min) / span
            near += int(normalized < 0.04 or normalized > 0.96)
        return near / len(self.stickman.actuated_joint_names)

    def _out_of_bounds(self) -> bool:
        margin = float(self.config["episode"]["out_of_bounds_margin"])
        width = float(self.config["physics"]["width"])
        height = float(self.config["physics"]["height"])
        for body in self.stickman.bodies.values():
            if body.position.x < -margin or body.position.x > width + margin:
                return True
            if body.position.y < -margin or body.position.y > height + margin:
                return True
        return False

    def _obstacle_ray_observation(self) -> np.ndarray:
        """Return world-fixed forward ray proximity values for configured obstacles.

        Each value is in [0, 1]: zero means no obstacle within range and one
        means an obstacle touches the ray origin. Room boundaries, the target,
        and the stickman's own shapes are intentionally excluded.
        """
        if not self.obstacle_ray_angles_deg:
            return np.empty(0, dtype=np.float32)
        if self.world is None:
            raise RuntimeError("Environment has not been reset")
        origin = self.stickman.torso.position
        obstacle_shape_ids = {id(shape) for shape in self.world.obstacle_shapes}
        readings: list[float] = []
        for angle_deg in self.obstacle_ray_angles_deg:
            angle = np.deg2rad(angle_deg)
            end = (
                origin.x + self.obstacle_ray_max_distance * float(np.cos(angle)),
                origin.y + self.obstacle_ray_max_distance * float(np.sin(angle)),
            )
            hits = self.world.space.segment_query(
                origin,
                end,
                0.0,
                pymunk.ShapeFilter(),
            )
            obstacle_alphas = [
                float(hit.alpha) for hit in hits if id(hit.shape) in obstacle_shape_ids
            ]
            proximity = 1.0 - min(obstacle_alphas) if obstacle_alphas else 0.0
            readings.append(float(np.clip(proximity, 0.0, 1.0)))
        return np.asarray(readings, dtype=np.float32)

    def _observation(self) -> np.ndarray:
        p = self.config["physics"]
        width, height = float(p["width"]), float(p["height"])
        torso = self.stickman.torso
        joint_angles, joint_velocities = self.stickman.joint_state()
        relative_positions = self.stickman.main_relative_positions().copy()
        relative_positions[0::2] /= width
        relative_positions[1::2] /= height
        navigation_goal = self._navigation_goal_position()
        target_delta = navigation_goal - np.asarray(torso.position, dtype=np.float32)
        distance = float(np.linalg.norm(target_delta))
        feet, hands, body_contact = self._contact_flags()
        obs = np.concatenate(
            [
                np.array(
                    [
                        torso.position.x / width,
                        torso.position.y / height,
                        torso.velocity.x / 10.0,
                        torso.velocity.y / 10.0,
                        np.sin(torso.angle),
                        np.cos(torso.angle),
                        torso.angular_velocity / 10.0,
                        self.step_count / max(1, int(self.config["episode"]["max_steps"])),
                    ],
                    dtype=np.float32,
                ),
                np.clip(joint_angles / pi, -2.0, 2.0),
                np.clip(joint_velocities / 10.0, -2.0, 2.0),
                relative_positions,
                np.array([target_delta[0] / width, target_delta[1] / height, distance / width], dtype=np.float32),
                np.array([feet, hands, body_contact, torso.position.y / height], dtype=np.float32),
                self._obstacle_ray_observation(),
            ]
        ).astype(np.float32)
        if obs.shape != self.observation_space.shape:
            raise RuntimeError(f"Observation shape mismatch: expected {self.observation_space.shape}, got {obs.shape}")
        if not np.all(np.isfinite(obs)):
            raise FloatingPointError("Non-finite observation generated")
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.world is None:
            raise RuntimeError("Call reset() before step()")
        clipped_action = self.stickman.apply_action(action)
        control_repeat = max(1, int(self.config["physics"].get("control_repeat", 1)))
        for _ in range(control_repeat):
            self.world.step()
            if not self.stickman.finite() or self._out_of_bounds():
                break
        self.step_count += 1
        distance = self._distance_to_target()
        previous_navigation_distance = self.previous_navigation_distance
        waypoint_advanced = self._advance_navigation_waypoint()
        navigation_distance = self._distance_to_navigation_goal()
        if waypoint_advanced:
            previous_navigation_distance = navigation_distance
        overlapping = self._target_overlap()
        self.goal_hold_count = self.goal_hold_count + 1 if overlapping else 0
        success = self.goal_hold_count >= int(self.config["target"]["hold_steps"])
        finite = self.stickman.finite()
        out_of_bounds = self._out_of_bounds()
        invalid = not finite
        terminated = bool(success or out_of_bounds or invalid)
        truncated = bool(self.step_count >= int(self.config["episode"]["max_steps"]))
        feet, hands, _ = self._contact_flags()
        self.episode_energy += float(np.mean(np.square(clipped_action)))
        final_progress_scale = float(
            self.config.get("navigation", {}).get("final_progress_scale", 1.0)
        )
        progress_scale = (
            final_progress_scale
            if self.active_waypoint_index >= len(self.navigation_waypoints)
            else 1.0
        )
        reward, components = self.reward_calculator.calculate(
            RewardInputs(
                previous_distance=previous_navigation_distance,
                distance=navigation_distance,
                success=success,
                action=clipped_action,
                previous_action=self.previous_action,
                torso_height=float(self.stickman.torso.position.y),
                torso_angle=float(self.stickman.torso.angle),
                feet_contact=feet,
                hands_contact=hands,
                joint_limit_fraction=self._joint_limit_fraction(),
                out_of_bounds=out_of_bounds,
                progress_scale=progress_scale,
            )
        )
        if invalid:
            reward += float(self.config["episode"]["invalid_state_penalty"])
            components["invalid_state"] = float(self.config["episode"]["invalid_state_penalty"])
            components["total"] = reward
        self.previous_distance = distance
        self.previous_navigation_distance = navigation_distance
        self.previous_action = clipped_action.copy()
        observation = self._observation() if finite else np.zeros(self.observation_space.shape, dtype=np.float32)
        info = self._build_info(components, success)
        self._last_info = info
        if self.render_mode == "human":
            self.render()
        return observation, float(reward), terminated, truncated, info

    def _build_info(self, components: dict[str, float], success: bool) -> dict[str, Any]:
        torso_height = float(self.stickman.torso.position.y) if self.world is not None else 0.0
        info: dict[str, Any] = {
            "step": self.step_count,
            "is_success": bool(success),
            "distance": float(self.previous_distance),
            "final_distance": float(self.previous_distance),
            "episode_energy": float(self.episode_energy),
            "mean_energy": float(self.episode_energy / max(1, self.step_count)),
            "torso_height": torso_height,
            "torso_x": float(self.stickman.torso.position.x) if self.world is not None else 0.0,
            "goal_hold_count": self.goal_hold_count,
            "active_waypoint_index": self.active_waypoint_index,
            "navigation_distance": float(self.previous_navigation_distance),
            "progress_scale": float(
                self.config.get("navigation", {}).get("final_progress_scale", 1.0)
                if self.active_waypoint_index >= len(self.navigation_waypoints)
                else 1.0
            ),
        }
        for key, value in components.items():
            info[f"reward_{key}"] = float(value)
        return info


    @staticmethod
    def _shape_geometry(shape: pymunk.Shape) -> dict[str, Any]:
        """Return JSON-safe geometry for browser-side live rendering."""
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
                "vertices": [
                    [float(vertex.x), float(vertex.y)] for vertex in shape.get_vertices()
                ],
            }
        raise TypeError(f"Unsupported live shape: {type(shape)!r}")

    def live_snapshot(self, include_metadata: bool = True) -> dict[str, Any]:
        """Return the current physics state without rendering or advancing time."""
        if self.world is None:
            raise RuntimeError("Call reset() before requesting a live snapshot")
        body_names = list(self.stickman.bodies)
        waypoints = [
            self._waypoint_position(waypoint).astype(float).tolist()
            for waypoint in self.navigation_waypoints
        ]
        payload: dict[str, Any] = {
            "frame": {
                "body_positions": [
                    [
                        float(self.stickman.bodies[name].position.x),
                        float(self.stickman.bodies[name].position.y),
                    ]
                    for name in body_names
                ],
                "body_angles": [
                    float(self.stickman.bodies[name].angle) for name in body_names
                ],
                "step": int(self.step_count),
                "target_position": self.target_position.astype(float).tolist(),
                "waypoints": waypoints,
                "active_waypoint_index": int(self.active_waypoint_index),
                "goal_hold_count": int(self.goal_hold_count),
                "info": dict(self._last_info),
            },
        }
        if include_metadata:
            payload["metadata"] = {
                "room": {
                    "width": float(self.config["physics"]["width"]),
                    "height": float(self.config["physics"]["height"]),
                },
                "target": {
                    "position": self.target_position.astype(float).tolist(),
                    "size": [float(value) for value in self.config["target"]["size"]],
                    "hold_steps": int(self.config["target"]["hold_steps"]),
                },
                "obstacles": self.config.get("obstacles", []),
                "waypoints": waypoints,
                "body_names": body_names,
                "body_geometry": {
                    name: self._shape_geometry(self.stickman.shapes[name]) for name in body_names
                },
                "action_names": list(self.stickman.actuated_joint_names),
                "max_steps": int(self.config["episode"]["max_steps"]),
                "control_repeat": int(self.config["physics"].get("control_repeat", 1)),
            }
        return payload

    def render(self) -> np.ndarray | None:
        if self.render_mode is None:
            return None
        if self.world is None:
            raise RuntimeError("Call reset() before render()")
        if self.renderer is None:
            self.renderer = PygameRenderer(self.config, self.render_mode)
        return self.renderer.render(self.world, self._last_info)

    def set_reward_weights(self, weights: dict[str, float]) -> None:
        """Update reward weights at runtime for smooth curriculum transitions."""
        self.reward_calculator.weights = {key: float(value) for key, value in weights.items()}

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        self.world = None
