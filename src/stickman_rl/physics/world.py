"""Room physics world and static geometry."""

from __future__ import annotations

from typing import Any

import pymunk

from stickman_rl.constants import COLLISION_ROOM, COLLISION_TARGET
from stickman_rl.obstacles import build_obstacles, trench_intervals
from stickman_rl.physics.stickman import Stickman


class PhysicsWorld:
    """Owns one PyMunk space, room boundaries, target sensor, obstacles, and stickman."""

    def __init__(self, config: dict[str, Any], target_position: tuple[float, float]) -> None:
        self.config = config
        physics = config["physics"]
        self.width = float(physics["width"])
        self.height = float(physics["height"])
        self.space = pymunk.Space(threaded=False)
        self.space.gravity = tuple(map(float, physics["gravity"]))
        self.space.damping = float(physics["damping"])
        self.space.iterations = int(physics["solver_iterations"])
        self.room_shapes: list[pymunk.Shape] = []
        self.obstacle_shapes: list[pymunk.Shape] = []
        self.target_shape: pymunk.Shape
        self.target_body: pymunk.Body
        self._build_room(config.get("obstacles", []))
        self.obstacle_shapes = build_obstacles(self.space, config.get("obstacles", []))
        self._build_target(target_position)
        self.stickman = Stickman(self.space, config["stickman"])

    def _room_segment(self, a: tuple[float, float], b: tuple[float, float], friction: float) -> None:
        shape = pymunk.Segment(self.space.static_body, a, b, 0.08)
        shape.friction = friction
        shape.elasticity = 0.0
        shape.collision_type = COLLISION_ROOM
        self.space.add(shape)
        self.room_shapes.append(shape)

    def _build_room(self, obstacle_specs: list[dict[str, Any]]) -> None:
        p = self.config["physics"]
        self._room_segment((0.0, 0.0), (0.0, self.height), float(p["wall_friction"]))
        self._room_segment((self.width, 0.0), (self.width, self.height), float(p["wall_friction"]))
        self._room_segment((0.0, self.height), (self.width, self.height), float(p["wall_friction"]))
        gaps = sorted(trench_intervals(obstacle_specs))
        cursor = 0.0
        for start, end in gaps:
            start = min(max(start, 0.0), self.width)
            end = min(max(end, 0.0), self.width)
            if start > cursor:
                self._room_segment((cursor, 0.0), (start, 0.0), float(p["floor_friction"]))
            cursor = max(cursor, end)
        if cursor < self.width:
            self._room_segment((cursor, 0.0), (self.width, 0.0), float(p["floor_friction"]))

    def _build_target(self, target_position: tuple[float, float]) -> None:
        width, height = map(float, self.config["target"]["size"])
        self.target_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.target_body.position = target_position
        self.target_shape = pymunk.Poly.create_box(self.target_body, (width, height))
        self.target_shape.sensor = True
        self.target_shape.collision_type = COLLISION_TARGET
        self.space.add(self.target_body, self.target_shape)

    def step(self) -> None:
        dt = float(self.config["physics"]["dt"])
        substeps = int(self.config["physics"]["substeps"])
        sub_dt = dt / substeps
        for _ in range(substeps):
            self.space.step(sub_dt)
