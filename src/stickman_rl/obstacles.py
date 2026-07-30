"""Configuration-driven static obstacle construction."""

from __future__ import annotations

from typing import Any

import pymunk

from stickman_rl.constants import COLLISION_OBSTACLE


def _style(shape: pymunk.Shape, friction: float = 1.0) -> pymunk.Shape:
    shape.friction = friction
    shape.elasticity = 0.0
    shape.collision_type = COLLISION_OBSTACLE
    return shape


def trench_intervals(obstacles: list[dict[str, Any]]) -> list[tuple[float, float]]:
    intervals: list[tuple[float, float]] = []
    for spec in obstacles:
        if spec.get("type") == "trench":
            x, _ = map(float, spec.get("position", [6.0, 0.0]))
            width = float(spec.get("width", spec.get("size", [1.0, 0.0])[0]))
            intervals.append((x - width * 0.5, x + width * 0.5))
    return intervals


def build_obstacles(space: pymunk.Space, specs: list[dict[str, Any]]) -> list[pymunk.Shape]:
    """Build boxes, walls, platforms, and slopes. Trenches are floor gaps handled by PhysicsWorld."""
    shapes: list[pymunk.Shape] = []
    static = space.static_body
    for spec in specs:
        kind = str(spec.get("type", "box")).lower()
        if kind == "trench":
            continue
        x, y = map(float, spec.get("position", [6.0, 0.5]))
        if kind in {"box", "platform", "wall"}:
            width, height = map(float, spec.get("size", [1.0, 0.5]))
            body = pymunk.Body(body_type=pymunk.Body.STATIC)
            body.position = (x, y)
            shape = _style(pymunk.Poly.create_box(body, (width, height)), float(spec.get("friction", 1.0)))
            space.add(body, shape)
            shapes.append(shape)
        elif kind == "slope":
            start = tuple(map(float, spec.get("start", [x - 1.0, y])))
            end = tuple(map(float, spec.get("end", [x + 1.0, y + 0.8])))
            radius = float(spec.get("radius", 0.05))
            shape = _style(pymunk.Segment(static, start, end, radius), float(spec.get("friction", 1.0)))
            space.add(shape)
            shapes.append(shape)
        else:
            raise ValueError(f"Unsupported obstacle type: {kind}")
    return shapes
