"""Articulated stickman construction and actuator control."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, pi
from typing import Any

import numpy as np
import pymunk

from stickman_rl.constants import ACTUATED_JOINTS, COLLISION_BODY


@dataclass(slots=True)
class JointAssembly:
    """A pivot, angular limit, and optional motor connecting two rigid bodies."""

    name: str
    parent: pymunk.Body
    child: pymunk.Body
    pivot: pymunk.PivotJoint
    limit: pymunk.RotaryLimitJoint
    motor: pymunk.SimpleMotor | None

    @property
    def angle(self) -> float:
        return float(self.child.angle - self.parent.angle)

    @property
    def angular_velocity(self) -> float:
        return float(self.child.angular_velocity - self.parent.angular_velocity)


class Stickman:
    """Multi-rigid-body humanoid with limited rotary joints and motor actuators."""

    def __init__(self, space: pymunk.Space, config: dict[str, Any]) -> None:
        self.space = space
        self.config = config
        self.bodies: dict[str, pymunk.Body] = {}
        self.shapes: dict[str, pymunk.Shape] = {}
        self.joints: dict[str, JointAssembly] = {}
        self.actuated_joint_names = tuple(ACTUATED_JOINTS)
        self._build()

    @property
    def torso(self) -> pymunk.Body:
        return self.bodies["torso"]

    def _add_box(self, name: str, center: tuple[float, float], size: tuple[float, float]) -> pymunk.Body:
        density = float(self.config["density"])
        mass = max(0.15, density * size[0] * size[1])
        moment = pymunk.moment_for_box(mass, size)
        body = pymunk.Body(mass, moment)
        body.position = center
        shape = pymunk.Poly.create_box(body, size)
        shape.friction = 0.75
        shape.elasticity = 0.0
        shape.collision_type = COLLISION_BODY
        shape.filter = pymunk.ShapeFilter(group=1 if not self.config.get("self_collision", False) else 0)
        self.space.add(body, shape)
        self.bodies[name] = body
        self.shapes[name] = shape
        return body

    def _add_circle(self, name: str, center: tuple[float, float], radius: float) -> pymunk.Body:
        density = float(self.config["density"])
        mass = max(0.1, density * pi * radius * radius)
        moment = pymunk.moment_for_circle(mass, 0.0, radius)
        body = pymunk.Body(mass, moment)
        body.position = center
        shape = pymunk.Circle(body, radius)
        shape.friction = 0.7
        shape.elasticity = 0.0
        shape.collision_type = COLLISION_BODY
        shape.filter = pymunk.ShapeFilter(group=1 if not self.config.get("self_collision", False) else 0)
        self.space.add(body, shape)
        self.bodies[name] = body
        self.shapes[name] = shape
        return body

    def _add_capsule(
        self,
        name: str,
        start: tuple[float, float],
        end: tuple[float, float],
        radius: float,
    ) -> pymunk.Body:
        start_v = pymunk.Vec2d(*start)
        end_v = pymunk.Vec2d(*end)
        delta = end_v - start_v
        length = max(delta.length, radius * 2.1)
        center = (start_v + end_v) * 0.5
        density = float(self.config["density"])
        area = 2.0 * radius * length + pi * radius * radius
        mass = max(0.08, density * area)
        local_a = (-length * 0.5, 0.0)
        local_b = (length * 0.5, 0.0)
        moment = pymunk.moment_for_segment(mass, local_a, local_b, radius)
        body = pymunk.Body(mass, moment)
        body.position = center
        body.angle = atan2(delta.y, delta.x)
        shape = pymunk.Segment(body, local_a, local_b, radius)
        shape.friction = 1.0
        shape.elasticity = 0.0
        shape.collision_type = COLLISION_BODY
        shape.filter = pymunk.ShapeFilter(group=1 if not self.config.get("self_collision", False) else 0)
        self.space.add(body, shape)
        self.bodies[name] = body
        self.shapes[name] = shape
        return body

    def _connect(
        self,
        name: str,
        parent: pymunk.Body,
        child: pymunk.Body,
        anchor: tuple[float, float],
        limits_deg: tuple[float, float],
        actuated: bool = True,
    ) -> None:
        pivot = pymunk.PivotJoint(parent, child, anchor)
        pivot.collide_bodies = False
        lower, upper = np.deg2rad(limits_deg)
        limit = pymunk.RotaryLimitJoint(parent, child, float(lower), float(upper))
        limit.collide_bodies = False
        motor: pymunk.SimpleMotor | None = None
        if actuated:
            motor = pymunk.SimpleMotor(parent, child, 0.0)
            motor.max_force = float(self.config["max_motor_force"])
            motor.collide_bodies = False
            self.space.add(pivot, limit, motor)
        else:
            self.space.add(pivot, limit)
        self.joints[name] = JointAssembly(name, parent, child, pivot, limit, motor)

    def _build(self) -> None:
        x, y = map(float, self.config["spawn"])
        torso_w, torso_h = map(float, self.config["torso_size"])
        head_r = float(self.config["head_radius"])
        ua = float(self.config["upper_arm_length"])
        fa = float(self.config["forearm_length"])
        thigh = float(self.config["thigh_length"])
        shin = float(self.config["shin_length"])
        radius = float(self.config["limb_radius"])

        torso = self._add_box("torso", (x, y), (torso_w, torso_h))
        head_center = (x, y + torso_h * 0.5 + head_r + 0.04)
        head = self._add_circle("head", head_center, head_r)

        left_shoulder = (x - torso_w * 0.45, y + torso_h * 0.30)
        right_shoulder = (x + torso_w * 0.45, y + torso_h * 0.30)
        left_elbow = (left_shoulder[0] - ua * 0.52, left_shoulder[1] - ua * 0.85)
        right_elbow = (right_shoulder[0] + ua * 0.52, right_shoulder[1] - ua * 0.85)
        left_hand = (left_elbow[0] - fa * 0.22, left_elbow[1] - fa * 0.98)
        right_hand = (right_elbow[0] + fa * 0.22, right_elbow[1] - fa * 0.98)

        left_upper_arm = self._add_capsule("left_upper_arm", left_shoulder, left_elbow, radius)
        right_upper_arm = self._add_capsule("right_upper_arm", right_shoulder, right_elbow, radius)
        left_forearm = self._add_capsule("left_forearm", left_elbow, left_hand, radius)
        right_forearm = self._add_capsule("right_forearm", right_elbow, right_hand, radius)

        left_hip = (x - torso_w * 0.28, y - torso_h * 0.48)
        right_hip = (x + torso_w * 0.28, y - torso_h * 0.48)
        left_knee = (left_hip[0] - thigh * 0.14, left_hip[1] - thigh * 0.99)
        right_knee = (right_hip[0] + thigh * 0.32, right_hip[1] - thigh * 0.95)
        left_foot = (left_knee[0] + shin * 0.05, left_knee[1] - shin * 0.99)
        right_foot = (right_knee[0] - shin * 0.05, right_knee[1] - shin * 0.99)

        left_thigh = self._add_capsule("left_thigh", left_hip, left_knee, radius * 1.05)
        right_thigh = self._add_capsule("right_thigh", right_hip, right_knee, radius * 1.05)
        left_shin = self._add_capsule("left_shin", left_knee, left_foot, radius * 0.9)
        right_shin = self._add_capsule("right_shin", right_knee, right_foot, radius * 0.9)

        limits = self.config["joint_limits_deg"]
        self._connect("neck", torso, head, (x, y + torso_h * 0.5), (-45.0, 45.0), actuated=False)
        self._connect("left_shoulder", torso, left_upper_arm, left_shoulder, tuple(limits["left_shoulder"]))
        self._connect("right_shoulder", torso, right_upper_arm, right_shoulder, tuple(limits["right_shoulder"]))
        self._connect("left_elbow", left_upper_arm, left_forearm, left_elbow, tuple(limits["left_elbow"]))
        self._connect("right_elbow", right_upper_arm, right_forearm, right_elbow, tuple(limits["right_elbow"]))
        self._connect("left_hip", torso, left_thigh, left_hip, tuple(limits["left_hip"]))
        self._connect("right_hip", torso, right_thigh, right_hip, tuple(limits["right_hip"]))
        self._connect("left_knee", left_thigh, left_shin, left_knee, tuple(limits["left_knee"]))
        self._connect("right_knee", right_thigh, right_shin, right_knee, tuple(limits["right_knee"]))

    def apply_action(self, action: np.ndarray) -> np.ndarray:
        """Clip normalized actions and convert them to target joint angular speeds."""
        clipped = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        if clipped.shape != (len(self.actuated_joint_names),):
            raise ValueError(f"Expected action shape {(len(self.actuated_joint_names),)}, got {clipped.shape}")
        max_speed = float(self.config["max_joint_speed"])
        for value, name in zip(clipped, self.actuated_joint_names, strict=True):
            motor = self.joints[name].motor
            if motor is not None:
                motor.rate = float(value) * max_speed
        return clipped

    def joint_state(self) -> tuple[np.ndarray, np.ndarray]:
        angles = np.array([self.joints[name].angle for name in self.actuated_joint_names], dtype=np.float32)
        velocities = np.array([self.joints[name].angular_velocity for name in self.actuated_joint_names], dtype=np.float32)
        return angles, velocities

    def main_relative_positions(self) -> np.ndarray:
        torso_pos = self.torso.position
        values: list[float] = []
        for name in (
            "head", "left_upper_arm", "right_upper_arm", "left_forearm", "right_forearm",
            "left_thigh", "right_thigh", "left_shin", "right_shin",
        ):
            delta = self.bodies[name].position - torso_pos
            values.extend((float(delta.x), float(delta.y)))
        return np.asarray(values, dtype=np.float32)

    def finite(self) -> bool:
        for body in self.bodies.values():
            state = (body.position.x, body.position.y, body.velocity.x, body.velocity.y, body.angle, body.angular_velocity)
            if not np.all(np.isfinite(state)):
                return False
        return True

    def set_motors_idle(self) -> None:
        for joint in self.joints.values():
            if joint.motor is not None:
                joint.motor.rate = 0.0
