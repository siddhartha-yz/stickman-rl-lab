"""Pygame renderer kept separate from environment physics and rewards."""

from __future__ import annotations

from typing import Any

import numpy as np
import pygame
import pymunk


class PygameRenderer:
    """Draw the room, articulated body, target, obstacles, and metrics."""

    def __init__(self, config: dict[str, Any], render_mode: str) -> None:
        pygame.init()
        self.config = config
        self.render_mode = render_mode
        self.screen_width = int(config["render"]["width"])
        self.screen_height = int(config["render"]["height"])
        flags = 0 if render_mode == "human" else pygame.HIDDEN
        self.surface = pygame.display.set_mode((self.screen_width, self.screen_height), flags=flags)
        pygame.display.set_caption("Stickman RL Lab")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 18)
        self.scale_x = self.screen_width / float(config["physics"]["width"])
        self.scale_y = self.screen_height / float(config["physics"]["height"])

    def _point(self, point: pymunk.Vec2d | tuple[float, float]) -> tuple[int, int]:
        x, y = float(point[0]), float(point[1])
        return int(x * self.scale_x), int(self.screen_height - y * self.scale_y)

    def _draw_shape(self, shape: pymunk.Shape, color: tuple[int, int, int]) -> None:
        if isinstance(shape, pymunk.Circle):
            center = self._point(shape.body.local_to_world(shape.offset))
            radius = max(2, int(shape.radius * (self.scale_x + self.scale_y) * 0.5))
            pygame.draw.circle(self.surface, color, center, radius)
            pygame.draw.circle(self.surface, (30, 30, 30), center, radius, 2)
        elif isinstance(shape, pymunk.Segment):
            a = self._point(shape.body.local_to_world(shape.a))
            b = self._point(shape.body.local_to_world(shape.b))
            width = max(2, int(shape.radius * (self.scale_x + self.scale_y)))
            pygame.draw.line(self.surface, color, a, b, width)
            pygame.draw.circle(self.surface, color, a, width // 2)
            pygame.draw.circle(self.surface, color, b, width // 2)
        elif isinstance(shape, pymunk.Poly):
            points = [self._point(shape.body.local_to_world(v)) for v in shape.get_vertices()]
            pygame.draw.polygon(self.surface, color, points)
            pygame.draw.polygon(self.surface, (35, 35, 35), points, 2)

    def render(self, world: Any, info: dict[str, Any]) -> np.ndarray | None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt("Renderer window closed")
        self.surface.fill((244, 247, 250))
        for shape in world.room_shapes:
            self._draw_shape(shape, (55, 62, 70))
        for shape in world.obstacle_shapes:
            self._draw_shape(shape, (95, 106, 117))
        self._draw_shape(world.target_shape, (235, 55, 65))
        for name, shape in world.stickman.shapes.items():
            color = (45, 105, 170) if name == "torso" else (30, 35, 40)
            self._draw_shape(shape, color)
        for joint in world.stickman.joints.values():
            p = self._point(joint.parent.local_to_world(joint.pivot.anchor_a))
            pygame.draw.circle(self.surface, (240, 165, 35), p, 4)
        if self.config["render"].get("show_debug", True):
            lines = [
                f"step: {info.get('step', 0)}",
                f"distance: {info.get('distance', 0.0):.3f}",
                f"success: {info.get('is_success', False)}",
                f"reward: {info.get('reward_total', 0.0):.3f}",
                f"torso height: {info.get('torso_height', 0.0):.3f}",
            ]
            for i, text in enumerate(lines):
                self.surface.blit(self.font.render(text, True, (20, 25, 30)), (12, 12 + 22 * i))
        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(int(self.config["render"]["fps"]))
            return None
        frame = pygame.surfarray.array3d(self.surface)
        return np.transpose(frame, (1, 0, 2)).copy()

    def close(self) -> None:
        pygame.display.quit()
        pygame.quit()
