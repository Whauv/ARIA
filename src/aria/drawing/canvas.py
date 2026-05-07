from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np

from .. import config
from ..config import (
    LINE_THICKNESS,
    MAX_UNDO_SNAPSHOTS,
    SMOOTHING_FAST_DISTANCE_PX,
    SMOOTHING_FAST_NEW_WEIGHT,
    SMOOTHING_FAST_PREV_WEIGHT,
    SMOOTHING_SLOW_DISTANCE_PX,
    SMOOTHING_SLOW_NEW_WEIGHT,
    SMOOTHING_SLOW_PREV_WEIGHT,
)


class DrawingCanvas:
    def __init__(self, frame_shape: tuple[int, int, int]) -> None:
        if len(frame_shape) != 3:
            raise ValueError("frame_shape must be a 3D image shape")

        self.canvas = np.zeros(frame_shape, dtype=np.uint8)
        self.strokes: list[dict[str, object]] = []
        self.undo_stack: list[np.ndarray] = [self.canvas.copy()]
        self._current_stroke: list[tuple[int, int]] = []
        self.brush_color = config.get_brush_color(config.DEFAULT_BRUSH_NAME) or config.DRAW_COLOR

    def set_brush_color(self, color: tuple[int, int, int]) -> None:
        self.brush_color = color

    def _rebuild_canvas(self) -> None:
        self.canvas[:] = 0
        for stroke_data in self.strokes:
            points = stroke_data["points"]
            color = stroke_data["color"]
            for start, end in zip(points, points[1:]):
                cv2.line(self.canvas, start, end, color, LINE_THICKNESS)

    def _push_snapshot(self) -> None:
        if self.undo_stack and np.array_equal(self.undo_stack[-1], self.canvas):
            return

        self.undo_stack.append(self.canvas.copy())
        if len(self.undo_stack) > MAX_UNDO_SNAPSHOTS:
            self.undo_stack = self.undo_stack[-MAX_UNDO_SNAPSHOTS:]

    def reset_stroke(self) -> None:
        if self._current_stroke:
            self.strokes.append(
                {
                    "points": self._current_stroke.copy(),
                    "color": self.brush_color,
                }
            )
            self._current_stroke.clear()
            self._push_snapshot()

    def add_segment(self, start: tuple[int, int], end: tuple[int, int]) -> None:
        if start is None or end is None:
            return
        if not self._current_stroke:
            self._current_stroke.append(start)
        self._current_stroke.append(end)
        cv2.line(self.canvas, start, end, self.brush_color, LINE_THICKNESS)
        radius = max(1, LINE_THICKNESS // 2)
        cv2.circle(self.canvas, start, radius, self.brush_color, -1)
        cv2.circle(self.canvas, end, radius, self.brush_color, -1)

    def add_path(self, points: list[tuple[int, int]]) -> None:
        if not points:
            return
        if len(points) == 1:
            self.add_segment(points[0], points[0])
            return

        for start, end in zip(points, points[1:]):
            self.add_segment(start, end)

    def clear(self) -> None:
        self.canvas[:] = 0
        self.strokes.clear()
        self._current_stroke.clear()
        self._push_snapshot()

    def undo_last_stroke(self) -> bool:
        if self._current_stroke:
            self._current_stroke.clear()
            if self.undo_stack:
                self.canvas[:] = self.undo_stack[-1]
            return True

        if len(self.undo_stack) <= 1:
            return False

        self.undo_stack.pop()
        self.canvas[:] = self.undo_stack[-1]
        if self.strokes:
            self.strokes.pop()
        return True

    def overlay_on(self, frame: np.ndarray, alpha: float) -> np.ndarray:
        if not hasattr(cv2, "addWeighted"):
            return np.where(self.canvas > 0, self.canvas, frame).astype(frame.dtype, copy=False)
        return cv2.addWeighted(frame, 1.0, self.canvas, alpha, 0.0)


def smooth_point(
    prev_point: Optional[tuple[float, float]],
    new_point: tuple[int, int],
    prev_weight: float,
    new_weight: float,
) -> tuple[float, float]:
    if prev_point is None:
        return float(new_point[0]), float(new_point[1])

    x = prev_weight * prev_point[0] + new_weight * new_point[0]
    y = prev_weight * prev_point[1] + new_weight * new_point[1]
    return x, y


def interpolate_segment(
    start: tuple[int, int],
    end: tuple[int, int],
    max_step_px: int,
) -> list[tuple[int, int]]:
    if max_step_px <= 0:
        return [start, end]

    distance = math.hypot(end[0] - start[0], end[1] - start[1])
    if distance <= max_step_px:
        return [start, end]

    steps = max(1, int(math.ceil(distance / max_step_px)))
    points = [start]
    for step_index in range(1, steps):
        ratio = step_index / steps
        x = int(round(start[0] + (end[0] - start[0]) * ratio))
        y = int(round(start[1] + (end[1] - start[1]) * ratio))
        interpolated = (x, y)
        if interpolated != points[-1]:
            points.append(interpolated)
    if end != points[-1]:
        points.append(end)
    return points


def adaptive_smooth_point(
    prev_point: Optional[tuple[float, float]],
    new_point: tuple[int, int],
    distance: float,
    slow_distance_px: float = SMOOTHING_SLOW_DISTANCE_PX,
    fast_distance_px: float = SMOOTHING_FAST_DISTANCE_PX,
    slow_prev_weight: float = SMOOTHING_SLOW_PREV_WEIGHT,
    slow_new_weight: float = SMOOTHING_SLOW_NEW_WEIGHT,
    fast_prev_weight: float = SMOOTHING_FAST_PREV_WEIGHT,
    fast_new_weight: float = SMOOTHING_FAST_NEW_WEIGHT,
) -> tuple[float, float]:
    if prev_point is None:
        return float(new_point[0]), float(new_point[1])

    if distance <= slow_distance_px:
        prev_weight, new_weight = slow_prev_weight, slow_new_weight
    elif distance >= fast_distance_px:
        prev_weight, new_weight = fast_prev_weight, fast_new_weight
    else:
        ratio = (distance - slow_distance_px) / max(1e-6, fast_distance_px - slow_distance_px)
        prev_weight = slow_prev_weight + (fast_prev_weight - slow_prev_weight) * ratio
        new_weight = slow_new_weight + (fast_new_weight - slow_new_weight) * ratio

    return smooth_point(prev_point, new_point, prev_weight, new_weight)
