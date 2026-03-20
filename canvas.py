from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

import config
from config import LINE_THICKNESS


class DrawingCanvas:
    def __init__(self, frame_shape: tuple[int, int, int]) -> None:
        self.canvas = np.zeros(frame_shape, dtype=np.uint8)
        self.strokes: list[dict[str, object]] = []
        self.undo_stack: list[dict[str, object]] = []
        self._current_stroke: list[tuple[int, int]] = []
        self.brush_color = config.get_active_brush_color()

    def set_brush_color(self, color: tuple[int, int, int]) -> None:
        self.brush_color = color

    def _rebuild_canvas(self) -> None:
        self.canvas[:] = 0
        for stroke_data in self.strokes:
            points = stroke_data["points"]
            color = stroke_data["color"]
            for start, end in zip(points, points[1:]):
                cv2.line(self.canvas, start, end, color, LINE_THICKNESS)

    def reset_stroke(self) -> None:
        if self._current_stroke:
            self.strokes.append(
                {
                    "points": self._current_stroke.copy(),
                    "color": self.brush_color,
                }
            )
            self._current_stroke.clear()

    def add_segment(self, start: tuple[int, int], end: tuple[int, int]) -> None:
        if not self._current_stroke:
            self._current_stroke.append(start)
        self._current_stroke.append(end)
        cv2.line(self.canvas, start, end, self.brush_color, LINE_THICKNESS)

    def clear(self) -> None:
        self.canvas[:] = 0
        self.undo_stack.extend(self.strokes)
        if self._current_stroke:
            self.undo_stack.append(
                {
                    "points": self._current_stroke.copy(),
                    "color": self.brush_color,
                }
            )
        self.strokes.clear()
        self._current_stroke.clear()

    def undo_last_stroke(self) -> bool:
        if self._current_stroke:
            self._current_stroke.clear()
            self._rebuild_canvas()
            return True

        if not self.strokes:
            return False

        self.undo_stack.append(self.strokes.pop())
        self._rebuild_canvas()
        return True

    def overlay_on(self, frame: np.ndarray, alpha: float) -> np.ndarray:
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
