from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from config import DRAW_COLOR, LINE_THICKNESS


class DrawingCanvas:
    def __init__(self, frame_shape: tuple[int, int, int]) -> None:
        self.canvas = np.zeros(frame_shape, dtype=np.uint8)
        self.strokes: list[list[tuple[int, int]]] = []
        self.undo_stack: list[list[tuple[int, int]]] = []
        self._current_stroke: list[tuple[int, int]] = []

    def reset_stroke(self) -> None:
        if self._current_stroke:
            self.strokes.append(self._current_stroke.copy())
            self._current_stroke.clear()

    def add_segment(self, start: tuple[int, int], end: tuple[int, int]) -> None:
        if not self._current_stroke:
            self._current_stroke.append(start)
        self._current_stroke.append(end)
        cv2.line(self.canvas, start, end, DRAW_COLOR, LINE_THICKNESS)

    def clear(self) -> None:
        self.canvas[:] = 0
        self.undo_stack.extend(self.strokes)
        if self._current_stroke:
            self.undo_stack.append(self._current_stroke.copy())
        self.strokes.clear()
        self._current_stroke.clear()

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
