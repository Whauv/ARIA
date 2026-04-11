from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..config import DEFAULT_BRUSH_NAME, STATUS_IDLE


@dataclass
class RuntimeState:
    frame_dimensions: dict[str, int] = field(default_factory=lambda: {"width": 0, "height": 0})
    prev_smoothed_point: Optional[tuple[float, float]] = None
    prev_draw_point: Optional[tuple[int, int]] = None
    prev_resize_distance: Optional[float] = None
    fist_start_time: Optional[float] = None
    hover_target_id: Optional[str] = None
    hover_start_time: Optional[float] = None
    last_pinch_time: Optional[float] = None
    last_pinched_sprite_ref: Optional[int] = None
    previous_pinch_active: bool = False
    status_text: str = STATUS_IDLE
    status_expires_at: Optional[float] = None
    interaction_mode: str = "draw_mode"
    brush_name: str = DEFAULT_BRUSH_NAME
    fps: float = 0.0
    ai_warning_text: Optional[str] = None
    latest_frame: Optional[np.ndarray] = None

    def set_status(self, text: str, now: float | None = None, duration: float | None = None) -> None:
        self.status_text = text
        if now is not None and duration is not None:
            self.status_expires_at = now + duration

    def clear_hover(self) -> None:
        self.hover_target_id = None
        self.hover_start_time = None

    def clear_drawing_path(self) -> None:
        self.prev_draw_point = None

    def clear_smoothing(self) -> None:
        self.prev_smoothed_point = None

    def clear_resize(self) -> None:
        self.prev_resize_distance = None

    def clear_fist_hold(self) -> None:
        self.fist_start_time = None

    def reset_interaction_transients(self) -> None:
        self.clear_hover()
        self.clear_drawing_path()
        self.clear_fist_hold()
        self.previous_pinch_active = False

    def expire_status(self, now: float) -> None:
        if self.status_expires_at and now >= self.status_expires_at:
            self.status_text = STATUS_IDLE
            self.status_expires_at = None
