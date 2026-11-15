from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ..config import DEFAULT_BRUSH_NAME, MAX_PENDING_EVENTS, STATUS_IDLE


@dataclass
class RuntimeState:
    frame_dimensions: dict[str, int] = field(
        default_factory=lambda: {"width": 0, "height": 0}
    )
    prev_smoothed_point: tuple[float, float] | None = None
    prev_draw_point: tuple[int, int] | None = None
    prev_resize_distance: float | None = None
    resize_anchor_distance: float | None = None
    resize_anchor_width: int | None = None
    resizing_sprite_ref: int | None = None
    fist_start_time: float | None = None
    hover_target_id: str | None = None
    hover_start_time: float | None = None
    # A dwell target that already fired; it cannot fire again until the
    # pointer leaves it, so a sustained hover never repeats an action.
    hover_latched_id: str | None = None
    # Set after a fist hold fires; the fist must open before it can re-arm.
    fist_latched: bool = False
    last_pinch_time: float | None = None
    last_pinched_sprite_ref: int | None = None
    last_draw_pose_time: float | None = None
    last_color_gesture_time: float | None = None
    color_toggle_hold_started_at: float | None = None
    color_toggle_triggered: bool = False
    pinch_started_at: float | None = None
    pinch_target_sprite_ref: int | None = None
    hovered_sprite_ref: int | None = None
    delete_pending_sprite_ref: int | None = None
    delete_pending_expires_at: float | None = None
    resize_clutch_started_at: float | None = None
    resize_anchor_first_point: tuple[int, int] | None = None
    resize_anchor_second_point: tuple[int, int] | None = None
    previous_pinch_active: bool = False
    status_text: str = STATUS_IDLE
    status_expires_at: float | None = None
    interaction_mode: str = "draw_mode"
    brush_name: str = DEFAULT_BRUSH_NAME
    fps: float = 0.0
    ai_warning_text: str | None = None
    latest_frame: np.ndarray | None = None
    # Camera + drawing + sprites without UI chrome (used for export and AI).
    latest_scene_frame: np.ndarray | None = None
    hand_scale_factor: float = 1.0
    calibrated: bool = False
    calibration_started_at: float | None = None
    calibration_sample_count: int = 0
    calibration_sample_total: float = 0.0
    tracking_stability: float = 0.0
    tracking_status_text: str = "Waiting for hand"
    tracking_recovery_hint: str = "Place your hand in view"
    guide_visible: bool = True
    help_visible: bool = False
    settings_visible: bool = False
    guide_drew: bool = False
    guide_made_sprite: bool = False
    guide_selected: bool = False
    hand_present: bool = False
    # Outbox for user-facing events raised by controllers (text, kind).
    events: deque[tuple[str, str]] = field(
        default_factory=lambda: deque(maxlen=MAX_PENDING_EVENTS)
    )

    def set_status(
        self, text: str, now: float | None = None, duration: float | None = None
    ) -> None:
        self.status_text = text
        if now is not None and duration is not None:
            self.status_expires_at = now + duration

    def set_activity(self, text: str, now: float | None = None) -> None:
        """Set the continuous activity label without hiding a timed hint."""
        if (
            self.status_expires_at is not None
            and now is not None
            and now < self.status_expires_at
        ):
            return
        self.status_text = text
        self.status_expires_at = None

    def post_event(self, text: str, kind: str = "info") -> None:
        self.events.append((text, kind))

    def drain_events(self) -> list[tuple[str, str]]:
        drained = list(self.events)
        self.events.clear()
        return drained

    def guide_complete(self) -> bool:
        return self.guide_drew and self.guide_made_sprite and self.guide_selected

    def clear_hover(self) -> None:
        self.hover_target_id = None
        self.hover_start_time = None

    def clear_drawing_path(self) -> None:
        self.prev_draw_point = None

    def clear_draw_pose(self) -> None:
        self.last_draw_pose_time = None

    def clear_smoothing(self) -> None:
        self.prev_smoothed_point = None

    def clear_resize(self) -> None:
        self.prev_resize_distance = None
        self.resize_anchor_distance = None
        self.resize_anchor_width = None
        self.resizing_sprite_ref = None
        self.resize_anchor_first_point = None
        self.resize_anchor_second_point = None

    def clear_resize_clutch(self) -> None:
        self.resize_clutch_started_at = None

    def clear_color_toggle(self) -> None:
        self.color_toggle_hold_started_at = None
        self.color_toggle_triggered = False

    def clear_delete_confirmation(self) -> None:
        self.delete_pending_sprite_ref = None
        self.delete_pending_expires_at = None

    def clear_fist_hold(self) -> None:
        self.fist_start_time = None

    def reset_for_hand_loss(self) -> None:
        """Cancel every pending gesture; nothing may fire while untracked."""
        self.reset_interaction_transients()
        self.clear_resize()
        self.clear_resize_clutch()
        self.clear_smoothing()
        self.hover_latched_id = None
        self.fist_latched = False

    def reset_interaction_transients(self) -> None:
        self.clear_hover()
        self.clear_drawing_path()
        self.clear_draw_pose()
        self.clear_fist_hold()
        self.clear_color_toggle()
        self.clear_delete_confirmation()
        self.previous_pinch_active = False
        self.pinch_started_at = None
        self.pinch_target_sprite_ref = None
        self.hovered_sprite_ref = None

    def stroke_active(self) -> bool:
        return self.prev_draw_point is not None

    def begin_calibration(self, now: float) -> None:
        if self.calibration_started_at is None:
            self.calibration_started_at = now

    def add_calibration_sample(self, sample: float) -> None:
        self.calibration_sample_total += sample
        self.calibration_sample_count += 1

    def finish_calibration(self, hand_scale_factor: float) -> None:
        self.hand_scale_factor = hand_scale_factor
        self.calibrated = True

    def set_tracking_stability(self, value: float) -> None:
        self.tracking_stability = max(0.0, min(1.0, value))

    def expire_status(self, now: float) -> None:
        if self.status_expires_at and now >= self.status_expires_at:
            self.status_text = STATUS_IDLE
            self.status_expires_at = None
