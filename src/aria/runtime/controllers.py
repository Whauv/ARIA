from __future__ import annotations

import math
from collections.abc import Iterable
from functools import lru_cache

import cv2
import numpy as np

from .. import config
from ..config import (
    COLOR_TOGGLE_GESTURE_SECONDS,
    DELETE_CONFIRM_SECONDS,
    DOUBLE_PINCH_SECONDS,
    DRAG_CLUTCH_SECONDS,
    DRAG_SMOOTHING_NEW_WEIGHT,
    DRAG_SMOOTHING_PREV_WEIGHT,
    DRAW_POSE_GRACE_SECONDS,
    FINISH_HOLD_SECONDS,
    MAX_MEDIAPIPE_HEIGHT,
    MAX_MEDIAPIPE_WIDTH,
    MAX_SEGMENT_STEP_PX,
    MIN_DRAW_MOVEMENT_PX,
    SPRITE_HIT_PADDING,
    STATUS_DRAWING,
    STATUS_IDLE,
    STATUS_PAUSED,
)
from ..drawing.canvas import DrawingCanvas, interpolate_segment
from ..drawing.sprite import create_sprite_from_canvas
from ..ui.layout import PaletteItem, ThumbnailItem, ToolbarItem, get_hover_padding
from ..vision.gestures import (
    is_closed_fist,
    is_double_pinch,
    is_draw_pose,
    is_index_and_middle_up,
    point_in_rect,
)
from .state import RuntimeState


def clear_sprite_selection(sprites) -> None:
    for sprite in sprites:
        sprite.selected = False
        sprite.dragging = False
        sprite.drag_offset_x = -1
        sprite.drag_offset_y = -1


def get_selected_sprite(sprites):
    for sprite in sorted(sprites, key=lambda item: item.z_index, reverse=True):
        if sprite.selected:
            return sprite
    return None


def get_topmost_sprite_near_point(sprites, point, padding: int = 18):
    if point is None:
        return None

    px, py = point
    for sprite in sorted(sprites, key=lambda item: item.z_index, reverse=True):
        if sprite.contains_point((px, py)):
            return sprite

    for sprite in sorted(sprites, key=lambda item: item.z_index, reverse=True):
        if sprite.contains_point_padded((px, py), padding):
            return sprite
    return None


def bring_sprite_to_front(sprites, sprite) -> None:
    if sprite is None:
        return

    next_z_index = max((item.z_index for item in sprites), default=0) + 1
    sprite.z_index = next_z_index


def resize_for_mediapipe(frame: np.ndarray) -> np.ndarray:
    frame_height, frame_width = frame.shape[:2]
    scale = min(
        1.0,
        MAX_MEDIAPIPE_WIDTH / frame_width,
        MAX_MEDIAPIPE_HEIGHT / frame_height,
    )
    if scale >= 1.0:
        return frame

    resized_width = int(frame_width * scale)
    resized_height = int(frame_height * scale)
    return cv2.resize(
        frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA
    )


@lru_cache(maxsize=8)
def _gamma_table(gamma: float) -> np.ndarray:
    """Cached 256-entry gamma LUT (it was rebuilt every frame)."""
    index = np.arange(256, dtype=np.float32) / 255.0
    return np.clip(np.power(index, 1.0 / gamma) * 255.0, 0, 255).astype(np.uint8)


def enhance_low_light(frame: np.ndarray) -> np.ndarray:
    required_attrs = (
        "cvtColor",
        "split",
        "createCLAHE",
        "merge",
        "LUT",
        "COLOR_BGR2LAB",
        "COLOR_LAB2BGR",
    )
    if not all(hasattr(cv2, attr) for attr in required_attrs):
        return frame

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    mean_luma = float(np.mean(l_channel))
    if mean_luma >= 115:
        return frame

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    enhanced = cv2.cvtColor(
        cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2BGR
    )
    gamma = 1.25 if mean_luma > 80 else 1.45
    return cv2.LUT(enhanced, _gamma_table(gamma))


def expand_rect(
    rect: tuple[int, int, int, int], padding: int
) -> tuple[int, int, int, int]:
    left, top, right, bottom = rect
    return left - padding, top - padding, right + padding, bottom + padding


def point_in_any_rect(
    point: tuple[int, int] | None, rects: list[tuple[int, int, int, int]]
) -> bool:
    if point is None:
        return False
    return any(point_in_rect(point, rect) for rect in rects)


def describe_services(
    ai_enabled: bool,
    voice_available: bool,
    voice_error: str | None,
    voice_listening: bool,
    voice_speaking: bool,
    transcription_enabled: bool,
    wake_word_ready: bool,
) -> dict[str, tuple[str, str]]:
    """Honest, compact service states as {service: (label, status_kind)}.

    Never reports a service as ready before it has actually started.
    """
    if voice_listening:
        voice = ("listening", "active")
    elif voice_speaking:
        voice = ("speaking", "active")
    elif not voice_available:
        voice = ("off", "off" if not voice_error else "warn")
    elif not transcription_enabled:
        voice = ("wake word only", "warn")
    elif wake_word_ready:
        voice = ("ready", "ok")
    else:
        voice = ("starting", "info")
    ai = ("ready", "ok") if ai_enabled else ("off - no key", "off")
    return {"voice": voice, "ai": ai}


class UIInteractionController:
    def resolve_hover_candidate(
        self,
        hover_point: tuple[int, int] | None,
        pinch_active: bool,
        interaction_mode: str,
        stroke_active: bool,
        toolbar_items: list[ToolbarItem],
        palette_items: list[PaletteItem],
        disabled_ids: Iterable[str] = (),
    ) -> str | None:
        if (
            hover_point is None
            or pinch_active
            or (interaction_mode == config.MODE_DRAW and stroke_active)
        ):
            return None

        disabled = set(disabled_ids)
        for item in toolbar_items:
            if item["id"] in disabled:
                continue
            if point_in_rect(
                hover_point,
                expand_rect(
                    item["rect"], get_hover_padding(interaction_mode, "toolbar")
                ),
            ):
                return item["id"]

        for item in palette_items:
            if point_in_rect(
                hover_point,
                expand_rect(
                    item["rect"], get_hover_padding(interaction_mode, "palette")
                ),
            ):
                return item["id"]

        return None

    def consume_hover(
        self, state: RuntimeState, hover_candidate: str | None, now: float
    ) -> str | None:
        """Advance dwell timing; return a target id exactly once per hover.

        After a target fires it stays latched until the pointer leaves it, so
        holding still over Undo or Clear never repeats the action.
        """
        if hover_candidate != state.hover_latched_id:
            state.hover_latched_id = None
        if hover_candidate is not None and hover_candidate == state.hover_latched_id:
            state.clear_hover()
            return None

        if hover_candidate != state.hover_target_id:
            state.hover_target_id = hover_candidate
            state.hover_start_time = now if hover_candidate else None
            return None

        required_dwell = config.dwell_seconds_for_mode(state.interaction_mode)
        if (
            hover_candidate
            and state.hover_start_time is not None
            and now - state.hover_start_time >= required_dwell
        ):
            state.clear_hover()
            state.hover_latched_id = hover_candidate
            return hover_candidate
        return None


class SpriteInteractionController:
    """Select-mode pinch handling.

    Priority (highest first) when a pinch starts:
      1. a pending delete on the pinched sprite -> delete
      2. a layer thumbnail -> select it and bring it to the front
      3. second quick pinch on the same sprite -> arm delete
      4. the topmost sprite under the fingertip -> select (hold to drag)
      5. empty space -> clear the selection
    A pinch only ever acts on the sprite it started on; sweeping a held pinch
    across other sprites never grabs them.
    """

    def handle_select_mode(
        self,
        state: RuntimeState,
        sprites,
        thumbnail_items: list[ThumbnailItem],
        hit_point: tuple[int, int] | None,
        current_point: tuple[int, int] | None,
        pinch_active: bool,
        pinch_started: bool,
        frame_width: int,
        frame_height: int,
        now: float,
    ) -> None:
        if current_point is None or hit_point is None:
            state.hovered_sprite_ref = None
            return

        if (
            state.delete_pending_expires_at is not None
            and now > state.delete_pending_expires_at
        ):
            state.clear_delete_confirmation()

        hovered_sprite = get_topmost_sprite_near_point(
            sprites, hit_point, padding=SPRITE_HIT_PADDING
        )
        state.hovered_sprite_ref = (
            id(hovered_sprite) if hovered_sprite is not None else None
        )

        if pinch_started:
            thumbnail_target = next(
                (
                    item["sprite"]
                    for item in thumbnail_items
                    if point_in_rect(hit_point, expand_rect(item["rect"], 10))
                ),
                None,
            )
            target = thumbnail_target or hovered_sprite
            if target is None:
                self._pinch_empty_space(state, sprites)
                return
            if state.delete_pending_sprite_ref == id(target):
                self._delete(state, sprites, target)
                return
            if thumbnail_target is not None:
                self._select(state, sprites, target, hit_point, now)
                state.pinch_target_sprite_ref = None
                state.pinch_started_at = None
                state.post_event("Layer brought to front", "info")
                return
            if is_double_pinch(
                state.last_pinch_time, now, DOUBLE_PINCH_SECONDS
            ) and state.last_pinched_sprite_ref == id(target):
                state.delete_pending_sprite_ref = id(target)
                state.delete_pending_expires_at = now + DELETE_CONFIRM_SECONDS
                state.pinch_target_sprite_ref = None
                state.pinch_started_at = None
                state.set_status(
                    "Pinch once more to delete", now, DELETE_CONFIRM_SECONDS
                )
                return
            self._select(state, sprites, target, hit_point, now)
            state.set_status(
                "Selected - hold pinch to drag", now, DRAG_CLUTCH_SECONDS + 0.45
            )

        dragging_sprite = next((sprite for sprite in sprites if sprite.dragging), None)
        held_sprite = dragging_sprite or next(
            (
                sprite
                for sprite in sprites
                if id(sprite) == state.pinch_target_sprite_ref
            ),
            None,
        )

        if not (pinch_active and held_sprite is not None):
            for sprite in sprites:
                sprite.dragging = False
            state.pinch_target_sprite_ref = None
            state.pinch_started_at = None
            return

        if state.pinch_started_at is None:
            state.pinch_started_at = now
        if now - state.pinch_started_at < DRAG_CLUTCH_SECONDS:
            state.set_status("Hold pinch to drag", now, 0.35)
            return

        if not held_sprite.selected:
            clear_sprite_selection(sprites)
            held_sprite.selected = True
        held_sprite.dragging = True
        if held_sprite.drag_offset_x < 0 or held_sprite.drag_offset_y < 0:
            held_sprite.drag_offset_x = held_sprite.w // 2
            held_sprite.drag_offset_y = held_sprite.h // 2
        target_x = current_point[0] - held_sprite.drag_offset_x
        target_y = current_point[1] - held_sprite.drag_offset_y
        held_sprite.x = int(
            DRAG_SMOOTHING_PREV_WEIGHT * held_sprite.x
            + DRAG_SMOOTHING_NEW_WEIGHT * target_x
        )
        held_sprite.y = int(
            DRAG_SMOOTHING_PREV_WEIGHT * held_sprite.y
            + DRAG_SMOOTHING_NEW_WEIGHT * target_y
        )
        held_sprite.clamp_to_frame(frame_width, frame_height)
        state.set_status("Dragging - release to place", now, 0.35)

    @staticmethod
    def _select(state: RuntimeState, sprites, target, hit_point, now: float) -> None:
        clear_sprite_selection(sprites)
        state.clear_delete_confirmation()
        target.selected = True
        bring_sprite_to_front(sprites, target)
        target.drag_offset_x = max(0, min(target.w, hit_point[0] - target.x))
        target.drag_offset_y = max(0, min(target.h, hit_point[1] - target.y))
        state.last_pinch_time = now
        state.last_pinched_sprite_ref = id(target)
        state.pinch_target_sprite_ref = id(target)
        state.pinch_started_at = now
        state.guide_selected = True

    @staticmethod
    def _delete(state: RuntimeState, sprites, target) -> None:
        if target in sprites:
            sprites.remove(target)
        clear_sprite_selection(sprites)
        state.clear_delete_confirmation()
        state.pinch_target_sprite_ref = None
        state.pinch_started_at = None
        state.last_pinch_time = None
        state.last_pinched_sprite_ref = None
        state.post_event("Sprite deleted - U to undo", "success")

    @staticmethod
    def _pinch_empty_space(state: RuntimeState, sprites) -> None:
        state.clear_delete_confirmation()
        state.pinch_target_sprite_ref = None
        state.pinch_started_at = None
        if get_selected_sprite(sprites) is not None:
            clear_sprite_selection(sprites)
            state.post_event("Selection cleared", "info")


class DrawingInteractionController:
    """Draw-mode gestures.

    Priority (highest first): closed-fist hold (make sprite) > pinch hold
    (quick color toggle) > index-only draw pose > index+middle pen lift.
    A fist can read as a pinch because the thumb rests on the index finger,
    so the fist always wins and suppresses the color toggle.
    """

    def _toggle_quick_color(
        self, state: RuntimeState, drawing_canvas: DrawingCanvas, now: float
    ) -> None:
        colors = config.QUICK_TOGGLE_COLORS
        if not colors or state.color_toggle_triggered:
            return

        if state.color_toggle_hold_started_at is None:
            state.color_toggle_hold_started_at = now
            state.set_status(
                "Hold pinch to switch color", now, COLOR_TOGGLE_GESTURE_SECONDS
            )
            return

        if now - state.color_toggle_hold_started_at < COLOR_TOGGLE_GESTURE_SECONDS:
            return

        current = state.brush_name
        if current not in colors:
            next_color = colors[0]
        else:
            next_color = colors[(colors.index(current) + 1) % len(colors)]

        next_color_value = config.get_brush_color(next_color)
        if next_color_value is None:
            return

        drawing_canvas.set_brush_color(next_color_value)
        state.brush_name = next_color
        state.last_color_gesture_time = now
        state.color_toggle_triggered = True
        state.post_event(f"Brush: {next_color.title()}", "info")

    def _handle_fist(
        self, state: RuntimeState, drawing_canvas: DrawingCanvas, sprites, now: float
    ) -> None:
        if state.fist_latched:
            return
        if state.fist_start_time is None:
            state.fist_start_time = now
            state.set_status("Hold fist to create sprite", now, FINISH_HOLD_SECONDS)
            return
        if now - state.fist_start_time < FINISH_HOLD_SECONDS:
            return

        state.clear_fist_hold()
        state.fist_latched = True
        drawing_canvas.reset_stroke()
        state.clear_drawing_path()
        state.clear_draw_pose()
        next_z_index = max((sprite.z_index for sprite in sprites), default=-1) + 1
        sprite = create_sprite_from_canvas(drawing_canvas.canvas, next_z_index)
        if sprite is None:
            state.post_event("Nothing to convert - draw something first", "warn")
            return
        clear_sprite_selection(sprites)
        sprites.append(sprite)
        sprite.selected = True
        drawing_canvas.clear()
        state.clear_smoothing()
        state.guide_made_sprite = True
        state.post_event("Sprite created - press V to move it", "success")

    def handle_draw_mode(
        self,
        state: RuntimeState,
        hand_landmarks,
        current_point: tuple[int, int] | None,
        raw_fingertip: tuple[int, int] | None,
        pinch_active: bool,
        drawing_canvas: DrawingCanvas,
        sprites,
        now: float,
        ui_rects: list[tuple[int, int, int, int]],
    ) -> None:
        scaled_min_draw_movement_px = MIN_DRAW_MOVEMENT_PX * state.hand_scale_factor
        scaled_max_segment_step_px = MAX_SEGMENT_STEP_PX * state.hand_scale_factor
        stroke_active = state.stroke_active()
        pointer_over_ui = point_in_any_rect(raw_fingertip or current_point, ui_rects)
        actively_triggering_ui = state.hover_target_id is not None
        drawing_blocked_by_ui = (not stroke_active) and (
            actively_triggering_ui or pointer_over_ui
        )

        if is_closed_fist(hand_landmarks):
            self._handle_fist(state, drawing_canvas, sprites, now)
            state.clear_color_toggle()
            if state.stroke_active():
                drawing_canvas.reset_stroke()
                state.clear_drawing_path()
            state.set_activity(STATUS_IDLE, now)
            return
        state.clear_fist_hold()
        state.fist_latched = False

        draw_pose_active = is_draw_pose(hand_landmarks)
        if draw_pose_active:
            state.last_draw_pose_time = now

        draw_pose_with_grace = draw_pose_active or (
            state.last_draw_pose_time is not None
            and now - state.last_draw_pose_time <= DRAW_POSE_GRACE_SECONDS
        )

        if pinch_active and not stroke_active and not drawing_blocked_by_ui:
            self._toggle_quick_color(state, drawing_canvas, now)
            return
        if not pinch_active:
            state.clear_color_toggle()

        if (
            draw_pose_with_grace
            and current_point is not None
            and not pinch_active
            and not drawing_blocked_by_ui
        ):
            state.set_activity(STATUS_DRAWING, now)
            state.guide_drew = True
            if state.prev_draw_point is not None:
                movement_px = math.hypot(
                    current_point[0] - state.prev_draw_point[0],
                    current_point[1] - state.prev_draw_point[1],
                )
                if movement_px >= scaled_min_draw_movement_px:
                    drawing_canvas.add_path(
                        interpolate_segment(
                            state.prev_draw_point,
                            current_point,
                            max(1, round(scaled_max_segment_step_px)),
                        )
                    )
                    state.prev_draw_point = current_point
            else:
                state.prev_smoothed_point = (
                    float(current_point[0]),
                    float(current_point[1]),
                )
                drawing_canvas.add_segment(current_point, current_point)
                state.prev_draw_point = current_point
        elif is_index_and_middle_up(hand_landmarks):
            drawing_canvas.reset_stroke()
            state.clear_drawing_path()
            state.clear_draw_pose()
            state.set_activity(STATUS_PAUSED, now)
        else:
            drawing_canvas.reset_stroke()
            state.clear_drawing_path()
            state.clear_draw_pose()
            state.set_activity(STATUS_IDLE, now)
