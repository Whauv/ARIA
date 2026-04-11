from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from .. import config
from ..config import (
    DOUBLE_PINCH_SECONDS,
    DRAG_SMOOTHING_NEW_WEIGHT,
    DRAG_SMOOTHING_PREV_WEIGHT,
    DWELL_SECONDS,
    FINISH_HOLD_SECONDS,
    MAX_MEDIAPIPE_HEIGHT,
    MAX_MEDIAPIPE_WIDTH,
    STATUS_DRAWING,
    STATUS_IDLE,
    STATUS_PAUSED,
    STATUS_SPRITE_CREATED,
)
from ..drawing.canvas import DrawingCanvas
from ..drawing.sprite import create_sprite_from_canvas
from ..ui.ui import PaletteItem, ThumbnailItem, ToolbarItem
from ..vision.gestures import is_closed_fist, is_double_pinch, is_index_and_middle_up, is_index_only_up, point_in_rect
from .state import RuntimeState


def clear_sprite_selection(sprites) -> None:
    for sprite in sprites:
        sprite.selected = False
        sprite.dragging = False


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
        if (
            sprite.x - padding <= px <= sprite.x + sprite.w + padding
            and sprite.y - padding <= py <= sprite.y + sprite.h + padding
        ):
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
    return cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)


def enhance_low_light(frame: np.ndarray) -> np.ndarray:
    required_attrs = ("cvtColor", "split", "createCLAHE", "merge", "LUT", "COLOR_BGR2LAB", "COLOR_LAB2BGR")
    if not all(hasattr(cv2, attr) for attr in required_attrs):
        return frame

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    mean_luma = float(np.mean(l_channel))
    if mean_luma >= 115:
        return frame

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    enhanced = cv2.cvtColor(cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2BGR)
    gamma = 1.25 if mean_luma > 80 else 1.45
    gamma_table = np.array([((index / 255.0) ** (1.0 / gamma)) * 255 for index in range(256)], dtype=np.uint8)
    return cv2.LUT(enhanced, gamma_table)


def expand_rect(rect: tuple[int, int, int, int], padding: int) -> tuple[int, int, int, int]:
    left, top, right, bottom = rect
    return left - padding, top - padding, right + padding, bottom + padding


def point_in_any_rect(point: tuple[int, int] | None, rects: list[tuple[int, int, int, int]]) -> bool:
    if point is None:
        return False
    return any(point_in_rect(point, rect) for rect in rects)


def build_warning_text(ai_warning_text: str | None, voice_available: bool, last_error: str | None) -> str | None:
    warning_parts = []
    if ai_warning_text:
        warning_parts.append("AI off")
    if not voice_available and last_error:
        warning_parts.append("Voice off")
    return " | ".join(warning_parts) if warning_parts else None


@dataclass
class UIActionCallbacks:
    set_brush_color: Callable[[str], bool]
    clear_canvas: Callable[[], None]
    save_snapshot: Callable[[str], bool]
    undo_last_stroke: Callable[[], bool]


class UIInteractionController:
    def resolve_hover_candidate(
        self,
        hover_point: tuple[int, int] | None,
        pinch_active: bool,
        toolbar_items: list[ToolbarItem],
        palette_items: list[PaletteItem],
    ) -> str | None:
        if hover_point is None or pinch_active:
            return None

        for item in toolbar_items:
            if point_in_rect(hover_point, expand_rect(item["rect"], 34)):
                return item["id"]

        for item in palette_items:
            if point_in_rect(hover_point, expand_rect(item["rect"], 20)):
                return item["id"]

        return None

    def consume_hover(self, state: RuntimeState, hover_candidate: str | None, now: float) -> str | None:
        if hover_candidate != state.hover_target_id:
            state.hover_target_id = hover_candidate
            state.hover_start_time = now if hover_candidate else None
            return None

        if hover_candidate and state.hover_start_time and now - state.hover_start_time >= DWELL_SECONDS:
            state.clear_hover()
            return hover_candidate
        return None

    def apply_action(self, state: RuntimeState, action_id: str, callbacks: UIActionCallbacks, now: float) -> None:
        if action_id.startswith("palette:"):
            color_name = action_id.split(":", 1)[1]
            if callbacks.set_brush_color(color_name):
                state.brush_name = color_name
                state.set_status(f"{color_name.title()} brush", now, 0.8)
            return

        if action_id == "toolbar:draw_mode":
            state.interaction_mode = "draw_mode"
        elif action_id == "toolbar:select_mode":
            state.interaction_mode = "select_mode"
        elif action_id == "toolbar:clear":
            callbacks.clear_canvas()
            state.set_status("Cleared", now, 0.8)
        elif action_id == "toolbar:save":
            if callbacks.save_snapshot(config.DEFAULT_SAVE_PATH):
                state.set_status("Saved", now, 0.8)
        elif action_id == "toolbar:undo":
            if callbacks.undo_last_stroke():
                state.set_status("Undid", now, 0.8)

        state.reset_interaction_transients()


class SpriteInteractionController:
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
            return

        hovered_sprite = get_topmost_sprite_near_point(sprites, hit_point, padding=22)
        if hovered_sprite is not None and not pinch_active:
            clear_sprite_selection(sprites)
            hovered_sprite.selected = True

        thumbnail_target_sprite = None
        if pinch_started:
            for item in thumbnail_items:
                if point_in_rect(hit_point, expand_rect(item["rect"], 10)):
                    thumbnail_target_sprite = item["sprite"]
                    clear_sprite_selection(sprites)
                    thumbnail_target_sprite.selected = True
                    bring_sprite_to_front(sprites, thumbnail_target_sprite)
                    break

        dragging_sprite = next((sprite for sprite in sprites if sprite.dragging), None)
        selected_sprite = get_selected_sprite(sprites)
        hover_sprite = get_topmost_sprite_near_point(sprites, hit_point, padding=22)
        pinch_target_sprite = thumbnail_target_sprite or dragging_sprite or hover_sprite or selected_sprite

        if pinch_started and pinch_target_sprite is not None:
            if is_double_pinch(state.last_pinch_time, now, DOUBLE_PINCH_SECONDS) and state.last_pinched_sprite_ref == id(
                pinch_target_sprite
            ):
                remove_index = next((index for index, sprite in enumerate(sprites) if sprite is pinch_target_sprite), None)
                if remove_index is not None:
                    sprites.pop(remove_index)
                    clear_sprite_selection(sprites)
                state.last_pinch_time = None
                state.last_pinched_sprite_ref = None
                state.set_status("Deleted sprite", now, 0.8)
                return

            clear_sprite_selection(sprites)
            pinch_target_sprite.selected = True
            bring_sprite_to_front(sprites, pinch_target_sprite)
            state.last_pinch_time = now
            state.last_pinched_sprite_ref = id(pinch_target_sprite)

        if pinch_active and pinch_target_sprite is not None:
            clear_sprite_selection(sprites)
            pinch_target_sprite.selected = True
            pinch_target_sprite.dragging = True
            target_x = current_point[0] - pinch_target_sprite.w // 2
            target_y = current_point[1] - pinch_target_sprite.h // 2
            pinch_target_sprite.x = int(
                DRAG_SMOOTHING_PREV_WEIGHT * pinch_target_sprite.x + DRAG_SMOOTHING_NEW_WEIGHT * target_x
            )
            pinch_target_sprite.y = int(
                DRAG_SMOOTHING_PREV_WEIGHT * pinch_target_sprite.y + DRAG_SMOOTHING_NEW_WEIGHT * target_y
            )
            pinch_target_sprite.clamp_to_frame(frame_width, frame_height)
        else:
            for sprite in sprites:
                sprite.dragging = False


class DrawingInteractionController:
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
        drawing_blocked_by_ui = state.hover_target_id is not None or point_in_any_rect(raw_fingertip or current_point, ui_rects)

        if is_closed_fist(hand_landmarks):
            if state.fist_start_time is None:
                state.fist_start_time = now
            elif now - state.fist_start_time >= FINISH_HOLD_SECONDS:
                drawing_canvas.reset_stroke()
                next_z_index = max((sprite.z_index for sprite in sprites), default=-1) + 1
                sprite = create_sprite_from_canvas(drawing_canvas.canvas, next_z_index)
                if sprite is not None:
                    clear_sprite_selection(sprites)
                    sprites.append(sprite)
                    sprite.selected = True
                    drawing_canvas.clear()
                    state.clear_smoothing()
                    state.clear_drawing_path()
                    state.set_status(STATUS_SPRITE_CREATED, now, 0.8)
                state.clear_fist_hold()
        else:
            state.clear_fist_hold()

        if is_index_only_up(hand_landmarks) and current_point is not None and not pinch_active and not drawing_blocked_by_ui:
            state.status_text = STATUS_DRAWING
            if state.prev_draw_point is not None:
                drawing_canvas.add_segment(state.prev_draw_point, current_point)
            else:
                state.prev_smoothed_point = (float(current_point[0]), float(current_point[1]))
                drawing_canvas.add_segment(current_point, current_point)
            state.prev_draw_point = current_point
        elif is_index_and_middle_up(hand_landmarks):
            drawing_canvas.reset_stroke()
            state.clear_drawing_path()
            state.status_text = STATUS_PAUSED
        elif state.status_text != STATUS_SPRITE_CREATED:
            drawing_canvas.reset_stroke()
            state.clear_drawing_path()
            state.status_text = STATUS_IDLE
