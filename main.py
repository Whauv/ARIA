from __future__ import annotations

import math
import threading
import time

import cv2
import mediapipe as mp
import numpy as np

import config
from ai_utils import gemini_enabled
from canvas import DrawingCanvas, smooth_point
from config import (
    ALPHA,
    DOUBLE_PINCH_SECONDS,
    DRAG_SMOOTHING_NEW_WEIGHT,
    DRAG_SMOOTHING_PREV_WEIGHT,
    DWELL_SECONDS,
    FINISH_HOLD_SECONDS,
    LINE_THICKNESS,
    MAX_MEDIAPIPE_HEIGHT,
    MAX_MEDIAPIPE_WIDTH,
    MODE_BG_COLOR,
    PINCH_RELEASE_DISTANCE,
    PINCH_START_DISTANCE,
    SMOOTHING_NEW_WEIGHT,
    SMOOTHING_PREV_WEIGHT,
    STATUS_BG_COLOR,
    STATUS_COLOR,
    STATUS_DRAWING,
    STATUS_IDLE,
    STATUS_PAUSED,
    STATUS_SPRITE_CREATED,
    TARGET_FPS,
    THUMB_TIP,
)
from gestures import (
    dwell_progress,
    get_fingertip_distance,
    get_index_fingertip,
    get_landmark_point,
    is_closed_fist,
    is_double_pinch,
    is_index_and_middle_up,
    is_index_only_up,
    is_pinching,
    is_two_hand_resize,
    point_in_rect,
)
from jarvis import JarvisAssistant, JarvisContext
from sprite import create_sprite_from_canvas, draw_sprite_selection, overlay_sprite
from ui import (
    draw_brush_preview,
    draw_fps,
    draw_palette,
    draw_thumbnail_strip,
    draw_toolbar,
    draw_warning_overlay,
    get_palette_items,
    get_thumbnail_items,
    get_toolbar_items,
)


def draw_status(frame, status: str) -> None:
    cv2.rectangle(frame, (10, 10), (180, 55), STATUS_BG_COLOR, -1)
    cv2.putText(
        frame,
        status,
        (20, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        STATUS_COLOR,
        2,
        cv2.LINE_AA,
    )


def draw_voice_indicator(frame, listening: bool, speaking: bool) -> None:
    frame_width = frame.shape[1]
    center = (frame_width - 35, 82)

    if listening:
        pulse_radius = 12 + int(4 * (1 + math.sin(time.time() * 6)))
        cv2.circle(frame, center, pulse_radius, config.VOICE_PULSE_COLOR, 2)
        cv2.circle(frame, center, 9, config.VOICE_LISTENING_COLOR, -1)
        cv2.putText(
            frame,
            "ARIA Listening...",
            (frame_width - 245, 88),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            config.VOICE_LISTENING_COLOR,
            2,
            cv2.LINE_AA,
        )
        return

    if speaking:
        cv2.circle(frame, center, 9, config.VOICE_SPEAKING_COLOR, -1)
        cv2.putText(
            frame,
            "ARIA Speaking...",
            (frame_width - 235, 88),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            config.VOICE_SPEAKING_COLOR,
            2,
            cv2.LINE_AA,
        )
        return

    cv2.circle(frame, center, 9, config.VOICE_IDLE_COLOR, 1)
    cv2.line(frame, (center[0], center[1] - 5), (center[0], center[1] + 4), config.VOICE_IDLE_COLOR, 2)
    cv2.ellipse(frame, (center[0], center[1] - 1), (4, 6), 0, 0, 360, config.VOICE_IDLE_COLOR, 2)
    cv2.line(frame, (center[0], center[1] + 8), (center[0], center[1] + 12), config.VOICE_IDLE_COLOR, 2)
    cv2.line(frame, (center[0] - 4, center[1] + 12), (center[0] + 4, center[1] + 12), config.VOICE_IDLE_COLOR, 2)


def draw_mode_indicator(frame, mode_text: str) -> None:
    text_width = max(150, 18 + len(mode_text) * 14)
    x2 = frame.shape[1] - 10
    x1 = max(10, x2 - text_width)
    cv2.rectangle(frame, (x1, 10), (x2, 55), MODE_BG_COLOR, -1)
    cv2.putText(
        frame,
        mode_text,
        (x1 + 12, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        STATUS_COLOR,
        2,
        cv2.LINE_AA,
    )


def clear_sprite_selection(sprites) -> None:
    for sprite in sprites:
        sprite.selected = False
        sprite.dragging = False


def get_selected_sprite(sprites):
    for sprite in sorted(sprites, key=lambda item: item.z_index, reverse=True):
        if sprite.selected:
            return sprite
    return None


def get_topmost_sprite_at_point(sprites, point):
    for sprite in sorted(sprites, key=lambda item: item.z_index, reverse=True):
        if sprite.contains_point(point):
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


def main() -> None:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    if not cap.isOpened():
        error_frame = np.zeros((220, 720, 3), dtype=np.uint8)
        cv2.putText(error_frame, "ARIA could not find a webcam.", (55, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.putText(error_frame, "Check your camera connection and try again.", (35, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("ARIA Error", error_frame)
        cv2.waitKey(2500)
        cv2.destroyAllWindows()
        return

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    drawing_canvas = None
    sprites = []
    sprites_lock = threading.Lock()
    canvas_lock = threading.Lock()
    frame_lock = threading.Lock()
    latest_frame = None
    frame_dimensions = {"width": 0, "height": 0}
    prev_smoothed_point = None
    prev_draw_point = None
    prev_resize_distance = None
    fist_start_time = None
    hover_target_id = None
    hover_start_time = None
    last_pinch_time = None
    last_pinched_sprite_ref = None
    previous_pinch_active = False
    status_text = STATUS_IDLE
    status_expires_at = None
    interaction_mode = "draw_mode"
    brush_state = {"name": "green"}
    frame_interval = 1.0 / TARGET_FPS
    fps = 0.0
    ai_warning_text = None if gemini_enabled() else "AI vision disabled: set GOOGLE_API_KEY or GEMINI_API_KEY to enable scene descriptions."

    def get_current_frame():
        with frame_lock:
            if latest_frame is None:
                return None
            return latest_frame.copy()

    def clear_canvas_state() -> None:
        if drawing_canvas is None:
            return
        with canvas_lock:
            drawing_canvas.clear()

    def delete_selected_sprite() -> bool:
        with sprites_lock:
            selected = get_selected_sprite(sprites)
            if selected is None:
                return False
            sprites.remove(selected)
            clear_sprite_selection(sprites)
            if sprites:
                top_sprite = max(sprites, key=lambda item: item.z_index)
                top_sprite.selected = True
            return True

    def scale_selected_sprite(scale_factor: float) -> bool:
        with sprites_lock:
            selected = get_selected_sprite(sprites)
            if selected is None:
                return False
            selected.scale_by_factor(scale_factor)
            selected.clamp_to_frame(frame_dimensions["width"], frame_dimensions["height"])
            return True

    def undo_last_stroke() -> bool:
        if drawing_canvas is None:
            return False
        with canvas_lock:
            return drawing_canvas.undo_last_stroke()

    def save_snapshot(path: str) -> bool:
        frame = get_current_frame()
        if frame is None:
            return False
        return cv2.imwrite(path, frame)

    def set_brush_color(color_name: str) -> bool:
        color = config.set_active_brush_color(color_name)
        if color is None or drawing_canvas is None:
            return False
        brush_state["name"] = color_name
        with canvas_lock:
            drawing_canvas.set_brush_color(color)
        return True

    jarvis = JarvisAssistant(
        JarvisContext(
            get_current_frame=get_current_frame,
            clear_canvas=clear_canvas_state,
            delete_selected_sprite=delete_selected_sprite,
            scale_selected_sprite=scale_selected_sprite,
            undo_last_stroke=undo_last_stroke,
            save_snapshot=save_snapshot,
            set_brush_color=set_brush_color,
        )
    )
    jarvis.start()

    try:
        while True:
            loop_start = time.time()
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            if drawing_canvas is None:
                drawing_canvas = DrawingCanvas(frame.shape)

            frame_height, frame_width = frame.shape[:2]
            frame_dimensions["width"] = frame_width
            frame_dimensions["height"] = frame_height
            current_point = None
            raw_fingertip = None
            try:
                mediapipe_input = resize_for_mediapipe(frame)
                rgb_frame = cv2.cvtColor(mediapipe_input, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)
                all_hands = results.multi_hand_landmarks if results.multi_hand_landmarks else []
            except Exception:
                all_hands = []
                status_text = "Hand tracking error"
                status_expires_at = time.time() + 1.0

            hand_landmarks = all_hands[0] if all_hands else None

            with sprites_lock:
                selected_sprite = get_selected_sprite(sprites)

            is_resize_mode = interaction_mode == "select_mode" and is_two_hand_resize(all_hands)
            if is_resize_mode:
                with canvas_lock:
                    drawing_canvas.reset_stroke()
                prev_draw_point = None
                prev_smoothed_point = None
                first_point = get_index_fingertip(all_hands[0], frame_width, frame_height)
                second_point = get_index_fingertip(all_hands[1], frame_width, frame_height)

                if selected_sprite and first_point and second_point:
                    current_distance = get_fingertip_distance(first_point, second_point)
                    if prev_resize_distance and prev_resize_distance > 0:
                        with sprites_lock:
                            selected_sprite.resize_from_original(current_distance / prev_resize_distance)
                            selected_sprite.clamp_to_frame(frame_width, frame_height)
                    prev_resize_distance = current_distance
                else:
                    prev_resize_distance = None
                previous_pinch_active = False
                fist_start_time = None
                hover_target_id = None
                hover_start_time = None
            elif hand_landmarks:
                prev_resize_distance = None
                raw_fingertip = get_index_fingertip(hand_landmarks, frame_width, frame_height)
                if raw_fingertip is not None:
                    smoothed = smooth_point(
                        prev_smoothed_point,
                        raw_fingertip,
                        SMOOTHING_PREV_WEIGHT,
                        SMOOTHING_NEW_WEIGHT,
                    )
                    current_point = (int(smoothed[0]), int(smoothed[1]))
                    prev_smoothed_point = smoothed

                is_currently_pinching = is_pinching(
                    hand_landmarks,
                    frame_width,
                    frame_height,
                    PINCH_START_DISTANCE,
                )
                pinch_distance = None
                if raw_fingertip is not None:
                    thumb_tip = get_landmark_point(hand_landmarks, THUMB_TIP, frame_width, frame_height)
                    pinch_distance = get_fingertip_distance(thumb_tip, raw_fingertip)

                with sprites_lock:
                    dragging_sprite = next((sprite for sprite in sprites if sprite.dragging), None)
                if dragging_sprite and pinch_distance is not None and pinch_distance > PINCH_RELEASE_DISTANCE:
                    with sprites_lock:
                        dragging_sprite.dragging = False

                pinch_active = is_currently_pinching or (
                    dragging_sprite is not None
                    and pinch_distance is not None
                    and pinch_distance <= PINCH_RELEASE_DISTANCE
                )
                pinch_started = pinch_active and not previous_pinch_active

                palette_items = get_palette_items(frame_width)
                toolbar_items = get_toolbar_items(frame_width, frame_height)
                with sprites_lock:
                    sprite_snapshot = list(sorted(sprites, key=lambda item: item.z_index))
                thumbnail_items = get_thumbnail_items(frame, sprite_snapshot)

                hover_candidate = None
                if current_point is not None and not pinch_active:
                    for item in palette_items + toolbar_items:
                        if point_in_rect(current_point, item["rect"]):
                            hover_candidate = item["id"]
                            break

                now = time.time()
                if hover_candidate != hover_target_id:
                    hover_target_id = hover_candidate
                    hover_start_time = now if hover_candidate else None
                elif hover_candidate and hover_start_time and now - hover_start_time >= DWELL_SECONDS:
                    if hover_candidate.startswith("palette:"):
                        color_name = hover_candidate.split(":", 1)[1]
                        if set_brush_color(color_name):
                            status_text = f"{color_name.title()} brush"
                            status_expires_at = now + 0.8
                    elif hover_candidate == "toolbar:draw_mode":
                        interaction_mode = "draw_mode"
                    elif hover_candidate == "toolbar:select_mode":
                        interaction_mode = "select_mode"
                    elif hover_candidate == "toolbar:clear":
                        clear_canvas_state()
                        status_text = "Cleared"
                        status_expires_at = now + 0.8
                    elif hover_candidate == "toolbar:save":
                        if save_snapshot("aria_snapshot.png"):
                            status_text = "Saved"
                            status_expires_at = now + 0.8
                    elif hover_candidate == "toolbar:undo":
                        if undo_last_stroke():
                            status_text = "Undid"
                            status_expires_at = now + 0.8
                    hover_target_id = None
                    hover_start_time = None

                pinch_target_sprite = None
                if interaction_mode == "select_mode" and current_point is not None:
                    if pinch_started:
                        for item in thumbnail_items:
                            if point_in_rect(current_point, item["rect"]):
                                pinch_target_sprite = item["sprite"]
                                with sprites_lock:
                                    clear_sprite_selection(sprites)
                                    pinch_target_sprite.selected = True
                                    bring_sprite_to_front(sprites, pinch_target_sprite)
                                break

                    with sprites_lock:
                        scene_sprite = dragging_sprite or get_topmost_sprite_at_point(sprites, current_point)

                    pinch_target_sprite = pinch_target_sprite or scene_sprite
                    if pinch_started and pinch_target_sprite is not None:
                        if is_double_pinch(last_pinch_time, now, DOUBLE_PINCH_SECONDS) and last_pinched_sprite_ref == id(pinch_target_sprite):
                            with sprites_lock:
                                if pinch_target_sprite in sprites:
                                    sprites.remove(pinch_target_sprite)
                                    clear_sprite_selection(sprites)
                            pinch_target_sprite = None
                            last_pinch_time = None
                            last_pinched_sprite_ref = None
                            status_text = "Deleted sprite"
                            status_expires_at = now + 0.8
                        else:
                            last_pinch_time = now
                            last_pinched_sprite_ref = id(pinch_target_sprite)

                    if pinch_active and current_point is not None and pinch_target_sprite is not None:
                        with sprites_lock:
                            clear_sprite_selection(sprites)
                            pinch_target_sprite.selected = True
                            pinch_target_sprite.dragging = True
                            target_x = current_point[0] - pinch_target_sprite.w // 2
                            target_y = current_point[1] - pinch_target_sprite.h // 2
                            pinch_target_sprite.x = int(
                                DRAG_SMOOTHING_PREV_WEIGHT * pinch_target_sprite.x
                                + DRAG_SMOOTHING_NEW_WEIGHT * target_x
                            )
                            pinch_target_sprite.y = int(
                                DRAG_SMOOTHING_PREV_WEIGHT * pinch_target_sprite.y
                                + DRAG_SMOOTHING_NEW_WEIGHT * target_y
                            )
                            pinch_target_sprite.clamp_to_frame(frame_width, frame_height)
                        with canvas_lock:
                            drawing_canvas.reset_stroke()
                        prev_draw_point = None
                    elif not pinch_active:
                        with sprites_lock:
                            for sprite in sprites:
                                sprite.dragging = False

                if interaction_mode == "draw_mode":
                    if is_closed_fist(hand_landmarks):
                        if fist_start_time is None:
                            fist_start_time = now
                        elif now - fist_start_time >= FINISH_HOLD_SECONDS:
                            with canvas_lock:
                                drawing_canvas.reset_stroke()
                                with sprites_lock:
                                    next_z_index = max((sprite.z_index for sprite in sprites), default=-1) + 1
                                sprite = create_sprite_from_canvas(drawing_canvas.canvas, next_z_index)
                            if sprite is not None:
                                with sprites_lock:
                                    clear_sprite_selection(sprites)
                                    sprites.append(sprite)
                                    sprite.selected = True
                                with canvas_lock:
                                    drawing_canvas.clear()
                                prev_smoothed_point = None
                                prev_draw_point = None
                                status_text = STATUS_SPRITE_CREATED
                                status_expires_at = now + 0.8
                            fist_start_time = None
                    else:
                        fist_start_time = None

                    if is_index_only_up(hand_landmarks) and current_point is not None and not pinch_active:
                        status_text = STATUS_DRAWING
                        if prev_draw_point is not None:
                            with canvas_lock:
                                drawing_canvas.add_segment(prev_draw_point, current_point)
                        prev_draw_point = current_point
                    elif is_index_and_middle_up(hand_landmarks):
                        with canvas_lock:
                            drawing_canvas.reset_stroke()
                        prev_draw_point = None
                        status_text = STATUS_PAUSED
                    elif status_text != STATUS_SPRITE_CREATED:
                        with canvas_lock:
                            drawing_canvas.reset_stroke()
                        prev_draw_point = None
                        status_text = STATUS_IDLE
                else:
                    with canvas_lock:
                        drawing_canvas.reset_stroke()
                    prev_draw_point = None
                    fist_start_time = None

                previous_pinch_active = pinch_active
            else:
                prev_resize_distance = None
                previous_pinch_active = False
                hover_target_id = None
                hover_start_time = None
                with canvas_lock:
                    drawing_canvas.reset_stroke()
                prev_smoothed_point = None
                prev_draw_point = None
                fist_start_time = None
                with sprites_lock:
                    for sprite in sprites:
                        sprite.dragging = False
                if status_text != STATUS_SPRITE_CREATED:
                    status_text = STATUS_IDLE

            with sprites_lock:
                sprite_snapshot = list(sorted(sprites, key=lambda item: item.z_index))
            palette_items = get_palette_items(frame_width)
            toolbar_items = get_toolbar_items(frame_width, frame_height)
            thumbnail_items = get_thumbnail_items(frame, sprite_snapshot)

            for sprite in sprite_snapshot:
                overlay_sprite(frame, sprite)
            for sprite in sprite_snapshot:
                draw_sprite_selection(frame, sprite)

            with canvas_lock:
                output_frame = drawing_canvas.overlay_on(frame, ALPHA)

            dwell_ratio = dwell_progress(hover_start_time, DWELL_SECONDS, time.time()) if hover_target_id else 0.0
            draw_palette(output_frame, palette_items, brush_state["name"], hover_target_id, dwell_ratio)
            draw_toolbar(output_frame, toolbar_items, interaction_mode, hover_target_id, dwell_ratio)
            draw_thumbnail_strip(output_frame, thumbnail_items, hover_target_id)
            draw_status(output_frame, status_text)
            draw_mode_indicator(output_frame, "RESIZING" if is_resize_mode else ("DRAWING" if interaction_mode == "draw_mode" else "SELECTING"))
            draw_voice_indicator(output_frame, jarvis.listening_event.is_set(), jarvis.speaking_event.is_set())
            draw_warning_overlay(output_frame, ai_warning_text)
            if interaction_mode == "draw_mode":
                draw_brush_preview(output_frame, current_point, LINE_THICKNESS, config.get_active_brush_color())

            elapsed = time.time() - loop_start
            if elapsed > 0:
                fps = 1.0 / elapsed
            draw_fps(output_frame, fps)
            with frame_lock:
                latest_frame = output_frame.copy()
            cv2.imshow("ARIA Phase 5", output_frame)

            remaining = max(0.0, frame_interval - (time.time() - loop_start))
            wait_ms = max(1, int(remaining * 1000))
            if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
                break

            if status_expires_at and time.time() >= status_expires_at:
                status_text = STATUS_IDLE
                status_expires_at = None
    finally:
        jarvis.stop()
        hands.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
