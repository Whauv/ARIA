from __future__ import annotations

import time

import cv2
import mediapipe as mp

from canvas import DrawingCanvas, smooth_point
from config import (
    ALPHA,
    DRAG_SMOOTHING_NEW_WEIGHT,
    DRAG_SMOOTHING_PREV_WEIGHT,
    FINISH_HOLD_SECONDS,
    THUMB_TIP,
    PINCH_RELEASE_DISTANCE,
    PINCH_START_DISTANCE,
    SMOOTHING_NEW_WEIGHT,
    SMOOTHING_PREV_WEIGHT,
    MODE_BG_COLOR,
    STATUS_BG_COLOR,
    STATUS_COLOR,
    STATUS_DRAWING,
    STATUS_IDLE,
    STATUS_PAUSED,
    STATUS_SPRITE_CREATED,
    TARGET_FPS,
)
from gestures import (
    get_fingertip_distance,
    get_index_fingertip,
    get_landmark_point,
    is_closed_fist,
    is_index_and_middle_up,
    is_index_only_up,
    is_pinching,
    is_two_hand_resize,
)
from sprite import create_sprite_from_canvas, draw_sprite_selection, overlay_sprite


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


def draw_mode_indicator(frame, mode_text: str | None) -> None:
    if not mode_text:
        return

    frame_width = frame.shape[1]
    x2 = frame_width - 10
    x1 = max(10, x2 - 130)
    y1, y2 = 10, 55
    cv2.rectangle(frame, (x1, y1), (x2, y2), MODE_BG_COLOR, -1)
    cv2.putText(
        frame,
        mode_text,
        (x1 + 15, y1 + 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
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


def main() -> None:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    drawing_canvas = None
    sprites = []
    prev_smoothed_point = None
    prev_draw_point = None
    fist_start_time = None
    prev_resize_distance = None
    status_text = STATUS_IDLE
    status_expires_at = None
    active_mode = None
    frame_interval = 1.0 / TARGET_FPS

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
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            all_hands = results.multi_hand_landmarks if results.multi_hand_landmarks else []
            hand_landmarks = all_hands[0] if all_hands else None

            current_point = None
            active_mode = None
            selected_sprite = get_selected_sprite(sprites)

            if is_two_hand_resize(all_hands):
                drawing_canvas.reset_stroke()
                prev_draw_point = None
                prev_smoothed_point = None

                first_point = get_index_fingertip(all_hands[0], frame_width, frame_height)
                second_point = get_index_fingertip(all_hands[1], frame_width, frame_height)

                if selected_sprite and first_point and second_point:
                    current_distance = get_fingertip_distance(first_point, second_point)
                    if prev_resize_distance and prev_resize_distance > 0:
                        scale_factor = current_distance / prev_resize_distance
                        selected_sprite.resize_from_original(scale_factor)
                        selected_sprite.clamp_to_frame(frame_width, frame_height)
                    prev_resize_distance = current_distance
                    status_text = STATUS_IDLE
                    active_mode = "RESIZE"
                else:
                    prev_resize_distance = None
            elif hand_landmarks:
                prev_resize_distance = None
                fingertip = get_index_fingertip(hand_landmarks, frame_width, frame_height)
                if fingertip is not None:
                    smoothed = smooth_point(
                        prev_smoothed_point,
                        fingertip,
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
                if fingertip is not None:
                    thumb_tip = get_landmark_point(hand_landmarks, THUMB_TIP, frame_width, frame_height)
                    pinch_distance = get_fingertip_distance(thumb_tip, fingertip)

                dragging_sprite = next((sprite for sprite in sprites if sprite.dragging), None)
                if dragging_sprite and pinch_distance is not None and pinch_distance > PINCH_RELEASE_DISTANCE:
                    dragging_sprite.dragging = False
                    active_mode = None

                pinch_active = is_currently_pinching or (
                    dragging_sprite is not None
                    and pinch_distance is not None
                    and pinch_distance <= PINCH_RELEASE_DISTANCE
                )

                if pinch_active and fingertip is not None:
                    target_sprite = dragging_sprite or get_topmost_sprite_at_point(sprites, fingertip)
                    if target_sprite is not None:
                        clear_sprite_selection(sprites)
                        target_sprite.selected = True
                        target_sprite.dragging = True
                        target_x = fingertip[0] - target_sprite.w // 2
                        target_y = fingertip[1] - target_sprite.h // 2
                        target_sprite.x = int(
                            DRAG_SMOOTHING_PREV_WEIGHT * target_sprite.x
                            + DRAG_SMOOTHING_NEW_WEIGHT * target_x
                        )
                        target_sprite.y = int(
                            DRAG_SMOOTHING_PREV_WEIGHT * target_sprite.y
                            + DRAG_SMOOTHING_NEW_WEIGHT * target_y
                        )
                        target_sprite.clamp_to_frame(frame_width, frame_height)
                        status_text = STATUS_IDLE
                        active_mode = "DRAG"
                        fist_start_time = None
                        drawing_canvas.reset_stroke()
                        prev_draw_point = None
                    elif status_text != STATUS_SPRITE_CREATED:
                        clear_sprite_selection(sprites)
                elif is_closed_fist(hand_landmarks):
                    if fist_start_time is None:
                        fist_start_time = time.time()
                    elif time.time() - fist_start_time >= FINISH_HOLD_SECONDS:
                        drawing_canvas.reset_stroke()
                        sprite = create_sprite_from_canvas(drawing_canvas.canvas, len(sprites))
                        if sprite is not None:
                            clear_sprite_selection(sprites)
                            sprites.append(sprite)
                            sprite.selected = True
                            drawing_canvas.clear()
                            prev_smoothed_point = None
                            prev_draw_point = None
                            status_text = STATUS_SPRITE_CREATED
                            status_expires_at = time.time() + 0.6
                        fist_start_time = None
                else:
                    fist_start_time = None

                if active_mode == "DRAG":
                    pass
                elif is_index_only_up(hand_landmarks) and current_point is not None:
                    status_text = STATUS_DRAWING
                    if prev_draw_point is not None:
                        start = (int(prev_draw_point[0]), int(prev_draw_point[1]))
                        drawing_canvas.add_segment(start, current_point)
                    prev_draw_point = current_point
                elif is_index_and_middle_up(hand_landmarks):
                    drawing_canvas.reset_stroke()
                    prev_draw_point = None
                    status_text = STATUS_PAUSED
                elif status_text != STATUS_SPRITE_CREATED:
                    drawing_canvas.reset_stroke()
                    prev_draw_point = None
                    status_text = STATUS_IDLE
            else:
                drawing_canvas.reset_stroke()
                prev_smoothed_point = None
                prev_draw_point = None
                fist_start_time = None
                prev_resize_distance = None
                for sprite in sprites:
                    sprite.dragging = False
                if status_text != STATUS_SPRITE_CREATED:
                    status_text = STATUS_IDLE

            for sprite in sorted(sprites, key=lambda item: item.z_index):
                overlay_sprite(frame, sprite)
            for sprite in sorted(sprites, key=lambda item: item.z_index):
                draw_sprite_selection(frame, sprite)

            output_frame = drawing_canvas.overlay_on(frame, ALPHA)
            draw_status(output_frame, status_text)
            draw_mode_indicator(output_frame, active_mode)
            cv2.imshow("ARIA Phase 3", output_frame)

            elapsed = time.time() - loop_start
            remaining = max(0.0, frame_interval - elapsed)
            wait_ms = max(1, int(remaining * 1000))
            if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
                break

            if status_text == STATUS_SPRITE_CREATED and status_expires_at and time.time() >= status_expires_at:
                status_text = STATUS_IDLE
    finally:
        hands.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
