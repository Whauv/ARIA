from __future__ import annotations

import time

import cv2
import mediapipe as mp

from canvas import DrawingCanvas, smooth_point
from config import (
    ALPHA,
    CLEAR_HOLD_SECONDS,
    SMOOTHING_NEW_WEIGHT,
    SMOOTHING_PREV_WEIGHT,
    STATUS_BG_COLOR,
    STATUS_CLEARED,
    STATUS_COLOR,
    STATUS_DRAWING,
    STATUS_IDLE,
    STATUS_PAUSED,
    TARGET_FPS,
)
from gestures import (
    get_index_fingertip,
    is_closed_fist,
    is_index_and_middle_up,
    is_index_only_up,
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


def main() -> None:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    drawing_canvas = None
    prev_smoothed_point = None
    prev_draw_point = None
    fist_start_time = None
    status_text = STATUS_IDLE
    cleared_at = None
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
            hand_landmarks = results.multi_hand_landmarks[0] if results.multi_hand_landmarks else None

            current_point = None
            if hand_landmarks:
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

                if is_closed_fist(hand_landmarks):
                    if fist_start_time is None:
                        fist_start_time = time.time()
                    elif time.time() - fist_start_time >= CLEAR_HOLD_SECONDS:
                        drawing_canvas.clear()
                        prev_smoothed_point = None
                        prev_draw_point = None
                        status_text = STATUS_CLEARED
                        cleared_at = time.time()
                        fist_start_time = None
                else:
                    fist_start_time = None

                if is_index_only_up(hand_landmarks) and current_point is not None:
                    status_text = STATUS_DRAWING
                    if prev_draw_point is not None:
                        start = (int(prev_draw_point[0]), int(prev_draw_point[1]))
                        drawing_canvas.add_segment(start, current_point)
                    prev_draw_point = current_point
                elif is_index_and_middle_up(hand_landmarks):
                    drawing_canvas.reset_stroke()
                    prev_draw_point = None
                    status_text = STATUS_PAUSED
                elif status_text != STATUS_CLEARED:
                    drawing_canvas.reset_stroke()
                    prev_draw_point = None
                    status_text = STATUS_IDLE
            else:
                drawing_canvas.reset_stroke()
                prev_smoothed_point = None
                prev_draw_point = None
                fist_start_time = None
                if status_text != STATUS_CLEARED:
                    status_text = STATUS_IDLE

            output_frame = drawing_canvas.overlay_on(frame, ALPHA)
            draw_status(output_frame, status_text)
            cv2.imshow("ARIA Phase 1", output_frame)

            elapsed = time.time() - loop_start
            remaining = max(0.0, frame_interval - elapsed)
            wait_ms = max(1, int(remaining * 1000))
            if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
                break

            if status_text == STATUS_CLEARED and cleared_at and time.time() - cleared_at >= 0.5:
                status_text = STATUS_IDLE
    finally:
        hands.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
