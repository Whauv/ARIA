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
    LINE_THICKNESS,
    MODE_BG_COLOR,
    PINCH_RELEASE_DISTANCE,
    PINCH_START_DISTANCE,
    SMOOTHING_NEW_WEIGHT,
    SMOOTHING_PREV_WEIGHT,
    STATUS_BG_COLOR,
    STATUS_COLOR,
    STATUS_IDLE,
    STATUS_SPRITE_CREATED,
    TARGET_FPS,
    THUMB_TIP,
)
from gestures import (
    dwell_progress,
    get_fingertip_distance,
    get_index_fingertip,
    get_landmark_point,
    is_pinching,
    is_two_hand_resize,
)
from jarvis import JarvisAssistant, JarvisContext
from runtime_controllers import (
    DrawingInteractionController,
    SpriteInteractionController,
    UIActionCallbacks,
    UIInteractionController,
    build_warning_text,
    clear_sprite_selection,
    enhance_low_light,
    get_selected_sprite,
    point_in_any_rect,
    resize_for_mediapipe,
)
from runtime_state import RuntimeState
from sprite import draw_sprite_selection, overlay_sprite
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


class AppRunner:
    def __init__(self, window_name: str = "ARIA Phase 5") -> None:
        self.window_name = window_name
        self.state = RuntimeState(
            ai_warning_text=None
            if gemini_enabled()
            else "AI vision disabled: set GOOGLE_API_KEY or GEMINI_API_KEY to enable scene descriptions."
        )
        self.frame_interval = 1.0 / TARGET_FPS
        self.ui_controller = UIInteractionController()
        self.sprite_controller = SpriteInteractionController()
        self.drawing_controller = DrawingInteractionController()
        self.sprites = []
        self.sprites_lock = threading.Lock()
        self.canvas_lock = threading.Lock()
        self.frame_lock = threading.Lock()
        self.drawing_canvas: DrawingCanvas | None = None

        self.jarvis = JarvisAssistant(
            JarvisContext(
                get_current_frame=self.get_current_frame,
                clear_canvas=self.clear_canvas_state,
                delete_selected_sprite=self.delete_selected_sprite,
                scale_selected_sprite=self.scale_selected_sprite,
                undo_last_stroke=self.undo_last_stroke,
                save_snapshot=self.save_snapshot,
                set_brush_color=self.set_brush_color,
            )
        )
        self.ui_callbacks = UIActionCallbacks(
            set_brush_color=self.set_brush_color,
            clear_canvas=self.clear_canvas_state,
            save_snapshot=self.save_snapshot,
            undo_last_stroke=self.undo_last_stroke,
        )

    def get_current_frame(self):
        with self.frame_lock:
            if self.state.latest_frame is None:
                return None
            return self.state.latest_frame.copy()

    def clear_canvas_state(self) -> None:
        if self.drawing_canvas is None:
            return
        with self.canvas_lock:
            self.drawing_canvas.clear()

    def delete_selected_sprite(self) -> bool:
        with self.sprites_lock:
            selected = get_selected_sprite(self.sprites)
            if selected is None:
                return False
            self.sprites.remove(selected)
            clear_sprite_selection(self.sprites)
            if self.sprites:
                top_sprite = max(self.sprites, key=lambda item: item.z_index)
                top_sprite.selected = True
            return True

    def scale_selected_sprite(self, scale_factor: float) -> bool:
        with self.sprites_lock:
            selected = get_selected_sprite(self.sprites)
            if selected is None:
                return False
            selected.scale_by_factor(scale_factor)
            selected.clamp_to_frame(self.state.frame_dimensions["width"], self.state.frame_dimensions["height"])
            return True

    def undo_last_stroke(self) -> bool:
        if self.drawing_canvas is None:
            return False
        with self.canvas_lock:
            return self.drawing_canvas.undo_last_stroke()

    def save_snapshot(self, path: str) -> bool:
        frame = self.get_current_frame()
        if frame is None:
            return False
        return cv2.imwrite(path, frame)

    def set_brush_color(self, color_name: str) -> bool:
        color = config.set_active_brush_color(color_name)
        if color is None or self.drawing_canvas is None:
            return False
        self.state.brush_name = color_name
        with self.canvas_lock:
            self.drawing_canvas.set_brush_color(color)
        return True

    def _show_webcam_error(self) -> None:
        error_frame = np.zeros((220, 720, 3), dtype=np.uint8)
        cv2.putText(error_frame, "ARIA could not find a webcam.", (55, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.putText(
            error_frame,
            "Check your camera connection and try again.",
            (35, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.imshow(self.window_name, error_frame)
        cv2.waitKey(2500)
        cv2.destroyAllWindows()

    def _render_output(self, frame, palette_items, toolbar_items, thumbnail_items, current_point, is_resize_mode) -> np.ndarray:
        with self.canvas_lock:
            output_frame = self.drawing_canvas.overlay_on(frame, ALPHA)

        dwell_ratio = (
            dwell_progress(self.state.hover_start_time, config.DWELL_SECONDS, time.time()) if self.state.hover_target_id else 0.0
        )
        draw_palette(output_frame, palette_items, self.state.brush_name, self.state.hover_target_id, dwell_ratio)
        draw_toolbar(output_frame, toolbar_items, self.state.interaction_mode, self.state.hover_target_id, dwell_ratio)
        draw_thumbnail_strip(output_frame, thumbnail_items, self.state.hover_target_id)
        draw_status(output_frame, self.state.status_text)
        draw_mode_indicator(
            output_frame,
            "RESIZING" if is_resize_mode else ("DRAWING" if self.state.interaction_mode == "draw_mode" else "SELECTING"),
        )
        draw_voice_indicator(output_frame, self.jarvis.listening_event.is_set(), self.jarvis.speaking_event.is_set())
        draw_warning_overlay(
            output_frame,
            build_warning_text(self.state.ai_warning_text, self.jarvis.voice_available, self.jarvis.last_error),
        )
        if self.state.interaction_mode == "draw_mode":
            draw_brush_preview(output_frame, current_point, LINE_THICKNESS, config.get_active_brush_color())
        draw_fps(output_frame, self.state.fps)
        return output_frame

    def run(self) -> None:
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1400, 900)
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_AUTOSIZE, cv2.WINDOW_NORMAL)

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
        if not cap.isOpened():
            self._show_webcam_error()
            return

        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )

        self.jarvis.start()

        try:
            while True:
                loop_start = time.time()
                success, frame = cap.read()
                if not success:
                    break

                frame = cv2.flip(frame, 1)
                frame = enhance_low_light(frame)
                if self.drawing_canvas is None:
                    self.drawing_canvas = DrawingCanvas(frame.shape)

                frame_height, frame_width = frame.shape[:2]
                self.state.frame_dimensions["width"] = frame_width
                self.state.frame_dimensions["height"] = frame_height
                current_point = None
                raw_fingertip = None
                try:
                    mediapipe_input = resize_for_mediapipe(frame)
                    rgb_frame = cv2.cvtColor(mediapipe_input, cv2.COLOR_BGR2RGB)
                    results = hands.process(rgb_frame)
                    all_hands = results.multi_hand_landmarks if results.multi_hand_landmarks else []
                except Exception:
                    all_hands = []
                    self.state.set_status("Hand tracking error", time.time(), 1.0)

                hand_landmarks = all_hands[0] if all_hands else None
                palette_items = get_palette_items(frame_width)
                toolbar_items = get_toolbar_items(frame_width, frame_height)

                with self.sprites_lock:
                    selected_sprite = get_selected_sprite(self.sprites)
                    sprite_snapshot = list(sorted(self.sprites, key=lambda item: item.z_index))
                thumbnail_items = get_thumbnail_items(frame, sprite_snapshot)
                palette_rects = [item["rect"] for item in palette_items]
                toolbar_rects = [item["rect"] for item in toolbar_items]
                thumbnail_rects = [item["rect"] for item in thumbnail_items]

                is_resize_mode = self.state.interaction_mode == "select_mode" and is_two_hand_resize(all_hands)
                if is_resize_mode:
                    with self.canvas_lock:
                        self.drawing_canvas.reset_stroke()
                    self.state.clear_drawing_path()
                    self.state.clear_smoothing()
                    first_point = get_index_fingertip(all_hands[0], frame_width, frame_height)
                    second_point = get_index_fingertip(all_hands[1], frame_width, frame_height)

                    if selected_sprite and first_point and second_point:
                        current_distance = get_fingertip_distance(first_point, second_point)
                        if self.state.prev_resize_distance and self.state.prev_resize_distance > 0:
                            with self.sprites_lock:
                                selected_sprite.resize_from_original(current_distance / self.state.prev_resize_distance)
                                selected_sprite.clamp_to_frame(frame_width, frame_height)
                        self.state.prev_resize_distance = current_distance
                    else:
                        self.state.clear_resize()
                    self.state.reset_interaction_transients()
                elif hand_landmarks:
                    self.state.clear_resize()
                    raw_fingertip = get_index_fingertip(hand_landmarks, frame_width, frame_height)
                    if raw_fingertip is not None:
                        smoothed = smooth_point(
                            self.state.prev_smoothed_point,
                            raw_fingertip,
                            SMOOTHING_PREV_WEIGHT,
                            SMOOTHING_NEW_WEIGHT,
                        )
                        current_point = (int(smoothed[0]), int(smoothed[1]))
                        self.state.prev_smoothed_point = smoothed

                    is_currently_pinching = is_pinching(hand_landmarks, frame_width, frame_height, PINCH_START_DISTANCE)
                    pinch_distance = None
                    if raw_fingertip is not None:
                        thumb_tip = get_landmark_point(hand_landmarks, THUMB_TIP, frame_width, frame_height)
                        pinch_distance = get_fingertip_distance(thumb_tip, raw_fingertip)

                    with self.sprites_lock:
                        dragging_sprite = next((sprite for sprite in self.sprites if sprite.dragging), None)
                    if dragging_sprite and pinch_distance is not None and pinch_distance > PINCH_RELEASE_DISTANCE:
                        with self.sprites_lock:
                            dragging_sprite.dragging = False

                    pinch_active = is_currently_pinching or (
                        dragging_sprite is not None and pinch_distance is not None and pinch_distance <= PINCH_RELEASE_DISTANCE
                    )
                    pinch_started = pinch_active and not self.state.previous_pinch_active

                    hover_point = raw_fingertip or current_point
                    hover_candidate = self.ui_controller.resolve_hover_candidate(
                        hover_point, pinch_active, toolbar_items, palette_items
                    )
                    pointer_over_controls = point_in_any_rect(hover_point, palette_rects + toolbar_rects)

                    now = time.time()
                    triggered_action = self.ui_controller.consume_hover(self.state, hover_candidate, now)
                    if triggered_action:
                        self.ui_controller.apply_action(self.state, triggered_action, self.ui_callbacks, now)

                    if self.state.interaction_mode == "select_mode" and current_point is not None and not pointer_over_controls:
                        hit_point = raw_fingertip or current_point
                        with self.sprites_lock:
                            self.sprite_controller.handle_select_mode(
                                state=self.state,
                                sprites=self.sprites,
                                thumbnail_items=thumbnail_items,
                                hit_point=hit_point,
                                current_point=current_point,
                                pinch_active=pinch_active,
                                pinch_started=pinch_started,
                                frame_width=frame_width,
                                frame_height=frame_height,
                                now=now,
                            )
                        with self.canvas_lock:
                            self.drawing_canvas.reset_stroke()
                        self.state.clear_drawing_path()

                    if self.state.interaction_mode == "draw_mode":
                        with self.sprites_lock:
                            self.drawing_controller.handle_draw_mode(
                                state=self.state,
                                hand_landmarks=hand_landmarks,
                                current_point=current_point,
                                raw_fingertip=raw_fingertip,
                                pinch_active=pinch_active,
                                drawing_canvas=self.drawing_canvas,
                                sprites=self.sprites,
                                now=now,
                                ui_rects=palette_rects + toolbar_rects + thumbnail_rects,
                            )
                    else:
                        with self.canvas_lock:
                            self.drawing_canvas.reset_stroke()
                        self.state.clear_drawing_path()
                        self.state.clear_fist_hold()

                    self.state.previous_pinch_active = pinch_active
                else:
                    self.state.clear_resize()
                    self.state.previous_pinch_active = False
                    self.state.clear_hover()
                    with self.canvas_lock:
                        self.drawing_canvas.reset_stroke()
                    self.state.clear_smoothing()
                    self.state.clear_drawing_path()
                    self.state.clear_fist_hold()
                    with self.sprites_lock:
                        for sprite in self.sprites:
                            sprite.dragging = False
                    if self.state.status_text != STATUS_SPRITE_CREATED:
                        self.state.status_text = STATUS_IDLE

                for sprite in sprite_snapshot:
                    overlay_sprite(frame, sprite)
                for sprite in sprite_snapshot:
                    draw_sprite_selection(frame, sprite)

                elapsed = time.time() - loop_start
                if elapsed > 0:
                    self.state.fps = 1.0 / elapsed
                output_frame = self._render_output(frame, palette_items, toolbar_items, thumbnail_items, current_point, is_resize_mode)

                with self.frame_lock:
                    self.state.latest_frame = output_frame.copy()
                cv2.imshow(self.window_name, output_frame)

                remaining = max(0.0, self.frame_interval - (time.time() - loop_start))
                wait_ms = max(1, int(remaining * 1000))
                if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
                    break

                self.state.expire_status(time.time())
        finally:
            self.jarvis.stop()
            hands.close()
            cap.release()
            cv2.destroyAllWindows()


def main() -> None:
    AppRunner().run()
