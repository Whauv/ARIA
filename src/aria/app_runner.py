from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from . import config
from .ai.ai_utils import gemini_enabled
from .assistant.jarvis import JarvisAssistant, JarvisContext
from .config import (
    ALPHA,
    BASE_HAND_SPREAD,
    CALIBRATION_SECONDS,
    LINE_THICKNESS,
    MODE_BG_COLOR,
    PINCH_RELEASE_DISTANCE,
    PINCH_START_DISTANCE,
    STATUS_BG_COLOR,
    STATUS_COLOR,
    STATUS_IDLE,
    STATUS_SPRITE_CREATED,
    TARGET_FPS,
    THUMB_TIP,
    TRACKING_STABLE_THRESHOLD,
    TRACKING_UNSTABLE_THRESHOLD,
)
from .drawing.canvas import DrawingCanvas, adaptive_smooth_point
from .drawing.sprite import draw_sprite_selection, overlay_sprite
from .runtime.controllers import (
    DrawingInteractionController,
    SpriteInteractionController,
    UIActionCallbacks,
    UIInteractionController,
    build_warning_text,
    clear_sprite_selection,
    enhance_low_light,
    get_selected_sprite,
    resize_for_mediapipe,
)
from .runtime.services import (
    CaptureDevice,
    CaptureFactory,
    HandTracker,
    HandTrackerFactory,
    RuntimeDiagnostics,
    create_default_capture,
    create_default_hand_tracker,
)
from .runtime.state import RuntimeState
from .ui.ui import (
    PaletteItem,
    ThumbnailItem,
    ToolbarItem,
    draw_brush_preview,
    draw_fingertip_marker,
    draw_fps,
    draw_palette,
    draw_tracking_feedback,
    draw_thumbnail_strip,
    draw_toolbar,
    draw_warning_overlay,
    get_palette_items,
    get_thumbnail_items,
    get_toolbar_items,
)
from .vision.gestures import (
    dwell_progress,
    get_fingertip_distance,
    get_index_fingertip,
    get_landmark_point,
    is_pinching,
    estimate_hand_spread,
    is_two_hand_resize,
)

logger = logging.getLogger(__name__)
OPEN_CV_ERROR = getattr(cv2, "error", Exception)


@dataclass(slots=True)
class FrameUiState:
    palette_items: list[PaletteItem]
    toolbar_items: list[ToolbarItem]
    thumbnail_items: list[ThumbnailItem]
    palette_rects: list[tuple[int, int, int, int]]
    toolbar_rects: list[tuple[int, int, int, int]]
    thumbnail_rects: list[tuple[int, int, int, int]]
    sprite_snapshot: list[object]
    selected_sprite: object | None


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


def mirror_frame(frame: np.ndarray) -> np.ndarray:
    if hasattr(cv2, "flip"):
        return cv2.flip(frame, 1)
    return np.ascontiguousarray(frame[:, ::-1])


class AppRunner:
    def __init__(
        self,
        window_name: str = "ARIA Phase 5",
        capture_factory: CaptureFactory = create_default_capture,
        hand_tracker_factory: HandTrackerFactory = create_default_hand_tracker,
        assistant_factory: Callable[[JarvisContext], JarvisAssistant] = JarvisAssistant,
    ) -> None:
        self.window_name = window_name
        self.capture_factory = capture_factory
        self.hand_tracker_factory = hand_tracker_factory
        self.state = RuntimeState(
            ai_warning_text=None
            if gemini_enabled()
            else "AI vision disabled: set GOOGLE_API_KEY or GEMINI_API_KEY to enable scene descriptions."
        )
        self.diagnostics = RuntimeDiagnostics()
        self.frame_interval = 1.0 / TARGET_FPS
        self.ui_controller = UIInteractionController()
        self.sprite_controller = SpriteInteractionController()
        self.drawing_controller = DrawingInteractionController()
        self.sprites = []
        self.sprites_lock = threading.Lock()
        self.canvas_lock = threading.Lock()
        self.frame_lock = threading.Lock()
        self.drawing_canvas: DrawingCanvas | None = None
        self._hands: HandTracker | None = None

        self.jarvis = assistant_factory(
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
        color = config.get_brush_color(color_name)
        if color is None or self.drawing_canvas is None:
            return False
        self.state.brush_name = color_name.lower()
        with self.canvas_lock:
            self.drawing_canvas.set_brush_color(color)
        return True

    def _show_webcam_error(self) -> None:
        error_frame = np.zeros((220, 720, 3), dtype=np.uint8)
        if hasattr(cv2, "putText") and hasattr(cv2, "FONT_HERSHEY_SIMPLEX"):
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
        if hasattr(cv2, "imshow"):
            cv2.imshow(self.window_name, error_frame)
        if hasattr(cv2, "waitKey"):
            cv2.waitKey(2500)
        if hasattr(cv2, "destroyAllWindows"):
            cv2.destroyAllWindows()

    def _setup_window(self) -> None:
        if hasattr(cv2, "namedWindow"):
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        if hasattr(cv2, "resizeWindow"):
            cv2.resizeWindow(self.window_name, 1400, 900)
        if hasattr(cv2, "setWindowProperty"):
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_AUTOSIZE, cv2.WINDOW_NORMAL)

    def _show_frame(self, frame: np.ndarray) -> None:
        if hasattr(cv2, "imshow"):
            cv2.imshow(self.window_name, frame)

    def _wait_for_exit(self, wait_ms: int) -> bool:
        if not hasattr(cv2, "waitKey"):
            return False
        return cv2.waitKey(wait_ms) & 0xFF == ord("q")

    def _initialize_capture(self):
        cap = self.capture_factory()
        if cap is None or not cap.isOpened():
            self.diagnostics.record_capture_failure("Webcam could not be opened.")
            self._show_webcam_error()
            return None
        return cap

    def _initialize_hands(self) -> HandTracker:
        try:
            return self.hand_tracker_factory()
        except Exception as exc:
            self.diagnostics.record_hand_tracking_failure(f"Hand tracker initialization failed: {exc}")
            raise

    def _update_calibration(self, hand_landmarks, now: float) -> None:
        if self.state.calibrated:
            return

        self.state.begin_calibration(now)
        if hand_landmarks is not None:
            self.state.add_calibration_sample(estimate_hand_spread(hand_landmarks))

        if self.state.calibration_started_at is None or now - self.state.calibration_started_at < CALIBRATION_SECONDS:
            self.state.tracking_status_text = "Calibrating hand..."
            return

        average_spread = (
            self.state.calibration_sample_total / self.state.calibration_sample_count
            if self.state.calibration_sample_count
            else BASE_HAND_SPREAD
        )
        hand_scale_factor = average_spread / BASE_HAND_SPREAD if BASE_HAND_SPREAD > 0 else 1.0
        self.state.finish_calibration(max(0.75, min(1.75, hand_scale_factor)))
        self.state.set_status("Calibration complete", now, 0.8)
        self.state.tracking_status_text = "DRAW READY"

    def _update_tracking_stability(self, raw_fingertip: tuple[int, int] | None) -> None:
        if raw_fingertip is None:
            self.state.set_tracking_stability(0.0)
            self.state.tracking_status_text = "Waiting for hand"
            return

        if self.state.prev_smoothed_point is None:
            self.state.set_tracking_stability(0.75)
            self.state.tracking_status_text = "DRAW READY" if self.state.calibrated else "Calibrating hand..."
            return

        movement = math.hypot(
            raw_fingertip[0] - self.state.prev_smoothed_point[0],
            raw_fingertip[1] - self.state.prev_smoothed_point[1],
        )
        stability = max(0.0, min(1.0, 1.0 - (movement / 30.0)))
        self.state.set_tracking_stability(stability)
        if stability >= TRACKING_STABLE_THRESHOLD:
            self.state.tracking_status_text = "DRAW READY"
        elif stability <= TRACKING_UNSTABLE_THRESHOLD:
            self.state.tracking_status_text = "TRACKING UNSTABLE"
        else:
            self.state.tracking_status_text = "Hold steady"

    def _prepare_ui_state(self, frame, frame_width: int, frame_height: int) -> FrameUiState:
        palette_items = get_palette_items(frame_width)
        toolbar_items = get_toolbar_items(frame_width, frame_height)
        with self.sprites_lock:
            selected_sprite = get_selected_sprite(self.sprites)
            sprite_snapshot = list(sorted(self.sprites, key=lambda item: item.z_index))
        thumbnail_items = get_thumbnail_items(frame, sprite_snapshot)
        return FrameUiState(
            palette_items=palette_items,
            toolbar_items=toolbar_items,
            thumbnail_items=thumbnail_items,
            palette_rects=[item["rect"] for item in palette_items],
            toolbar_rects=[item["rect"] for item in toolbar_items],
            thumbnail_rects=[item["rect"] for item in thumbnail_items],
            sprite_snapshot=sprite_snapshot,
            selected_sprite=selected_sprite,
        )

    def _process_hand_tracking(self, frame):
        assert self._hands is not None
        try:
            mediapipe_input = resize_for_mediapipe(frame)
            if hasattr(cv2, "cvtColor") and hasattr(cv2, "COLOR_BGR2RGB"):
                rgb_frame = cv2.cvtColor(mediapipe_input, cv2.COLOR_BGR2RGB)
            else:
                rgb_frame = mediapipe_input
            return self._hands.process(rgb_frame)
        except OPEN_CV_ERROR as exc:
            logger.warning("OpenCV hand-tracking preprocessing failed: %s", exc)
            self.diagnostics.record_hand_tracking_failure(f"OpenCV preprocessing failed: {exc}")
        except Exception as exc:
            logger.warning("Hand tracking failed: %s", exc)
            self.diagnostics.record_hand_tracking_failure(f"Hand tracking failed: {exc}")
        self.state.set_status("Hand tracking error", time.time(), 1.0)
        return None

    def _handle_resize_mode(
        self,
        all_hands,
        frame_width: int,
        frame_height: int,
        selected_sprite,
    ) -> None:
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

    def _handle_no_hands_detected(self) -> None:
        self.state.clear_resize()
        self.state.previous_pinch_active = False
        self.state.clear_hover()
        with self.canvas_lock:
            self.drawing_canvas.reset_stroke()
        self.state.clear_smoothing()
        self.state.clear_drawing_path()
        self.state.clear_draw_pose()
        self.state.clear_fist_hold()
        self.state.set_tracking_stability(0.0)
        self.state.tracking_status_text = "Waiting for hand"
        with self.sprites_lock:
            for sprite in self.sprites:
                sprite.dragging = False
        if self.state.status_text != STATUS_SPRITE_CREATED:
            self.state.status_text = STATUS_IDLE

    def _handle_hand_interactions(
        self,
        hand_landmarks,
        ui_state: FrameUiState,
        frame_width: int,
        frame_height: int,
    ) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
        self.state.clear_resize()
        current_point = None
        raw_fingertip = get_index_fingertip(hand_landmarks, frame_width, frame_height)
        self._update_tracking_stability(raw_fingertip)
        if raw_fingertip is not None:
            previous_raw = None
            if self.state.prev_smoothed_point is not None:
                previous_raw = (
                    int(round(self.state.prev_smoothed_point[0])),
                    int(round(self.state.prev_smoothed_point[1])),
                )
            movement_distance = get_fingertip_distance(previous_raw, raw_fingertip) if previous_raw is not None else 0.0
            smoothed = adaptive_smooth_point(
                self.state.prev_smoothed_point,
                raw_fingertip,
                movement_distance,
            )
            current_point = (int(smoothed[0]), int(smoothed[1]))
            self.state.prev_smoothed_point = smoothed

        scaled_pinch_start = PINCH_START_DISTANCE * self.state.hand_scale_factor
        scaled_pinch_release = PINCH_RELEASE_DISTANCE * self.state.hand_scale_factor
        is_currently_pinching = is_pinching(hand_landmarks, frame_width, frame_height, scaled_pinch_start)
        pinch_distance = None
        if raw_fingertip is not None:
            thumb_tip = get_landmark_point(hand_landmarks, THUMB_TIP, frame_width, frame_height)
            pinch_distance = get_fingertip_distance(thumb_tip, raw_fingertip)

        with self.sprites_lock:
            dragging_sprite = next((sprite for sprite in self.sprites if sprite.dragging), None)
        if dragging_sprite and pinch_distance is not None and pinch_distance > scaled_pinch_release:
            with self.sprites_lock:
                dragging_sprite.dragging = False

        pinch_active = is_currently_pinching or (
            dragging_sprite is not None and pinch_distance is not None and pinch_distance <= scaled_pinch_release
        )
        pinch_started = pinch_active and not self.state.previous_pinch_active

        hover_point = raw_fingertip or current_point
        hover_candidate = self.ui_controller.resolve_hover_candidate(
            hover_point,
            pinch_active,
            self.state.interaction_mode,
            self.state.stroke_active(),
            ui_state.toolbar_items,
            ui_state.palette_items,
        )
        pointer_over_controls = point_in_any_rect(hover_point, ui_state.palette_rects + ui_state.toolbar_rects)

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
                    thumbnail_items=ui_state.thumbnail_items,
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
                    ui_rects=ui_state.palette_rects + ui_state.toolbar_rects + ui_state.thumbnail_rects,
                )
        else:
            with self.canvas_lock:
                self.drawing_canvas.reset_stroke()
            self.state.clear_drawing_path()
            self.state.clear_fist_hold()

        self.state.previous_pinch_active = pinch_active
        return current_point, raw_fingertip

    def _render_output(self, frame, palette_items, toolbar_items, thumbnail_items, current_point, is_resize_mode) -> np.ndarray:
        with self.canvas_lock:
            output_frame = self.drawing_canvas.overlay_on(frame, ALPHA)

        dwell_ratio = (
            dwell_progress(
                self.state.hover_start_time,
                config.UI_DRAW_MODE_DWELL_SECONDS if self.state.interaction_mode == "draw_mode" else config.DWELL_SECONDS,
                time.time(),
            )
            if self.state.hover_target_id
            else 0.0
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
            brush_color = config.get_brush_color(self.state.brush_name) or config.DRAW_COLOR
            draw_brush_preview(output_frame, current_point, LINE_THICKNESS, brush_color)
            draw_fingertip_marker(output_frame, current_point, self.state.tracking_stability)
            draw_tracking_feedback(
                output_frame,
                self.state.calibrated,
                self.state.tracking_stability,
                self.state.tracking_status_text,
            )
        draw_fps(output_frame, self.state.fps)
        return output_frame

    def run_iteration(self, capture: CaptureDevice) -> bool:
        loop_start = time.time()
        success, frame = capture.read()
        if not success:
            self.diagnostics.record_capture_failure("Frame read failed.")
            return False

        frame = mirror_frame(frame)
        frame = enhance_low_light(frame)
        if self.drawing_canvas is None:
            self.drawing_canvas = DrawingCanvas(frame.shape)

        frame_height, frame_width = frame.shape[:2]
        self.state.frame_dimensions["width"] = frame_width
        self.state.frame_dimensions["height"] = frame_height
        current_point = None
        results = self._process_hand_tracking(frame)
        all_hands = results.multi_hand_landmarks if results and results.multi_hand_landmarks else []
        hand_landmarks = all_hands[0] if all_hands else None
        self._update_calibration(hand_landmarks, time.time())
        ui_state = self._prepare_ui_state(frame, frame_width, frame_height)

        is_resize_mode = self.state.interaction_mode == "select_mode" and is_two_hand_resize(all_hands)
        if not self.state.calibrated:
            if hand_landmarks is None:
                self._handle_no_hands_detected()
            else:
                self.state.status_text = "Calibrating"
                self._update_tracking_stability(get_index_fingertip(hand_landmarks, frame_width, frame_height))
        elif is_resize_mode:
            self._handle_resize_mode(
                all_hands=all_hands,
                frame_width=frame_width,
                frame_height=frame_height,
                selected_sprite=ui_state.selected_sprite,
            )
        elif hand_landmarks:
            current_point, _ = self._handle_hand_interactions(
                hand_landmarks=hand_landmarks,
                ui_state=ui_state,
                frame_width=frame_width,
                frame_height=frame_height,
            )
        else:
            self._handle_no_hands_detected()

        for sprite in ui_state.sprite_snapshot:
            overlay_sprite(frame, sprite)
        for sprite in ui_state.sprite_snapshot:
            draw_sprite_selection(frame, sprite)

        elapsed = time.time() - loop_start
        if elapsed > 0:
            self.state.fps = 1.0 / elapsed
        output_frame = self._render_output(
            frame,
            ui_state.palette_items,
            ui_state.toolbar_items,
            ui_state.thumbnail_items,
            current_point,
            is_resize_mode,
        )

        with self.frame_lock:
            self.state.latest_frame = output_frame.copy()
        self.diagnostics.record_frame(elapsed, self.state.status_text)
        return True

    def run(self) -> None:
        self._setup_window()

        cap = self._initialize_capture()
        if cap is None:
            return

        try:
            self._hands = self._initialize_hands()
        except Exception:
            return

        self.jarvis.start()

        try:
            while True:
                if not self.run_iteration(cap):
                    break
                output_frame = self.get_current_frame()
                if output_frame is not None:
                    self._show_frame(output_frame)

                remaining = max(0.0, self.frame_interval - (self.diagnostics.last_frame_latency_ms / 1000.0))
                wait_ms = max(1, int(remaining * 1000))
                if self._wait_for_exit(wait_ms):
                    break

                self.state.expire_status(time.time())
        finally:
            self.jarvis.stop()
            if self._hands is not None:
                self._hands.close()
            cap.release()
            if hasattr(cv2, "destroyAllWindows"):
                cv2.destroyAllWindows()


def main() -> None:
    AppRunner().run()
