"""ARIA application loop.

Frame pipeline (render thread):
    capture -> mirror -> low-light enhance -> workspace scale
    -> drain queued commands (voice) -> hand tracking -> primary hand
    -> gesture interpretation (controllers) -> commands / state updates
    -> scene composite (camera + sprites + drawing) -> UI overlay -> imshow

Input methods (gesture dwell, keyboard, mouse, voice) all go through the
same ``CommandDispatcher`` on this thread. Voice runs on its own thread and
submits through a bounded ``CommandQueue``.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

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
    MAX_SCENE_HISTORY,
    PINCH_RELEASE_DISTANCE,
    PINCH_START_DISTANCE,
    RESIZE_CLUTCH_SECONDS,
    STATUS_IDLE,
    TARGET_FPS,
    THUMB_TIP,
    TRACKING_STABLE_THRESHOLD,
    TRACKING_UNSTABLE_THRESHOLD,
)
from .drawing.canvas import DrawingCanvas, adaptive_smooth_point
from .drawing.sprite import (
    draw_sprite_candidate,
    draw_sprite_delete_confirmation,
    draw_sprite_selection,
    overlay_sprite,
)
from .runtime.commands import (
    HELP_CLOSE_ITEM_ID,
    Action,
    Command,
    CommandDispatcher,
    CommandQueue,
    CommandResult,
    command_for_key,
    command_for_ui_item,
    is_quit_key,
    wait_for_result,
)
from .runtime.controllers import (
    DrawingInteractionController,
    SpriteInteractionController,
    UIInteractionController,
    bring_sprite_to_front,
    clear_sprite_selection,
    describe_services,
    enhance_low_light,
    expand_rect,
    get_selected_sprite,
    get_topmost_sprite_near_point,
    point_in_any_rect,
    resize_for_mediapipe,
)
from .runtime.notifications import Notifier
from .runtime.scene import (
    SceneHistory,
    SceneSnapshot,
    clone_sprite,
    load_project,
    save_project,
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
from .ui import help_content, theme
from .ui.layout import UiLayout, compute_layout
from .ui.ui import (
    draw_gesture_progress,
    draw_guide,
    draw_hand_skeleton,
    draw_help,
    draw_layers,
    draw_mode_frame,
    draw_palette,
    draw_pointer,
    draw_resize_anchor,
    draw_status_panel,
    draw_toasts,
    draw_toolbar,
    draw_top_bar,
    draw_voice_card,
)
from .vision.gestures import (
    dwell_progress,
    estimate_hand_spread,
    get_fingertip_distance,
    get_index_fingertip,
    get_landmark_point,
    is_pinching,
    is_two_hand_resize,
    select_primary_hand,
)

logger = logging.getLogger(__name__)
OPEN_CV_ERROR = getattr(cv2, "error", Exception)
MOUSE_SPRITE_PADDING = 6


def mirror_frame(frame: np.ndarray) -> np.ndarray:
    if hasattr(cv2, "flip"):
        return cv2.flip(frame, 1)
    return np.ascontiguousarray(frame[:, ::-1])


def scale_to_workspace(
    frame: np.ndarray, min_height: int = config.WORKSPACE_MIN_HEIGHT
) -> np.ndarray:
    """Upscale small camera frames (aspect preserved) so overlays stay legible.

    All interaction coordinates are derived from this workspace frame:
    MediaPipe landmarks are normalized, so they map onto it directly.
    """
    height, width = frame.shape[:2]
    if height >= min_height or height <= 0 or not hasattr(cv2, "resize"):
        return frame
    scale = min_height / height
    return cv2.resize(
        frame,
        (round(width * scale), min_height),
        interpolation=cv2.INTER_LINEAR,
    )


def fit_window_size(
    frame_width: int, frame_height: int, max_width: int, max_height: int
) -> tuple[int, int]:
    """Largest window size within the bounds that keeps the frame aspect."""
    if frame_width <= 0 or frame_height <= 0:
        return max_width, max_height
    scale = min(max_width / frame_width, max_height / frame_height)
    return max(1, round(frame_width * scale)), max(1, round(frame_height * scale))


@dataclass(slots=True)
class MouseDrag:
    sprite: object
    offset_x: int
    offset_y: int


class AppRunner:
    def __init__(
        self,
        window_name: str = "ARIA",
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
        self.notifier = Notifier()
        self.frame_interval = 1.0 / TARGET_FPS
        self.ui_controller = UIInteractionController()
        self.sprite_controller = SpriteInteractionController()
        self.drawing_controller = DrawingInteractionController()
        self.sprites = []
        self.sprites_lock = threading.Lock()
        self.canvas_lock = threading.Lock()
        self.frame_lock = threading.Lock()
        self.drawing_canvas: DrawingCanvas | None = None
        self.scene_history = SceneHistory(MAX_SCENE_HISTORY)
        self._last_scene_signature: tuple[object, ...] | None = None
        self._hands: HandTracker | None = None
        self._layout: UiLayout | None = None
        self._mouse_drag: MouseDrag | None = None
        self._window_fitted = False
        self._camera_size: tuple[int, int] | None = None
        self._tracking_error_at: float | None = None
        self._last_tick: float | None = None
        self._owner_thread = threading.current_thread()

        self.command_queue = CommandQueue()
        self.dispatcher = CommandDispatcher(
            {
                Action.MODE_DRAW: self._cmd_mode,
                Action.MODE_SELECT: self._cmd_mode,
                Action.BRUSH: self._cmd_brush,
                Action.UNDO: self._cmd_undo,
                Action.REDO: self._cmd_redo,
                Action.CLEAR: self._cmd_clear,
                Action.DELETE: self._cmd_delete,
                Action.SCALE: self._cmd_scale,
                Action.DESELECT: self._cmd_deselect,
                Action.EXPORT_PNG: self._cmd_export_png,
                Action.SAVE_PROJECT: self._cmd_save_project,
                Action.LOAD_PROJECT: self._cmd_load_project,
                Action.TOGGLE_HELP: self._cmd_toggle_help,
                Action.TOGGLE_STATUS: self._cmd_toggle_status,
                Action.CANCEL: self._cmd_cancel,
            },
            on_result=self._on_command_result,
        )

        self.jarvis = assistant_factory(
            JarvisContext(
                get_current_frame=self.get_scene_frame,
                clear_canvas=lambda: self._voice(Action.CLEAR),
                delete_selected_sprite=lambda: self._voice(Action.DELETE),
                scale_selected_sprite=lambda factor: self._voice(Action.SCALE, factor),
                undo_last_stroke=lambda: self._voice(Action.UNDO),
                save_snapshot=lambda path: self._voice(Action.EXPORT_PNG, path),
                set_brush_color=lambda name: self._voice(Action.BRUSH, name),
                set_interaction_mode=lambda mode: self._voice(
                    Action.MODE_SELECT
                    if mode == config.MODE_SELECT
                    else Action.MODE_DRAW
                    if mode == config.MODE_DRAW
                    else None,
                ),
                redo_last_scene=lambda: self._voice(Action.REDO),
            )
        )

    # --- Commands ---------------------------------------------------------

    def run_command(self, command: Command) -> CommandResult:
        """Execute on the render thread; from other threads, queue and wait."""
        if threading.current_thread() is self._owner_thread:
            return self.dispatcher.dispatch(command)
        future = self.command_queue.submit(command)
        return wait_for_result(future, config.VOICE_COMMAND_RESULT_TIMEOUT_SECONDS)

    def _voice(self, action: Action | None, arg: object = None) -> bool:
        if action is None:
            return False
        return self.run_command(Command(action, arg, "voice")).ok

    def _on_command_result(self, command: Command, result: CommandResult) -> None:
        if result.message:
            self.notifier.push(
                result.message,
                result.kind,
                now=time.monotonic(),
                key=command.action.value,
            )

    def _cmd_mode(self, command: Command) -> CommandResult:
        mode = (
            config.MODE_DRAW
            if command.action == Action.MODE_DRAW
            else config.MODE_SELECT
        )
        if mode == self.state.interaction_mode:
            label = "Draw" if mode == config.MODE_DRAW else "Select"
            return CommandResult.info(f"Already in {label} mode")
        self.set_interaction_mode(mode)
        if mode == config.MODE_DRAW:
            return CommandResult.info("Draw mode - raise your index finger to draw")
        return CommandResult.info("Select mode - pinch a sprite to select it")

    def _cmd_brush(self, command: Command) -> CommandResult:
        name = str(command.arg or "")
        if self.drawing_canvas is None:
            return CommandResult.unavailable("Camera is not ready yet")
        if not self.set_brush_color(name):
            return CommandResult.unavailable(f"Unknown color: {name}")
        if name.lower() == "eraser":
            return CommandResult.info("Eraser - draw over strokes to erase")
        return CommandResult.info(f"Brush: {name.title()}")

    def _cmd_undo(self, _command: Command) -> CommandResult:
        if self.undo_last_stroke():
            return CommandResult.info("Undone - R to redo")
        return CommandResult.unavailable("Nothing to undo")

    def _cmd_redo(self, _command: Command) -> CommandResult:
        if self.redo_last_scene():
            return CommandResult.info("Redone")
        return CommandResult.unavailable("Nothing to redo")

    def _cmd_clear(self, _command: Command) -> CommandResult:
        if self.drawing_canvas is None:
            return CommandResult.unavailable("Camera is not ready yet")
        with self.canvas_lock:
            has_content = self.drawing_canvas.has_content()
        if not has_content:
            return CommandResult.info("Drawing is already empty")
        self.clear_canvas_state()
        return CommandResult.success("Drawing cleared - U to undo")

    def _cmd_delete(self, _command: Command) -> CommandResult:
        if self.delete_selected_sprite():
            return CommandResult.success("Sprite deleted - U to undo")
        return CommandResult.unavailable("Select a sprite first")

    def _cmd_scale(self, command: Command) -> CommandResult:
        try:
            factor = float(command.arg)
        except (TypeError, ValueError):
            return CommandResult.unavailable("Invalid size")
        with self.sprites_lock:
            selected = get_selected_sprite(self.sprites)
            before = (selected.w, selected.h) if selected is not None else None
        if before is None:
            return CommandResult.unavailable("Select a sprite first")
        self.scale_selected_sprite(factor)
        with self.sprites_lock:
            after = (selected.w, selected.h)
        if after == before:
            return CommandResult.info("Sprite is at its size limit")
        return CommandResult.info(f"Sprite {after[0]} x {after[1]}")

    def _cmd_deselect(self, _command: Command) -> CommandResult:
        with self.sprites_lock:
            had_selection = get_selected_sprite(self.sprites) is not None
            clear_sprite_selection(self.sprites)
        return CommandResult.info("Selection cleared" if had_selection else "")

    def _cmd_export_png(self, command: Command) -> CommandResult:
        path = str(command.arg or config.DEFAULT_SAVE_PATH)
        if self.save_snapshot(path):
            return CommandResult.success(f"Saved {path}")
        return CommandResult.failure(f"Could not save {path}")

    def _cmd_save_project(self, _command: Command) -> CommandResult:
        path = config.DEFAULT_PROJECT_PATH
        if self.save_project(path):
            return CommandResult.success(f"Project saved: {path}")
        return CommandResult.failure("Could not save the project")

    def _cmd_load_project(self, _command: Command) -> CommandResult:
        path = Path(config.DEFAULT_PROJECT_PATH).with_suffix(".npz")
        if not path.exists():
            return CommandResult.unavailable(f"No saved project ({path.name})")
        if self.load_project(str(path)):
            return CommandResult.success("Project loaded - U to undo")
        return CommandResult.failure("Project file could not be read")

    def _cmd_toggle_help(self, _command: Command) -> CommandResult:
        self.state.help_visible = not self.state.help_visible
        if self.state.help_visible:
            self._suspend_gestures()
        return CommandResult.info()

    def _cmd_toggle_status(self, _command: Command) -> CommandResult:
        self.state.settings_visible = not self.state.settings_visible
        return CommandResult.info()

    def _cmd_cancel(self, _command: Command) -> CommandResult:
        if self.state.help_visible:
            self.state.help_visible = False
            return CommandResult.info()
        if self.state.delete_pending_sprite_ref is not None:
            self.state.clear_delete_confirmation()
            return CommandResult.info("Delete cancelled")
        if self.state.guide_visible:
            self.state.guide_visible = False
            return CommandResult.info("Quick start hidden - press H for help")
        return self._cmd_deselect(_command)

    # --- Workspace operations (render thread) -----------------------------

    def get_current_frame(self):
        with self.frame_lock:
            if self.state.latest_frame is None:
                return None
            return self.state.latest_frame.copy()

    def get_scene_frame(self):
        """Camera + drawing + sprites without UI chrome (export and AI)."""
        with self.frame_lock:
            frame = self.state.latest_scene_frame
            if frame is None:
                frame = self.state.latest_frame
            return None if frame is None else frame.copy()

    def clear_canvas_state(self) -> None:
        if self.drawing_canvas is None:
            return
        with self.canvas_lock:
            self.drawing_canvas.clear()
        self.state.clear_drawing_path()
        self._record_scene()

    def delete_selected_sprite(self) -> bool:
        with self.sprites_lock:
            selected = get_selected_sprite(self.sprites)
            if selected is None:
                return False
            self.sprites.remove(selected)
            clear_sprite_selection(self.sprites)
        if self._mouse_drag is not None and self._mouse_drag.sprite is selected:
            self._mouse_drag = None
        self.state.clear_delete_confirmation()
        self._record_scene()
        return True

    def scale_selected_sprite(self, scale_factor: float) -> bool:
        if scale_factor <= 0:
            return False
        with self.sprites_lock:
            selected = get_selected_sprite(self.sprites)
            if selected is None:
                return False
            selected.scale_by_factor(scale_factor)
            self._clamp_sprite(selected)
        self._record_scene()
        return True

    def _clamp_sprite(self, sprite) -> None:
        width = self.state.frame_dimensions["width"]
        height = self.state.frame_dimensions["height"]
        if width and height:
            sprite.clamp_to_frame(width, height)

    def _capture_scene(self) -> SceneSnapshot | None:
        if self.drawing_canvas is None:
            return None
        with self.canvas_lock, self.sprites_lock:
            return SceneSnapshot(
                canvas=self.drawing_canvas.canvas.copy(),
                strokes=[dict(stroke) for stroke in self.drawing_canvas.strokes],
                brush_color=self.drawing_canvas.brush_color,
                sprites=[clone_sprite(sprite) for sprite in self.sprites],
                interaction_mode=self.state.interaction_mode,
                brush_name=self.state.brush_name,
            )

    def _record_scene(self) -> None:
        snapshot = self._capture_scene()
        if snapshot is not None:
            self.scene_history.record(snapshot)
            self._last_scene_signature = self._scene_signature()

    def _scene_signature(self) -> tuple[object, ...] | None:
        """Content-only signature: selection, mode and brush are not history."""
        if self.drawing_canvas is None:
            return None
        with self.canvas_lock, self.sprites_lock:
            sprite_state = tuple(
                (sprite.x, sprite.y, sprite.w, sprite.h, sprite.z_index)
                for sprite in self.sprites
            )
            return (self.drawing_canvas.revision, sprite_state)

    def _record_scene_if_changed(self) -> None:
        with self.sprites_lock:
            interaction_in_progress = any(sprite.dragging for sprite in self.sprites)
        if (
            self.state.stroke_active()
            or interaction_in_progress
            or self._mouse_drag is not None
            or self.state.resizing_sprite_ref is not None
        ):
            return
        signature = self._scene_signature()
        if signature != self._last_scene_signature:
            self._record_scene()

    def _restore_scene(self, snapshot: SceneSnapshot, restore_tools: bool) -> None:
        canvas_image = snapshot.canvas
        if self.drawing_canvas is None:
            self.drawing_canvas = DrawingCanvas(canvas_image.shape)
        target_shape = self.drawing_canvas.canvas.shape
        if canvas_image.shape != target_shape and hasattr(cv2, "resize"):
            canvas_image = cv2.resize(
                canvas_image,
                (target_shape[1], target_shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        with self.canvas_lock, self.sprites_lock:
            self.drawing_canvas.canvas = canvas_image.copy()
            self.drawing_canvas.strokes = [dict(stroke) for stroke in snapshot.strokes]
            self.drawing_canvas.undo_stack = [canvas_image.copy()]
            self.drawing_canvas.discard_stroke()
            self.drawing_canvas.mark_changed()
            self.sprites = [clone_sprite(sprite) for sprite in snapshot.sprites]
            for sprite in self.sprites:
                self._clamp_sprite(sprite)
            if restore_tools:
                self.drawing_canvas.brush_color = snapshot.brush_color
        if restore_tools:
            self.state.interaction_mode = snapshot.interaction_mode
            self.state.brush_name = snapshot.brush_name
        self._mouse_drag = None
        self.state.reset_interaction_transients()
        self._last_scene_signature = self._scene_signature()

    def undo_last_stroke(self) -> bool:
        snapshot = self.scene_history.undo()
        if snapshot is None:
            return False
        self._restore_scene(snapshot, restore_tools=False)
        return True

    def redo_last_scene(self) -> bool:
        snapshot = self.scene_history.redo()
        if snapshot is None:
            return False
        self._restore_scene(snapshot, restore_tools=False)
        return True

    def save_snapshot(self, path: str) -> bool:
        frame = self.get_scene_frame()
        if frame is None:
            return False
        try:
            return bool(cv2.imwrite(path, frame))
        except (OPEN_CV_ERROR, OSError) as exc:
            logger.warning("Snapshot save failed: %s", exc)
            return False

    def save_project(self, path: str = config.DEFAULT_PROJECT_PATH) -> bool:
        snapshot = self._capture_scene()
        if snapshot is None:
            return False
        try:
            save_project(path, snapshot)
        except (OSError, ValueError) as exc:
            logger.warning("Project save failed: %s", exc)
            return False
        return True

    def load_project(self, path: str = config.DEFAULT_PROJECT_PATH) -> bool:
        try:
            snapshot = load_project(path)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            logger.warning("Project load failed: %s", exc)
            return False
        self._restore_scene(snapshot, restore_tools=True)
        self._record_scene()
        return True

    def set_brush_color(self, color_name: str) -> bool:
        color = config.get_brush_color(color_name)
        if color is None or self.drawing_canvas is None:
            return False
        self.state.brush_name = color_name.lower()
        with self.canvas_lock:
            self.drawing_canvas.set_brush_color(color)
        return True

    def set_interaction_mode(self, mode_name: str) -> bool:
        if mode_name not in {config.MODE_DRAW, config.MODE_SELECT}:
            return False
        self.state.interaction_mode = mode_name
        self.state.clear_resize()
        self.state.clear_resize_clutch()
        self.state.reset_interaction_transients()
        self._mouse_drag = None
        if self.drawing_canvas is not None:
            with self.canvas_lock:
                self.drawing_canvas.reset_stroke()
        with self.sprites_lock:
            for sprite in self.sprites:
                sprite.dragging = False
        return True

    def _suspend_gestures(self) -> None:
        """Cancel in-flight gestures (used when help opens)."""
        self.state.reset_for_hand_loss()
        if self.drawing_canvas is not None:
            with self.canvas_lock:
                self.drawing_canvas.reset_stroke()
        with self.sprites_lock:
            for sprite in self.sprites:
                sprite.dragging = False

    # --- Window, keyboard and mouse ---------------------------------------

    def _show_message_screen(self, title: str, detail: str) -> None:
        error_frame = np.full((240, 760, 3), theme.GRAPHITE_950, dtype=np.uint8)
        if hasattr(cv2, "putText"):
            theme.draw_text(error_frame, title, (40, 100), 0.85, theme.STATUS_ERROR, 2)
            theme.draw_text(error_frame, detail, (40, 145), 0.6, theme.TEXT)
            theme.draw_text(
                error_frame, "Closing in a few seconds.", (40, 185), 0.5,
                theme.TEXT_MUTED,
            )  # fmt: skip
        if hasattr(cv2, "imshow"):
            cv2.imshow(self.window_name, error_frame)
        if hasattr(cv2, "waitKey"):
            cv2.waitKey(3000)
        if hasattr(cv2, "destroyAllWindows"):
            cv2.destroyAllWindows()

    def _show_webcam_error(self) -> None:
        self._show_message_screen(
            "ARIA could not open a webcam.",
            "Check the camera connection and that no other app is using it.",
        )

    def _setup_window(self) -> None:
        if hasattr(cv2, "namedWindow"):
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        if hasattr(cv2, "setMouseCallback"):
            cv2.setMouseCallback(self.window_name, self._handle_mouse_event)

    def _fit_window(self, frame: np.ndarray) -> None:
        """Size the window to the frame aspect once, so nothing is stretched
        and mouse coordinates map 1:1 onto the workspace axes."""
        if self._window_fitted or not hasattr(cv2, "resizeWindow"):
            return
        self._window_fitted = True
        height, width = frame.shape[:2]
        size = fit_window_size(
            width, height, config.WINDOW_MAX_WIDTH, config.WINDOW_MAX_HEIGHT
        )
        try:
            cv2.resizeWindow(self.window_name, *size)
        except OPEN_CV_ERROR as exc:
            logger.debug("Window resize unavailable: %s", exc)

    def _show_frame(self, frame: np.ndarray) -> None:
        if hasattr(cv2, "imshow"):
            cv2.imshow(self.window_name, frame)
            self._fit_window(frame)

    def _window_closed(self) -> bool:
        if not hasattr(cv2, "getWindowProperty"):
            return False
        try:
            return cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1
        except OPEN_CV_ERROR:
            return True

    def handle_key(self, key: int) -> bool:
        """Process a key code. Returns True when the app should quit."""
        if key == -1:
            return False
        key &= 0xFF
        if is_quit_key(key):
            return True
        command = command_for_key(key)
        if command is not None:
            self.dispatcher.dispatch(command)
        return False

    def _wait_for_exit(self, wait_ms: int) -> bool:
        if not hasattr(cv2, "waitKey"):
            return False
        if self.handle_key(cv2.waitKey(wait_ms)):
            return True
        return self._window_closed()

    def _handle_mouse_event(self, event, x: int, y: int, flags, _param) -> None:
        layout = self._layout
        if layout is None:
            return
        point = (int(x), int(y))
        if event == getattr(cv2, "EVENT_LBUTTONDOWN", 1):
            self._mouse_down(layout, point)
        elif event == getattr(cv2, "EVENT_MOUSEMOVE", 0):
            if self._mouse_drag is not None:
                self._mouse_move(point)
        elif event == getattr(cv2, "EVENT_LBUTTONUP", 4):
            self._mouse_drag = None
        elif event == getattr(cv2, "EVENT_MOUSEWHEEL", 10):
            if self.state.interaction_mode != config.MODE_SELECT:
                return
            delta = (
                cv2.getMouseWheelDelta(flags)
                if hasattr(cv2, "getMouseWheelDelta")
                else 0
            )
            if delta:
                factor = (
                    config.SPRITE_SCALE_STEP_UP
                    if delta > 0
                    else config.SPRITE_SCALE_STEP_DOWN
                )
                self.dispatcher.dispatch(Command(Action.SCALE, factor, "mouse"))

    def _mouse_down(self, layout: UiLayout, point: tuple[int, int]) -> None:
        if self.state.help_visible:
            self.state.help_visible = False
            return
        for item in layout.toolbar_items + layout.palette_items:
            if point_in_any_rect(point, [item["rect"]]):
                command = command_for_ui_item(item["id"], "mouse")
                if command is not None:
                    self.dispatcher.dispatch(command)
                return
        for item in layout.thumbnail_items:
            if point_in_any_rect(point, [item["rect"]]):
                with self.sprites_lock:
                    if item["sprite"] in self.sprites:
                        clear_sprite_selection(self.sprites)
                        item["sprite"].selected = True
                        bring_sprite_to_front(self.sprites, item["sprite"])
                        self.state.guide_selected = True
                self.notifier.push("Layer brought to front", "info", time.monotonic())
                return
        if self.state.interaction_mode != config.MODE_SELECT:
            return
        with self.sprites_lock:
            sprite = get_topmost_sprite_near_point(
                self.sprites, point, padding=MOUSE_SPRITE_PADDING
            )
            if sprite is None:
                had_selection = get_selected_sprite(self.sprites) is not None
                clear_sprite_selection(self.sprites)
            else:
                clear_sprite_selection(self.sprites)
                sprite.selected = True
                bring_sprite_to_front(self.sprites, sprite)
                self._mouse_drag = MouseDrag(
                    sprite, point[0] - sprite.x, point[1] - sprite.y
                )
                self.state.guide_selected = True
        if sprite is None and had_selection:
            self.notifier.push("Selection cleared", "info", time.monotonic())

    def _mouse_move(self, point: tuple[int, int]) -> None:
        drag = self._mouse_drag
        if drag is None:
            return
        with self.sprites_lock:
            if drag.sprite not in self.sprites:
                self._mouse_drag = None
                return
            drag.sprite.x = point[0] - drag.offset_x
            drag.sprite.y = point[1] - drag.offset_y
            self._clamp_sprite(drag.sprite)

    # --- Capture and tracking ---------------------------------------------

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
            self.diagnostics.record_hand_tracking_failure(
                f"Hand tracker initialization failed: {exc}"
            )
            raise

    def _update_calibration(self, hand_landmarks, now: float) -> None:
        if self.state.calibrated:
            return

        self.state.begin_calibration(now)
        if hand_landmarks is not None:
            self.state.add_calibration_sample(estimate_hand_spread(hand_landmarks))

        if (
            self.state.calibration_started_at is None
            or now - self.state.calibration_started_at < CALIBRATION_SECONDS
        ):
            self.state.tracking_status_text = "Calibrating hand..."
            return

        average_spread = (
            self.state.calibration_sample_total / self.state.calibration_sample_count
            if self.state.calibration_sample_count
            else BASE_HAND_SPREAD
        )
        hand_scale_factor = (
            average_spread / BASE_HAND_SPREAD if BASE_HAND_SPREAD > 0 else 1.0
        )
        self.state.finish_calibration(max(0.75, min(1.75, hand_scale_factor)))
        self.state.tracking_status_text = "DRAW READY"

    def _update_tracking_stability(self, raw_fingertip: tuple[int, int] | None) -> None:
        if raw_fingertip is None:
            self.state.set_tracking_stability(0.0)
            self.state.tracking_status_text = "Waiting for hand"
            self.state.tracking_recovery_hint = "Place your hand in view"
            return

        if self.state.prev_smoothed_point is None:
            self.state.set_tracking_stability(0.75)
            self.state.tracking_status_text = (
                "DRAW READY" if self.state.calibrated else "Calibrating hand..."
            )
            self.state.tracking_recovery_hint = "Move slowly to begin"
            return

        movement = math.hypot(
            raw_fingertip[0] - self.state.prev_smoothed_point[0],
            raw_fingertip[1] - self.state.prev_smoothed_point[1],
        )
        stability = max(0.0, min(1.0, 1.0 - (movement / 30.0)))
        self.state.set_tracking_stability(stability)
        if stability >= TRACKING_STABLE_THRESHOLD:
            self.state.tracking_status_text = "DRAW READY"
            self.state.tracking_recovery_hint = "Tracking stable"
        elif stability <= TRACKING_UNSTABLE_THRESHOLD:
            self.state.tracking_status_text = "TRACKING UNSTABLE"
            self.state.tracking_recovery_hint = "Improve lighting or hold steady"
        else:
            self.state.tracking_status_text = "Hold steady"
            self.state.tracking_recovery_hint = "Slow your fingertip movement"

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
            self.diagnostics.record_hand_tracking_failure(
                f"OpenCV preprocessing failed: {exc}"
            )
        except Exception as exc:  # noqa: BLE001 - MediaPipe backend boundary
            logger.warning("Hand tracking failed: %s", exc)
            self.diagnostics.record_hand_tracking_failure(
                f"Hand tracking failed: {exc}"
            )
        now = time.monotonic()
        self._tracking_error_at = now
        self.notifier.push(
            "Hand tracking error - retrying", "error", now, key="tracking-error"
        )
        return None

    def _ensure_canvas(self, frame: np.ndarray) -> None:
        if self.drawing_canvas is None:
            self.drawing_canvas = DrawingCanvas(frame.shape)
            self._record_scene()
            return
        if self.drawing_canvas.canvas.shape == frame.shape:
            return
        # Resolution changed mid-session: rescale content instead of crashing.
        old_h, old_w = self.drawing_canvas.canvas.shape[:2]
        new_h, new_w = frame.shape[:2]
        with self.canvas_lock:
            self.drawing_canvas.reset_stroke()
            self.drawing_canvas.canvas = cv2.resize(
                self.drawing_canvas.canvas,
                (new_w, new_h),
                interpolation=cv2.INTER_NEAREST,
            )
            self.drawing_canvas.undo_stack = [self.drawing_canvas.canvas.copy()]
            self.drawing_canvas.mark_changed()
        with self.sprites_lock:
            for sprite in self.sprites:
                sprite.x = round(sprite.x * new_w / old_w)
                sprite.y = round(sprite.y * new_h / old_h)
                sprite.clamp_to_frame(new_w, new_h)
        self.state.reset_for_hand_loss()
        self._mouse_drag = None
        self.state.post_event("Camera resolution changed - workspace rescaled", "warn")

    # --- Gesture handling ---------------------------------------------------

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
            self.state.resize_anchor_first_point = first_point
            self.state.resize_anchor_second_point = second_point
            current_distance = get_fingertip_distance(first_point, second_point)
            if (
                self.state.resizing_sprite_ref != id(selected_sprite)
                or self.state.resize_anchor_distance is None
                or self.state.resize_anchor_width is None
            ):
                self.state.resizing_sprite_ref = id(selected_sprite)
                self.state.resize_anchor_distance = max(1.0, current_distance)
                self.state.resize_anchor_width = selected_sprite.w
            if self.state.resize_anchor_distance and self.state.resize_anchor_width:
                with self.sprites_lock:
                    scale_factor = current_distance / self.state.resize_anchor_distance
                    target_width = self.state.resize_anchor_width * scale_factor
                    selected_sprite.resize_to_dimensions(target_width)
                    selected_sprite.clamp_to_frame(frame_width, frame_height)
            self.state.prev_resize_distance = current_distance
        else:
            self.state.clear_resize()
        self.state.reset_interaction_transients()

    def _resize_clutch_active(
        self,
        all_hands,
        selected_sprite,
        frame_width: int,
        frame_height: int,
        now: float,
    ) -> bool:
        if selected_sprite is None or not is_two_hand_resize(all_hands):
            self.state.clear_resize_clutch()
            return False

        pinch_threshold = PINCH_START_DISTANCE * self.state.hand_scale_factor
        both_pinching = all(
            is_pinching(hand, frame_width, frame_height, pinch_threshold)
            for hand in all_hands[:2]
        )
        if not both_pinching:
            self.state.clear_resize_clutch()
            return False

        if self.state.resize_clutch_started_at is None:
            self.state.resize_clutch_started_at = now
            self.state.set_status(
                "Hold both pinches to resize", now, RESIZE_CLUTCH_SECONDS + 0.45
            )
            return False

        return now - self.state.resize_clutch_started_at >= RESIZE_CLUTCH_SECONDS

    def _pending_gesture_label(self) -> str | None:
        state = self.state
        with self.sprites_lock:
            dragging = any(sprite.dragging for sprite in self.sprites)
        if dragging:
            return "Drag"
        if state.resizing_sprite_ref is not None:
            return "Resize"
        if state.fist_start_time is not None:
            return "Sprite hold"
        if state.hover_target_id is not None:
            return "Button hold"
        if (
            state.color_toggle_hold_started_at is not None
            and not state.color_toggle_triggered
        ):
            return "Color hold"
        return None

    def _handle_no_hands_detected(
        self, now: float | None = None, announce: bool = True
    ) -> None:
        now = time.monotonic() if now is None else now
        pending = self._pending_gesture_label() if announce else None
        if self.drawing_canvas is not None:
            with self.canvas_lock:
                self.drawing_canvas.reset_stroke()
        self.state.reset_for_hand_loss()
        self.state.clear_draw_pose()
        self.state.set_tracking_stability(0.0)
        self.state.tracking_status_text = "Waiting for hand"
        self.state.tracking_recovery_hint = "Place your hand in view"
        with self.sprites_lock:
            for sprite in self.sprites:
                sprite.dragging = False
        if pending is not None:
            self.notifier.push(
                f"Hand lost - {pending.lower()} cancelled", "warn", now, key="hand-lost"
            )
        self.state.set_activity(STATUS_IDLE, now)

    def _handle_help_overlay(
        self, hand_landmarks, layout: UiLayout, width: int, height: int, now: float
    ) -> tuple[int, int] | None:
        """Help is open: only its Close button accepts a dwell.

        Every drawing and sprite gesture stays suspended, but the pointer keeps
        tracking so the overlay can be dismissed by pointing at Close rather
        than only with H or Esc.
        """
        # Suspend drawing and sprite gestures WITHOUT touching hover state:
        # _suspend_gestures() clears the dwell timer, so calling it per frame
        # here would reset the Close dwell before it could ever complete.
        if self.drawing_canvas is not None:
            with self.canvas_lock:
                self.drawing_canvas.reset_stroke()
        self.state.clear_drawing_path()
        self.state.clear_draw_pose()
        with self.sprites_lock:
            for sprite in self.sprites:
                sprite.dragging = False
        if hand_landmarks is None:
            self.state.clear_hover()
            self.state.hover_latched_id = None
            self._update_tracking_stability(None)
            return None
        point = get_index_fingertip(hand_landmarks, width, height)
        self._update_tracking_stability(point)
        candidate = (
            HELP_CLOSE_ITEM_ID
            if point is not None
            and point_in_any_rect(
                point,
                [expand_rect(layout.help_close_rect, layout.px(10))],
            )
            else None
        )
        if self.ui_controller.consume_hover(self.state, candidate, now):
            self.dispatcher.dispatch(Command(Action.TOGGLE_HELP, None, "gesture"))
        return point

    def _selected_sprite(self):
        with self.sprites_lock:
            return get_selected_sprite(self.sprites)

    def _disabled_toolbar_ids(self) -> set[str]:
        """Buttons that cannot act right now: greyed out and not dwell-able."""
        disabled: set[str] = set()
        if not self.scene_history.can_undo():
            disabled.add("toolbar:undo")
        if not self.scene_history.can_redo():
            disabled.add("toolbar:redo")
        if self.drawing_canvas is None or not self.drawing_canvas.has_content():
            disabled.add("toolbar:clear")
        if self._selected_sprite() is None:
            disabled.add("toolbar:delete")
        if self.drawing_canvas is None:
            disabled.update({"toolbar:export", "toolbar:save_project"})
        if not Path(config.DEFAULT_PROJECT_PATH).with_suffix(".npz").exists():
            disabled.add("toolbar:load_project")
        return disabled

    def _handle_hand_interactions(
        self,
        hand_landmarks,
        layout: UiLayout,
        frame_width: int,
        frame_height: int,
        now: float | None = None,
    ) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
        now = time.monotonic() if now is None else now
        self.state.clear_resize()
        current_point = None
        raw_fingertip = get_index_fingertip(hand_landmarks, frame_width, frame_height)
        self._update_tracking_stability(raw_fingertip)
        if raw_fingertip is not None:
            previous_raw = None
            if self.state.prev_smoothed_point is not None:
                previous_raw = (
                    round(self.state.prev_smoothed_point[0]),
                    round(self.state.prev_smoothed_point[1]),
                )
            movement_distance = (
                get_fingertip_distance(previous_raw, raw_fingertip)
                if previous_raw is not None
                else 0.0
            )
            smoothed = adaptive_smooth_point(
                self.state.prev_smoothed_point,
                raw_fingertip,
                movement_distance,
            )
            current_point = (int(smoothed[0]), int(smoothed[1]))
            self.state.prev_smoothed_point = smoothed

        scaled_pinch_start = PINCH_START_DISTANCE * self.state.hand_scale_factor
        scaled_pinch_release = PINCH_RELEASE_DISTANCE * self.state.hand_scale_factor
        is_currently_pinching = is_pinching(
            hand_landmarks, frame_width, frame_height, scaled_pinch_start
        )
        pinch_distance = None
        if raw_fingertip is not None:
            thumb_tip = get_landmark_point(
                hand_landmarks, THUMB_TIP, frame_width, frame_height
            )
            pinch_distance = get_fingertip_distance(thumb_tip, raw_fingertip)

        with self.sprites_lock:
            dragging_sprite = next(
                (sprite for sprite in self.sprites if sprite.dragging), None
            )
        if (
            dragging_sprite
            and pinch_distance is not None
            and pinch_distance > scaled_pinch_release
        ):
            with self.sprites_lock:
                dragging_sprite.dragging = False

        # Hysteresis: once a drag is underway, the looser release distance
        # keeps it alive so tracking noise does not drop the sprite.
        pinch_active = is_currently_pinching or (
            dragging_sprite is not None
            and pinch_distance is not None
            and pinch_distance <= scaled_pinch_release
        )
        pinch_started = pinch_active and not self.state.previous_pinch_active

        # Priority 1: toolbar / palette dwell (never while pinching or mid-stroke).
        hover_point = raw_fingertip or current_point
        hover_candidate = self.ui_controller.resolve_hover_candidate(
            hover_point,
            pinch_active,
            self.state.interaction_mode,
            self.state.stroke_active(),
            layout.toolbar_items,
            layout.palette_items,
            self._disabled_toolbar_ids(),
        )
        pointer_over_controls = point_in_any_rect(hover_point, layout.control_rects)

        triggered_id = self.ui_controller.consume_hover(
            self.state, hover_candidate, now
        )
        if triggered_id:
            command = command_for_ui_item(triggered_id, "gesture")
            if command is not None:
                self.dispatcher.dispatch(command)

        # Priority 2: mode-specific gestures, never over the controls.
        if (
            self.state.interaction_mode == config.MODE_SELECT
            and current_point is not None
            and not pointer_over_controls
        ):
            hit_point = raw_fingertip or current_point
            with self.sprites_lock:
                self.sprite_controller.handle_select_mode(
                    state=self.state,
                    sprites=self.sprites,
                    thumbnail_items=layout.thumbnail_items,
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
        elif self.state.interaction_mode == config.MODE_SELECT:
            self.state.hovered_sprite_ref = None

        if self.state.interaction_mode == config.MODE_DRAW:
            with self.canvas_lock, self.sprites_lock:
                self.drawing_controller.handle_draw_mode(
                    state=self.state,
                    hand_landmarks=hand_landmarks,
                    current_point=current_point,
                    raw_fingertip=raw_fingertip,
                    pinch_active=pinch_active,
                    drawing_canvas=self.drawing_canvas,
                    sprites=self.sprites,
                    now=now,
                    ui_rects=layout.draw_block_rects,
                )
        else:
            with self.canvas_lock:
                self.drawing_canvas.reset_stroke()
            self.state.clear_drawing_path()
            self.state.clear_fist_hold()

        self.state.previous_pinch_active = pinch_active
        return current_point, raw_fingertip

    # --- Rendering ----------------------------------------------------------

    def _hand_chip(self, now: float) -> tuple[str, str, str]:
        if self._tracking_error_at is not None and now - self._tracking_error_at < 2.0:
            return ("HAND", "error", "error")
        if not self.state.hand_present:
            return ("HAND", "no hand", "off")
        if not self.state.calibrated:
            return ("HAND", "calibrating", "info")
        if self.state.tracking_stability >= TRACKING_STABLE_THRESHOLD:
            return ("HAND", "tracking", "ok")
        if self.state.tracking_stability <= TRACKING_UNSTABLE_THRESHOLD:
            return ("HAND", "unstable", "warn")
        return ("HAND", "hold steady", "info")

    def _services(self) -> dict[str, tuple[str, str]]:
        jarvis = self.jarvis
        return describe_services(
            ai_enabled=self.state.ai_warning_text is None,
            voice_available=getattr(jarvis, "voice_available", False),
            voice_error=getattr(jarvis, "last_error", None),
            voice_listening=jarvis.listening_event.is_set(),
            voice_speaking=jarvis.speaking_event.is_set(),
            transcription_enabled=getattr(jarvis, "transcription_enabled", True),
            wake_word_ready=getattr(jarvis, "wake_word_ready", False),
        )

    def _activity_text(self, now: float) -> str:
        state = self.state
        if not state.calibrated and state.calibration_started_at is not None:
            progress = dwell_progress(
                state.calibration_started_at, CALIBRATION_SECONDS, now
            )
            return f"Calibrating - hold one hand in view ({int(progress * 100)}%)"
        if not state.hand_present:
            return "Show one hand to the camera"
        if state.status_text == STATUS_IDLE:
            if state.interaction_mode == config.MODE_DRAW:
                return "Raise your index finger to draw"
            if self._selected_sprite() is not None:
                return "Pinch and hold to drag - pinch empty space to deselect"
            return "Pinch a sprite to select it"
        return state.status_text

    def _status_lines(self) -> list[tuple[str, str]]:
        width = self.state.frame_dimensions["width"]
        height = self.state.frame_dimensions["height"]
        camera = (
            f"{self._camera_size[0]}x{self._camera_size[1]}"
            if self._camera_size
            else "-"
        )
        services = self._services()
        jarvis = self.jarvis
        voice_detail = services["voice"][0]
        voice_error = getattr(jarvis, "last_error", None)
        if voice_error and not getattr(jarvis, "voice_available", True):
            voice_detail = voice_error
        with self.sprites_lock:
            sprite_count = len(self.sprites)
        return [
            ("Camera", f"{camera} -> {width}x{height} workspace"),
            ("Dropped", f"{self.diagnostics.capture_failures} frames"),
            (
                "Tracking",
                f"{int(self.state.tracking_stability * 100)}%  hand scale {self.state.hand_scale_factor:.2f}",
            ),
            ("Voice", voice_detail),
            ("Wake word", getattr(jarvis, "wake_word_label", "-")),
            (
                "Transcribe",
                "cloud (Google) on"
                if getattr(jarvis, "transcription_enabled", True)
                else "off",
            ),
            (
                "AI",
                config.GEMINI_MODEL
                if self.state.ai_warning_text is None
                else "off - no API key",
            ),
            ("History", f"{len(self.scene_history)} of {MAX_SCENE_HISTORY} steps"),
            ("Sprites", str(sprite_count)),
        ]

    def _draw_gesture_feedback(
        self,
        frame: np.ndarray,
        layout: UiLayout,
        current_point,
        is_resize_mode: bool,
        now: float,
    ) -> None:
        state = self.state
        mode = state.interaction_mode
        if state.resize_clutch_started_at is not None and not is_resize_mode:
            selected = self._selected_sprite()
            if selected is not None:
                center = (int(selected.center[0]), int(selected.center[1]))
                draw_gesture_progress(
                    frame, layout, center,
                    dwell_progress(state.resize_clutch_started_at, RESIZE_CLUTCH_SECONDS, now),
                    "Hold both pinches to resize", theme.SIGNAL_SELECT,
                )  # fmt: skip
            return
        if is_resize_mode:
            scale_factor = None
            if state.resize_anchor_distance and state.prev_resize_distance:
                scale_factor = state.prev_resize_distance / state.resize_anchor_distance
            draw_resize_anchor(
                frame, layout, state.resize_anchor_first_point,
                state.resize_anchor_second_point, scale_factor,
            )  # fmt: skip
            return
        if current_point is None:
            return
        if state.hover_target_id is not None:
            label = (
                state.hover_target_id.split(":", 1)[1]
                .replace("_mode", "")
                .replace("_", " ")
            )
            draw_gesture_progress(
                frame, layout, current_point,
                dwell_progress(state.hover_start_time, config.dwell_seconds_for_mode(mode), now),
                f"Hold: {label.title()}", theme.mode_color(mode),
            )  # fmt: skip
        elif mode == config.MODE_DRAW and state.fist_start_time is not None:
            draw_gesture_progress(
                frame, layout, current_point,
                dwell_progress(state.fist_start_time, config.FINISH_HOLD_SECONDS, now),
                "Hold fist: make sprite", theme.SIGNAL_DRAW,
            )  # fmt: skip
        elif (
            mode == config.MODE_DRAW
            and state.color_toggle_hold_started_at is not None
            and not state.color_toggle_triggered
        ):
            draw_gesture_progress(
                frame, layout, current_point,
                dwell_progress(state.color_toggle_hold_started_at, config.COLOR_TOGGLE_GESTURE_SECONDS, now),
                "Hold pinch: switch color", theme.SIGNAL_DRAW,
            )  # fmt: skip
        elif (
            mode == config.MODE_SELECT
            and state.pinch_started_at is not None
            and state.pinch_target_sprite_ref is not None
        ):
            with self.sprites_lock:
                dragging = any(sprite.dragging for sprite in self.sprites)
            if not dragging:
                draw_gesture_progress(
                    frame, layout, current_point,
                    dwell_progress(state.pinch_started_at, config.DRAG_CLUTCH_SECONDS, now),
                    "Hold to drag", theme.SIGNAL_SELECT,
                )  # fmt: skip

    def _draw_sprite_annotations(self, frame: np.ndarray) -> None:
        if self.state.interaction_mode != config.MODE_SELECT:
            return
        with self.sprites_lock:
            sprites = list(self.sprites)
        hovered = delete_pending = None
        for sprite in sprites:
            if id(sprite) == self.state.hovered_sprite_ref:
                hovered = sprite
            if id(sprite) == self.state.delete_pending_sprite_ref:
                delete_pending = sprite
            label = None
            if sprite.dragging or (
                self._mouse_drag and self._mouse_drag.sprite is sprite
            ):
                label = "Moving"
            elif id(sprite) == self.state.resizing_sprite_ref:
                label = f"{sprite.w} x {sprite.h}"
            draw_sprite_selection(frame, sprite, label)
        if hovered is not None and delete_pending is not hovered:
            draw_sprite_candidate(frame, hovered)
        if delete_pending is not None:
            draw_sprite_delete_confirmation(frame, delete_pending)

    def _render_output(
        self,
        frame: np.ndarray,
        layout: UiLayout,
        current_point,
        hand_landmarks,
        is_resize_mode: bool,
        now: float,
    ) -> np.ndarray:
        state = self.state
        mode = state.interaction_mode
        height, width = frame.shape[:2]

        if hand_landmarks is not None and not state.help_visible:
            draw_hand_skeleton(
                frame,
                [
                    get_landmark_point(hand_landmarks, index, width, height)
                    for index in range(21)
                ],
            )
        self._draw_sprite_annotations(frame)
        draw_mode_frame(frame, layout, mode)

        dwell_ratio = (
            dwell_progress(
                state.hover_start_time, config.dwell_seconds_for_mode(mode), now
            )
            if state.hover_target_id
            else 0.0
        )
        draw_palette(
            frame, layout, state.brush_name, state.hover_target_id, dwell_ratio, mode
        )
        draw_layers(frame, layout, state.hover_target_id)
        draw_toolbar(
            frame,
            layout,
            mode,
            state.hover_target_id,
            dwell_ratio,
            self._disabled_toolbar_ids(),
        )

        brush_color = config.get_brush_color(state.brush_name) or config.DRAW_COLOR
        draw_pointer(
            frame,
            layout,
            current_point,
            mode,
            brush_color,
            LINE_THICKNESS,
            state.tracking_stability,
        )
        if not state.help_visible:
            self._draw_gesture_feedback(
                frame, layout, current_point, is_resize_mode, now
            )

        selected = self._selected_sprite()
        if mode == config.MODE_DRAW:
            detail = state.brush_name.title()
            swatch = config.get_brush_color(state.brush_name)
        else:
            detail = "Sprite selected" if selected is not None else "No selection"
            swatch = None
        services = self._services()
        chips = [
            self._hand_chip(now),
            ("VOICE", *services["voice"]),
            ("AI", *services["ai"]),
        ]
        draw_top_bar(
            frame,
            layout,
            mode,
            detail,
            swatch,
            self._activity_text(now),
            chips,
            state.fps,
        )

        jarvis = self.jarvis
        draw_voice_card(
            frame, layout,
            jarvis.listening_event.is_set(), jarvis.speaking_event.is_set(),
            getattr(jarvis, "last_transcript", None), getattr(jarvis, "last_response", None),
            getattr(jarvis, "last_update_at", None), now,
        )  # fmt: skip
        draw_toasts(frame, layout, self.notifier.active(now), now)

        if state.guide_visible and not state.help_visible:
            draw_guide(
                frame, layout,
                help_content.quick_start_steps(state.guide_drew, state.guide_made_sprite, state.guide_selected),
            )  # fmt: skip
        if state.settings_visible and not state.help_visible:
            draw_status_panel(frame, layout, self._status_lines())
        if state.help_visible:
            close_hovered = state.hover_target_id == HELP_CLOSE_ITEM_ID
            draw_help(
                frame, layout,
                help_content.help_columns(
                    getattr(jarvis, "wake_word_label", "hey jarvis"),
                    getattr(jarvis, "transcription_enabled", True),
                ),
                help_content.HELP_FOOTER,
                close_hovered,
                dwell_progress(
                    state.hover_start_time,
                    config.dwell_seconds_for_mode(mode),
                    now,
                ) if close_hovered else 0.0,
            )  # fmt: skip
        return frame

    def _flush_events(self, now: float) -> None:
        for text, kind in self.state.drain_events():
            self.notifier.push(text, kind, now)
        if self.state.guide_visible and self.state.guide_complete():
            self.state.guide_visible = False
            self.notifier.push(
                "Quick start complete - press H for every control", "success", now
            )

    def _update_fps(self, now: float) -> None:
        if self._last_tick is not None:
            interval = now - self._last_tick
            if interval > 0:
                instant = 1.0 / interval
                self.state.fps = (
                    instant
                    if self.state.fps <= 0
                    else 0.9 * self.state.fps + 0.1 * instant
                )
        self._last_tick = now

    def run_iteration(self, capture: CaptureDevice) -> bool:
        loop_start = time.perf_counter()
        success, frame = capture.read()
        if not success or frame is None:
            self.diagnostics.record_capture_failure("Frame read failed.")
            return False

        now = time.monotonic()
        self._update_fps(now)
        self._camera_size = (frame.shape[1], frame.shape[0])
        frame = mirror_frame(frame)
        frame = enhance_low_light(frame)
        frame = scale_to_workspace(frame)
        self._ensure_canvas(frame)

        frame_height, frame_width = frame.shape[:2]
        self.state.frame_dimensions["width"] = frame_width
        self.state.frame_dimensions["height"] = frame_height

        # Voice commands queued since the last frame run here, on this thread.
        self.command_queue.drain(self.dispatcher)

        results = self._process_hand_tracking(frame)
        all_hands = (
            list(results.multi_hand_landmarks)
            if results and getattr(results, "multi_hand_landmarks", None)
            else []
        )
        hand_landmarks = select_primary_hand(
            all_hands, self.state.prev_smoothed_point, frame_width, frame_height
        )
        self.state.hand_present = hand_landmarks is not None
        self._update_calibration(hand_landmarks, now)

        with self.sprites_lock:
            sprite_snapshot = sorted(self.sprites, key=lambda item: item.z_index)
        layout = compute_layout(frame_width, frame_height, sprite_snapshot)
        self._layout = layout

        current_point = None
        selected_sprite = self._selected_sprite()
        gestures_suspended = self.state.help_visible
        is_resize_mode = (
            not gestures_suspended
            and self.state.interaction_mode == config.MODE_SELECT
            and self._resize_clutch_active(
                all_hands, selected_sprite, frame_width, frame_height, now
            )
        )
        if gestures_suspended:
            current_point = self._handle_help_overlay(
                hand_landmarks, layout, frame_width, frame_height, now
            )
        elif not self.state.calibrated:
            if hand_landmarks is None:
                self._handle_no_hands_detected(now)
            else:
                self.state.status_text = "Calibrating"
                self._update_tracking_stability(
                    get_index_fingertip(hand_landmarks, frame_width, frame_height)
                )
        elif is_resize_mode:
            self._handle_resize_mode(
                all_hands=all_hands,
                frame_width=frame_width,
                frame_height=frame_height,
                selected_sprite=selected_sprite,
            )
        elif hand_landmarks is not None:
            current_point, _ = self._handle_hand_interactions(
                hand_landmarks, layout, frame_width, frame_height, now
            )
        else:
            self._handle_no_hands_detected(now)
        self._flush_events(now)

        with self.sprites_lock:
            sprite_snapshot = sorted(self.sprites, key=lambda item: item.z_index)
        for sprite in sprite_snapshot:
            overlay_sprite(frame, sprite)
        with self.canvas_lock:
            scene = self.drawing_canvas.overlay_on(frame, ALPHA)
        with self.frame_lock:
            self.state.latest_scene_frame = scene

        output_frame = self._render_output(
            scene.copy(), layout, current_point, hand_landmarks, is_resize_mode, now
        )
        with self.frame_lock:
            self.state.latest_frame = output_frame

        self.diagnostics.record_frame(
            time.perf_counter() - loop_start, self.state.status_text
        )
        self._record_scene_if_changed()
        return True

    def _close_resources(self, capture) -> None:
        self.command_queue.close()
        try:
            self.jarvis.stop()
        except Exception as exc:  # noqa: BLE001 - shutdown must continue
            logger.warning("Voice shutdown failed: %s", exc)
        if self._hands is not None:
            try:
                self._hands.close()
            except Exception as exc:  # noqa: BLE001 - shutdown must continue
                logger.warning("Hand tracker shutdown failed: %s", exc)
        if capture is not None:
            capture.release()
        if hasattr(cv2, "destroyAllWindows"):
            cv2.destroyAllWindows()

    def run(self) -> None:
        self._owner_thread = threading.current_thread()
        self._setup_window()

        cap = self._initialize_capture()
        if cap is None:
            return

        try:
            self._hands = self._initialize_hands()
        except Exception as exc:  # noqa: BLE001 - optional hand-tracking backend
            cap.release()
            self._show_message_screen(
                "Hand tracking could not start.",
                f"MediaPipe Hands is unavailable ({type(exc).__name__}).",
            )
            return

        consecutive_failures = 0
        try:
            # Voice is optional: a failed start must not take the workspace
            # down, and the finally still releases camera/tracker/window.
            try:
                self.jarvis.start()
            except Exception as exc:  # noqa: BLE001 - optional voice backend
                logger.warning("Voice assistant could not start: %s", exc)
                self.jarvis.voice_available = False
                self.jarvis.last_error = (
                    f"Voice disabled: assistant failed to start ({type(exc).__name__})."
                )
                self.notifier.push(
                    "Voice unavailable - drawing and keyboard still work",
                    "warn",
                    time.monotonic(),
                    key="voice-start",
                )
            while True:
                if not self.run_iteration(cap):
                    consecutive_failures += 1
                    if consecutive_failures >= config.CAPTURE_MAX_CONSECUTIVE_FAILURES:
                        self._show_message_screen(
                            "The camera stopped sending frames.",
                            "Reconnect the camera, then restart ARIA.",
                        )
                        break
                    if self._wait_for_exit(15):
                        break
                    continue
                consecutive_failures = 0

                output_frame = self.get_current_frame()
                if output_frame is not None:
                    self._show_frame(output_frame)

                remaining = max(
                    0.0,
                    self.frame_interval
                    - (self.diagnostics.last_frame_latency_ms / 1000.0),
                )
                wait_ms = max(1, int(remaining * 1000))
                if self._wait_for_exit(wait_ms):
                    break

                self.state.expire_status(time.monotonic())
        except KeyboardInterrupt:
            logger.info("Interrupted; shutting down.")
        finally:
            self._close_resources(cap)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    AppRunner().run()
