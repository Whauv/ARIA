"""Behavioral tests for input routing, gestures, feedback, layout and rendering.

Hardware, clocks and network are faked; nothing here needs a camera,
microphone, API key or download.
"""

from __future__ import annotations

import os
import tempfile
import threading
import time
import unittest
from itertools import pairwise
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from aria import config
from aria.app_runner import AppRunner, fit_window_size, scale_to_workspace
from aria.assistant import jarvis
from aria.assistant.jarvis import JarvisAssistant, JarvisContext, command_handler
from aria.drawing.canvas import DrawingCanvas
from aria.drawing.sprite import Sprite, fit_sprite_dimensions
from aria.runtime.commands import (
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
from aria.runtime.controllers import (
    DrawingInteractionController,
    SpriteInteractionController,
)
from aria.runtime.notifications import Notifier
from aria.runtime.state import RuntimeState
from aria.ui.layout import compute_layout
from aria.vision.gestures import select_primary_hand


def make_sprite(x: int, y: int, w: int = 40, h: int = 30, z: int = 0) -> Sprite:
    image = np.zeros((h, w, 4), dtype=np.uint8)
    image[:, :, 1] = 255
    image[:, :, 3] = 255
    return Sprite(image.copy(), image.copy(), x, y, w, h, False, z, False)


class FakeLandmark:
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y


def make_hand(tip_x: float = 0.5, tip_y: float = 0.4, pose: str = "point"):
    """Synthetic MediaPipe-style hand. ``pose``: point | fist | pinch."""
    landmarks = [FakeLandmark(tip_x, tip_y + 0.2) for _ in range(21)]
    for tip, pip in (
        (config.MIDDLE_TIP, config.MIDDLE_PIP),
        (config.RING_TIP, config.RING_PIP),
        (config.PINKY_TIP, config.PINKY_PIP),
    ):
        landmarks[pip] = FakeLandmark(tip_x + 0.03, tip_y + 0.1)
        landmarks[tip] = FakeLandmark(tip_x + 0.03, tip_y + 0.13)
    landmarks[config.INDEX_PIP] = FakeLandmark(tip_x, tip_y + 0.1)
    landmarks[config.INDEX_TIP] = FakeLandmark(tip_x, tip_y)
    landmarks[config.THUMB_TIP] = FakeLandmark(tip_x - 0.2, tip_y + 0.1)
    if pose == "fist":
        landmarks[config.INDEX_TIP] = FakeLandmark(tip_x, tip_y + 0.13)
    if pose == "pinch":
        landmarks[config.THUMB_TIP] = FakeLandmark(tip_x + 0.005, tip_y)
    return SimpleNamespace(landmark=landmarks)


class FakeCapture:
    def __init__(self, frames) -> None:
        self.frames = list(frames)
        self.released = False

    def isOpened(self) -> bool:
        return True

    def read(self):
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def release(self) -> None:
        self.released = True


class FakeTracker:
    def __init__(self, hands=None) -> None:
        self.hands = hands or []
        self.closed = False

    def process(self, _frame):
        return SimpleNamespace(multi_hand_landmarks=list(self.hands))

    def close(self) -> None:
        self.closed = True


class FakeAssistant:
    def __init__(self, context: JarvisContext) -> None:
        self.context = context
        self.voice_available = True
        self.last_error = None
        self.listening_event = threading.Event()
        self.speaking_event = threading.Event()
        self.started = self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True


def make_runner(shape=(720, 1280, 3)) -> AppRunner:
    runner = AppRunner(window_name="ARIA Test", assistant_factory=FakeAssistant)
    runner.drawing_canvas = DrawingCanvas(shape)
    runner.state.frame_dimensions.update(width=shape[1], height=shape[0])
    runner._record_scene()
    return runner


class CommandMappingTests(unittest.TestCase):
    def test_keyboard_map(self) -> None:
        self.assertEqual(command_for_key(ord("d")).action, Action.MODE_DRAW)
        self.assertEqual(command_for_key(ord("V")).action, Action.MODE_SELECT)
        self.assertEqual(command_for_key(8).action, Action.DELETE)
        self.assertEqual(command_for_key(27).action, Action.CANCEL)
        self.assertEqual(command_for_key(ord("1")).arg, config.PALETTE_ORDER[0])
        self.assertEqual(command_for_key(ord("e")).action, Action.EXPORT_PNG)
        self.assertGreater(command_for_key(ord("+")).arg, 1.0)
        self.assertLess(command_for_key(ord("-")).arg, 1.0)
        self.assertIsNone(command_for_key(255))
        self.assertIsNone(command_for_key(ord("z")))
        self.assertTrue(is_quit_key(ord("q")))

    def test_every_toolbar_and_palette_item_maps_to_a_command(self) -> None:
        layout = compute_layout(1280, 720)
        for item in layout.toolbar_items + layout.palette_items:
            self.assertIsNotNone(command_for_ui_item(item["id"], "gesture"), item["id"])
        self.assertEqual(
            command_for_ui_item("palette:red", "mouse"),
            Command(Action.BRUSH, "red", "mouse"),
        )

    def test_every_keyboard_action_is_also_reachable_by_pointing(self) -> None:
        """Pointing is a full alternative to the keyboard, not a subset.

        SCALE and CANCEL are deliberately absent: resizing is the two-hand
        pinch gesture and cancel/deselect is a pinch on empty space, so both
        already have a pointing route that is not a button.
        """
        from aria.runtime.commands import _KEY_ACTIONS

        layout = compute_layout(1280, 720)
        on_screen = {
            command_for_ui_item(item["id"], "gesture").action
            for item in layout.toolbar_items + layout.palette_items
        }
        keyboard_actions = {action for action, _arg in _KEY_ACTIONS.values()}
        gesture_native = {Action.SCALE, Action.CANCEL}
        missing = keyboard_actions - on_screen - gesture_native
        self.assertEqual(missing, set(), f"keyboard-only actions: {missing}")

    def test_toolbar_buttons_fit_without_overlapping_at_every_size(self) -> None:
        for width, height in ((960, 720), (1280, 720), (1920, 1080), (1280, 960)):
            layout = compute_layout(width, height)
            rects = [item["rect"] for item in layout.toolbar_items]
            self.assertEqual(len(rects), len(config.TOOLBAR_ACTIONS))
            for left, _top, right, _bottom in rects:
                self.assertGreaterEqual(left, 0, (width, height))
                self.assertLessEqual(right, width, (width, height))
                self.assertGreater(right, left)
            for earlier, later in pairwise(rects):
                self.assertLessEqual(earlier[2], later[0], (width, height))


class CommandQueueTests(unittest.TestCase):
    def test_dispatcher_survives_handler_errors(self) -> None:
        seen = []
        dispatcher = CommandDispatcher(
            {Action.UNDO: Mock(side_effect=RuntimeError("boom"))},
            on_result=lambda command, result: seen.append(result),
        )
        result = dispatcher.dispatch(Command(Action.UNDO))
        self.assertFalse(result.ok)
        self.assertEqual(result.kind, "error")
        self.assertEqual(seen, [result])
        self.assertFalse(dispatcher.dispatch(Command(Action.REDO)).ok)

    def test_queue_is_bounded_and_resolves_on_drain(self) -> None:
        queue = CommandQueue(maxsize=1)
        first = queue.submit(Command(Action.UNDO))
        overflow = queue.submit(Command(Action.UNDO))
        self.assertFalse(overflow.result(timeout=0).ok)
        dispatcher = CommandDispatcher(
            {Action.UNDO: lambda c: CommandResult.info("ok")}
        )
        self.assertEqual(queue.drain(dispatcher), 1)
        self.assertTrue(first.result(timeout=0).ok)

    def test_timeout_and_close_never_hang(self) -> None:
        queue = CommandQueue()
        future = queue.submit(Command(Action.UNDO))
        self.assertFalse(wait_for_result(future, 0.01).ok)
        pending = queue.submit(Command(Action.REDO))
        queue.close()
        self.assertFalse(pending.result(timeout=0).ok)
        self.assertFalse(queue.submit(Command(Action.UNDO)).result(timeout=0).ok)


class NotifierTests(unittest.TestCase):
    def test_bounded_deduplicated_and_expiring(self) -> None:
        notifier = Notifier(max_items=2)
        notifier.push("a", now=0.0)
        notifier.push("b", now=0.0)
        notifier.push("c", now=0.0)
        self.assertEqual([t.text for t in notifier.active(0.1)], ["b", "c"])
        notifier.push("Nothing to undo", "warn", 0.0, key="undo")
        notifier.push("Nothing to undo", "warn", 0.5, key="undo")
        self.assertEqual(sum(t.key == "undo" for t in notifier.active(0.6)), 1)
        self.assertEqual(notifier.active(100.0), [])


class DrawGestureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.controller = DrawingInteractionController()
        self.state = RuntimeState()
        self.canvas = DrawingCanvas((120, 160, 3))

    def _fist(self, now: float, sprites) -> None:
        with patch("aria.runtime.controllers.is_closed_fist", return_value=True):
            self.controller.handle_draw_mode(
                self.state, object(), (50, 50), (50, 50), True, self.canvas,
                sprites, now, [],
            )  # fmt: skip

    def test_sustained_fist_creates_exactly_one_sprite(self) -> None:
        self.canvas.add_segment((10, 10), (60, 40))
        self.canvas.reset_stroke()
        sprites: list = []
        for step in range(60):
            self._fist(step * 0.1, sprites)
        self.assertEqual(len(sprites), 1)
        self.assertTrue(sprites[0].selected)
        self.assertIn(
            ("Sprite created - press V to move it", "success"), list(self.state.events)
        )

    def test_fist_on_empty_canvas_warns_instead_of_claiming_success(self) -> None:
        sprites: list = []
        self._fist(0.0, sprites)
        self._fist(config.FINISH_HOLD_SECONDS + 0.01, sprites)
        self.assertEqual(sprites, [])
        kinds = [kind for _text, kind in self.state.drain_events()]
        self.assertEqual(kinds, ["warn"])

    def test_fist_wins_over_pinch_color_toggle(self) -> None:
        self.state.brush_name = "green"
        sprites: list = []
        for step in range(15):
            self._fist(step * 0.1, sprites)
        self.assertEqual(self.state.brush_name, "green")
        self.assertIsNone(self.state.color_toggle_hold_started_at)

    def test_fist_rearms_after_hand_opens(self) -> None:
        sprites: list = []
        self._fist(0.0, sprites)
        self._fist(2.0, sprites)
        self.assertTrue(self.state.fist_latched)
        with (
            patch("aria.runtime.controllers.is_closed_fist", return_value=False),
            patch("aria.runtime.controllers.is_draw_pose", return_value=False),
            patch(
                "aria.runtime.controllers.is_index_and_middle_up", return_value=False
            ),
        ):
            self.controller.handle_draw_mode(
                self.state,
                object(),
                (5, 5),
                (5, 5),
                False,
                self.canvas,
                sprites,
                2.1,
                [],
            )
        self.assertFalse(self.state.fist_latched)


class SelectGestureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.controller = SpriteInteractionController()
        self.state = RuntimeState(interaction_mode="select_mode")

    def _step(self, sprites, point, pinch, started, now, thumbs=()):
        self.controller.handle_select_mode(
            self.state, sprites, list(thumbs), point, point, pinch, started,
            640, 480, now,
        )  # fmt: skip

    def test_pinch_on_empty_space_deselects(self) -> None:
        sprite = make_sprite(10, 10)
        sprite.selected = True
        self._step([sprite], (300, 300), True, True, 1.0)
        self.assertFalse(sprite.selected)

    def test_sweeping_a_held_pinch_never_grabs_another_sprite(self) -> None:
        sprite = make_sprite(200, 200)
        sprites = [sprite]
        self._step(sprites, (20, 20), True, True, 1.0)
        for step in range(1, 10):
            self._step(sprites, (210, 210), True, False, 1.0 + step * 0.1)
        self.assertFalse(sprite.dragging)
        self.assertEqual((sprite.x, sprite.y), (200, 200))

    def test_overlap_selects_topmost(self) -> None:
        low, high = make_sprite(10, 10, z=1), make_sprite(20, 20, z=5)
        self._step([low, high], (30, 30), True, True, 1.0)
        self.assertTrue(high.selected)
        self.assertFalse(low.selected)

    def test_thumbnail_pinch_selects_and_brings_to_front(self) -> None:
        low, high = make_sprite(10, 10, z=1), make_sprite(300, 300, z=5)
        thumbs = [{"id": "thumb", "rect": (600, 100, 640, 140), "sprite": low}]
        self._step([low, high], (620, 120), True, True, 1.0, thumbs)
        self.assertTrue(low.selected)
        self.assertGreater(low.z_index, high.z_index)


class GeometryTests(unittest.TestCase):
    def test_thin_artwork_keeps_aspect_and_never_hits_zero(self) -> None:
        width, height = fit_sprite_dimensions(300, 5, 10)
        self.assertEqual(width, config.MIN_SPRITE_SIZE)
        self.assertGreaterEqual(height, 1)
        self.assertLess(height, 5)
        width, height = fit_sprite_dimensions(100, 50, 100000)
        self.assertEqual(width, config.MAX_SPRITE_SIZE)
        self.assertEqual(height, config.MAX_SPRITE_SIZE // 2)
        self.assertEqual(fit_sprite_dimensions(0, 0, 0), (1, 1))

    def test_resize_keeps_center(self) -> None:
        sprite = make_sprite(100, 100, 40, 20)
        center = sprite.center
        sprite.resize_to_dimensions(80)
        self.assertEqual((sprite.w, sprite.h), (80, 40))
        self.assertAlmostEqual(sprite.center[0], center[0], delta=1)
        self.assertAlmostEqual(sprite.center[1], center[1], delta=1)
        self.assertEqual(sprite.img.shape[:2], (40, 80))

    def test_contains_point_respects_bounds_and_alpha(self) -> None:
        sprite = make_sprite(10, 10, 20, 20)
        sprite.img[:, :10, 3] = 0
        self.assertTrue(sprite.contains_point((25, 15)))
        self.assertFalse(sprite.contains_point((12, 15)))
        self.assertFalse(sprite.contains_point((30, 15)))  # right edge is exclusive

    def test_primary_hand_follows_previous_pointer(self) -> None:
        left, right = make_hand(0.2, 0.4), make_hand(0.8, 0.4)
        self.assertIs(select_primary_hand([left, right], (790, 280), 1000, 700), right)
        self.assertIs(select_primary_hand([right, left], (190, 280), 1000, 700), left)
        self.assertIs(select_primary_hand([left, right], None, 1000, 700), left)
        self.assertIsNone(select_primary_hand([], None, 10, 10))

    def test_workspace_and_window_scaling_preserve_aspect(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        scaled = scale_to_workspace(frame)
        self.assertEqual(scaled.shape[:2], (720, 960))
        big = np.zeros((1080, 1920, 3), dtype=np.uint8)
        self.assertIs(scale_to_workspace(big), big)
        self.assertEqual(fit_window_size(960, 720, 1400, 900), (1200, 900))
        self.assertEqual(fit_window_size(1920, 1080, 1400, 900), (1400, 788))


class LayoutTests(unittest.TestCase):
    RESOLUTIONS = ((960, 720), (1280, 720), (1920, 1080), (720, 960), (1440, 720))

    def test_controls_stay_in_frame_and_do_not_overlap(self) -> None:
        sprites = [make_sprite(0, 0, z=index) for index in range(40)]
        for width, height in self.RESOLUTIONS:
            layout = compute_layout(width, height, sprites)
            rects = layout.palette_rects + layout.toolbar_rects + layout.thumbnail_rects
            for left, top, right, bottom in rects:
                self.assertTrue(0 <= left < right <= width, (width, height))
                self.assertTrue(0 <= top < bottom <= height, (width, height))
            for index, first in enumerate(rects):
                for second in rects[index + 1 :]:
                    overlap = not (
                        first[2] <= second[0] or second[2] <= first[0]
                        or first[3] <= second[1] or second[3] <= first[1]
                    )  # fmt: skip
                    self.assertFalse(overlap, (width, height, first, second))
            for rect in layout.thumbnail_rects:
                self.assertLessEqual(rect[3], layout.toolbar_rect[1])
            self.assertEqual(
                len(layout.thumbnail_items) + layout.hidden_layer_count, len(sprites)
            )


class AppCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = make_runner((240, 320, 3))

    def test_equivalent_inputs_produce_the_same_result(self) -> None:
        for trigger in (
            lambda: self.runner.handle_key(ord("v")),
            lambda: self.runner.dispatcher.dispatch(
                command_for_ui_item("toolbar:select_mode", "gesture")
            ),
            lambda: self.runner.jarvis.context.set_interaction_mode("select_mode"),
        ):
            self.runner.set_interaction_mode("draw_mode")
            trigger()
            self.assertEqual(self.runner.state.interaction_mode, "select_mode")

    def test_voice_from_background_thread_runs_on_render_thread(self) -> None:
        result: dict = {}
        worker = threading.Thread(
            target=lambda: result.update(
                ok=self.runner.jarvis.context.set_brush_color("red")
            )
        )
        worker.start()
        for _ in range(200):
            self.runner.command_queue.drain(self.runner.dispatcher)
            if not worker.is_alive():
                break
            worker.join(0.01)
        worker.join(1)
        self.assertTrue(result["ok"])
        self.assertEqual(self.runner.state.brush_name, "red")

    def test_delete_is_recoverable_with_undo(self) -> None:
        sprite = make_sprite(20, 20)
        sprite.selected = True
        self.runner.sprites.append(sprite)
        self.runner._record_scene()
        result = self.runner.dispatcher.dispatch(Command(Action.DELETE))
        self.assertTrue(result.ok)
        self.assertIn("undo", result.message.lower())
        self.assertEqual(self.runner.sprites, [])
        self.runner.handle_key(ord("u"))
        self.assertEqual(len(self.runner.sprites), 1)
        self.runner.handle_key(ord("r"))
        self.assertEqual(self.runner.sprites, [])

    def test_delete_works_with_several_differently_sized_sprites(self) -> None:
        """Regression: Sprite must not be compared by value.

        A value-comparing dataclass compares the numpy image fields, so
        ``sprites.remove(target)`` raised ValueError ("operands could not be
        broadcast together") whenever the target was not the first sprite.
        """
        first = make_sprite(10, 10, w=7, h=13)
        second = make_sprite(80, 80, w=182, h=235)
        third = make_sprite(150, 150, w=40, h=30)
        self.runner.sprites.extend([first, second, third])
        second.selected = True
        self.runner._record_scene()

        result = self.runner.dispatcher.dispatch(Command(Action.DELETE))

        self.assertTrue(result.ok, result.message)
        self.assertEqual(len(self.runner.sprites), 2)
        self.assertTrue(all(s is not second for s in self.runner.sprites))
        self.runner.handle_key(ord("u"))
        self.assertEqual(len(self.runner.sprites), 3)

    def test_gesture_delete_with_mixed_sprite_sizes(self) -> None:
        """The gesture path deletes through the controller, not the command."""
        state = RuntimeState()
        sprites = [make_sprite(0, 0, w=7, h=13), make_sprite(60, 60, w=182, h=235)]
        target = sprites[1]
        SpriteInteractionController._delete(state, sprites, target)
        self.assertEqual(len(sprites), 1)
        self.assertTrue(all(s is not target for s in sprites))

    def test_sprite_identity_lookups_never_compare_images(self) -> None:
        small = make_sprite(0, 0, w=7, h=13)
        large = make_sprite(0, 0, w=182, h=235)
        same_shape = make_sprite(0, 0, w=7, h=13)
        # Each of these raised ValueError under value equality.
        self.assertIn(large, [small, large])
        self.assertNotIn(same_shape, [small, large])
        self.assertNotEqual(small, same_shape)
        self.assertEqual(small, small)

    def test_selection_and_mode_changes_do_not_create_history(self) -> None:
        sprite = make_sprite(20, 20)
        self.runner.sprites.append(sprite)
        self.runner._record_scene_if_changed()
        depth = len(self.runner.scene_history)
        sprite.selected = True
        self.runner.set_interaction_mode("select_mode")
        self.runner.set_brush_color("red")
        for _ in range(5):
            self.runner._record_scene_if_changed()
        self.assertEqual(len(self.runner.scene_history), depth)
        sprite.x += 10
        self.runner._record_scene_if_changed()
        self.assertEqual(len(self.runner.scene_history), depth + 1)

    def test_undo_keeps_current_mode_and_brush(self) -> None:
        self.runner.drawing_canvas.add_segment((5, 5), (50, 50))
        self.runner.drawing_canvas.reset_stroke()
        self.runner._record_scene_if_changed()
        self.runner.set_brush_color("red")
        self.runner.set_interaction_mode("select_mode")
        self.assertTrue(self.runner.undo_last_stroke())
        self.assertEqual(self.runner.state.brush_name, "red")
        self.assertEqual(self.runner.state.interaction_mode, "select_mode")
        self.assertFalse(self.runner.drawing_canvas.has_content())

    def test_clear_and_failures_report_honestly(self) -> None:
        result = self.runner.dispatcher.dispatch(Command(Action.CLEAR))
        self.assertIn("already empty", result.message)
        self.assertFalse(self.runner.dispatcher.dispatch(Command(Action.UNDO)).ok)
        self.runner.state.latest_scene_frame = np.zeros((4, 4, 3), dtype=np.uint8)
        with patch("aria.app_runner.cv2.imwrite", return_value=False):
            result = self.runner.dispatcher.dispatch(Command(Action.EXPORT_PNG))
        self.assertFalse(result.ok)
        self.assertNotIn("Saved", result.message)
        messages = [
            toast.text for toast in self.runner.notifier.active(time.monotonic())
        ]
        self.assertTrue(any("Could not save" in text for text in messages))

    def test_export_uses_clean_scene_frame(self) -> None:
        self.runner.state.latest_frame = np.full((4, 4, 3), 9, dtype=np.uint8)
        self.runner.state.latest_scene_frame = np.full((4, 4, 3), 1, dtype=np.uint8)
        with patch("aria.app_runner.cv2.imwrite", return_value=True) as imwrite:
            self.assertTrue(self.runner.save_snapshot("x.png"))
        self.assertEqual(int(imwrite.call_args.args[1][0, 0, 0]), 1)

    def test_scale_reports_limits_and_requires_selection(self) -> None:
        self.assertFalse(self.runner.dispatcher.dispatch(Command(Action.SCALE, 2.0)).ok)
        sprite = make_sprite(10, 10, 30, 30)
        sprite.selected = True
        self.runner.sprites.append(sprite)
        result = self.runner.dispatcher.dispatch(Command(Action.SCALE, 0.1))
        self.assertIn("limit", result.message)

    def test_escape_closes_help_then_hides_guide_then_deselects(self) -> None:
        sprite = make_sprite(10, 10)
        sprite.selected = True
        self.runner.sprites.append(sprite)
        self.runner.state.help_visible = True
        self.runner.handle_key(27)
        self.assertFalse(self.runner.state.help_visible)
        self.assertTrue(self.runner.state.guide_visible)
        self.runner.handle_key(27)
        self.assertFalse(self.runner.state.guide_visible)
        self.runner.handle_key(27)
        self.assertFalse(sprite.selected)

    def test_project_round_trip_and_missing_file(self) -> None:
        with tempfile.TemporaryDirectory(dir=os.getcwd()) as directory:
            path = os.path.join(directory, "proj")
            with patch.object(config, "DEFAULT_PROJECT_PATH", path):
                self.assertFalse(
                    self.runner.dispatcher.dispatch(Command(Action.LOAD_PROJECT)).ok
                )
                self.runner.sprites.append(make_sprite(5, 5))
                self.assertTrue(self.runner.save_project(path))
                self.runner.sprites.clear()
                self.assertTrue(
                    self.runner.dispatcher.dispatch(Command(Action.LOAD_PROJECT)).ok
                )
        self.assertEqual(len(self.runner.sprites), 1)


class HandLossTests(unittest.TestCase):
    def test_hand_loss_cancels_pending_gestures_and_says_so(self) -> None:
        runner = make_runner((240, 320, 3))
        sprite = make_sprite(10, 10)
        sprite.dragging = True
        runner.sprites.append(sprite)
        runner.state.fist_start_time = 1.0
        runner.state.hover_target_id = "toolbar:clear"
        runner.state.hover_start_time = 1.0
        runner.state.hover_latched_id = "toolbar:undo"
        runner._handle_no_hands_detected(now=2.0)
        self.assertFalse(sprite.dragging)
        self.assertIsNone(runner.state.fist_start_time)
        self.assertIsNone(runner.state.hover_target_id)
        self.assertIsNone(runner.state.hover_latched_id)
        texts = [toast.text for toast in runner.notifier.active(2.0)]
        self.assertTrue(any("cancelled" in text for text in texts))


class RenderSmokeTests(unittest.TestCase):
    """Run whole frames through the real renderer with synthetic input."""

    def _run(self, shape, hands, setup=None, frames=3):
        runner = AppRunner(
            window_name="ARIA Test",
            hand_tracker_factory=lambda: FakeTracker(hands),
            assistant_factory=FakeAssistant,
        )
        runner._hands = FakeTracker(hands)
        runner.state.calibrated = True
        if setup:
            setup(runner)
        capture = FakeCapture(
            [np.full(shape, 60, dtype=np.uint8) for _ in range(frames)]
        )
        for _ in range(frames):
            self.assertTrue(runner.run_iteration(capture))
        return runner

    def test_renders_across_resolutions_modes_and_panels(self) -> None:
        for shape in ((480, 640, 3), (720, 1280, 3), (1080, 1920, 3), (640, 480, 3)):
            for mode in ("draw_mode", "select_mode"):

                def setup(runner, mode=mode):
                    runner.state.interaction_mode = mode
                    runner.state.settings_visible = True
                    for index in range(30):
                        runner.sprites.append(make_sprite(20 * index, 30, z=index))
                    runner.sprites[0].selected = True

                runner = self._run(shape, [make_hand()], setup)
                output = runner.get_current_frame()
                self.assertGreaterEqual(output.shape[0], config.WORKSPACE_MIN_HEIGHT)
                self.assertEqual(
                    output.shape[:2], runner.state.latest_scene_frame.shape[:2]
                )
                self.assertFalse(
                    np.array_equal(output, runner.state.latest_scene_frame)
                )

    def test_help_overlay_suspends_gestures(self) -> None:
        runner = self._run(
            (480, 640, 3),
            [make_hand()],
            lambda r: setattr(r.state, "help_visible", True),
        )
        self.assertFalse(runner.drawing_canvas.has_content())

    def test_drawing_with_synthetic_hand_marks_canvas(self) -> None:
        hand = make_hand(0.5, 0.45)
        runner = AppRunner(window_name="ARIA Test", assistant_factory=FakeAssistant)
        runner._hands = FakeTracker([hand])
        runner.state.calibrated = True
        capture = FakeCapture(
            [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(8)]
        )
        for step in range(8):
            hand.landmark[config.INDEX_TIP].x = 0.4 + step * 0.02
            runner.run_iteration(capture)
        self.assertTrue(runner.drawing_canvas.has_content())
        self.assertTrue(runner.state.guide_drew)

    def test_shutdown_releases_capture_when_hand_tracking_fails(self) -> None:
        capture = FakeCapture([])
        runner = AppRunner(
            window_name="ARIA Test",
            capture_factory=lambda: capture,
            hand_tracker_factory=Mock(side_effect=RuntimeError("no mediapipe")),
            assistant_factory=FakeAssistant,
        )
        with (
            patch.object(AppRunner, "_setup_window"),
            patch.object(AppRunner, "_show_message_screen") as message,
        ):
            runner.run()
        self.assertTrue(capture.released)
        self.assertFalse(runner.jarvis.started)
        message.assert_called_once()

    def test_voice_start_failure_keeps_workspace_and_releases_resources(self) -> None:
        """Voice is optional: a failed start must not kill the workspace."""

        class FailingAssistant(FakeAssistant):
            def start(self) -> None:
                raise RuntimeError("no audio device")

        capture = FakeCapture([])
        tracker = FakeTracker()
        runner = AppRunner(
            window_name="ARIA Test",
            capture_factory=lambda: capture,
            hand_tracker_factory=lambda: tracker,
            assistant_factory=FailingAssistant,
        )
        with (
            patch.object(AppRunner, "_setup_window"),
            patch.object(AppRunner, "_wait_for_exit", return_value=False),
            patch.object(AppRunner, "_show_message_screen"),
        ):
            runner.run()  # must not raise
        self.assertTrue(capture.released)
        self.assertTrue(tracker.closed)
        self.assertTrue(runner.jarvis.stopped)
        self.assertFalse(runner.jarvis.voice_available)
        texts = [toast.text for toast in runner.notifier.active(time.monotonic())]
        self.assertTrue(any("Voice unavailable" in text for text in texts))

    def test_help_can_be_opened_and_closed_by_pointing(self) -> None:
        """Help must not be a dead end for a user who never touches the keyboard."""
        runner = AppRunner(window_name="ARIA Test", assistant_factory=FakeAssistant)
        runner._hands = FakeTracker([])
        runner.state.calibrated = True
        capture = FakeCapture([np.full((720, 1280, 3), 60, dtype=np.uint8)])
        runner.run_iteration(capture)
        layout = runner._layout
        self.assertIsNotNone(layout)

        # Open it the way a dwell on the Help button does.
        runner.dispatcher.dispatch(command_for_ui_item("toolbar:help", "gesture"))
        self.assertTrue(runner.state.help_visible)

        close_rect = layout.help_close_rect
        centre = (
            (close_rect[0] + close_rect[2]) // 2,
            (close_rect[1] + close_rect[3]) // 2,
        )
        hand = make_hand()
        with patch("aria.app_runner.get_index_fingertip", return_value=centre):
            dwell = config.dwell_seconds_for_mode(runner.state.interaction_mode)
            runner._handle_help_overlay(hand, layout, 1280, 720, now=100.0)
            self.assertTrue(runner.state.help_visible, "should need a full dwell")
            runner._handle_help_overlay(
                hand, layout, 1280, 720, now=100.0 + dwell + 0.1
            )
        self.assertFalse(runner.state.help_visible)

    def test_help_overlay_ignores_a_pointer_away_from_close(self) -> None:
        runner = AppRunner(window_name="ARIA Test", assistant_factory=FakeAssistant)
        runner._hands = FakeTracker([])
        runner.state.calibrated = True
        capture = FakeCapture([np.full((720, 1280, 3), 60, dtype=np.uint8)])
        runner.run_iteration(capture)
        layout = runner._layout
        runner.state.help_visible = True
        with patch("aria.app_runner.get_index_fingertip", return_value=(5, 5)):
            dwell = config.dwell_seconds_for_mode(runner.state.interaction_mode)
            runner._handle_help_overlay(hand := make_hand(), layout, 1280, 720, 100.0)
            runner._handle_help_overlay(hand, layout, 1280, 720, 100.0 + dwell + 0.1)
        self.assertTrue(runner.state.help_visible)

    def test_camera_dropout_is_tolerated_then_reported(self) -> None:
        capture = FakeCapture([])
        tracker = FakeTracker()
        runner = AppRunner(
            window_name="ARIA Test",
            capture_factory=lambda: capture,
            hand_tracker_factory=lambda: tracker,
            assistant_factory=FakeAssistant,
        )
        with (
            patch.object(AppRunner, "_setup_window"),
            patch.object(AppRunner, "_wait_for_exit", return_value=False),
            patch.object(AppRunner, "_show_message_screen") as message,
            patch("aria.app_runner.cv2.destroyAllWindows", create=True),
        ):
            runner.run()
        self.assertEqual(
            runner.diagnostics.capture_failures, config.CAPTURE_MAX_CONSECUTIVE_FAILURES
        )
        message.assert_called_once()
        self.assertTrue(capture.released and tracker.closed and runner.jarvis.stopped)


class AssistantTests(unittest.TestCase):
    def make_context(self, **overrides) -> JarvisContext:
        values = {
            "get_current_frame": lambda: None,
            "clear_canvas": lambda: True,
            "delete_selected_sprite": lambda: False,
            "scale_selected_sprite": lambda factor: False,
            "undo_last_stroke": lambda: False,
            "save_snapshot": lambda path: False,
            "set_brush_color": lambda name: False,
            "set_interaction_mode": lambda mode: False,
        }
        values.update(overrides)
        return JarvisContext(**values)

    def test_redo_and_failed_save_responses(self) -> None:
        self.assertIn("nothing to redo", command_handler("redo", self.make_context()))
        self.assertEqual(
            command_handler("redo", self.make_context(redo_last_scene=lambda: True)),
            "Redone.",
        )
        self.assertIn("could not save", command_handler("save", self.make_context()))

    def test_disabled_transcription_explains_itself_without_listening(self) -> None:
        assistant = JarvisAssistant(self.make_context())
        assistant.transcription_enabled = False
        with patch.object(assistant, "_capture_command") as capture:
            assistant._handle_wake_word()
        capture.assert_not_called()
        self.assertIn("ALLOW_CLOUD_SPEECH_RECOGNITION", assistant.last_response or "")
        self.assertFalse(assistant.listening_event.is_set())

    def test_tts_queue_is_bounded_and_keeps_latest(self) -> None:
        assistant = JarvisAssistant(self.make_context())
        for index in range(10):
            assistant._speak_async(f"reply {index}")
        queued = []
        while not assistant._tts_queue.empty():
            queued.append(assistant._tts_queue.get_nowait())
        self.assertEqual(len(queued), config.TTS_QUEUE_SIZE)
        self.assertEqual(queued[-1], "reply 9")

    def test_stop_joins_worker_threads(self) -> None:
        assistant = JarvisAssistant(self.make_context())
        with patch.object(jarvis, "openwakeword", None):
            assistant.start()
            assistant.stop()
        self.assertFalse(assistant.thread.is_alive())
        self.assertFalse(assistant._tts_thread.is_alive())
        self.assertFalse(assistant.voice_available)

    def test_gemini_failures_do_not_leak_error_details(self) -> None:
        from aria.ai import ai_utils

        failing = Mock()
        failing.Client.return_value.models.generate_content.side_effect = RuntimeError(
            "secret-request-detail"
        )
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        with (
            patch.dict(os.environ, {"GOOGLE_API_KEY": "k"}),
            patch.object(ai_utils, "google_genai", failing),
        ):
            reply = ai_utils.describe_image_with_gemini(frame, "prompt-unique-123")
        self.assertNotIn("secret-request-detail", reply)


if __name__ == "__main__":
    unittest.main()
