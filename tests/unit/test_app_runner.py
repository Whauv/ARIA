from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import cv2

from aria import config
from aria.app_runner import AppRunner
from aria.assistant.jarvis import JarvisContext
from aria.drawing.canvas import DrawingCanvas


HAS_CV2_IMWRITE = hasattr(cv2, "imwrite")


class FakeCapture:
    def __init__(self, frames: list[np.ndarray], opened: bool = True) -> None:
        self.frames = list(frames)
        self.opened = opened
        self.released = False

    def isOpened(self) -> bool:
        return self.opened

    def read(self) -> tuple[bool, np.ndarray | None]:
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def release(self) -> None:
        self.released = True


class FakeHandTracker:
    def __init__(self, results: object | Exception | None = None) -> None:
        self.results = results if results is not None else SimpleNamespace(multi_hand_landmarks=[])
        self.closed = False

    def process(self, frame: np.ndarray) -> object:
        if isinstance(self.results, Exception):
            raise self.results
        return self.results

    def close(self) -> None:
        self.closed = True


class FakeAssistant:
    def __init__(self, context: JarvisContext) -> None:
        self.context = context
        self.started = False
        self.stopped = False
        self.voice_available = True
        self.last_error = None
        self.listening_event = Mock()
        self.speaking_event = Mock()
        self.listening_event.is_set.return_value = False
        self.speaking_event.is_set.return_value = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True


class AppRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = AppRunner(window_name="ARIA Test", assistant_factory=FakeAssistant)

    def test_get_current_frame_returns_copy(self) -> None:
        frame = np.ones((8, 8, 3), dtype=np.uint8)
        self.runner.state.latest_frame = frame

        copied = self.runner.get_current_frame()

        self.assertIsNot(copied, frame)
        copied[0, 0, 0] = 99
        self.assertNotEqual(int(frame[0, 0, 0]), 99)

    def test_set_brush_color_updates_canvas_and_state(self) -> None:
        self.runner.drawing_canvas = DrawingCanvas((20, 20, 3))

        changed = self.runner.set_brush_color("red")

        self.assertTrue(changed)
        self.assertEqual(self.runner.state.brush_name, "red")
        self.assertEqual(self.runner.drawing_canvas.brush_color, config.BRUSH_COLORS["red"])

    @unittest.skipUnless(HAS_CV2_IMWRITE, "Full OpenCV bindings are not available in this Python environment")
    @patch("aria.app_runner.cv2.imwrite", return_value=True)
    def test_save_snapshot_uses_current_frame(self, mocked_imwrite) -> None:
        self.runner.state.latest_frame = np.zeros((5, 5, 3), dtype=np.uint8)

        saved = self.runner.save_snapshot("snapshot.png")

        self.assertTrue(saved)
        mocked_imwrite.assert_called_once()

    def test_run_iteration_updates_latest_frame_and_diagnostics(self) -> None:
        capture = FakeCapture([np.zeros((24, 32, 3), dtype=np.uint8)])
        self.runner._hands = FakeHandTracker()

        with patch.object(AppRunner, "_render_output", side_effect=lambda frame, *_args: frame.copy()):
            processed = self.runner.run_iteration(capture)

        self.assertTrue(processed)
        self.assertIsNotNone(self.runner.state.latest_frame)
        self.assertEqual(self.runner.diagnostics.frames_processed, 1)
        self.assertGreaterEqual(self.runner.state.frame_dimensions["width"], 1)

    def test_run_iteration_records_hand_tracking_failures(self) -> None:
        capture = FakeCapture([np.zeros((20, 20, 3), dtype=np.uint8)])
        self.runner._hands = FakeHandTracker(RuntimeError("tracker offline"))

        with patch.object(AppRunner, "_render_output", side_effect=lambda frame, *_args: frame.copy()):
            processed = self.runner.run_iteration(capture)

        self.assertTrue(processed)
        self.assertEqual(self.runner.diagnostics.hand_tracking_failures, 1)
        self.assertIn("tracker offline", self.runner.diagnostics.last_error or "")

    @patch("aria.app_runner.cv2.destroyAllWindows", create=True)
    @patch.object(AppRunner, "_wait_for_exit", side_effect=[True])
    @patch.object(AppRunner, "_show_frame")
    @patch.object(AppRunner, "_setup_window")
    @patch.object(AppRunner, "_render_output", side_effect=lambda frame, *_args: frame.copy())
    def test_run_starts_and_stops_runtime_services(
        self,
        _render_output,
        _setup_window,
        mocked_show_frame,
        _wait_for_exit,
        _destroy_windows,
    ) -> None:
        capture = FakeCapture([np.zeros((18, 18, 3), dtype=np.uint8)])
        hand_tracker = FakeHandTracker()
        runner = AppRunner(
            window_name="ARIA Test",
            capture_factory=lambda: capture,
            hand_tracker_factory=lambda: hand_tracker,
            assistant_factory=FakeAssistant,
        )

        runner.run()

        self.assertTrue(runner.jarvis.started)
        self.assertTrue(runner.jarvis.stopped)
        self.assertTrue(hand_tracker.closed)
        self.assertTrue(capture.released)
        mocked_show_frame.assert_called()


if __name__ == "__main__":
    unittest.main()
