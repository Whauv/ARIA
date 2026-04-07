from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import cv2

import config
from app_runner import AppRunner
from canvas import DrawingCanvas


HAS_CV2_IMWRITE = hasattr(cv2, "imwrite")


class AppRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = AppRunner(window_name="ARIA Test")

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
    @patch("app_runner.cv2.imwrite", return_value=True)
    def test_save_snapshot_uses_current_frame(self, mocked_imwrite) -> None:
        self.runner.state.latest_frame = np.zeros((5, 5, 3), dtype=np.uint8)

        saved = self.runner.save_snapshot("snapshot.png")

        self.assertTrue(saved)
        mocked_imwrite.assert_called_once()


if __name__ == "__main__":
    unittest.main()
