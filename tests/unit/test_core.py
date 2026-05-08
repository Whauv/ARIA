from __future__ import annotations

import os
import unittest

import numpy as np
import cv2

from aria import config
from aria.ai import ai_utils
from aria.drawing.canvas import DrawingCanvas, adaptive_smooth_point, interpolate_segment, smooth_point
from aria.drawing.sprite import create_sprite_from_canvas
from aria.vision.gestures import dwell_progress, get_fingertip_distance, is_double_pinch, is_draw_pose, point_in_rect


HAS_FULL_CV2 = all(hasattr(cv2, attr) for attr in ("line", "imencode", "cvtColor"))


@unittest.skipUnless(HAS_FULL_CV2, "Full OpenCV bindings are not available in this Python environment")
class DrawingCanvasTests(unittest.TestCase):
    def test_add_segment_and_undo_restore_previous_snapshot(self) -> None:
        canvas = DrawingCanvas((60, 60, 3))
        canvas.add_segment((5, 5), (20, 20))
        canvas.reset_stroke()

        self.assertEqual(len(canvas.strokes), 1)
        self.assertTrue(np.count_nonzero(canvas.canvas) > 0)

        undone = canvas.undo_last_stroke()

        self.assertTrue(undone)
        self.assertEqual(len(canvas.strokes), 0)
        self.assertEqual(np.count_nonzero(canvas.canvas), 0)

    def test_clear_pushes_snapshot(self) -> None:
        canvas = DrawingCanvas((40, 40, 3))
        canvas.add_segment((1, 1), (10, 10))
        canvas.reset_stroke()

        canvas.clear()

        self.assertEqual(np.count_nonzero(canvas.canvas), 0)
        self.assertGreaterEqual(len(canvas.undo_stack), 2)


@unittest.skipUnless(HAS_FULL_CV2, "Full OpenCV bindings are not available in this Python environment")
class SpriteTests(unittest.TestCase):
    def test_create_sprite_from_canvas_returns_bgra_sprite(self) -> None:
        canvas = np.zeros((80, 80, 3), dtype=np.uint8)
        canvas[10:20, 15:30] = (0, 255, 0)

        sprite = create_sprite_from_canvas(canvas, z_index=3)

        self.assertIsNotNone(sprite)
        assert sprite is not None
        self.assertEqual(sprite.img.shape[2], 4)
        self.assertEqual(sprite.z_index, 3)
        self.assertEqual((sprite.x, sprite.y), (15, 10))


class GestureUtilityTests(unittest.TestCase):
    def make_hand_landmarks(self, y_values: dict[int, float]):
        class Landmark:
            def __init__(self, y: float) -> None:
                self.y = y
                self.x = 0.5

        landmarks = [Landmark(0.5) for _ in range(21)]
        for index, y_value in y_values.items():
            landmarks[index] = Landmark(y_value)

        class HandLandmarks:
            def __init__(self, landmark_list) -> None:
                self.landmark = landmark_list

        return HandLandmarks(landmarks)

    def test_geometry_helpers(self) -> None:
        self.assertAlmostEqual(get_fingertip_distance((0, 0), (3, 4)), 5.0)
        self.assertTrue(point_in_rect((5, 5), (0, 0, 10, 10)))
        self.assertFalse(point_in_rect((11, 5), (0, 0, 10, 10)))
        self.assertTrue(is_double_pinch(1.0, 1.3, 0.5))
        self.assertFalse(is_double_pinch(1.0, 1.6, 0.5))
        self.assertEqual(dwell_progress(10.0, 1.0, 10.5), 0.5)

    def test_draw_pose_allows_slightly_open_non_index_fingers(self) -> None:
        hand_landmarks = self.make_hand_landmarks(
            {
                config.INDEX_TIP: 0.35,
                config.INDEX_PIP: 0.40,
                config.MIDDLE_TIP: 0.405,
                config.MIDDLE_PIP: 0.41,
                config.RING_TIP: 0.42,
                config.RING_PIP: 0.425,
                config.PINKY_TIP: 0.43,
                config.PINKY_PIP: 0.435,
            }
        )

        self.assertTrue(is_draw_pose(hand_landmarks))

    def test_smooth_point(self) -> None:
        self.assertEqual(smooth_point(None, (10, 20), 0.7, 0.3), (10.0, 20.0))
        smoothed = smooth_point((10.0, 10.0), (20, 30), 0.6, 0.4)
        self.assertEqual(smoothed, (14.0, 18.0))

    def test_adaptive_smooth_point_is_more_responsive_for_fast_motion(self) -> None:
        slow = adaptive_smooth_point((10.0, 10.0), (20, 20), distance=4.0)
        fast = adaptive_smooth_point((10.0, 10.0), (20, 20), distance=40.0)

        self.assertGreater(fast[0], slow[0])
        self.assertGreater(fast[1], slow[1])

    def test_interpolate_segment_adds_intermediate_points_for_large_gap(self) -> None:
        path = interpolate_segment((0, 0), (25, 0), 10)

        self.assertEqual(path[0], (0, 0))
        self.assertEqual(path[-1], (25, 0))
        self.assertGreater(len(path), 2)


class AiUtilsTests(unittest.TestCase):
    @unittest.skipUnless(HAS_FULL_CV2, "Full OpenCV bindings are not available in this Python environment")
    def test_frame_to_base64_png_returns_content(self) -> None:
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        encoded = ai_utils.frame_to_base64_png(frame)
        self.assertIsInstance(encoded, str)
        self.assertTrue(len(encoded) > 0)

    def test_gemini_enabled_requires_key_and_sdk(self) -> None:
        original_google = os.environ.get("GOOGLE_API_KEY")
        try:
            os.environ["GOOGLE_API_KEY"] = "test-key"
            expected = ai_utils.google_genai is not None or ai_utils.legacy_genai is not None
            self.assertEqual(ai_utils.gemini_enabled(), expected)
        finally:
            if original_google is None:
                os.environ.pop("GOOGLE_API_KEY", None)
            else:
                os.environ["GOOGLE_API_KEY"] = original_google


class ConfigTests(unittest.TestCase):
    def test_get_brush_color(self) -> None:
        self.assertEqual(config.get_brush_color("red"), config.BRUSH_COLORS["red"])
        self.assertEqual(config.get_brush_color("RED"), config.BRUSH_COLORS["red"])
        self.assertIsNone(config.get_brush_color("missing"))


if __name__ == "__main__":
    unittest.main()
