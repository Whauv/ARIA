from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

import numpy as np

from aria import config
from aria.drawing.sprite import Sprite
from aria.runtime.controllers import (
    DrawingInteractionController,
    SpriteInteractionController,
    UIActionCallbacks,
    UIInteractionController,
    build_warning_text,
    clear_sprite_selection,
    get_selected_sprite,
)
from aria.runtime.state import RuntimeState


class UIInteractionControllerTests(unittest.TestCase):
    def test_hover_candidate_is_blocked_while_drawing_stroke_is_active(self) -> None:
        controller = UIInteractionController()
        toolbar_items = [{"id": "toolbar:draw_mode", "action": "draw_mode", "label": "Draw Mode", "rect": (0, 0, 100, 40)}]

        candidate = controller.resolve_hover_candidate(
            hover_point=(20, 20),
            pinch_active=False,
            interaction_mode="draw_mode",
            stroke_active=True,
            toolbar_items=toolbar_items,
            palette_items=[],
        )

        self.assertIsNone(candidate)

    def test_hover_then_dwell_triggers_action(self) -> None:
        state = RuntimeState()
        controller = UIInteractionController()

        first = controller.consume_hover(state, "toolbar:draw_mode", 10.0)
        second = controller.consume_hover(state, "toolbar:draw_mode", 10.6)

        self.assertIsNone(first)
        self.assertEqual(second, "toolbar:draw_mode")
        self.assertIsNone(state.hover_target_id)

    def test_apply_palette_action_updates_brush_state(self) -> None:
        state = RuntimeState()
        controller = UIInteractionController()
        callbacks = UIActionCallbacks(
            set_brush_color=lambda color_name: color_name == "red",
            clear_canvas=lambda: None,
            save_snapshot=lambda path: True,
            undo_last_stroke=lambda: True,
        )

        controller.apply_action(state, "palette:red", callbacks, now=5.0)

        self.assertEqual(state.brush_name, "red")
        self.assertIn("Red", state.status_text)


class SpriteInteractionControllerTests(unittest.TestCase):
    def make_sprite(self, x: int, y: int, z_index: int) -> Sprite:
        image = np.zeros((20, 20, 4), dtype=np.uint8)
        return Sprite(
            original_img=image.copy(),
            img=image.copy(),
            x=x,
            y=y,
            w=20,
            h=20,
            dragging=False,
            z_index=z_index,
            selected=False,
        )

    def test_select_mode_prefers_topmost_hovered_sprite(self) -> None:
        state = RuntimeState()
        controller = SpriteInteractionController()
        low = self.make_sprite(10, 10, 1)
        high = self.make_sprite(12, 12, 2)
        sprites = [low, high]

        controller.handle_select_mode(
            state=state,
            sprites=sprites,
            thumbnail_items=[],
            hit_point=(15, 15),
            current_point=(15, 15),
            pinch_active=False,
            pinch_started=False,
            frame_width=200,
            frame_height=200,
            now=1.0,
        )

        self.assertTrue(high.selected)
        self.assertFalse(low.selected)

    def test_dragging_updates_selected_sprite_position(self) -> None:
        state = RuntimeState()
        controller = SpriteInteractionController()
        sprite = self.make_sprite(10, 10, 1)
        sprite.selected = True
        sprites = [sprite]

        controller.handle_select_mode(
            state=state,
            sprites=sprites,
            thumbnail_items=[],
            hit_point=(25, 25),
            current_point=(40, 50),
            pinch_active=True,
            pinch_started=True,
            frame_width=200,
            frame_height=200,
            now=2.0,
        )

        self.assertTrue(sprite.dragging)
        self.assertTrue(sprite.selected)
        self.assertNotEqual((sprite.x, sprite.y), (10, 10))


class DrawingInteractionControllerTests(unittest.TestCase):
    def test_draw_mode_adds_segment_when_index_only_up(self) -> None:
        state = RuntimeState()
        state.hover_target_id = None
        canvas = Mock()
        controller = DrawingInteractionController()

        with patch("aria.runtime.controllers.is_closed_fist", return_value=False), patch(
            "aria.runtime.controllers.is_index_only_up", return_value=True
        ), patch("aria.runtime.controllers.is_index_and_middle_up", return_value=False):
            controller.handle_draw_mode(
                state=state,
                hand_landmarks=object(),
                current_point=(30, 40),
                raw_fingertip=(30, 40),
                pinch_active=False,
                drawing_canvas=canvas,
                sprites=[],
                now=1.0,
                ui_rects=[],
            )

        canvas.add_segment.assert_called_once_with((30, 40), (30, 40))
        self.assertEqual(state.prev_draw_point, (30, 40))
        self.assertEqual(state.status_text, config.STATUS_DRAWING)

    def test_draw_mode_grace_period_keeps_stroke_alive_for_one_noisy_frame(self) -> None:
        state = RuntimeState()
        state.hover_target_id = None
        state.last_draw_pose_time = 1.0
        state.prev_draw_point = (30, 40)
        canvas = Mock()
        controller = DrawingInteractionController()

        with patch("aria.runtime.controllers.is_closed_fist", return_value=False), patch(
            "aria.runtime.controllers.is_draw_pose", return_value=False
        ), patch("aria.runtime.controllers.is_index_and_middle_up", return_value=False):
            controller.handle_draw_mode(
                state=state,
                hand_landmarks=object(),
                current_point=(34, 44),
                raw_fingertip=(34, 44),
                pinch_active=False,
                drawing_canvas=canvas,
                sprites=[],
                now=1.1,
                ui_rects=[],
            )

        canvas.add_segment.assert_called_once_with((30, 40), (34, 44))
        self.assertEqual(state.prev_draw_point, (34, 44))
        self.assertEqual(state.status_text, config.STATUS_DRAWING)

    def test_draw_mode_ignores_tiny_motion_jitter(self) -> None:
        state = RuntimeState()
        state.hover_target_id = None
        state.prev_draw_point = (30, 40)
        canvas = Mock()
        controller = DrawingInteractionController()

        with patch("aria.runtime.controllers.is_closed_fist", return_value=False), patch(
            "aria.runtime.controllers.is_draw_pose", return_value=True
        ), patch("aria.runtime.controllers.is_index_and_middle_up", return_value=False):
            controller.handle_draw_mode(
                state=state,
                hand_landmarks=object(),
                current_point=(32, 42),
                raw_fingertip=(32, 42),
                pinch_active=False,
                drawing_canvas=canvas,
                sprites=[],
                now=1.0,
                ui_rects=[],
            )

        canvas.add_path.assert_not_called()
        canvas.add_segment.assert_not_called()
        self.assertEqual(state.prev_draw_point, (30, 40))

    def test_draw_mode_interpolates_large_motion_into_path(self) -> None:
        state = RuntimeState()
        state.hover_target_id = None
        state.prev_draw_point = (10, 10)
        canvas = Mock()
        controller = DrawingInteractionController()

        with patch("aria.runtime.controllers.is_closed_fist", return_value=False), patch(
            "aria.runtime.controllers.is_draw_pose", return_value=True
        ), patch("aria.runtime.controllers.is_index_and_middle_up", return_value=False):
            controller.handle_draw_mode(
                state=state,
                hand_landmarks=object(),
                current_point=(40, 10),
                raw_fingertip=(40, 10),
                pinch_active=False,
                drawing_canvas=canvas,
                sprites=[],
                now=1.0,
                ui_rects=[],
            )

        canvas.add_path.assert_called_once()
        drawn_path = canvas.add_path.call_args.args[0]
        self.assertEqual(drawn_path[0], (10, 10))
        self.assertEqual(drawn_path[-1], (40, 10))
        self.assertGreater(len(drawn_path), 2)
        self.assertEqual(state.prev_draw_point, (40, 10))


class RuntimeUtilityTests(unittest.TestCase):
    def test_warning_text_combines_flags(self) -> None:
        self.assertEqual(build_warning_text("AI disabled", False, "voice failed"), "AI off | Voice off")
        self.assertEqual(build_warning_text(None, True, None), None)

    def test_selection_helpers(self) -> None:
        sprite = Sprite(np.zeros((5, 5, 4), dtype=np.uint8), np.zeros((5, 5, 4), dtype=np.uint8), 0, 0, 5, 5, True, 1, True)
        other = Sprite(np.zeros((5, 5, 4), dtype=np.uint8), np.zeros((5, 5, 4), dtype=np.uint8), 0, 0, 5, 5, True, 2, False)
        sprites = [sprite, other]

        clear_sprite_selection(sprites)

        self.assertIsNone(get_selected_sprite(sprites))
        self.assertFalse(sprite.dragging)


if __name__ == "__main__":
    unittest.main()
