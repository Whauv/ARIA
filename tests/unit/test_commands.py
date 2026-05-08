from __future__ import annotations

import unittest
from unittest.mock import patch

from aria import config
from aria.assistant import jarvis
from aria.assistant.command_intents import CommandIntent, parse_command
from aria.assistant.jarvis import JarvisAssistant, JarvisContext, command_handler


class CommandIntentTests(unittest.TestCase):
    def test_parse_color_change_command(self) -> None:
        parsed = parse_command("change color to red")
        self.assertEqual(parsed.intent, CommandIntent.CHANGE_BRUSH_COLOR)
        self.assertEqual(parsed.color_name, "red")

    def test_parse_scale_commands(self) -> None:
        self.assertEqual(parse_command("make it bigger").scale_factor, 1.5)
        self.assertEqual(parse_command("make it smaller").scale_factor, 0.5)

    def test_unknown_command(self) -> None:
        parsed = parse_command("launch rockets")
        self.assertEqual(parsed.intent, CommandIntent.UNKNOWN)


class CommandHandlerTests(unittest.TestCase):
    def make_context(self) -> JarvisContext:
        return JarvisContext(
            get_current_frame=lambda: None,
            clear_canvas=lambda: None,
            delete_selected_sprite=lambda: False,
            scale_selected_sprite=lambda factor: factor > 1.0,
            undo_last_stroke=lambda: True,
            save_snapshot=lambda path: True,
            set_brush_color=lambda color_name: color_name == "red",
            set_interaction_mode=lambda mode_name: mode_name in {"draw_mode", "select_mode"},
        )

    def test_command_handler_routes_palette_change(self) -> None:
        response = command_handler("change color to red", self.make_context())
        self.assertIn("Changed the brush color to red", response)

    def test_command_handler_handles_unknown_intent(self) -> None:
        response = command_handler("launch rockets", self.make_context())
        self.assertIn("do not know how to handle it", response)


class JarvisAssistantTests(unittest.TestCase):
    def make_context(self) -> JarvisContext:
        return JarvisContext(
            get_current_frame=lambda: None,
            clear_canvas=lambda: None,
            delete_selected_sprite=lambda: False,
            scale_selected_sprite=lambda factor: False,
            undo_last_stroke=lambda: False,
            save_snapshot=lambda path: False,
            set_brush_color=lambda color_name: False,
            set_interaction_mode=lambda mode_name: False,
        )

    def test_capture_command_respects_cloud_speech_toggle(self) -> None:
        assistant = JarvisAssistant(self.make_context())
        original_value = config.ALLOW_CLOUD_SPEECH_RECOGNITION
        try:
            config.ALLOW_CLOUD_SPEECH_RECOGNITION = False
            with patch.object(jarvis, "sr", object()):
                self.assertIsNone(assistant._capture_command())
            self.assertIn("disabled by configuration", assistant.last_error or "")
        finally:
            config.ALLOW_CLOUD_SPEECH_RECOGNITION = original_value


if __name__ == "__main__":
    unittest.main()
