from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .. import config


class CommandIntent(str, Enum):
    DESCRIBE_SCENE = "describe_scene"
    CHANGE_BRUSH_COLOR = "change_brush_color"
    CLEAR_CANVAS = "clear_canvas"
    DELETE_SELECTION = "delete_selection"
    SCALE_SELECTION = "scale_selection"
    UNDO = "undo"
    SAVE = "save"
    SWITCH_MODE = "switch_mode"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ParsedCommand:
    intent: CommandIntent
    raw_text: str
    color_name: str | None = None
    scale_factor: float | None = None
    mode_name: str | None = None


def parse_command(command: str) -> ParsedCommand:
    normalized = command.lower().strip()

    if "what did i draw" in normalized or "what is this" in normalized:
        return ParsedCommand(intent=CommandIntent.DESCRIBE_SCENE, raw_text=normalized)

    if "change color to" in normalized:
        color_phrase = normalized.split("change color to", 1)[1].strip()
        for supported_color in config.BRUSH_COLORS:
            if supported_color in color_phrase:
                return ParsedCommand(
                    intent=CommandIntent.CHANGE_BRUSH_COLOR,
                    raw_text=normalized,
                    color_name=supported_color,
                )
        return ParsedCommand(intent=CommandIntent.CHANGE_BRUSH_COLOR, raw_text=normalized)

    if normalized in {"clear", "clear canvas"}:
        return ParsedCommand(intent=CommandIntent.CLEAR_CANVAS, raw_text=normalized)

    if normalized in {"delete", "remove it"}:
        return ParsedCommand(intent=CommandIntent.DELETE_SELECTION, raw_text=normalized)

    if normalized in {"make it bigger", "enlarge"}:
        return ParsedCommand(intent=CommandIntent.SCALE_SELECTION, raw_text=normalized, scale_factor=1.5)

    if normalized in {"make it smaller", "shrink"}:
        return ParsedCommand(intent=CommandIntent.SCALE_SELECTION, raw_text=normalized, scale_factor=0.5)

    if normalized == "undo":
        return ParsedCommand(intent=CommandIntent.UNDO, raw_text=normalized)

    if normalized == "save":
        return ParsedCommand(intent=CommandIntent.SAVE, raw_text=normalized)

    select_mode_phrases = {
        "select mode",
        "switch to select",
        "go to select",
        "select tool",
        "selection mode",
    }
    draw_mode_phrases = {
        "draw mode",
        "drawing mode",
        "switch to draw",
        "go to draw",
        "draw tool",
    }

    if any(phrase in normalized for phrase in select_mode_phrases):
        return ParsedCommand(intent=CommandIntent.SWITCH_MODE, raw_text=normalized, mode_name="select_mode")

    if any(phrase in normalized for phrase in draw_mode_phrases):
        return ParsedCommand(intent=CommandIntent.SWITCH_MODE, raw_text=normalized, mode_name="draw_mode")

    return ParsedCommand(intent=CommandIntent.UNKNOWN, raw_text=normalized)
