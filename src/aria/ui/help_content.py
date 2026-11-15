"""Text for the in-app help reference and quick start.

Values are derived from config so the help can never drift from the real
thresholds. Keep README.md and landing/ in sync when this changes.
"""

from __future__ import annotations

from .. import config
from ..runtime.commands import KEYBOARD_REFERENCE

Entry = tuple[str, str]


def _seconds(value: float) -> str:
    return f"{value:g}s"


def gesture_entries() -> list[Entry]:
    return [
        ("#DRAW MODE", ""),
        ("Index finger up", "draw"),
        ("Index + middle up", "lift pen"),
        ("Hold fist", f"{_seconds(config.FINISH_HOLD_SECONDS)} - make sprite"),
        (
            "Pinch and hold",
            f"{_seconds(config.COLOR_TOGGLE_GESTURE_SECONDS)} - "
            + " / ".join(config.QUICK_TOGGLE_COLORS),
        ),
        ("#SELECT MODE", ""),
        ("Pinch sprite", "select (topmost wins)"),
        ("Keep pinching", "drag, release to place"),
        ("Pinch empty space", "deselect"),
        ("Pinch a layer", "select + bring to front"),
        ("Pinch 3x quickly", "delete (undoable)"),
        ("Both hands pinch", "resize around center"),
        ("#ANY MODE", ""),
        (
            "Point at a button",
            (
                f"hold {_seconds(config.UI_DRAW_MODE_DWELL_SECONDS)} draw / "
                f"{_seconds(config.UI_SELECT_MODE_DWELL_SECONDS)} select"
            ),
        ),
        ("Every toolbar action", "works by pointing"),
        ("Greyed out button", "not available yet"),
    ]


def voice_entries(wake_word: str, transcription_enabled: bool) -> list[Entry]:
    entries: list[Entry] = [
        ("Say first", f'"{wake_word}"'),
        ("what did I draw", "Gemini, sends 1 frame"),
        ("change color to red", "any palette color"),
        ("clear canvas", "clear drawing"),
        ("undo / redo", "scene history"),
        ("save", "export PNG"),
        ("delete", "remove selection"),
        ("make it bigger", "or smaller"),
        ("draw mode", "or select mode"),
    ]
    if not transcription_enabled:
        entries.append(("#COMMANDS OFF", ""))
        entries.append(("Transcription", "disabled in .env"))
    return entries


def keyboard_entries() -> list[Entry]:
    entries: list[Entry] = list(KEYBOARD_REFERENCE)
    entries.append(("#MOUSE", ""))
    entries.append(("Click", "toolbar, colors, layers"))
    entries.append(("Select mode", "click / drag sprites"))
    entries.append(("Wheel", "resize selected sprite"))
    return entries


def help_columns(
    wake_word: str, transcription_enabled: bool
) -> list[tuple[str, list[Entry]]]:
    return [
        ("GESTURES", gesture_entries()),
        ("VOICE", voice_entries(wake_word, transcription_enabled)),
        ("KEYS + MOUSE", keyboard_entries()),
    ]


HELP_FOOTER = "Other gestures pause while help is open  |  Point at Close, or H / Esc"


def quick_start_steps(
    drew: bool, made_sprite: bool, selected: bool
) -> list[tuple[str, str, bool]]:
    return [
        ("1  Draw", "Raise your index finger", drew),
        (
            "2  Make a sprite",
            f"Hold a fist {_seconds(config.FINISH_HOLD_SECONDS)}",
            made_sprite,
        ),
        ("3  Move it", "Point at Select, pinch + hold", selected),
    ]
