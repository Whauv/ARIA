DRAW_COLOR = (0, 255, 0)
ACTIVE_BRUSH_COLOR = DRAW_COLOR
STATUS_COLOR = (255, 255, 255)
STATUS_BG_COLOR = (30, 30, 30)
MODE_BG_COLOR = (20, 60, 20)
SELECTED_BORDER_COLOR = (0, 255, 255)
VOICE_IDLE_COLOR = (180, 180, 180)
VOICE_LISTENING_COLOR = (255, 180, 40)
VOICE_SPEAKING_COLOR = (255, 255, 255)
VOICE_PULSE_COLOR = (255, 120, 40)

LINE_THICKNESS = 5
ALPHA = 0.75
SELECTION_BORDER_THICKNESS = 2

THUMB_TIP = 4
INDEX_TIP = 8
INDEX_PIP = 6
MIDDLE_TIP = 12
MIDDLE_PIP = 10
RING_TIP = 16
RING_PIP = 14
PINKY_TIP = 20
PINKY_PIP = 18

SMOOTHING_PREV_WEIGHT = 0.7
SMOOTHING_NEW_WEIGHT = 0.3
DRAG_SMOOTHING_PREV_WEIGHT = 0.6
DRAG_SMOOTHING_NEW_WEIGHT = 0.4

CLEAR_HOLD_SECONDS = 1.0
FINISH_HOLD_SECONDS = 1.5
TARGET_FPS = 30
PINCH_START_DISTANCE = 40
PINCH_RELEASE_DISTANCE = 50
MIN_SPRITE_SIZE = 30
MAX_SPRITE_SIZE = 600

STATUS_DRAWING = "Drawing"
STATUS_PAUSED = "Paused"
STATUS_CLEARED = "Cleared"
STATUS_IDLE = "Idle"
STATUS_SPRITE_CREATED = "Sprite Created"

BRUSH_COLORS = {
    "black": (0, 0, 0),
    "blue": (255, 0, 0),
    "cyan": (255, 255, 0),
    "green": (0, 255, 0),
    "orange": (0, 165, 255),
    "pink": (203, 192, 255),
    "purple": (255, 0, 255),
    "red": (0, 0, 255),
    "white": (255, 255, 255),
    "yellow": (0, 255, 255),
}


def get_active_brush_color() -> tuple[int, int, int]:
    return ACTIVE_BRUSH_COLOR


def set_active_brush_color(color_name: str) -> tuple[int, int, int] | None:
    global ACTIVE_BRUSH_COLOR

    color = BRUSH_COLORS.get(color_name.lower())
    if color is None:
        return None

    ACTIVE_BRUSH_COLOR = color
    return color
