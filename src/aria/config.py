from __future__ import annotations

import os

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None

# Load .env before any environment-dependent setting below is evaluated.
# Without this, values such as ALLOW_CLOUD_SPEECH_RECOGNITION=0 in .env were
# read before the file was loaded and silently ignored.
if load_dotenv is not None:
    load_dotenv()

DRAW_COLOR = (0, 255, 0)  # BGR fallback brush color
# Visual tokens (colors, type ramp) live in aria/ui/theme.py.
DEFAULT_BRUSH_NAME = "green"
DEFAULT_SAVE_PATH = "aria_snapshot.png"
DEFAULT_PROJECT_PATH = "aria_project.npz"

LINE_THICKNESS = 5
ALPHA = 0.75
DOUBLE_PINCH_SECONDS = 0.5
DRAW_POSE_GRACE_SECONDS = 0.15
MIN_DRAW_MOVEMENT_PX = 4
MAX_SEGMENT_STEP_PX = 10
UI_DRAW_MODE_PADDING = 10
UI_SELECT_MODE_PADDING = 24
UI_DRAW_MODE_DWELL_SECONDS = 0.8
UI_SELECT_MODE_DWELL_SECONDS = 0.55
COLOR_TOGGLE_GESTURE_SECONDS = 0.6
DRAG_CLUTCH_SECONDS = 0.18
RESIZE_CLUTCH_SECONDS = 0.28
DELETE_CONFIRM_SECONDS = 1.5
CALIBRATION_SECONDS = 2.5
BASE_HAND_SPREAD = 0.35
TRACKING_STABLE_THRESHOLD = 0.65
TRACKING_UNSTABLE_THRESHOLD = 0.4

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
SMOOTHING_SLOW_DISTANCE_PX = 8
SMOOTHING_FAST_DISTANCE_PX = 30
SMOOTHING_SLOW_PREV_WEIGHT = 0.82
SMOOTHING_SLOW_NEW_WEIGHT = 0.18
SMOOTHING_FAST_PREV_WEIGHT = 0.45
SMOOTHING_FAST_NEW_WEIGHT = 0.55

CLEAR_HOLD_SECONDS = 1.0
FINISH_HOLD_SECONDS = 1.5
TARGET_FPS = 30
PINCH_START_DISTANCE = 65
PINCH_RELEASE_DISTANCE = 90
MIN_SPRITE_SIZE = 30
MAX_SPRITE_SIZE = 1200
SPRITE_HIT_PADDING = 22
MAX_MEDIAPIPE_WIDTH = 640
MAX_MEDIAPIPE_HEIGHT = 480
MIN_FPS_WARNING = 20
GEMINI_CACHE_SECONDS = 5
MAX_UNDO_SNAPSHOTS = 10
MAX_SCENE_HISTORY = 20
OPENWAKEWORD_THRESHOLD = 0.5
VOICE_SAMPLE_RATE = 16000
VOICE_CHUNK_SECONDS = 0.08
VOICE_WAKEWORD_COOLDOWN_SECONDS = 0.5
VOICE_COMMAND_TIMEOUT_SECONDS = 5
VOICE_COMMAND_PHRASE_LIMIT_SECONDS = 6
VOICE_AMBIENT_CALIBRATION_SECONDS = 0.3
VOICE_WAKEWORD_CALIBRATION_SECONDS = 0.5
DRAW_INDEX_MIN_OPENNESS = 0.02
DRAW_NON_INDEX_MAX_OPENNESS = 0.015
DRAW_NON_INDEX_ALLOWED_OPEN_FINGERS = 1
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip() or (
    "gemini-2.5-flash"
)
ALLOW_CLOUD_SPEECH_RECOGNITION = os.getenv(
    "ALLOW_CLOUD_SPEECH_RECOGNITION", "1"
).strip().lower() not in {
    "0",
    "false",
    "no",
}

STATUS_DRAWING = "Drawing"
STATUS_PAUSED = "Paused"
STATUS_IDLE = "Idle"

BRUSH_COLORS = {
    "black": (0, 0, 0),
    "blue": (255, 0, 0),
    "cyan": (255, 255, 0),
    "eraser": (0, 0, 0),
    "green": (0, 255, 0),
    "orange": (0, 165, 255),
    "pink": (203, 192, 255),
    "purple": (255, 0, 255),
    "red": (0, 0, 255),
    "white": (255, 255, 255),
    "yellow": (0, 255, 255),
}

PALETTE_ORDER = [
    "red",
    "green",
    "blue",
    "yellow",
    "white",
    "purple",
    "orange",
    "eraser",
]
# Colors cycled by the draw-mode "pinch and hold" gesture.
QUICK_TOGGLE_COLORS = ["green", "white"]

MODE_DRAW = "draw_mode"
MODE_SELECT = "select_mode"
MODE_LABELS = {MODE_DRAW: "DRAW", MODE_SELECT: "SELECT"}

# (action id, label, keyboard hint). Mode buttons come first, then edits.
# Every command reachable by pointing: each entry becomes an on-screen button
# that a dwell (hover-and-hold) activates, so the keyboard is an alternative
# rather than the only way to reach an action.
TOOLBAR_ACTIONS = [
    ("draw_mode", "Draw", "D"),
    ("select_mode", "Select", "V"),
    ("undo", "Undo", "U"),
    ("redo", "Redo", "R"),
    ("clear", "Clear", "C"),
    ("delete", "Delete", "X"),
    ("export", "Export", "E"),
    ("save_project", "Save", "P"),
    ("load_project", "Load", "L"),
    ("help", "Help", "H"),
    ("status", "Status", "S"),
]
# Wider gaps after these actions separate the groups: modes | edit | file | view.
TOOLBAR_GROUP_BREAKS = ("select_mode", "delete", "load_project")

# --- Feedback, workspace, and service reliability -------------------------
# Frames smaller than this are upscaled (aspect preserved) so overlays stay
# legible; all coordinates (landmarks, UI, mouse) live in this workspace space.
WORKSPACE_MIN_HEIGHT = 720
WINDOW_MAX_WIDTH = 1400
WINDOW_MAX_HEIGHT = 900
CAPTURE_MAX_CONSECUTIVE_FAILURES = 30

# Requested camera stream. Most webcams default to 640x480 unless asked for
# more, which the workspace then had to upscale (soft, blurry image). We ask
# for 720p natively and fall back to whatever the device actually grants.
# Override per machine with ARIA_CAMERA_WIDTH / ARIA_CAMERA_HEIGHT.
CAPTURE_WIDTH = int(os.getenv("ARIA_CAMERA_WIDTH", "1280"))
CAPTURE_HEIGHT = int(os.getenv("ARIA_CAMERA_HEIGHT", "720"))
# Many webcams only deliver 720p+ at full frame rate as MJPG; the default
# YUY2 stream drops to 5-10 fps at that size. Set ARIA_CAMERA_MJPG=0 if a
# device misbehaves with MJPG.
CAPTURE_PREFER_MJPG = os.getenv("ARIA_CAMERA_MJPG", "1").strip().lower() not in {
    "0",
    "false",
    "no",
}
TOAST_SECONDS = 2.2
TOAST_ERROR_SECONDS = 4.0
MAX_TOASTS = 3
MAX_PENDING_EVENTS = 16
VOICE_DISPLAY_SECONDS = 8.0
VOICE_COMMAND_QUEUE_SIZE = 8
VOICE_COMMAND_RESULT_TIMEOUT_SECONDS = 2.0
TTS_QUEUE_SIZE = 2
SPRITE_SCALE_STEP_UP = 1.15
SPRITE_SCALE_STEP_DOWN = 1 / 1.15
VOICE_SCALE_UP = 1.5
VOICE_SCALE_DOWN = 0.5


def get_brush_color(color_name: str) -> tuple[int, int, int] | None:
    return BRUSH_COLORS.get(color_name.lower())


def dwell_seconds_for_mode(interaction_mode: str) -> float:
    """Single source of truth for toolbar/palette dwell time per mode."""
    if interaction_mode == MODE_DRAW:
        return UI_DRAW_MODE_DWELL_SECONDS
    return UI_SELECT_MODE_DWELL_SECONDS
