"""Visual tokens and low-level drawing primitives for the OpenCV overlay.

"The Spatial Instrument": deep graphite backplates, neutral text, restrained
signal accents, thin outlines. The website (landing/) uses the same hex
values. OpenCV expects BGR, so tokens are declared as hex and converted once.
"""

from __future__ import annotations

import cv2
import numpy as np

Color = tuple[int, int, int]
Rect = tuple[int, int, int, int]


def hex_to_bgr(value: str) -> Color:
    value = value.lstrip("#")
    red, green, blue = (int(value[index : index + 2], 16) for index in (0, 2, 4))
    return blue, green, red


# Surfaces and text
GRAPHITE_950 = hex_to_bgr("#0E1012")
GRAPHITE_900 = hex_to_bgr("#15181C")
GRAPHITE_800 = hex_to_bgr("#1E2228")
GRAPHITE_700 = hex_to_bgr("#2A2F37")
GRAPHITE_500 = hex_to_bgr("#5A616C")
TEXT = hex_to_bgr("#E8E6E1")
TEXT_MUTED = hex_to_bgr("#A3A7AE")

# Signal accents
SIGNAL_DRAW = hex_to_bgr("#F5B841")
SIGNAL_SELECT = hex_to_bgr("#5EC8F2")
STATUS_OK = hex_to_bgr("#6FD39B")
STATUS_WARN = hex_to_bgr("#F08A4B")
STATUS_ERROR = hex_to_bgr("#F0605D")

STATUS_COLORS: dict[str, Color] = {
    "ok": STATUS_OK,
    "success": STATUS_OK,
    "info": TEXT_MUTED,
    "warn": STATUS_WARN,
    "error": STATUS_ERROR,
    "off": GRAPHITE_500,
    "active": SIGNAL_SELECT,
}

BACKPLATE_ALPHA = 0.78
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX
LINE = cv2.LINE_AA


def mode_color(interaction_mode: str) -> Color:
    return SIGNAL_DRAW if interaction_mode == "draw_mode" else SIGNAL_SELECT


def ui_scale(frame_width: int, frame_height: int) -> float:
    """Scale factor relative to a 1280x720 reference workspace."""
    if frame_width <= 0 or frame_height <= 0:
        return 1.0
    return max(0.6, min(1.6, min(frame_width / 1280.0, frame_height / 720.0)))


# Type ramp (font scale at ui_scale == 1.0)
TYPE_LABEL = 0.48
TYPE_BODY = 0.52
TYPE_TITLE = 0.62
TYPE_CAPTION = 0.42


def text_size(text: str, scale: float, thickness: int = 1) -> tuple[int, int]:
    (width, height), _baseline = cv2.getTextSize(text, FONT, scale, thickness)
    return width, height


def draw_text(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float,
    color: Color = TEXT,
    thickness: int = 1,
    font: int = FONT,
) -> None:
    cv2.putText(
        frame,
        text,
        (int(origin[0]), int(origin[1])),
        font,
        scale,
        color,
        thickness,
        LINE,
    )


def fit_text(text: str, max_width: int, scale: float, thickness: int = 1) -> str:
    """Truncate text with an ellipsis so it fits within max_width pixels."""
    if max_width <= 0:
        return ""
    if text_size(text, scale, thickness)[0] <= max_width:
        return text
    trimmed = text
    while trimmed and text_size(trimmed + "...", scale, thickness)[0] > max_width:
        trimmed = trimmed[:-1]
    return (trimmed.rstrip() + "...") if trimmed else ""


def wrap_text(
    text: str, max_width: int, scale: float, max_lines: int, thickness: int = 1
) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if text_size(candidate, scale, thickness)[0] <= max_width or not current:
            current = candidate
            continue
        lines.append(current)
        current = word
        if len(lines) == max_lines:
            break
    if current and len(lines) < max_lines:
        lines.append(current)
    if len(lines) == max_lines and " ".join(lines) != " ".join(words):
        lines[-1] = fit_text(lines[-1] + " ...", max_width, scale, thickness)
    return [fit_text(line, max_width, scale, thickness) for line in lines]


def clip_rect(rect: Rect, frame: np.ndarray) -> Rect | None:
    left, top, right, bottom = rect
    height, width = frame.shape[:2]
    left, right = max(0, left), min(width, right)
    top, bottom = max(0, top), min(height, bottom)
    if left >= right or top >= bottom:
        return None
    return left, top, right, bottom


def fill_rect(
    frame: np.ndarray, rect: Rect, color: Color, alpha: float = BACKPLATE_ALPHA
) -> None:
    """Blend a solid rectangle into the frame, touching only that region."""
    clipped = clip_rect(rect, frame)
    if clipped is None:
        return
    left, top, right, bottom = clipped
    if alpha >= 1.0:
        frame[top:bottom, left:right] = color
        return
    region = frame[top:bottom, left:right]
    plate = np.empty_like(region)
    plate[:] = color
    cv2.addWeighted(plate, alpha, region, 1.0 - alpha, 0.0, dst=region)


def outline_rect(
    frame: np.ndarray, rect: Rect, color: Color, thickness: int = 1
) -> None:
    left, top, right, bottom = rect
    cv2.rectangle(frame, (left, top), (right - 1, bottom - 1), color, thickness, LINE)


def panel(
    frame: np.ndarray,
    rect: Rect,
    border: Color | None = GRAPHITE_700,
    alpha: float = BACKPLATE_ALPHA,
    fill: Color = GRAPHITE_900,
) -> None:
    fill_rect(frame, rect, fill, alpha)
    if border is not None:
        outline_rect(frame, rect, border, 1)


def dashed_rect(
    frame: np.ndarray, rect: Rect, color: Color, dash: int = 8, gap: int = 6
) -> None:
    left, top, right, bottom = rect
    step = dash + gap
    for x in range(left, right, step):
        cv2.line(frame, (x, top), (min(x + dash, right), top), color, 1, LINE)
        cv2.line(frame, (x, bottom), (min(x + dash, right), bottom), color, 1, LINE)
    for y in range(top, bottom, step):
        cv2.line(frame, (left, y), (left, min(y + dash, bottom)), color, 1, LINE)
        cv2.line(frame, (right, y), (right, min(y + dash, bottom)), color, 1, LINE)


def status_dot(
    frame: np.ndarray, center: tuple[int, int], radius: int, color: Color
) -> None:
    cv2.circle(frame, center, radius, color, -1, LINE)


def progress_arc(
    frame: np.ndarray,
    center: tuple[int, int],
    radius: int,
    progress: float,
    color: Color,
    thickness: int = 3,
) -> None:
    progress = max(0.0, min(1.0, progress))
    cv2.circle(frame, center, radius, GRAPHITE_700, max(1, thickness - 1), LINE)
    if progress > 0:
        cv2.ellipse(
            frame,
            center,
            (radius, radius),
            -90,
            0,
            int(progress * 360),
            color,
            thickness,
            LINE,
        )
