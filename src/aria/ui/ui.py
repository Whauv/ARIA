from __future__ import annotations

from typing import Iterable, TypedDict

import cv2
import numpy as np

from ..config import (
    BRUSH_COLORS,
    BRUSH_PREVIEW_RADIUS,
    BUTTON_ACTIVE_COLOR,
    BUTTON_HOVER_COLOR,
    BUTTON_TEXT_COLOR,
    ERASER_COLOR,
    FPS_COLOR,
    PALETTE_ORDER,
    PANEL_BG_COLOR,
    THUMBNAIL_BORDER_COLOR,
    THUMBNAIL_WIDTH,
    TOOLBAR_ACTIONS,
    TOOLBAR_BG_COLOR,
    TOOLBAR_HEIGHT,
    TRACKING_STABLE_THRESHOLD,
    TRACKING_UNSTABLE_THRESHOLD,
    UI_DRAW_MODE_PADDING,
    UI_SELECT_MODE_PADDING,
    WARNING_BG_COLOR,
    WARNING_TEXT_COLOR,
)


class PaletteItem(TypedDict):
    id: str
    label: str
    color_name: str
    preview_color: tuple[int, int, int]
    rect: tuple[int, int, int, int]


class ToolbarItem(TypedDict):
    id: str
    action: str
    label: str
    rect: tuple[int, int, int, int]


class ThumbnailItem(TypedDict):
    id: str
    rect: tuple[int, int, int, int]
    sprite: object


def get_hover_padding(interaction_mode: str, target_kind: str) -> int:
    base_padding = UI_DRAW_MODE_PADDING if interaction_mode == "draw_mode" else UI_SELECT_MODE_PADDING
    if target_kind == "toolbar":
        return base_padding + 8
    if target_kind == "palette":
        return base_padding
    return base_padding


def _draw_panel(frame: np.ndarray, rect: tuple[int, int, int, int], color: tuple[int, int, int], alpha: float) -> None:
    left, top, right, bottom = rect
    overlay = frame.copy()
    cv2.rectangle(overlay, (left, top), (right, bottom), color, -1)
    frame[:] = cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0.0)


def get_palette_items(frame_width: int) -> list[PaletteItem]:
    items: list[PaletteItem] = []
    button_width = 112
    button_height = 30
    spacing = 8
    x = 14
    y = 68
    palette_order = config.TEST_PALETTE_ORDER if hasattr(config, "TEST_PALETTE_ORDER") else PALETTE_ORDER
    for color_name in palette_order:
        items.append(
            {
                "id": f"palette:{color_name}",
                "label": color_name.title(),
                "color_name": color_name,
                "preview_color": BRUSH_COLORS[color_name],
                "rect": (x, y, x + button_width, y + button_height),
            }
        )
        y += button_height + spacing
    return items


def draw_palette(frame: np.ndarray, items: list[PaletteItem], active_color_name: str, hover_target_id: str | None, dwell_ratio: float) -> None:
    if not items:
        return

    first_left, first_top, _, _ = items[0]["rect"]
    _, _, last_right, last_bottom = items[-1]["rect"]
    _draw_panel(frame, (first_left - 8, first_top - 8, last_right + 8, last_bottom + 8), PANEL_BG_COLOR, 0.4)
    for item in items:
        left, top, right, bottom = item["rect"]
        color_name = item["color_name"]
        fill_color = ERASER_COLOR if color_name == "eraser" else item["preview_color"]
        background = BUTTON_ACTIVE_COLOR if color_name == active_color_name else (38, 42, 54)
        cv2.rectangle(frame, (left, top), (right, bottom), background, -1)
        swatch_left = left + 8
        swatch_right = left + 34
        swatch_top = top + 5
        swatch_bottom = bottom - 5
        cv2.rectangle(frame, (swatch_left, swatch_top), (swatch_right, swatch_bottom), fill_color, -1)
        cv2.putText(
            frame,
            item["label"],
            (left + 42, top + 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            BUTTON_TEXT_COLOR,
            1,
            cv2.LINE_AA,
        )
        if hover_target_id == item["id"]:
            progress_width = int((right - left) * dwell_ratio)
            cv2.rectangle(frame, (left, bottom - 4), (left + progress_width, bottom), BUTTON_HOVER_COLOR, -1)


def get_toolbar_items(frame_width: int, frame_height: int) -> list[ToolbarItem]:
    items: list[ToolbarItem] = []
    side_margin = 16
    spacing = 10
    available_width = max(320, frame_width - side_margin * 2)
    button_width = max(100, (available_width - (len(TOOLBAR_ACTIONS) - 1) * spacing) // len(TOOLBAR_ACTIONS))
    total_width = len(TOOLBAR_ACTIONS) * button_width + (len(TOOLBAR_ACTIONS) - 1) * spacing
    if total_width > available_width:
        button_width = max(88, (available_width - (len(TOOLBAR_ACTIONS) - 1) * spacing) // len(TOOLBAR_ACTIONS))
        total_width = len(TOOLBAR_ACTIONS) * button_width + (len(TOOLBAR_ACTIONS) - 1) * spacing
    start_x = max(side_margin, (frame_width - total_width) // 2)
    top = frame_height - TOOLBAR_HEIGHT + 14
    bottom = frame_height - 12

    for index, (action_id, label) in enumerate(TOOLBAR_ACTIONS):
        left = start_x + index * (button_width + spacing)
        right = min(frame_width - side_margin, left + button_width)
        items.append(
            {
                "id": f"toolbar:{action_id}",
                "action": action_id,
                "label": label,
                "rect": (left, top, right, bottom),
            }
        )
    return items


def draw_toolbar(frame: np.ndarray, items: list[ToolbarItem], active_action: str, hover_target_id: str | None, dwell_ratio: float) -> None:
    if not items:
        return

    top = frame.shape[0] - TOOLBAR_HEIGHT
    _draw_panel(frame, (12, top - 4, frame.shape[1] - 12, frame.shape[0] - 8), TOOLBAR_BG_COLOR, 0.46)
    for item in items:
        left, item_top, right, bottom = item["rect"]
        background = BUTTON_ACTIVE_COLOR if item["action"] == active_action else (38, 42, 54)
        cv2.rectangle(frame, (left, item_top), (right, bottom), background, -1)
        text_scale = 0.52 if (right - left) < 120 else 0.58
        text_x = left + 10 if (right - left) < 120 else left + 12
        cv2.putText(
            frame,
            item["label"],
            (text_x, item_top + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_scale,
            BUTTON_TEXT_COLOR,
            2,
            cv2.LINE_AA,
        )
        if hover_target_id == item["id"]:
            progress_width = int((right - left) * dwell_ratio)
            cv2.rectangle(frame, (left, bottom - 4), (left + progress_width, bottom), BUTTON_HOVER_COLOR, -1)


def get_thumbnail_items(frame: np.ndarray, sprites: Iterable) -> list[ThumbnailItem]:
    items: list[ThumbnailItem] = []
    x1 = frame.shape[1] - THUMBNAIL_WIDTH
    y = 104
    for index, sprite in enumerate(sorted(sprites, key=lambda item: item.z_index, reverse=True)):
        rect = (x1, y, frame.shape[1] - 12, y + 52)
        items.append({"id": f"thumb:{id(sprite)}", "rect": rect, "sprite": sprite})
        y += 58
    return items


def draw_thumbnail_strip(frame: np.ndarray, items: list[ThumbnailItem], hover_target_id: str | None) -> None:
    if not items:
        return

    _draw_panel(frame, (frame.shape[1] - THUMBNAIL_WIDTH - 8, 98, frame.shape[1] - 8, frame.shape[0] - TOOLBAR_HEIGHT - 12), PANEL_BG_COLOR, 0.2)
    for item in items:
        left, top, right, bottom = item["rect"]
        sprite = item["sprite"]
        cv2.rectangle(frame, (left, top), (right, bottom), THUMBNAIL_BORDER_COLOR, 1)
        thumb = cv2.resize(sprite.img[:, :, :3], (right - left - 8, bottom - top - 8), interpolation=cv2.INTER_AREA)
        frame[top + 4 : bottom - 4, left + 4 : right - 4] = thumb
        if sprite.selected:
            cv2.rectangle(frame, (left, top), (right, bottom), BUTTON_ACTIVE_COLOR, 2)
        elif hover_target_id == item["id"]:
            cv2.rectangle(frame, (left, top), (right, bottom), BUTTON_HOVER_COLOR, 2)


def draw_brush_preview(frame: np.ndarray, fingertip: tuple[int, int] | None, brush_size: int, brush_color: tuple[int, int, int]) -> None:
    if fingertip is None:
        return

    center = (fingertip[0] + 18, fingertip[1] - 18)
    radius = max(BRUSH_PREVIEW_RADIUS, brush_size)
    cv2.circle(frame, center, radius, brush_color, 2)


def draw_fingertip_marker(frame: np.ndarray, fingertip: tuple[int, int] | None, tracking_stability: float) -> None:
    if fingertip is None:
        return

    if tracking_stability >= TRACKING_STABLE_THRESHOLD:
        color = (80, 255, 120)
    elif tracking_stability <= TRACKING_UNSTABLE_THRESHOLD:
        color = (0, 140, 255)
    else:
        color = (0, 220, 255)

    cv2.circle(frame, fingertip, 6, color, -1)
    cv2.circle(frame, fingertip, 10, color, 1)


def draw_fps(frame: np.ndarray, fps: float) -> None:
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (14, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        FPS_COLOR,
        2,
        cv2.LINE_AA,
    )


def draw_warning_overlay(frame: np.ndarray, warning_text: str | None) -> None:
    if not warning_text:
        return

    overlay = frame.copy()
    text = warning_text[:28]
    left = 212
    right = min(frame.shape[1] - THUMBNAIL_WIDTH - 24, left + 200)
    top = 14
    bottom = 38
    cv2.rectangle(overlay, (left, top), (right, bottom), WARNING_BG_COLOR, -1)
    frame[:] = cv2.addWeighted(overlay, 0.28, frame, 0.72, 0.0)
    cv2.putText(
        frame,
        text,
        (left + 10, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        WARNING_TEXT_COLOR,
        1,
        cv2.LINE_AA,
    )


def draw_tracking_feedback(frame: np.ndarray, calibrated: bool, tracking_stability: float, tracking_status_text: str) -> None:
    if not calibrated:
        text = "Calibrating hand..."
        color = (255, 220, 140)
    elif tracking_stability >= TRACKING_STABLE_THRESHOLD:
        text = "DRAW READY"
        color = (120, 255, 120)
    elif tracking_stability <= TRACKING_UNSTABLE_THRESHOLD:
        text = "TRACKING UNSTABLE"
        color = (0, 140, 255)
    else:
        text = tracking_status_text
        color = (0, 220, 255)

    cv2.putText(
        frame,
        text,
        (14, 108),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )
