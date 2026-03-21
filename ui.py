from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np

from config import (
    BRUSH_COLORS,
    BRUSH_PREVIEW_RADIUS,
    BUTTON_ACTIVE_COLOR,
    BUTTON_HOVER_COLOR,
    BUTTON_TEXT_COLOR,
    ERASER_COLOR,
    FPS_COLOR,
    PALETTE_HEIGHT,
    PALETTE_ORDER,
    PANEL_BG_COLOR,
    THUMBNAIL_BORDER_COLOR,
    THUMBNAIL_WIDTH,
    TOOLBAR_ACTIONS,
    TOOLBAR_BG_COLOR,
    TOOLBAR_HEIGHT,
    WARNING_BG_COLOR,
    WARNING_TEXT_COLOR,
)


def _draw_panel(frame: np.ndarray, rect: tuple[int, int, int, int], color: tuple[int, int, int], alpha: float) -> None:
    left, top, right, bottom = rect
    overlay = frame.copy()
    cv2.rectangle(overlay, (left, top), (right, bottom), color, -1)
    frame[:] = cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0.0)


def get_palette_items(frame_width: int) -> list[dict[str, object]]:
    items = []
    button_width = 74
    x = 10
    for color_name in PALETTE_ORDER:
        items.append(
            {
                "id": f"palette:{color_name}",
                "label": color_name.title(),
                "color_name": color_name,
                "preview_color": BRUSH_COLORS[color_name],
                "rect": (x, 10, x + button_width, 10 + PALETTE_HEIGHT - 14),
            }
        )
        x += button_width + 8
    return items


def draw_palette(
    frame: np.ndarray,
    items: list[dict[str, object]],
    active_color_name: str,
    hover_target_id: str | None,
    dwell_ratio: float,
) -> None:
    if not items:
        return

    _draw_panel(frame, (6, 6, min(frame.shape[1] - 110, 8 + len(items) * 82), PALETTE_HEIGHT), PANEL_BG_COLOR, 0.45)
    for item in items:
        left, top, right, bottom = item["rect"]
        color_name = item["color_name"]
        fill_color = ERASER_COLOR if color_name == "eraser" else item["preview_color"]
        background = BUTTON_ACTIVE_COLOR if color_name == active_color_name else (60, 60, 60)
        cv2.rectangle(frame, (left, top), (right, bottom), background, -1)
        swatch_left = left + 6
        swatch_right = right - 6
        swatch_top = top + 6
        swatch_bottom = top + 30
        cv2.rectangle(frame, (swatch_left, swatch_top), (swatch_right, swatch_bottom), fill_color, -1)
        cv2.putText(
            frame,
            item["label"],
            (left + 6, bottom - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            BUTTON_TEXT_COLOR,
            1,
            cv2.LINE_AA,
        )
        if hover_target_id == item["id"]:
            progress_width = int((right - left) * dwell_ratio)
            cv2.rectangle(frame, (left, bottom - 4), (left + progress_width, bottom), BUTTON_HOVER_COLOR, -1)


def get_toolbar_items(frame_width: int, frame_height: int) -> list[dict[str, object]]:
    items = []
    button_width = 112
    spacing = 12
    total_width = len(TOOLBAR_ACTIONS) * button_width + (len(TOOLBAR_ACTIONS) - 1) * spacing
    start_x = max(12, (frame_width - total_width) // 2)
    top = frame_height - TOOLBAR_HEIGHT + 12
    bottom = frame_height - 14

    for index, (action_id, label) in enumerate(TOOLBAR_ACTIONS):
        left = start_x + index * (button_width + spacing)
        items.append(
            {
                "id": f"toolbar:{action_id}",
                "action": action_id,
                "label": label,
                "rect": (left, top, left + button_width, bottom),
            }
        )
    return items


def draw_toolbar(
    frame: np.ndarray,
    items: list[dict[str, object]],
    active_action: str,
    hover_target_id: str | None,
    dwell_ratio: float,
) -> None:
    if not items:
        return

    top = frame.shape[0] - TOOLBAR_HEIGHT
    _draw_panel(frame, (0, top, frame.shape[1], frame.shape[0]), TOOLBAR_BG_COLOR, 0.5)
    for item in items:
        left, item_top, right, bottom = item["rect"]
        background = BUTTON_ACTIVE_COLOR if item["action"] == active_action else (55, 55, 55)
        cv2.rectangle(frame, (left, item_top), (right, bottom), background, -1)
        cv2.putText(
            frame,
            item["label"],
            (left + 10, item_top + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            BUTTON_TEXT_COLOR,
            1,
            cv2.LINE_AA,
        )
        if hover_target_id == item["id"]:
            progress_width = int((right - left) * dwell_ratio)
            cv2.rectangle(frame, (left, bottom - 4), (left + progress_width, bottom), BUTTON_HOVER_COLOR, -1)


def get_thumbnail_items(frame: np.ndarray, sprites: Iterable) -> list[dict[str, object]]:
    items = []
    x1 = frame.shape[1] - THUMBNAIL_WIDTH
    y = PALETTE_HEIGHT + 18
    for index, sprite in enumerate(sorted(sprites, key=lambda item: item.z_index, reverse=True)):
        rect = (x1, y, frame.shape[1] - 10, y + 64)
        items.append({"id": f"thumb:{id(sprite)}", "rect": rect, "sprite": sprite})
        y += 72
    return items


def draw_thumbnail_strip(frame: np.ndarray, items: list[dict[str, object]], hover_target_id: str | None) -> None:
    if not items:
        return

    _draw_panel(frame, (frame.shape[1] - THUMBNAIL_WIDTH - 6, PALETTE_HEIGHT + 10, frame.shape[1], frame.shape[0] - TOOLBAR_HEIGHT - 10), PANEL_BG_COLOR, 0.35)
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
    cv2.rectangle(overlay, (190, 10), (frame.shape[1] - 150, 44), WARNING_BG_COLOR, -1)
    frame[:] = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0.0)
    cv2.putText(
        frame,
        warning_text,
        (202, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        WARNING_TEXT_COLOR,
        2,
        cv2.LINE_AA,
    )
