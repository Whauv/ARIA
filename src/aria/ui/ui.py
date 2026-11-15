"""Overlay renderers. Geometry comes from ``layout``; tokens from ``theme``."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import cv2
import numpy as np

from .. import config
from ..config import TRACKING_STABLE_THRESHOLD, TRACKING_UNSTABLE_THRESHOLD
from . import theme
from .layout import (
    PaletteItem,
    ThumbnailItem,
    ToolbarItem,
    UiLayout,
    compute_layout,
    get_hover_padding,
)

__all__ = [
    "PaletteItem",
    "ThumbnailItem",
    "ToolbarItem",
    "UiLayout",
    "compute_layout",
    "draw_gesture_progress",
    "draw_guide",
    "draw_hand_skeleton",
    "draw_help",
    "draw_layers",
    "draw_mode_frame",
    "draw_palette",
    "draw_pointer",
    "draw_resize_anchor",
    "draw_status_panel",
    "draw_toasts",
    "draw_toolbar",
    "draw_top_bar",
    "draw_voice_card",
    "get_hover_padding",
]

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
    (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)  # fmt: skip


def _scaled(layout: UiLayout, font_scale: float) -> float:
    return font_scale * layout.scale


def _center_text(
    frame: np.ndarray,
    text: str,
    rect: theme.Rect,
    scale: float,
    color: theme.Color,
    thickness: int = 1,
) -> None:
    left, top, right, bottom = rect
    text = theme.fit_text(text, right - left - 8, scale, thickness)
    width, height = theme.text_size(text, scale, thickness)
    origin = (left + (right - left - width) // 2, top + (bottom - top + height) // 2)
    theme.draw_text(frame, text, origin, scale, color, thickness)


# --- Workspace chrome ----------------------------------------------------


def draw_mode_frame(frame: np.ndarray, layout: UiLayout, interaction_mode: str) -> None:
    """Thin inset border in the mode color: always-visible mode cue."""
    left, top, right, bottom = layout.workspace_rect
    cv2.rectangle(
        frame,
        (left + 1, top + 1),
        (right - 2, bottom - 2),
        theme.mode_color(interaction_mode),
        max(1, layout.px(2)),
    )


def draw_top_bar(
    frame: np.ndarray,
    layout: UiLayout,
    interaction_mode: str,
    tool_detail: str,
    tool_swatch: theme.Color | None,
    activity: str,
    chips: Sequence[tuple[str, str, str]],
    fps: float,
) -> None:
    """Mode + tool on the left, activity in the middle, services on the right.

    ``chips`` is a sequence of (name, label, status_kind).
    """
    left, top, right, bottom = layout.top_bar
    theme.fill_rect(frame, layout.top_bar, theme.GRAPHITE_950, 0.84)
    cv2.line(frame, (left, bottom - 1), (right, bottom - 1), theme.GRAPHITE_700, 1)
    mid_y = (top + bottom) // 2
    label_scale = _scaled(layout, theme.TYPE_LABEL)
    title_scale = _scaled(layout, theme.TYPE_TITLE)
    caption = _scaled(layout, theme.TYPE_CAPTION)

    # Mode chip
    color = theme.mode_color(interaction_mode)
    x = layout.margin
    theme.status_dot(frame, (x + layout.px(6), mid_y), layout.px(5), color)
    x += layout.px(18)
    mode_label = config.MODE_LABELS.get(interaction_mode, interaction_mode.upper())
    width, height = theme.text_size(mode_label, title_scale, 2)
    theme.draw_text(
        frame, mode_label, (x, mid_y + height // 2), title_scale, color, 2,
        theme.FONT_BOLD,
    )  # fmt: skip
    x += width + layout.px(14)
    cv2.line(
        frame,
        (x, top + layout.px(12)),
        (x, bottom - layout.px(12)),
        theme.GRAPHITE_700,
        1,
    )
    x += layout.px(14)
    if tool_swatch is not None:
        size = layout.px(14)
        swatch = (x, mid_y - size // 2, x + size, mid_y + size // 2)
        theme.fill_rect(frame, swatch, tool_swatch, 1.0)
        theme.outline_rect(frame, swatch, theme.TEXT_MUTED)
        x += size + layout.px(8)
    width, height = theme.text_size(tool_detail, label_scale)
    theme.draw_text(
        frame, tool_detail, (x, mid_y + height // 2), label_scale, theme.TEXT
    )
    left_end = x + width + layout.px(20)

    # Service chips, right-aligned
    x = right - layout.margin
    fps_text = f"{fps:4.0f} fps"
    width, height = theme.text_size(fps_text, caption)
    x -= width
    theme.draw_text(
        frame, fps_text, (x, mid_y + height // 2), caption, theme.TEXT_MUTED
    )
    for name, label, kind in reversed(list(chips)):
        text = f"{name} {label}"
        width, height = theme.text_size(text, caption)
        x -= width + layout.px(30)
        dot_color = theme.STATUS_COLORS.get(kind, theme.TEXT_MUTED)
        theme.status_dot(frame, (x + layout.px(5), mid_y), layout.px(4), dot_color)
        theme.draw_text(
            frame, text, (x + layout.px(14), mid_y + height // 2), caption, theme.TEXT
        )
    right_start = x - layout.px(20)

    # Activity, centered in the remaining space
    if activity and right_start - left_end > layout.px(60):
        _center_text(
            frame, activity, (left_end, top, right_start, bottom), label_scale,
            theme.TEXT_MUTED,
        )  # fmt: skip


def draw_palette(
    frame: np.ndarray,
    layout: UiLayout,
    active_color_name: str,
    hover_target_id: str | None,
    dwell_ratio: float,
    interaction_mode: str,
) -> None:
    if not layout.palette_items or layout.palette_rect is None:
        return
    theme.panel(frame, layout.palette_rect, border=None, alpha=0.7)
    label_scale = _scaled(layout, theme.TYPE_LABEL)
    caption = _scaled(layout, theme.TYPE_CAPTION)
    accent = theme.mode_color(interaction_mode)
    for item in layout.palette_items:
        left, top, right, bottom = item["rect"]
        active = item["color_name"] == active_color_name
        hovered = hover_target_id == item["id"]
        theme.fill_rect(frame, item["rect"], theme.GRAPHITE_800, 0.92)
        border = (
            theme.TEXT
            if active
            else (theme.TEXT_MUTED if hovered else theme.GRAPHITE_700)
        )
        theme.outline_rect(frame, item["rect"], border, 2 if active else 1)

        inset = layout.px(5)
        swatch = (
            left + inset,
            top + inset,
            left + (bottom - top) - inset,
            bottom - inset,
        )
        if item["color_name"] == "eraser":
            theme.fill_rect(frame, swatch, theme.GRAPHITE_950, 1.0)
            cv2.line(
                frame,
                (swatch[0], swatch[3] - 1),
                (swatch[2] - 1, swatch[1]),
                theme.TEXT_MUTED,
                1,
                theme.LINE,
            )
        else:
            theme.fill_rect(frame, swatch, item["preview_color"], 1.0)
        theme.outline_rect(frame, swatch, theme.GRAPHITE_500)

        text_x = swatch[2] + layout.px(8)
        shortcut_w = (
            theme.text_size(item["shortcut"], caption)[0] if item["shortcut"] else 0
        )
        label = theme.fit_text(
            item["label"], right - text_x - shortcut_w - layout.px(12), label_scale
        )
        _width, height = theme.text_size(item["label"], label_scale)
        baseline = top + (bottom - top + height) // 2
        theme.draw_text(
            frame, label, (text_x, baseline), label_scale,
            theme.TEXT if active else theme.TEXT_MUTED,
        )  # fmt: skip
        if item["shortcut"]:
            width = shortcut_w
            theme.draw_text(
                frame, item["shortcut"], (right - width - layout.px(7), baseline),
                caption, theme.GRAPHITE_500,
            )  # fmt: skip
        if hovered and dwell_ratio > 0:
            fill = int((right - left) * dwell_ratio)
            theme.fill_rect(
                frame, (left, bottom - layout.px(3), left + fill, bottom), accent, 1.0
            )


def draw_toolbar(
    frame: np.ndarray,
    layout: UiLayout,
    interaction_mode: str,
    hover_target_id: str | None,
    dwell_ratio: float,
    disabled_ids: Iterable[str] = (),
) -> None:
    if not layout.toolbar_items:
        return
    disabled = set(disabled_ids)
    theme.fill_rect(frame, layout.toolbar_rect, theme.GRAPHITE_950, 0.8)
    left, top, right, _ = layout.toolbar_rect
    cv2.line(frame, (left, top), (right, top), theme.GRAPHITE_700, 1)
    label_scale = _scaled(layout, theme.TYPE_BODY)
    caption = _scaled(layout, theme.TYPE_CAPTION)
    accent = theme.mode_color(interaction_mode)
    for item in layout.toolbar_items:
        rect = item["rect"]
        b_left, b_top, b_right, b_bottom = rect
        is_mode = item["action"] in (config.MODE_DRAW, config.MODE_SELECT)
        active = is_mode and item["action"] == interaction_mode
        hovered = hover_target_id == item["id"]
        is_disabled = item["id"] in disabled

        theme.fill_rect(frame, rect, theme.GRAPHITE_800, 0.94)
        if active:
            border = accent
        elif hovered and not is_disabled:
            border = theme.TEXT_MUTED
        else:
            border = theme.GRAPHITE_700
        theme.outline_rect(frame, rect, border, 2 if active else 1)
        if active:
            theme.fill_rect(
                frame, (b_left, b_bottom - layout.px(4), b_right, b_bottom), accent, 1.0
            )

        if is_disabled:
            text_color = theme.GRAPHITE_500
        elif active:
            text_color = accent
        else:
            text_color = theme.TEXT
        _center_text(frame, item["label"], rect, label_scale, text_color, 1)
        if item["shortcut"]:
            width, height = theme.text_size(item["shortcut"], caption)
            theme.draw_text(
                frame, item["shortcut"],
                (b_right - width - layout.px(6), b_top + height + layout.px(5)),
                caption, theme.GRAPHITE_500 if is_disabled else theme.TEXT_MUTED,
            )  # fmt: skip
        if hovered and dwell_ratio > 0 and not is_disabled:
            fill = int((b_right - b_left) * dwell_ratio)
            theme.fill_rect(
                frame,
                (b_left, b_bottom - layout.px(4), b_left + fill, b_bottom),
                accent,
                1.0,
            )


def _paste_thumbnail(frame: np.ndarray, sprite, rect: theme.Rect) -> None:
    left, top, right, bottom = rect
    inner_w, inner_h = right - left - 6, bottom - top - 6
    image = getattr(sprite, "img", None)
    if image is None or image.size == 0 or inner_w <= 0 or inner_h <= 0:
        return
    src_h, src_w = image.shape[:2]
    scale = min(inner_w / src_w, inner_h / src_h)
    thumb_w, thumb_h = max(1, int(src_w * scale)), max(1, int(src_h * scale))
    thumb = cv2.resize(image, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
    x = left + 3 + (inner_w - thumb_w) // 2
    y = top + 3 + (inner_h - thumb_h) // 2
    clipped = theme.clip_rect((x, y, x + thumb_w, y + thumb_h), frame)
    if clipped is None:
        return
    region = frame[y : y + thumb_h, x : x + thumb_w]
    if region.shape[:2] != (thumb_h, thumb_w):
        return
    if thumb.ndim == 3 and thumb.shape[2] == 4:
        alpha = thumb[:, :, 3:4].astype(np.float32) / 255.0
        blended = alpha * thumb[:, :, :3] + (1.0 - alpha) * region
        region[:] = blended.astype(np.uint8)
    else:
        region[:] = thumb[:, :, :3]


def draw_layers(
    frame: np.ndarray, layout: UiLayout, hover_target_id: str | None
) -> None:
    if layout.layers_rect is None:
        return
    theme.panel(frame, layout.layers_rect, border=None, alpha=0.7)
    caption = _scaled(layout, theme.TYPE_CAPTION)
    left, top, _right, _bottom = layout.layers_rect
    _, height = theme.text_size("LAYERS", caption)
    theme.draw_text(
        frame, "LAYERS", (left + layout.px(6), top + height + layout.px(6)), caption,
        theme.TEXT_MUTED,
    )  # fmt: skip
    for item in layout.thumbnail_items:
        rect = item["rect"]
        sprite = item["sprite"]
        theme.fill_rect(frame, rect, theme.GRAPHITE_950, 0.9)
        _paste_thumbnail(frame, sprite, rect)
        if getattr(sprite, "selected", False):
            theme.outline_rect(frame, rect, theme.SIGNAL_SELECT, 2)
        elif hover_target_id == item["id"]:
            theme.outline_rect(frame, rect, theme.TEXT_MUTED, 1)
        else:
            theme.outline_rect(frame, rect, theme.GRAPHITE_700, 1)
    if layout.hidden_layer_count:
        text = f"+{layout.hidden_layer_count} more"
        last_bottom = (
            layout.thumbnail_items[-1]["rect"][3]
            if layout.thumbnail_items
            else top + height + layout.px(10)
        )
        theme.draw_text(
            frame, text, (left + layout.px(6), last_bottom + layout.px(16)), caption,
            theme.TEXT_MUTED,
        )  # fmt: skip


# --- Pointer and gesture feedback -----------------------------------------


def draw_hand_skeleton(frame: np.ndarray, points: Sequence[tuple[int, int]]) -> None:
    if len(points) < 21:
        return
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], theme.GRAPHITE_500, 1, theme.LINE)
    for point in points:
        cv2.circle(frame, point, 2, theme.TEXT_MUTED, -1, theme.LINE)


def _stability_color(stability: float) -> theme.Color:
    if stability >= TRACKING_STABLE_THRESHOLD:
        return theme.STATUS_OK
    if stability <= TRACKING_UNSTABLE_THRESHOLD:
        return theme.STATUS_WARN
    return theme.TEXT


def draw_pointer(
    frame: np.ndarray,
    layout: UiLayout,
    point: tuple[int, int] | None,
    interaction_mode: str,
    brush_color: theme.Color,
    brush_thickness: int,
    stability: float,
) -> None:
    """Fingertip indicator. Draw mode: brush-sized ring in the brush color.
    Select mode: cyan reticle. The center dot reflects tracking stability."""
    if point is None:
        return
    x, y = point
    if interaction_mode == config.MODE_DRAW:
        radius = max(layout.px(7), brush_thickness // 2 + layout.px(5))
        cv2.circle(frame, point, radius + 1, theme.GRAPHITE_950, 3, theme.LINE)
        ring = brush_color if any(brush_color) else theme.TEXT_MUTED
        cv2.circle(frame, point, radius, ring, 2, theme.LINE)
    else:
        gap, length = layout.px(5), layout.px(9)
        color = theme.SIGNAL_SELECT
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            start = (x + dx * gap, y + dy * gap)
            end = (x + dx * (gap + length), y + dy * (gap + length))
            cv2.line(frame, start, end, theme.GRAPHITE_950, 4, theme.LINE)
            cv2.line(frame, start, end, color, 2, theme.LINE)
    cv2.circle(
        frame, point, layout.px(2) + 1, _stability_color(stability), -1, theme.LINE
    )


def draw_gesture_progress(
    frame: np.ndarray,
    layout: UiLayout,
    center: tuple[int, int] | None,
    progress: float,
    label: str,
    color: theme.Color,
) -> None:
    if center is None:
        return
    radius = layout.px(22)
    theme.progress_arc(frame, center, radius, progress, color, max(2, layout.px(3)))
    caption = _scaled(layout, theme.TYPE_CAPTION)
    width, height = theme.text_size(label, caption)
    x = center[0] + radius + layout.px(8)
    if x + width + layout.px(10) > frame.shape[1]:
        x = center[0] - radius - layout.px(8) - width
    y = center[1] - radius
    theme.fill_rect(
        frame, (x - 5, y - height - 6, x + width + 5, y + 5), theme.GRAPHITE_950, 0.8
    )
    theme.draw_text(frame, label, (x, y), caption, color)


def draw_resize_anchor(
    frame: np.ndarray,
    layout: UiLayout,
    first_point: tuple[int, int] | None,
    second_point: tuple[int, int] | None,
    scale_factor: float | None,
) -> None:
    if first_point is None or second_point is None:
        return
    color = theme.SIGNAL_SELECT
    cv2.line(frame, first_point, second_point, color, 1, theme.LINE)
    for point in (first_point, second_point):
        cv2.circle(frame, point, layout.px(7), color, 2, theme.LINE)
    midpoint = (
        (first_point[0] + second_point[0]) // 2,
        (first_point[1] + second_point[1]) // 2,
    )
    label = "Resize" if scale_factor is None else f"Resize {scale_factor:.2f}x"
    caption = _scaled(layout, theme.TYPE_LABEL)
    width, height = theme.text_size(label, caption)
    x, y = midpoint[0] + layout.px(10), midpoint[1] - layout.px(10)
    theme.fill_rect(
        frame, (x - 5, y - height - 6, x + width + 5, y + 5), theme.GRAPHITE_950, 0.8
    )
    theme.draw_text(frame, label, (x, y), caption, color)


# --- Feedback surfaces ----------------------------------------------------


def draw_toasts(
    frame: np.ndarray, layout: UiLayout, toasts: Sequence[object], now: float
) -> None:
    """Stack transient notifications upward from above the toolbar."""
    if not toasts:
        return
    label_scale = _scaled(layout, theme.TYPE_BODY)
    pad_x, line_h = layout.px(14), layout.px(32)
    gap = layout.px(6)
    center_x, bottom = layout.toast_anchor
    for toast in reversed(list(toasts)):
        text = theme.fit_text(
            toast.text, layout.toast_max_width - 2 * pad_x, label_scale
        )
        width, height = theme.text_size(text, label_scale)
        box_w = width + 2 * pad_x + layout.px(4)
        rect = (center_x - box_w // 2, bottom - line_h, center_x + box_w // 2, bottom)
        alpha = 0.9
        remaining = toast.expires_at - now
        if remaining < 0.3:
            alpha *= max(0.0, remaining / 0.3)
        theme.panel(frame, rect, border=theme.GRAPHITE_700, alpha=alpha)
        accent = theme.STATUS_COLORS.get(toast.kind, theme.TEXT_MUTED)
        theme.fill_rect(
            frame, (rect[0], rect[1], rect[0] + layout.px(3), rect[3]), accent, 1.0
        )
        theme.draw_text(
            frame, text, (rect[0] + pad_x + layout.px(2), rect[1] + (line_h + height) // 2),
            label_scale, theme.TEXT,
        )  # fmt: skip
        bottom -= line_h + gap


def draw_voice_card(
    frame: np.ndarray,
    layout: UiLayout,
    listening: bool,
    speaking: bool,
    transcript: str | None,
    response: str | None,
    last_update_at: float | None,
    now: float,
) -> None:
    recent = (
        last_update_at is not None
        and now - last_update_at <= config.VOICE_DISPLAY_SECONDS
    )
    if not (listening or speaking or (recent and (transcript or response))):
        return
    label_scale = _scaled(layout, theme.TYPE_LABEL)
    caption = _scaled(layout, theme.TYPE_CAPTION)
    pad = layout.px(10)
    width = layout.voice_max_width
    inner = width - 2 * pad
    lines: list[tuple[str, float, theme.Color]] = []
    if listening:
        header, color = "Listening...", theme.SIGNAL_SELECT
    elif speaking:
        header, color = "ARIA speaking", theme.TEXT
    else:
        header, color = "Voice", theme.TEXT_MUTED
    if transcript and not listening:
        lines.append(
            (
                theme.fit_text(f'Heard: "{transcript}"', inner, caption),
                caption,
                theme.TEXT_MUTED,
            )
        )
    if response and not listening:
        for line in theme.wrap_text(response, inner, label_scale, 3):
            lines.append((line, label_scale, theme.TEXT))
    line_h = layout.px(20)
    height = pad * 2 + line_h + len(lines) * line_h
    left, bottom = layout.voice_anchor
    rect = (left, bottom - height, left + width, bottom)
    theme.panel(frame, rect, border=theme.GRAPHITE_700, alpha=0.88)
    dot_x, dot_y = left + pad + layout.px(4), rect[1] + pad + line_h // 2
    radius = layout.px(4)
    if listening:
        radius += round(layout.px(2) * (0.5 + 0.5 * math.sin(now * 5.0)))
    theme.status_dot(frame, (dot_x, dot_y), radius, color)
    _, text_h = theme.text_size(header, caption)
    theme.draw_text(
        frame, header, (dot_x + layout.px(12), dot_y + text_h // 2), caption, color
    )
    y = rect[1] + pad + line_h
    for text, scale, text_color in lines:
        _, text_h = theme.text_size(text, scale)
        theme.draw_text(
            frame, text, (left + pad, y + (line_h + text_h) // 2), scale, text_color
        )
        y += line_h


def draw_guide(
    frame: np.ndarray, layout: UiLayout, steps: Sequence[tuple[str, str, bool]]
) -> None:
    """Compact, non-blocking quick start that checks off steps as they happen."""
    rect = layout.guide_rect
    left, top, right, _bottom = rect
    theme.panel(frame, rect, border=theme.GRAPHITE_700, alpha=0.86)
    caption = _scaled(layout, theme.TYPE_CAPTION)
    label_scale = _scaled(layout, theme.TYPE_LABEL)
    pad = layout.px(10)
    _, cap_h = theme.text_size("Q", caption)
    title_y = top + pad + cap_h
    theme.draw_text(
        frame, "QUICK START", (left + pad, title_y), caption, theme.TEXT_MUTED
    )
    hint = "Help button  |  Esc hide"
    hint_w, _ = theme.text_size(hint, caption)
    theme.draw_text(
        frame, hint, (right - pad - hint_w, title_y), caption, theme.GRAPHITE_500
    )
    if not steps:
        return
    column_w = (right - left - 2 * pad) // len(steps)
    row_y = title_y + layout.px(10)
    for index, (title, detail, done) in enumerate(steps):
        x = left + pad + index * column_w
        center = (x + layout.px(7), row_y + layout.px(8))
        if done:
            theme.status_dot(frame, center, layout.px(6), theme.STATUS_OK)
            cv2.line(
                frame,
                (center[0] - 3, center[1]),
                (center[0] - 1, center[1] + 3),
                theme.GRAPHITE_950,
                2,
                theme.LINE,
            )
            cv2.line(
                frame,
                (center[0] - 1, center[1] + 3),
                (center[0] + 4, center[1] - 3),
                theme.GRAPHITE_950,
                2,
                theme.LINE,
            )
        else:
            cv2.circle(frame, center, layout.px(6), theme.TEXT_MUTED, 1, theme.LINE)
        text_x = x + layout.px(20)
        max_w = column_w - layout.px(32)
        _, label_h = theme.text_size(title, label_scale)
        theme.draw_text(
            frame, theme.fit_text(title, max_w, label_scale), (text_x, row_y + label_h + layout.px(2)),
            label_scale, theme.TEXT_MUTED if done else theme.TEXT,
        )  # fmt: skip
        theme.draw_text(
            frame, theme.fit_text(detail, max_w, caption),
            (text_x, row_y + label_h + cap_h + layout.px(9)), caption, theme.TEXT_MUTED,
        )  # fmt: skip


def draw_status_panel(
    frame: np.ndarray, layout: UiLayout, lines: Sequence[tuple[str, str]]
) -> None:
    rect = layout.status_panel_rect
    left, top, right, bottom = rect
    theme.panel(frame, rect, border=theme.GRAPHITE_700, alpha=0.9)
    caption = _scaled(layout, theme.TYPE_CAPTION)
    pad = layout.px(10)
    line_h = layout.px(20)
    _, cap_h = theme.text_size("Q", caption)
    y = top + pad + cap_h
    theme.draw_text(frame, "STATUS", (left + pad, y), caption, theme.TEXT_MUTED)
    hint = "S to close"
    hint_w, _ = theme.text_size(hint, caption)
    theme.draw_text(frame, hint, (right - pad - hint_w, y), caption, theme.GRAPHITE_500)
    key_w = layout.px(96)
    for key, value in lines:
        y += line_h
        if y > bottom - pad:
            break
        theme.draw_text(frame, key, (left + pad, y), caption, theme.TEXT_MUTED)
        theme.draw_text(
            frame, theme.fit_text(value, right - left - 2 * pad - key_w, caption),
            (left + pad + key_w, y), caption, theme.TEXT,
        )  # fmt: skip


def draw_help(
    frame: np.ndarray,
    layout: UiLayout,
    columns: Sequence[tuple[str, Sequence[tuple[str, str]]]],
    footer: str,
    close_hovered: bool = False,
    close_dwell: float = 0.0,
) -> None:
    """Reference panel. ``columns`` = [(title, [(term, description), ...])].

    A term starting with "#" renders as a sub-heading.
    """
    rect = layout.help_rect
    left, top, right, bottom = rect
    theme.panel(
        frame, rect, border=theme.GRAPHITE_700, alpha=0.94, fill=theme.GRAPHITE_950
    )
    pad = layout.px(18)
    title_scale = _scaled(layout, theme.TYPE_TITLE)
    caption = _scaled(layout, theme.TYPE_CAPTION)
    label_scale = _scaled(layout, theme.TYPE_LABEL)
    _, title_h = theme.text_size("ARIA", title_scale, 2)
    theme.draw_text(
        frame, "ARIA HELP", (left + pad, top + pad + title_h), title_scale, theme.TEXT, 2,
        theme.FONT_BOLD,
    )  # fmt: skip
    footer_w, _ = theme.text_size(footer, caption)
    theme.draw_text(
        frame, theme.fit_text(footer, right - left - 2 * pad - layout.px(160), caption),
        (max(left + pad + layout.px(160), right - pad - footer_w), top + pad + title_h),
        caption, theme.TEXT_MUTED,
    )  # fmt: skip
    if not columns:
        return
    content_top = top + pad + title_h + layout.px(18)
    cv2.line(
        frame,
        (left + pad, content_top - layout.px(8)),
        (right - pad, content_top - layout.px(8)),
        theme.GRAPHITE_700,
        1,
    )
    column_w = (right - left - 2 * pad) // len(columns)
    line_h = layout.px(19)
    _, cap_h = theme.text_size("Q", caption)
    for index, (title, entries) in enumerate(columns):
        x = left + pad + index * column_w
        y = content_top + cap_h
        theme.draw_text(frame, title, (x, y), caption, theme.SIGNAL_SELECT)
        # Size the term column to its longest term (bounded) so short keys
        # leave room for descriptions and long gesture names are not cut.
        longest = max(
            (
                theme.text_size(term, label_scale)[0]
                for term, _ in entries
                if not term.startswith("#")
            ),
            default=0,
        )
        term_w = min(int(column_w * 0.58), longest + layout.px(14))
        for term, description in entries:
            y += line_h
            if y > bottom - pad:
                break
            if term.startswith("#"):
                y += layout.px(4)
                theme.draw_text(frame, term[1:], (x, y), caption, theme.TEXT_MUTED)
                continue
            theme.draw_text(
                frame, theme.fit_text(term, term_w - layout.px(8), label_scale), (x, y),
                label_scale, theme.TEXT,
            )  # fmt: skip
            theme.draw_text(
                frame, theme.fit_text(description, column_w - term_w - layout.px(14), caption),
                (x + term_w, y), caption, theme.TEXT_MUTED,
            )  # fmt: skip
    _draw_help_close(frame, layout, close_hovered, close_dwell)


def _draw_help_close(
    frame: np.ndarray, layout: UiLayout, hovered: bool, dwell: float
) -> None:
    """Close button for the help overlay: point at it to dismiss help."""
    rect = layout.help_close_rect
    label_scale = _scaled(layout, theme.TYPE_LABEL)
    theme.fill_rect(frame, rect, theme.GRAPHITE_800, 0.96)
    theme.outline_rect(
        frame, rect, theme.SIGNAL_SELECT if hovered else theme.GRAPHITE_700, 2 if hovered else 1
    )  # fmt: skip
    if hovered and dwell > 0:
        left, _top, right, bottom = rect
        fill_right = left + int((right - left) * max(0.0, min(1.0, dwell)))
        theme.fill_rect(
            frame, (left, bottom - layout.px(4), fill_right, bottom), theme.SIGNAL_SELECT, 1.0
        )  # fmt: skip
    _center_text(
        frame, "Close", rect, label_scale,
        theme.TEXT if hovered else theme.TEXT_MUTED, 1,
    )  # fmt: skip
