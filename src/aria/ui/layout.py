"""Centralized UI geometry.

Rendering and hit testing both read from ``UiLayout`` so what the user sees is
exactly what fingertip dwell, mouse clicks, and draw-blocking respond to. All
coordinates are in workspace (displayed frame) pixels.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TypedDict

from .. import config
from .theme import Rect, ui_scale


class PaletteItem(TypedDict):
    id: str
    label: str
    color_name: str
    preview_color: tuple[int, int, int]
    rect: Rect
    shortcut: str


class ToolbarItem(TypedDict):
    id: str
    action: str
    label: str
    rect: Rect
    shortcut: str


class ThumbnailItem(TypedDict):
    id: str
    rect: Rect
    sprite: object


@dataclass(slots=True)
class UiLayout:
    width: int
    height: int
    scale: float
    margin: int
    top_bar: Rect
    toolbar_rect: Rect
    palette_rect: Rect | None
    palette_items: list[PaletteItem]
    toolbar_items: list[ToolbarItem]
    layers_rect: Rect | None
    thumbnail_items: list[ThumbnailItem]
    hidden_layer_count: int
    guide_rect: Rect
    help_rect: Rect
    # Dwell target that closes the help overlay, so help can be dismissed by
    # pointing: while help is open every other gesture is suspended.
    help_close_rect: Rect
    status_panel_rect: Rect
    voice_anchor: tuple[int, int]
    voice_max_width: int
    toast_anchor: tuple[int, int]
    toast_max_width: int
    workspace_rect: Rect = field(default=(0, 0, 0, 0))

    def px(self, value: float) -> int:
        return max(1, round(value * self.scale))

    @property
    def palette_rects(self) -> list[Rect]:
        return [item["rect"] for item in self.palette_items]

    @property
    def toolbar_rects(self) -> list[Rect]:
        return [item["rect"] for item in self.toolbar_items]

    @property
    def thumbnail_rects(self) -> list[Rect]:
        return [item["rect"] for item in self.thumbnail_items]

    @property
    def control_rects(self) -> list[Rect]:
        """Regions whose hover dwell triggers actions."""
        return self.palette_rects + self.toolbar_rects

    @property
    def draw_block_rects(self) -> list[Rect]:
        """Regions where a new stroke must not start."""
        return self.control_rects + self.thumbnail_rects + [self.toolbar_rect]


def get_hover_padding(interaction_mode: str, target_kind: str) -> int:
    base_padding = (
        config.UI_DRAW_MODE_PADDING
        if interaction_mode == config.MODE_DRAW
        else config.UI_SELECT_MODE_PADDING
    )
    if target_kind == "toolbar":
        return base_padding + 8
    return base_padding


def _palette_items(
    margin: int, top: int, bottom_limit: int, scale: float
) -> list[PaletteItem]:
    order = list(config.PALETTE_ORDER)
    if not order:
        return []
    width = round(122 * scale)
    gap = max(3, round(6 * scale))
    height = round(32 * scale)
    available = bottom_limit - top
    needed = len(order) * height + (len(order) - 1) * gap
    if needed > available:
        height = max(
            round(20 * scale), (available - (len(order) - 1) * gap) // len(order)
        )
    items: list[PaletteItem] = []
    y = top
    for index, color_name in enumerate(order):
        if y + height > bottom_limit:
            break
        items.append(
            {
                "id": f"palette:{color_name}",
                "label": color_name.title(),
                "color_name": color_name,
                "preview_color": config.BRUSH_COLORS[color_name],
                "rect": (margin, y, margin + width, y + height),
                "shortcut": str(index + 1) if index < 9 else "",
            }
        )
        y += height + gap
    return items


def _toolbar_items(
    width: int, top: int, bottom: int, scale: float
) -> list[ToolbarItem]:
    actions = config.TOOLBAR_ACTIONS
    count = len(actions)
    side = round(16 * scale)
    gap = max(4, round(8 * scale))
    group_gap = max(gap, round(22 * scale))
    breaks = {
        action
        for action, *_ in actions
        if action in config.TOOLBAR_GROUP_BREAKS and action != actions[-1][0]
    }
    gaps_total = (count - 1) * gap + len(breaks) * (group_gap - gap)
    available = width - 2 * side - gaps_total
    button_width = min(round(140 * scale), available // count if count else 0)
    # Never wider than the frame allows: with many actions the buttons shrink
    # rather than running off the edge, and hit rects follow the same math.
    button_width = max(1, min(button_width, available // count if count else 1))
    button_width = max(round(40 * scale), button_width)
    total = count * button_width + gaps_total
    x = max(side, (width - total) // 2)
    items: list[ToolbarItem] = []
    for action_id, label, shortcut in actions:
        right = min(width - side, x + button_width)
        items.append(
            {
                "id": f"toolbar:{action_id}",
                "action": action_id,
                "label": label,
                "rect": (x, top, right, bottom),
                "shortcut": shortcut,
            }
        )
        x += button_width + (group_gap if action_id in breaks else gap)
    return items


def compute_layout(
    frame_width: int, frame_height: int, sprites: Sequence[object] = ()
) -> UiLayout:
    scale = ui_scale(frame_width, frame_height)

    def px(value: float) -> int:
        return max(1, round(value * scale))

    margin = px(12)
    top_bar = (0, 0, frame_width, px(44))
    toolbar_height = px(64)
    toolbar_rect = (0, frame_height - toolbar_height, frame_width, frame_height)
    toolbar_items = _toolbar_items(
        frame_width,
        frame_height - toolbar_height + px(10),
        frame_height - px(10),
        scale,
    )

    content_top = top_bar[3] + margin
    content_bottom = toolbar_rect[1] - margin

    palette_items = _palette_items(margin, content_top, content_bottom, scale)
    palette_rect = None
    if palette_items:
        palette_rect = (
            palette_items[0]["rect"][0] - px(6),
            palette_items[0]["rect"][1] - px(6),
            palette_items[-1]["rect"][2] + px(6),
            palette_items[-1]["rect"][3] + px(6),
        )

    # Layers rail (sprite thumbnails), top layer first, bounded to the space
    # available so it can never run past the toolbar or the frame.
    ordered = sorted(sprites, key=lambda item: item.z_index, reverse=True)
    thumb_w, thumb_h, thumb_gap = px(72), px(54), px(6)
    rail_right = frame_width - margin
    rail_left = rail_right - thumb_w
    header_h = px(22)
    thumbs_top = content_top + header_h
    capacity = max(
        0, (content_bottom - thumbs_top + thumb_gap) // (thumb_h + thumb_gap)
    )
    hidden = 0
    visible = ordered
    if len(ordered) > capacity:
        visible_count = max(0, capacity - 1)
        visible = ordered[:visible_count]
        hidden = len(ordered) - visible_count
    thumbnail_items: list[ThumbnailItem] = []
    y = thumbs_top
    for sprite in visible:
        thumbnail_items.append(
            {
                "id": f"thumb:{id(sprite)}",
                "rect": (rail_left, y, rail_right, y + thumb_h),
                "sprite": sprite,
            }
        )
        y += thumb_h + thumb_gap
    layers_rect = None
    if ordered:
        layers_bottom = y - thumb_gap + (px(20) if hidden else 0)
        layers_rect = (
            rail_left - px(6),
            content_top - px(4),
            rail_right + px(6),
            min(content_bottom, layers_bottom + px(6)),
        )

    side_left = (palette_rect[2] if palette_rect else 0) + margin
    side_right = (layers_rect[0] if layers_rect else frame_width) - margin
    guide_width = max(px(200), min(px(640), side_right - side_left))
    guide_left = max(side_left, (frame_width - guide_width) // 2)
    # Tall enough for the title row plus each step's label AND its detail
    # line; at px(60) the detail text was clipped by the panel edge.
    guide_rect = (
        guide_left,
        content_top,
        min(frame_width - margin, guide_left + guide_width),
        content_top + px(78),
    )

    help_width = min(frame_width - 2 * margin, px(1000))
    help_height = min(content_bottom - content_top, px(500))
    help_left = (frame_width - help_width) // 2
    help_top = content_top + max(0, (content_bottom - content_top - help_height) // 2)
    help_rect = (help_left, help_top, help_left + help_width, help_top + help_height)
    close_w = px(104)
    close_h = px(34)
    help_close_rect = (
        help_rect[2] - px(14) - close_w,
        help_rect[3] - px(12) - close_h,
        help_rect[2] - px(14),
        help_rect[3] - px(12),
    )

    status_right = (layers_rect[0] - margin) if layers_rect else frame_width - margin
    # Below the quick start card so the two can be visible together.
    status_top = guide_rect[3] + margin
    status_panel_rect = (
        max(margin, status_right - px(320)),
        status_top,
        status_right,
        min(content_bottom, status_top + px(232)),
    )

    # Voice card bottom-left; toasts centered in the space to its right so
    # the two never overlap.
    voice_max_width = max(px(200), min(px(440), int(frame_width * 0.34)))
    toast_left = margin + voice_max_width + margin
    toast_right = frame_width - margin
    toast_max_width = max(
        px(160), min(int(frame_width * 0.42), toast_right - toast_left)
    )

    return UiLayout(
        width=frame_width,
        height=frame_height,
        scale=scale,
        margin=margin,
        top_bar=top_bar,
        toolbar_rect=toolbar_rect,
        palette_rect=palette_rect,
        palette_items=palette_items,
        toolbar_items=toolbar_items,
        layers_rect=layers_rect,
        thumbnail_items=thumbnail_items,
        hidden_layer_count=hidden,
        guide_rect=guide_rect,
        help_rect=help_rect,
        help_close_rect=help_close_rect,
        status_panel_rect=status_panel_rect,
        voice_anchor=(margin, content_bottom),
        voice_max_width=voice_max_width,
        toast_anchor=((toast_left + toast_right) // 2, content_bottom),
        toast_max_width=toast_max_width,
        workspace_rect=(0, top_bar[3], frame_width, toolbar_rect[1]),
    )
