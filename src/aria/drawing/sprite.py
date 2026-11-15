from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..config import MAX_SPRITE_SIZE, MIN_SPRITE_SIZE
from ..ui import theme


def fit_sprite_dimensions(
    original_width: int, original_height: int, target_width: float
) -> tuple[int, int]:
    """Aspect-preserving size for a sprite scaled to ``target_width``.

    Bounds apply to the longest side, so thin artwork (a single line) keeps its
    proportions instead of being stretched to a minimum height, and neither
    side can collapse to zero.
    """
    if original_width <= 0 or original_height <= 0:
        side = max(1, int(target_width))
        return side, side
    longest = max(original_width, original_height)
    scale = max(target_width, 1.0) / original_width
    scale = max(MIN_SPRITE_SIZE / longest, min(MAX_SPRITE_SIZE / longest, scale))
    return (
        max(1, round(original_width * scale)),
        max(1, round(original_height * scale)),
    )


@dataclass(eq=False)
class Sprite:
    """A placed image. Compared by identity, never by value.

    ``eq=False`` is load-bearing: a value-comparing dataclass would compare the
    numpy image fields, so ``sprites.remove(sprite)`` and ``sprite in sprites``
    would raise ValueError as soon as the list held more than one sprite (numpy
    arrays of different shapes cannot be broadcast, and equal shapes give an
    ambiguous truth value). The runtime tracks sprites by ``id(...)``, so
    identity is also the semantics every call site already expects.
    """

    original_img: np.ndarray
    img: np.ndarray
    x: int
    y: int
    w: int
    h: int
    dragging: bool
    z_index: int
    selected: bool
    drag_offset_x: int = -1
    drag_offset_y: int = -1

    def contains_point(self, point: tuple[int, int]) -> bool:
        px, py = point
        local_x = int(px) - self.x
        local_y = int(py) - self.y
        image_height, image_width = self.img.shape[:2]
        if not (0 <= local_x < min(self.w, image_width)):
            return False
        if not (0 <= local_y < min(self.h, image_height)):
            return False
        if self.img.ndim == 3 and self.img.shape[2] == 4:
            return int(self.img[local_y, local_x, 3]) > 20
        return True

    def contains_point_padded(self, point: tuple[int, int], padding: int) -> bool:
        px, py = point
        return (
            self.x - padding <= px <= self.x + self.w + padding
            and self.y - padding <= py <= self.y + self.h + padding
        )

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.w / 2.0, self.y + self.h / 2.0

    def clamp_to_frame(self, frame_width: int, frame_height: int) -> None:
        self.x = max(0, min(frame_width - self.w, self.x))
        self.y = max(0, min(frame_height - self.h, self.y))

    def resize_from_original(self, scale_factor: float) -> None:
        if scale_factor <= 0:
            return
        self.resize_to_dimensions(self.w * scale_factor)

    def resize_to_dimensions(
        self, target_width: float, keep_center: bool = True
    ) -> None:
        original_h, original_w = self.original_img.shape[:2]
        new_w, new_h = fit_sprite_dimensions(original_w, original_h, target_width)
        if (new_w, new_h) == (self.w, self.h) and self.img.shape[:2] == (new_h, new_w):
            return
        center_x, center_y = self.center
        interpolation = cv2.INTER_AREA if new_w < original_w else cv2.INTER_LINEAR
        self.img = cv2.resize(
            self.original_img, (new_w, new_h), interpolation=interpolation
        )
        self.w = new_w
        self.h = new_h
        if keep_center:
            self.x = round(center_x - new_w / 2.0)
            self.y = round(center_y - new_h / 2.0)

    def scale_by_factor(self, factor: float) -> None:
        self.resize_from_original(factor)


def create_sprite_from_canvas(canvas: np.ndarray, z_index: int) -> Sprite | None:
    canvas_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    non_zero_points = cv2.findNonZero(canvas_gray)
    if non_zero_points is None:
        return None

    x, y, w, h = cv2.boundingRect(non_zero_points)
    cropped = canvas[y : y + h, x : x + w]
    sprite_bgra = cv2.cvtColor(cropped, cv2.COLOR_BGR2BGRA)

    alpha_mask = np.where(np.any(cropped > 0, axis=2), 255, 0).astype(np.uint8)
    sprite_bgra[:, :, 3] = alpha_mask

    return Sprite(
        original_img=sprite_bgra.copy(),
        img=sprite_bgra.copy(),
        x=x,
        y=y,
        w=w,
        h=h,
        dragging=False,
        z_index=z_index,
        selected=False,
    )


def overlay_sprite(frame: np.ndarray, sprite: Sprite) -> None:
    frame_height, frame_width = frame.shape[:2]
    x1 = max(0, sprite.x)
    y1 = max(0, sprite.y)
    x2 = min(frame_width, sprite.x + sprite.w)
    y2 = min(frame_height, sprite.y + sprite.h)

    if x1 >= x2 or y1 >= y2:
        return

    sprite_x1 = x1 - sprite.x
    sprite_y1 = y1 - sprite.y
    sprite_x2 = sprite_x1 + (x2 - x1)
    sprite_y2 = sprite_y1 + (y2 - y1)

    sprite_region = sprite.img[sprite_y1:sprite_y2, sprite_x1:sprite_x2]
    alpha = sprite_region[:, :, 3].astype(np.float32) / 255.0
    alpha = alpha[:, :, np.newaxis]

    frame_region = frame[y1:y2, x1:x2].astype(np.float32)
    sprite_rgb = sprite_region[:, :, :3].astype(np.float32)
    blended = alpha * sprite_rgb + (1.0 - alpha) * frame_region
    frame[y1:y2, x1:x2] = blended.astype(np.uint8)


def _selection_rect(sprite: Sprite, offset: int) -> tuple[int, int, int, int]:
    return (
        sprite.x - offset,
        sprite.y - offset,
        sprite.x + sprite.w + offset,
        sprite.y + sprite.h + offset,
    )


def _tag(frame: np.ndarray, sprite: Sprite, text: str, color, offset: int) -> None:
    scale = theme.TYPE_CAPTION
    width, height = theme.text_size(text, scale)
    left = max(0, sprite.x - offset)
    bottom = sprite.y - offset - 4
    if bottom - height - 8 < 0:
        bottom = sprite.y + sprite.h + offset + height + 12
    theme.fill_rect(
        frame,
        (left, bottom - height - 8, left + width + 12, bottom),
        theme.GRAPHITE_900,
    )
    theme.draw_text(frame, text, (left + 6, bottom - 5), scale, color)


def draw_sprite_selection(
    frame: np.ndarray, sprite: Sprite, label: str | None = None
) -> None:
    """Thin outline with corner handles, kept outside the artwork."""
    if not sprite.selected:
        return
    offset = 4
    left, top, right, bottom = _selection_rect(sprite, offset)
    color = theme.SIGNAL_SELECT
    cv2.rectangle(frame, (left, top), (right, bottom), color, 1, theme.LINE)
    handle = 4
    for cx, cy in ((left, top), (right, top), (left, bottom), (right, bottom)):
        cv2.rectangle(
            frame,
            (cx - handle, cy - handle),
            (cx + handle, cy + handle),
            theme.GRAPHITE_950,
            -1,
        )
        cv2.rectangle(
            frame, (cx - handle, cy - handle), (cx + handle, cy + handle), color, 1
        )
    if label:
        _tag(frame, sprite, label, color, offset)


def draw_sprite_candidate(frame: np.ndarray, sprite: Sprite) -> None:
    """Hover state: dashed outline only, never confused with selection."""
    if sprite.selected:
        return
    theme.dashed_rect(frame, _selection_rect(sprite, 4), theme.SIGNAL_SELECT)


def draw_sprite_delete_confirmation(frame: np.ndarray, sprite: Sprite) -> None:
    left, top, right, bottom = _selection_rect(sprite, 6)
    cv2.rectangle(
        frame, (left, top), (right, bottom), theme.STATUS_ERROR, 2, theme.LINE
    )
    _tag(frame, sprite, "Pinch again to delete", theme.STATUS_ERROR, 6)
