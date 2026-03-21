from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from config import MAX_SPRITE_SIZE, MIN_SPRITE_SIZE, SELECTED_BORDER_COLOR, SELECTION_BORDER_THICKNESS


@dataclass
class Sprite:
    original_img: np.ndarray
    img: np.ndarray
    x: int
    y: int
    w: int
    h: int
    dragging: bool
    z_index: int
    selected: bool

    def contains_point(self, point: tuple[int, int]) -> bool:
        px, py = point
        return self.x <= px <= self.x + self.w and self.y <= py <= self.y + self.h

    def clamp_to_frame(self, frame_width: int, frame_height: int) -> None:
        self.x = max(0, min(frame_width - self.w, self.x))
        self.y = max(0, min(frame_height - self.h, self.y))

    def resize_from_original(self, scale_factor: float) -> None:
        if scale_factor <= 0:
            return

        new_w = max(MIN_SPRITE_SIZE, min(MAX_SPRITE_SIZE, int(self.w * scale_factor)))
        original_h, original_w = self.original_img.shape[:2]
        aspect_ratio = original_h / original_w if original_w else 1.0
        new_h = max(MIN_SPRITE_SIZE, min(MAX_SPRITE_SIZE, int(new_w * aspect_ratio)))

        self.img = cv2.resize(self.original_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        self.w = new_w
        self.h = new_h

    def scale_by_factor(self, factor: float) -> None:
        self.resize_from_original(factor)


def create_sprite_from_canvas(canvas: np.ndarray, z_index: int) -> Optional[Sprite]:
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


def draw_sprite_selection(frame: np.ndarray, sprite: Sprite) -> None:
    if not sprite.selected:
        return

    top_left = (sprite.x, sprite.y)
    bottom_right = (sprite.x + sprite.w, sprite.y + sprite.h)
    cv2.rectangle(frame, top_left, bottom_right, SELECTED_BORDER_COLOR, SELECTION_BORDER_THICKNESS)
