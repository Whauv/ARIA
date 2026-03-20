from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


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
