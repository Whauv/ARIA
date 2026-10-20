from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..drawing.sprite import Sprite


@dataclass
class SceneSnapshot:
    canvas: np.ndarray
    strokes: list[dict[str, object]]
    brush_color: tuple[int, int, int]
    sprites: list[Sprite]
    interaction_mode: str
    brush_name: str


def clone_sprite(sprite: Sprite) -> Sprite:
    return Sprite(
        original_img=sprite.original_img.copy(),
        img=sprite.img.copy(),
        x=sprite.x,
        y=sprite.y,
        w=sprite.w,
        h=sprite.h,
        dragging=False,
        z_index=sprite.z_index,
        selected=sprite.selected,
        drag_offset_x=sprite.drag_offset_x,
        drag_offset_y=sprite.drag_offset_y,
    )


class SceneHistory:
    """Bounded whole-scene history for canvas and sprite mutations."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self._snapshots: list[SceneSnapshot] = []
        self._index = -1

    def record(self, snapshot: SceneSnapshot) -> None:
        self._snapshots = self._snapshots[: self._index + 1]
        self._snapshots.append(snapshot)
        if len(self._snapshots) > self.limit:
            self._snapshots.pop(0)
        self._index = len(self._snapshots) - 1

    def undo(self) -> SceneSnapshot | None:
        if self._index <= 0:
            return None
        self._index -= 1
        return self._snapshots[self._index]

    def redo(self) -> SceneSnapshot | None:
        if self._index >= len(self._snapshots) - 1:
            return None
        self._index += 1
        return self._snapshots[self._index]


def save_project(path: str, snapshot: SceneSnapshot) -> str:
    project_path = Path(path).with_suffix(".npz")
    metadata = {
        "interaction_mode": snapshot.interaction_mode,
        "brush_name": snapshot.brush_name,
        "brush_color": snapshot.brush_color,
        "strokes": snapshot.strokes,
        "sprites": [
            {
                "x": sprite.x,
                "y": sprite.y,
                "w": sprite.w,
                "h": sprite.h,
                "z_index": sprite.z_index,
                "selected": sprite.selected,
            }
            for sprite in snapshot.sprites
        ],
    }
    arrays: dict[str, np.ndarray] = {"canvas": snapshot.canvas}
    for index, sprite in enumerate(snapshot.sprites):
        arrays[f"sprite_{index}_original"] = sprite.original_img
        arrays[f"sprite_{index}_image"] = sprite.img
    np.savez_compressed(project_path, metadata=json.dumps(metadata), **arrays)
    return str(project_path)


def load_project(path: str) -> SceneSnapshot:
    with np.load(Path(path).with_suffix(".npz"), allow_pickle=False) as project:
        metadata = json.loads(str(project["metadata"].item()))
        sprites = []
        for index, sprite_data in enumerate(metadata["sprites"]):
            sprites.append(
                Sprite(
                    original_img=project[f"sprite_{index}_original"].copy(),
                    img=project[f"sprite_{index}_image"].copy(),
                    dragging=False,
                    drag_offset_x=-1,
                    drag_offset_y=-1,
                    **sprite_data,
                )
            )
        return SceneSnapshot(
            canvas=project["canvas"].copy(),
            strokes=metadata["strokes"],
            brush_color=tuple(metadata["brush_color"]),
            sprites=sprites,
            interaction_mode=metadata["interaction_mode"],
            brush_name=metadata["brush_name"],
        )
