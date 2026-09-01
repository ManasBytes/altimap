"""Locate and load the raw GAMUS tiles for classifier training.

The gamus-terrain frontend's public/*-{rgb,classes}.jpg files are lossy,
JPEG-recompressed derivatives of this data -- decoding a classes.jpg back
into a class index requires nearest-palette-color matching because
compression drifts every pixel by a few RGB values. That drift is label
noise with no signal in it. The .h5 sources here carry the real thing: exact
integer class indices (float32-encoded but exactly 0.0-6.0, no rounding
needed beyond casting) and true metric AGL (height above ground level, in
metres) at native 1024x1024 resolution, so training against these instead of
the JPEGs removes an entire, avoidable source of noise from the labels.

Two source trees, same {rgb,classes,heights}/{split}/{id}_{RGB,CLS,AGL}.h5
layout, no manifest needed for the second:
    GAMUS_50_each/   -- 50 scenes per split (150 total)
    GAMUS_extra_15/  -- 5 more scenes per split (15 total)
165 total, matching exactly what's baked into gamus-terrain/public.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_ROOTS = (
    Path("/home/biplab-dev/GAMUS_50_each"),
    Path("/home/biplab-dev/GAMUS_extra_15"),
)


@dataclass(frozen=True)
class Tile:
    scene_id: str
    split: str
    rgb: Path
    classes: Path
    height: Path


def find_tiles(roots: tuple[Path, ...] = DEFAULT_ROOTS) -> list[Tile]:
    tiles = []
    seen = set()
    for root in roots:
        images_dir = root / "images"
        if not images_dir.is_dir():
            continue
        for split_dir in sorted(images_dir.iterdir()):
            split = split_dir.name
            for rgb_path in sorted(split_dir.glob("*_RGB.h5")):
                scene_id = rgb_path.name.removesuffix("_RGB.h5")
                key = (split, scene_id)
                if key in seen:
                    continue  # extra_15 and 50_each never overlap in practice, but don't double-count if they did
                classes_path = root / "classes" / split / f"{scene_id}_CLS.h5"
                height_path = root / "heights" / split / f"{scene_id}_AGL.h5"
                if not (classes_path.exists() and height_path.exists()):
                    continue
                seen.add(key)
                tiles.append(Tile(scene_id, split, rgb_path, classes_path, height_path))
    return tiles


def load_h5(path: Path):
    import h5py

    with h5py.File(path, "r") as f:
        return f["image"][()]


def load_tile(tile: Tile):
    """-> (rgb uint8 HxWx3, class_map uint8 HxW, agl_metres float32 HxW)."""
    import numpy as np

    rgb = load_h5(tile.rgb)
    classes = load_h5(tile.classes).astype(np.uint8)
    height = load_h5(tile.height).astype(np.float32)
    return rgb, classes, height
