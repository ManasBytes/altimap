"""Height field -> textured glTF terrain mesh.

The GLB is a standalone deliverable: openable in Blender or any glTF viewer
without the dashboard, and it is what satisfies the brief's "project the
original optical image onto a generated 3D terrain mesh".

trimesh and Pillow are imported lazily inside build_terrain so that
height_field stays testable in the main venv.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from viewer.metrics import detrend


def height_field(depth: np.ndarray, plane: tuple[float, float, float] | None = None) -> np.ndarray:
    """Depth -> height in [0, 1], where 1 is the highest ground.

    Depth is distance from the sensor, so for a nadir view larger depth means
    LOWER ground -- hence the inversion. Constant input returns all zeros
    rather than nan, so a flat scene renders flat instead of vanishing.
    """
    field = detrend(depth, plane) if plane is not None else np.asarray(depth, dtype=np.float64)
    lo, hi = float(field.min()), float(field.max())
    if hi == lo:
        return np.zeros_like(field, dtype=np.float64)
    return (hi - field) / (hi - lo)


def _resample(field: np.ndarray, res: int) -> np.ndarray:
    from PIL import Image

    if field.shape == (res, res):
        return field
    image = Image.fromarray(field.astype(np.float32), mode="F")
    return np.asarray(image.resize((res, res), Image.BILINEAR), dtype=np.float64)


def terrain_grid(res: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Grid geometry for a res x res terrain patch: (xx, zz, faces, uv).

    Row 0 of the height field is the TOP of the source image and sits at
    zz = -0.5. glTF puts UV (0,0) at the image's upper-left, so that row must
    get v = 0. Using `1 - (zz + 0.5)` instead paints the image upside-down onto
    the mesh -- which is invisible on a symmetric scene and obvious on a road.
    """
    lin = np.linspace(-0.5, 0.5, res)
    xx, zz = np.meshgrid(lin, lin)

    idx = np.arange(res * res).reshape(res, res)
    tl, tr = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel()
    bl, br = idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    faces = np.concatenate([
        np.column_stack([tl, bl, tr]),
        np.column_stack([tr, bl, br]),
    ]).astype(np.int64)

    uv = np.column_stack([xx.ravel() + 0.5, zz.ravel() + 0.5]).astype(np.float32)
    return xx, zz, faces, uv


def build_terrain(
    height01: np.ndarray,
    texture_path: Path,
    out_path: Path,
    res: int = 256,
    exaggeration: float = 0.15,
) -> None:
    """Write a textured, displaced grid as GLB.

    The grid spans [-0.5, 0.5] in X and Z, so every scene arrives at the same
    footprint regardless of source resolution and exaggeration means the same
    thing everywhere. UVs are identity per design doc 5.1 -- DSM and RGB share
    one grid, so there is no reprojection step and no reprojection error.
    """
    import trimesh
    from PIL import Image

    field = _resample(np.asarray(height01, dtype=np.float64), res)
    xx, zz, faces, uv = terrain_grid(res)
    vertices = np.column_stack([
        xx.ravel(),
        (field * exaggeration).ravel(),
        zz.ravel(),
    ]).astype(np.float32)

    material = trimesh.visual.material.SimpleMaterial(image=Image.open(texture_path))
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        visual=trimesh.visual.TextureVisuals(uv=uv, material=material),
        process=False,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(out_path)
