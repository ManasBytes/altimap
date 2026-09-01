"""Runs the DA3 -> geo/metrics -> terrain pipeline against an uploaded image.

This is the same processing viewer/server.py (FastAPI) used to do, ported to
run in-process inside Django instead of behind a second HTTP service --
viewer/ itself is untouched; this module is just a second caller of it. Kept
free of any Django import (models, settings) so it stays a plain function of
(path in, files + dict out), the same shape the project's other exporters use.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
from PIL import Image

from viewer.geo import encode_rg16, fit_absolute_elevation, read_geo_meta
from viewer.metrics import luminance, scene_metrics
from viewer.terrain import build_terrain, height_field

MAX_UPLOAD_BYTES = 200 * 1024 * 1024
ALLOWED_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
PROCESS_RES = 504

_model = None
_device = None
_dem = None


class ProcessingError(Exception):
    """A user-facing processing failure (bad/degenerate imagery), as opposed
    to a bug -- callers surface `str(exc)` to the client."""


def _get_model():
    """Loaded on first use, not at import -- so `manage.py runserver` starts
    instantly and the multi-second model load only happens on the first
    upload, not on every process restart."""
    global _model, _device
    if _model is None:
        import torch
        from depth_anything_3.api import DepthAnything3

        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _model = DepthAnything3.from_pretrained("depth-anything/DA3-SMALL").to(device=_device)
    return _model, _device


def _get_dem():
    global _dem
    if _dem is None:
        from viewer.dem import DemSource

        _dem = DemSource()
    return _dem


def safe_stem(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", Path(name).stem)[:40] or "upload"


def _read_rgb(path: Path) -> np.ndarray:
    """Uploaded imagery as HxWx3 uint8, via rasterio then PIL."""
    try:
        import rasterio

        with rasterio.open(path) as src:
            bands = min(3, src.count)
            arr = np.transpose(src.read(list(range(1, bands + 1))), (1, 2, 0))
        if arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        if arr.dtype != np.uint8:
            a = arr.astype(np.float64)
            lo, hi = np.percentile(a, [2, 98])
            arr = np.clip((a - lo) / max(hi - lo, 1e-9) * 255, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(arr[:, :, :3])
    except Exception:
        return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def process_scene(source_path: Path, scene_id: str, original_filename: str, scene_dir: Path) -> dict:
    """Runs the model + geo/metrics + terrain build and writes depth.png,
    rgb.jpg, terrain.glb and meta.json into `scene_dir`. Returns the same
    record dict that gets written to meta.json; the caller (scenes.views)
    copies the relevant fields onto the Scene row."""
    model, _ = _get_model()
    rgb = _read_rgb(source_path)

    started = time.perf_counter()
    prediction = model.inference([rgb], process_res=PROCESS_RES)
    depth = np.asarray(prediction.depth[0], dtype=np.float32)
    if not np.isfinite(depth).all():
        raise ProcessingError("model produced non-finite depth for this image")
    conf = np.asarray(prediction.conf[0], dtype=np.float32) if prediction.conf is not None else None
    elapsed = time.perf_counter() - started

    h, w = depth.shape
    rgb_small = np.asarray(Image.fromarray(rgb).resize((w, h), Image.BILINEAR), dtype=np.uint8)
    lum = luminance(rgb_small.astype(np.float64) / 255.0)

    record = {
        "id": scene_id,
        "source_image": original_filename,
        "source_size": [int(rgb.shape[1]), int(rgb.shape[0])],
        "width": w,
        "height": h,
        "is_metric": bool(prediction.is_metric),
        "encoding": "rg16-png",
        "seconds": round(elapsed, 3),
        **scene_metrics(depth, lum, conf),
    }

    geo = read_geo_meta(source_path)
    record["geo"] = geo
    record["absolute"] = None

    if geo["georeferenced"]:
        try:
            patch = _get_dem().patch(geo["bounds"], geo["crs"], (h, w))
        except Exception:
            patch = None
        if patch is not None:
            scale, offset, r2 = fit_absolute_elevation(height_field(depth), patch)
            valid = patch[np.isfinite(patch)]
            usable = bool(np.isfinite(r2) and r2 >= 0.3 and np.isfinite(scale) and scale > 0)
            record["absolute"] = {
                "source": "3dep-seamless",
                "scale_m": scale, "offset_m": offset, "fit_r2": r2,
                "usable": usable,
                "reject_reason": None if usable else (
                    "inverted (negative scale)" if np.isfinite(scale) and scale <= 0
                    else "weak fit" if np.isfinite(r2) else "unfittable"),
                "dem_min_m": float(valid.min()) if valid.size else None,
                "dem_max_m": float(valid.max()) if valid.size else None,
                "reference_is_bare_earth": True,
                "reference_posting_m": 10.0,
            }

    scene_dir.mkdir(parents=True, exist_ok=True)
    encoded, lo, hi = encode_rg16(depth)
    record["depth_lo"], record["depth_hi"] = lo, hi
    Image.fromarray(encoded).save(scene_dir / "depth.png", compress_level=6)
    Image.fromarray(rgb_small).save(scene_dir / "rgb.jpg", quality=88)

    plane = (record["plane"]["a"], record["plane"]["b"], record["plane"]["c"])
    build_terrain(height_field(depth, plane), scene_dir / "rgb.jpg", scene_dir / "terrain.glb", res=256)
    record["has_glb"] = True

    (scene_dir / "meta.json").write_text(json.dumps(record))
    return record


def clean_metric(value) -> float | None:
    """JSON/JS have no NaN in a strict sense; unmeasurable metrics travel as
    null on the API rather than a float that fails Number.isFinite checks."""
    if value is None:
        return None
    f = float(value)
    return None if not np.isfinite(f) else round(f, 4)
