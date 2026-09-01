"""Local app server: serves the dashboards and runs DA3 on uploaded imagery.

The static dashboards cannot do this themselves -- DA3 needs a GPU and a Python
process -- so this wraps the same pipeline modules the batch exporters use. An
uploaded file therefore gets byte-identical treatment to a batch-exported scene:
same metrics, same rg16 encoding, same georeferencing and calibration rules.

    .venv-da3/bin/python -m viewer.server
    -> http://localhost:8000/

Bound to 127.0.0.1 by default. This accepts file uploads and runs inference on
them; do not expose it to a network you do not trust.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import uuid
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from viewer.geo import (
    encode_rg16,
    fit_absolute_elevation,
    geo_meta_from_dataset,
    read_geo_meta,
)
from viewer.metrics import luminance, scene_metrics
from viewer.terrain import build_terrain, height_field

WEB_DIR = Path(__file__).resolve().parent / "web"
UPLOAD_DIR = WEB_DIR / "data-uploads"
SCENES_DIR = UPLOAD_DIR / "scenes"
CLASSIFICATIONS_DIR = UPLOAD_DIR / "classifications"
RECONSTRUCTIONS_DIR = UPLOAD_DIR / "reconstructions"

MAX_UPLOAD_BYTES = 200 * 1024 * 1024
TIFF_SUFFIXES = {".tif", ".tiff", ".geotif", ".geotiff"}
ALLOWED_SUFFIXES = {*TIFF_SUFFIXES, ".png", ".jpg", ".jpeg"}
PROCESS_RES = 504
CLASSIFY_RES = 513  # matches gamus-terrain's HEIGHT_SAMPLE_WIDTH/HEIGHT
RECONSTRUCT_RES = CLASSIFY_RES

# Direct GeoTIFF reconstruction uses only georeferenced elevation values.
# Class labels are visual/semantic output; they never modify the height field.
DIRECT_RECONSTRUCT_RES = 1025  # 4× the 513² interactive-grid cell count

app = FastAPI(title="AltiMap")

# The gamus-terrain viewer (Vite dev server) runs on a different origin than
# this API, and the classify-static response is consumed straight into a
# canvas for pixel readback -- an uncorsed image would taint that canvas and
# make getImageData throw, so this has to be wide open on responses, not just
# reachable.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None
_device = None
_class_model = None
_class_device = None
_dem = None
_global_dem = None
_building_indexes: dict[str, object] = {}


def _get_model():
    """Loaded on first use, not at import -- otherwise every static file request
    would wait behind a multi-second model load at startup."""
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


def _get_global_dem():
    """Copernicus GLO-30 fallback when the 3DEP US DEM has no coverage."""
    global _global_dem
    if _global_dem is None:
        from viewer.dem import DemSource

        _global_dem = DemSource(collection="cop-dem-glo-30")
    return _global_dem


def _get_class_model():
    """Loaded on first use, same reasoning as _get_model: a multi-second
    checkpoint load must not block every static file request at startup."""
    global _class_model, _class_device
    if _class_model is None:
        from viewer.classify import load_model

        _class_model, _class_device = load_model()
    return _class_model, _class_device


def _safe_stem(name: str) -> str:
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
            # 16-bit imagery is common in remote sensing; stretch to 8-bit for
            # both the model and the browser texture.
            a = arr.astype(np.float64)
            lo, hi = np.percentile(a, [2, 98])
            arr = np.clip((a - lo) / max(hi - lo, 1e-9) * 255, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(arr[:, :, :3])
    except Exception:
        return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _process(path: Path, scene_id: str, want_glb: bool, original_name: str) -> dict:
    model, _ = _get_model()
    rgb = _read_rgb(path)

    started = time.perf_counter()
    prediction = model.inference([rgb], process_res=PROCESS_RES)
    depth = np.asarray(prediction.depth[0], dtype=np.float32)
    if not np.isfinite(depth).all():
        raise HTTPException(422, "model produced non-finite depth for this image")
    conf = np.asarray(prediction.conf[0], dtype=np.float32) if prediction.conf is not None else None
    elapsed = time.perf_counter() - started

    h, w = depth.shape
    rgb_small = np.asarray(Image.fromarray(rgb).resize((w, h), Image.BILINEAR), dtype=np.uint8)
    lum = luminance(rgb_small.astype(np.float64) / 255.0)

    record = {
        "id": scene_id,
        "class": "upload",
        "source_image": original_name,
        "source_size": [int(rgb.shape[1]), int(rgb.shape[0])],
        "width": w,
        "height": h,
        "is_metric": bool(prediction.is_metric),
        "encoding": "rg16-png",
        "seconds": round(elapsed, 3),
        **scene_metrics(depth, lum, conf),
    }

    geo = read_geo_meta(path)
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

    scene_dir = SCENES_DIR / scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)
    encoded, lo, hi = encode_rg16(depth)
    record["depth_lo"], record["depth_hi"] = lo, hi
    Image.fromarray(encoded).save(scene_dir / "depth.png", compress_level=6)
    Image.fromarray(rgb_small).save(scene_dir / "rgb.jpg", quality=88)

    if want_glb:
        plane = (record["plane"]["a"], record["plane"]["b"], record["plane"]["c"])
        build_terrain(height_field(depth, plane), scene_dir / "rgb.jpg",
                      scene_dir / "terrain.glb", res=256)
        record["has_glb"] = True

    (scene_dir / "meta.json").write_text(json.dumps(record))
    return record


@app.post("/api/upload")
async def upload(file: UploadFile = File(...), glb: bool = True):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(415, f"unsupported type {suffix or '(none)'}; "
                                 f"expected one of {sorted(ALLOWED_SUFFIXES)}")

    data = await file.read()
    if not data:
        raise HTTPException(400, "empty upload")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"file is {len(data)/1e6:.0f} MB; limit is "
                                 f"{MAX_UPLOAD_BYTES/1e6:.0f} MB")

    scene_id = f"{_safe_stem(file.filename or 'upload')}__{uuid.uuid4().hex[:8]}"
    staged = SCENES_DIR / scene_id / f"source{suffix}"
    staged.parent.mkdir(parents=True, exist_ok=True)
    staged.write_bytes(data)

    try:
        record = _process(staged, scene_id, want_glb=glb,
                          original_name=Path(file.filename or 'upload').name)
    finally:
        # The source is not served; the derived assets are what the viewer needs.
        staged.unlink(missing_ok=True)

    _write_upload_index()
    return JSONResponse(record)


def _png_data_uri(arr: np.ndarray) -> str:
    import base64
    import io

    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _finite_json_number(value):
    """Use None, rather than NaN, for values carried in the JSON response."""
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def _rgb_from_raster_dataset(source) -> np.ndarray:
    """Read the visible bands from an open GDAL dataset as model-ready RGB."""
    bands = min(3, source.count)
    if bands < 1:
        raise ValueError("the GeoTIFF has no raster bands")
    rgb = np.transpose(source.read(list(range(1, bands + 1))), (1, 2, 0))
    if rgb.shape[2] == 1:
        rgb = np.repeat(rgb, 3, axis=2)
    elif rgb.shape[2] == 2:
        rgb = np.concatenate([rgb, rgb[:, :, 1:2]], axis=2)
    if rgb.dtype != np.uint8:
        values = rgb.astype(np.float64)
        low, high = np.percentile(values, [2, 98])
        rgb = np.clip(
            (values - low) / max(high - low, 1e-9) * 255, 0, 255
        ).astype(np.uint8)
    return np.ascontiguousarray(rgb[:, :, :3])


def _read_upload_rgb_and_geo(
    data: bytes, suffix: str
) -> tuple[np.ndarray, dict | None, dict | None, dict | None]:
    """Read an uploaded image while retaining its GeoTIFF spatial profile.

    The RGB array is a display/model representation. The original profile,
    CRS, affine transform, and dataset tags are retained separately so the
    predicted class raster can be written back on exactly the source grid.
    """
    import io

    if suffix in TIFF_SUFFIXES:
        try:
            import rasterio

            with rasterio.MemoryFile(data) as memfile, memfile.open() as source:
                rgb = _rgb_from_raster_dataset(source)
                geo = geo_meta_from_dataset(source)
                geo.update(
                    {
                        "transform": [float(value) for value in source.transform],
                        "nodata": _finite_json_number(source.nodata),
                        "driver": source.driver,
                        "band_descriptions": list(source.descriptions),
                    }
                )
                return rgb, geo, source.profile.copy(), source.tags().copy()
        except Exception as error:
            # Do not silently use Pillow here: that would display the pixels
            # while dropping the coordinate reference the user supplied.
            raise HTTPException(422, f"could not read GeoTIFF metadata: {error}") from error
    try:
        return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"), dtype=np.uint8), None, None, None
    except Exception as error:
        raise HTTPException(422, f"could not read image: {error}") from error


def _write_georeferenced_class_mask(
    class_map: np.ndarray,
    source_profile: dict,
    source_tags: dict,
    source_name: str,
) -> str:
    """Persist a class mask on the original GeoTIFF grid with CRS intact."""
    import rasterio

    CLASSIFICATIONS_DIR.mkdir(parents=True, exist_ok=True)
    mask_name = f"{_safe_stem(source_name)}__{uuid.uuid4().hex[:8]}_classes.tif"
    destination_path = CLASSIFICATIONS_DIR / mask_name
    profile = source_profile.copy()
    # Semantic IDs are uint8; spatial keys (crs/transform/width/height) stay
    # unchanged because class_map is restored to the source image dimensions.
    profile.pop("photometric", None)
    profile.pop("interleave", None)
    profile.pop("predictor", None)
    profile.update(count=1, dtype="uint8", nodata=255, compress="deflate")
    with rasterio.open(destination_path, "w", **profile) as destination:
        destination.write(class_map.astype(np.uint8), 1)
        if source_tags:
            destination.update_tags(**source_tags)
        destination.update_tags(
            semantic_classes="background,ground,low_vegetation,buildings,water,roads,trees",
            model="seven-class static land-cover classifier",
            source_image=source_name,
        )
    return f"/data-uploads/classifications/{mask_name}"


def _resize_rgb(image: np.ndarray, width: int = RECONSTRUCT_RES, height: int = RECONSTRUCT_RES) -> np.ndarray:
    return np.asarray(
        Image.fromarray(image).resize((width, height), Image.Resampling.BILINEAR),
        dtype=np.uint8,
    )


def _resize_height(field: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize a float elevation/depth raster without propagating nodata."""
    values = np.asarray(field, dtype=np.float32)
    finite = np.isfinite(values)
    if not finite.any():
        raise HTTPException(422, "elevation source contains no finite values")
    filled = np.where(finite, values, float(np.nanmedian(values[finite])))
    return np.asarray(
        Image.fromarray(filled, mode="F").resize(
            (width, height), Image.Resampling.BILINEAR
        ),
        dtype=np.float32,
    )


def _height_preview(field: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Float field -> browser-readable grayscale while retaining its range."""
    values = np.asarray(field, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise HTTPException(422, "elevation source contains no finite values")
    low, high = float(finite.min()), float(finite.max())
    span = max(high - low, 1e-9)
    preview = np.clip((values - low) / span * 255, 0, 255)
    preview[~np.isfinite(preview)] = 0
    return preview.astype(np.uint8), low, high


def _extract_embedded_elevation(
    data: bytes, requested_band: int | None = None
) -> tuple[np.ndarray | None, str | None, str | None]:
    """Find a genuine elevation band embedded in a GeoTIFF, when declared.

    RGB GeoTIFFs usually contain *location* metadata but not a DSM. We only
    treat an internal band as elevation when it is clearly labelled as such,
    or when a georeferenced source has a single non-image band. This avoids
    accidentally turning a near-infrared image band into terrain.
    """
    import rasterio

    pattern = re.compile(r"(?:^|[_\s-])(dem|dsm|dtm|elevation|height|altitude|hag|ndsm)(?:$|[_\s-])", re.I)
    with rasterio.MemoryFile(data) as memfile, memfile.open() as source:
        explicit = source.tags().get("elevation_band")
        candidates: list[tuple[int, str]] = []
        if requested_band is not None:
            if not 1 <= requested_band <= source.count:
                raise HTTPException(
                    422,
                    f"requested elevation band {requested_band} is outside the GeoTIFF's 1–{source.count} bands",
                )
            candidates.append((requested_band, "user-selected GeoTIFF elevation band"))
        if explicit and str(explicit).isdigit():
            band = int(explicit)
            if 1 <= band <= source.count:
                candidates.append((band, "GeoTIFF elevation_band tag"))
        for band in range(1, source.count + 1):
            name = " ".join(
                [
                    source.descriptions[band - 1] or "",
                    *source.tags(band).keys(),
                    *source.tags(band).values(),
                ]
            )
            if pattern.search(name):
                candidates.append((band, "embedded GeoTIFF elevation band"))
        if not candidates:
            for band, dtype in enumerate(source.dtypes, start=1):
                if np.dtype(dtype).kind == "f":
                    candidates.append((band, "embedded floating-point GeoTIFF elevation band"))
        if not candidates and source.count == 1 and source.crs is not None:
            candidates.append((1, "single-band georeferenced elevation raster"))
        if not candidates:
            return None, None, None
        band, label = candidates[0]
        band_text = " ".join(
            [
                source.descriptions[band - 1] or "",
                *source.tags(band).keys(),
                *source.tags(band).values(),
            ]
        ).lower()
        # A DSM already contains roof/canopy height and must not receive a
        # second building extrusion. A DEM/DTM is ground-only and can safely
        # be refined with measured footprint heights.
        elevation_kind = (
            "surface"
            if "dsm" in band_text or "surface" in band_text
            else "terrain"
            if any(token in band_text for token in ("dem", "dtm", "hag", "bare earth"))
            else "unknown"
        )
        field = source.read(band).astype(np.float32)
        if source.nodata is not None and np.isfinite(source.nodata):
            field[field == source.nodata] = np.nan
        return field, label, elevation_kind


def _reference_elevation(geo: dict, shape: tuple[int, int]) -> tuple[np.ndarray, str]:
    """Read real terrain for a georeferenced RGB image, with global fallback."""
    if not geo.get("georeferenced"):
        raise HTTPException(422, "a direct terrain path requires CRS and affine transform")
    patch = _get_dem().patch(geo["bounds"], geo["crs"], shape)
    source = "3dep-seamless"
    if patch is None:
        patch = _get_global_dem().patch(geo["bounds"], geo["crs"], shape)
        source = "cop-dem-glo-30"
    if patch is None:
        raise HTTPException(
            422,
            "no reference DEM coverage for this GeoTIFF; include a DEM/DSM band or use a location with DEM coverage",
        )
    return np.asarray(patch, dtype=np.float32), source


def _bbox_intersects(left, right) -> bool:
    west, south, east, north = left
    other_west, other_south, other_east, other_north = right
    return west < other_east and other_west < east and south < other_north and other_south < north


def _building_index_for_geo(geo: dict):
    """Load an Overture footprint cache only when it covers this GeoTIFF."""
    from rasterio.warp import transform_bounds

    bounds_ll = tuple(transform_bounds(geo["crs"], "EPSG:4326", *geo["bounds"]))
    cache_dir = Path(__file__).resolve().parent / "cache"
    for stem in ("buildings_atlanta", "buildings_inria"):
        parquet_path = cache_dir / f"{stem}.parquet"
        meta_path = cache_dir / f"{stem}.json"
        if not parquet_path.is_file() or not meta_path.is_file():
            continue
        metadata = json.loads(meta_path.read_text())
        coverage = []
        if metadata.get("bbox"):
            coverage.append(tuple(metadata["bbox"]))
        coverage.extend(tuple(bounds) for bounds in metadata.get("regions", {}).values())
        if not any(_bbox_intersects(bounds_ll, area) for area in coverage):
            continue
        if stem not in _building_indexes:
            from viewer.footprints import BuildingIndex

            _building_indexes[stem] = BuildingIndex(parquet_path)
        return _building_indexes[stem], stem
    return None, None


def _refine_with_known_buildings(elevation: np.ndarray, geo: dict) -> tuple[np.ndarray, dict]:
    """Mirror demdirect: extrude only footprints carrying a known height."""
    index, cache_name = _building_index_for_geo(geo)
    if index is None:
        return elevation, {
            "source": None,
            "cache": None,
            "n_footprints": 0,
            "n_with_height": 0,
            "n_extruded": 0,
        }
    from viewer.refine import extrude_known_buildings

    mask, known_m, stats = index.rasterize(
        geo["bounds"], geo["crs"], elevation.shape
    )
    fused, n_extruded = extrude_known_buildings(elevation, mask, known_m)
    return fused.astype(np.float32), {
        "source": "overture-building-heights",
        "cache": cache_name,
        **stats,
        "n_extruded": n_extruded,
    }


def _write_georeferenced_elevation(
    elevation: np.ndarray,
    source_profile: dict,
    source_tags: dict,
    source_name: str,
    elevation_source: str,
) -> str:
    """Write a float32 elevation product on the exact source GeoTIFF grid."""
    import rasterio

    RECONSTRUCTIONS_DIR.mkdir(parents=True, exist_ok=True)
    output_name = f"{_safe_stem(source_name)}__{uuid.uuid4().hex[:8]}_elevation.tif"
    output_path = RECONSTRUCTIONS_DIR / output_name
    profile = source_profile.copy()
    profile.pop("photometric", None)
    profile.pop("interleave", None)
    profile.pop("predictor", None)
    profile.update(count=1, dtype="float32", nodata=np.nan, compress="deflate")
    with rasterio.open(output_path, "w", **profile) as destination:
        destination.write(elevation.astype(np.float32), 1)
        if source_tags:
            destination.update_tags(**source_tags)
        destination.update_tags(
            vertical_unit="m",
            elevation_source=elevation_source,
            source_image=source_name,
        )
    return f"/data-uploads/reconstructions/{output_name}"


def _write_terrain_glb(
    height: np.ndarray,
    rgb: np.ndarray,
    source_name: str,
    height_world_scale: float,
    resolution: int = 512,
) -> str:
    """Export the same textured terrain used by the Three.js viewer as GLB."""
    artifact_name = f"{_safe_stem(source_name)}__{uuid.uuid4().hex[:8]}"
    artifact_dir = RECONSTRUCTIONS_DIR / artifact_name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    texture_path = artifact_dir / "rgb.jpg"
    glb_path = artifact_dir / "terrain.glb"
    Image.fromarray(rgb).save(texture_path, quality=92)
    values = np.asarray(height, dtype=np.float32)
    low, high = float(np.nanmin(values)), float(np.nanmax(values))
    height01 = (
        (values - low) / (high - low)
        if high > low
        else np.zeros_like(values, dtype=np.float32)
    )
    # build_terrain uses a one-unit-wide world; the interactive mesh uses
    # eight units, hence the /8 that retains the same real aspect ratio.
    build_terrain(
        height01,
        texture_path,
        glb_path,
        res=resolution,
        exaggeration=height_world_scale / 8.0,
    )
    return f"/data-uploads/reconstructions/{artifact_name}/terrain.glb"


def _true_scale_world_height(elevation_range_m: float, geo: dict | None) -> float:
    """Map metre relief to the viewer's fixed 8-unit horizontal world width."""
    footprint = max((geo or {}).get("ground_m") or [0.0])
    if not np.isfinite(footprint) or footprint <= 0:
        return 0.62
    return float(np.clip(8.0 * elevation_range_m / footprint, 0.03, 3.0))


@app.post("/api/classify-static")
async def classify_static(file: UploadFile = File(...)):
    """Non-georeferenced path: classify land cover, elevate by fixed
    per-class height constants. No depth model involved -- see
    viewer/classify.py's module docstring for why, and for the plan to
    calibrate a depth model against these same static values later.

    Returns everything as data URIs rather than files under /data-uploads/,
    so the response needs no follow-up fetch and (more importantly) never
    taints the canvas the frontend reads pixels back from -- a real
    same-origin URL would need CORS-correct caching semantics to guarantee
    that on every browser, a data URI just structurally can't fail it.
    """
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(415, f"unsupported type {suffix or '(none)'}; "
                                 f"expected one of {sorted(ALLOWED_SUFFIXES)}")
    data = await file.read()
    if not data:
        raise HTTPException(400, "empty upload")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"file is {len(data)/1e6:.0f} MB; limit is "
                                 f"{MAX_UPLOAD_BYTES/1e6:.0f} MB")

    from viewer.classify import classes_to_rgb, classes_to_static_height, predict_classes

    started = time.perf_counter()
    rgb_full, geo, source_profile, source_tags = _read_upload_rgb_and_geo(data, suffix)
    rgb = np.asarray(
        Image.fromarray(rgb_full).resize((CLASSIFY_RES, CLASSIFY_RES), Image.BILINEAR),
        dtype=np.uint8,
    )

    model, device = _get_class_model()
    class_map = predict_classes(rgb, model, device)
    height = classes_to_static_height(class_map)
    height_u8 = np.clip(height * 255.0, 0, 255).astype(np.uint8)
    elapsed = time.perf_counter() - started

    counts = {name: int((class_map == i).sum()) for i, name in enumerate(
        ["background", "ground", "low_vegetation", "buildings", "water", "roads", "trees"]
    )}

    mask_geo_tiff_url = None
    if geo and geo["georeferenced"]:
        # The classifier runs at its stable 513×513 working size. Reproject its
        # labels back to the *native source grid* with nearest neighbour so no
        # class IDs are blended, while the original affine transform remains
        # exact and the bounds/ground sampling distance stay unchanged.
        full_class_map = np.asarray(
            Image.fromarray(class_map).resize(
                (rgb_full.shape[1], rgb_full.shape[0]), Image.Resampling.NEAREST
            ),
            dtype=np.uint8,
        )
        mask_geo_tiff_url = _write_georeferenced_class_mask(
            full_class_map,
            source_profile,
            source_tags,
            Path(file.filename or "upload.tif").name,
        )

    return JSONResponse({
        "seconds": round(elapsed, 3),
        "width": CLASSIFY_RES,
        "height_px": CLASSIFY_RES,
        "class_pixel_counts": counts,
        "rgb": _png_data_uri(rgb),
        "height": _png_data_uri(np.stack([height_u8] * 3, axis=-1)),
        "classes": _png_data_uri(classes_to_rgb(class_map)),
        "geo": geo,
        "mask_geo_tiff_url": mask_geo_tiff_url,
    })


@app.post("/api/reconstruct")
async def reconstruct(
    file: UploadFile = File(...), elevation_band: int | None = Form(None)
):
    """Build a terrain-ready RGB, height, and class triple from one upload.

    GeoTIFFs follow the direct, georeferenced path: retain their exact grid
    and use an embedded elevation band or a reference DEM. PNG/JPEG files have
    no spatial anchor, so they follow the image path: relative depth estimate
    fused with semantic classes. The response is deliberately shaped like a
    regular viewer scene, so the Three.js renderer does not have to know which
    backend produced it.
    """
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            415,
            f"unsupported type {suffix or '(none)'}; expected one of {sorted(ALLOWED_SUFFIXES)}",
        )
    data = await file.read()
    if not data:
        raise HTTPException(400, "empty upload")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            413,
            f"file is {len(data) / 1e6:.0f} MB; limit is {MAX_UPLOAD_BYTES / 1e6:.0f} MB",
        )

    from viewer.classify import (
        CLASS_NAMES,
        classes_to_rgb,
        classes_to_static_height,
        predict_classes,
    )

    started = time.perf_counter()
    rgb_full, geo, source_profile, source_tags = _read_upload_rgb_and_geo(data, suffix)
    direct_geo = bool(geo and geo.get("georeferenced"))
    render_resolution = DIRECT_RECONSTRUCT_RES if direct_geo else RECONSTRUCT_RES
    render_rgb = _resize_rgb(rgb_full, render_resolution, render_resolution)
    class_model, class_device = _get_class_model()
    class_map = predict_classes(render_rgb, class_model, class_device)
    class_rgb = classes_to_rgb(class_map)
    class_counts = {
        name: int((class_map == index).sum()) for index, name in enumerate(CLASS_NAMES)
    }

    source_name = Path(file.filename or "upload").name
    if direct_geo:
        embedded_elevation, elevation_source, elevation_kind = _extract_embedded_elevation(
            data, elevation_band
        )
        if embedded_elevation is None:
            ground_full, dem_source = _reference_elevation(geo, rgb_full.shape[:2])
            elevation_full, building_refinement = _refine_with_known_buildings(ground_full, geo)
            elevation_source = (
                f"{dem_source} terrain + Overture building heights"
                if building_refinement["n_extruded"]
                else f"{dem_source} terrain"
            )
        else:
            elevation_full = embedded_elevation
            if elevation_kind == "terrain":
                elevation_full, building_refinement = _refine_with_known_buildings(elevation_full, geo)
                if building_refinement["n_extruded"]:
                    elevation_source = f"{elevation_source} + Overture building heights"
            else:
                building_refinement = {
                    "source": "embedded surface model",
                    "cache": None,
                    "n_footprints": 0,
                    "n_with_height": 0,
                    "n_extruded": 0,
                }
        render_height = _resize_height(
            elevation_full, render_resolution, render_resolution
        )
        height_preview, height_low, height_high = _height_preview(render_height)
        class_map_full = np.asarray(
            Image.fromarray(class_map).resize(
                (rgb_full.shape[1], rgb_full.shape[0]), Image.Resampling.NEAREST
            ),
            dtype=np.uint8,
        )
        mask_geo_tiff_url = _write_georeferenced_class_mask(
            class_map_full, source_profile, source_tags, source_name
        )
        elevation_geo_tiff_url = _write_georeferenced_elevation(
            elevation_full, source_profile, source_tags, source_name, elevation_source
        )
        mode = "geotiff-direct"
        height_mode = "absolute"
        height_baseline = 0.0
        height_world_scale = _true_scale_world_height(height_high - height_low, geo)
    else:
        # PNG/JPEG has no map position from which to obtain absolute terrain.
        # DA3 supplies the structural ordering and the semantic map stabilizes
        # buildings, roads, water, and vegetation into a usable relative DSM.
        depth_model, _ = _get_model()
        prediction = depth_model.inference([render_rgb], process_res=RECONSTRUCT_RES)
        relative_depth = height_field(np.asarray(prediction.depth[0], dtype=np.float32))
        relative_depth = _resize_height(relative_depth, RECONSTRUCT_RES, RECONSTRUCT_RES)
        semantic_height = classes_to_static_height(class_map).astype(np.float32)
        render_height = np.clip(0.68 * relative_depth + 0.32 * semantic_height, 0.0, 1.0)
        height_preview, height_low, height_high = _height_preview(render_height)
        elevation_source = "relative depth + semantic class priors"
        mask_geo_tiff_url = None
        elevation_geo_tiff_url = None
        building_refinement = None
        mode = "image-relative"
        # This field is already fused and regularized by the RGB pipeline.
        # Mark it separately so the viewer does not apply its catalog-scene
        # class remapping a second time (which causes terraced, uneven roofs).
        height_mode = "relative-final"
        height_baseline = 0.32
        height_world_scale = 0.62

    export_resolution = (
        min(1024, elevation_full.shape[0] - 1, elevation_full.shape[1] - 1)
        if direct_geo
        else 512
    )
    terrain_glb_url = _write_terrain_glb(
        elevation_full if direct_geo else render_height,
        rgb_full if direct_geo else render_rgb,
        source_name,
        height_world_scale,
        max(64, export_resolution),
    )
    elapsed = time.perf_counter() - started
    height_rgb = np.repeat(height_preview[:, :, None], 3, axis=2)
    return JSONResponse(
        {
            "seconds": round(elapsed, 3),
            "width": render_resolution,
            "height_px": render_resolution,
            "mesh_resolution": render_resolution,
            "source_width": int(rgb_full.shape[1]),
            "source_height": int(rgb_full.shape[0]),
            "rgb": _png_data_uri(render_rgb),
            "height": _png_data_uri(height_rgb),
            "classes": _png_data_uri(class_rgb),
            "class_pixel_counts": class_counts,
            "class_names": CLASS_NAMES,
            "geo": geo,
            "mode": mode,
            "height_mode": height_mode,
            "height_baseline": height_baseline,
            "height_world_scale": height_world_scale,
            "height_range": {"min": height_low, "max": height_high, "unit": "m" if direct_geo else "relative"},
            "elevation_source": elevation_source,
            "building_refinement": building_refinement,
            "mask_geo_tiff_url": mask_geo_tiff_url,
            "elevation_geo_tiff_url": elevation_geo_tiff_url,
            "terrain_glb_url": terrain_glb_url,
        }
    )


@app.get("/api/uploads")
def list_uploads():
    return JSONResponse(_write_upload_index())


@app.delete("/api/uploads/{scene_id}")
def delete_upload(scene_id: str):
    if not re.fullmatch(r"[A-Za-z0-9._-]+", scene_id):
        raise HTTPException(400, "bad scene id")
    target = SCENES_DIR / scene_id
    if not target.is_dir():
        raise HTTPException(404, "no such scene")
    for child in target.iterdir():
        child.unlink()
    target.rmdir()
    return JSONResponse(_write_upload_index())


def _write_upload_index() -> dict:
    scenes = []
    if SCENES_DIR.is_dir():
        for meta_path in sorted(SCENES_DIR.glob("*/meta.json")):
            try:
                m = json.loads(meta_path.read_text())
            except json.JSONDecodeError:
                continue
            scenes.append({
                "id": m["id"], "source_image": m.get("source_image"),
                "plane_r2": _clean(m.get("plane_r2")),
                "structure_alignment": _clean(m.get("structure_alignment")),
                "residual_relief": _clean(m.get("residual_relief")),
                "georeferenced": bool(m.get("geo", {}).get("georeferenced")),
                "calibrated": bool((m.get("absolute") or {}).get("usable")),
                "glb": bool(m.get("has_glb")),
            })
    index = {"dataset": "uploads", "total": len(scenes), "scenes": scenes}
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    (UPLOAD_DIR / "index.json").write_text(json.dumps(index))
    return index


def _clean(v):
    """JSON has no NaN; unmeasurable metrics travel as null."""
    if v is None:
        return None
    f = float(v)
    return None if not np.isfinite(f) else round(f, 4)


@app.get("/api/health")
def health():
    return {"ok": True, "model_loaded": _model is not None, "device": str(_device)}


# Mounted last so /api/* wins over a same-named static path.
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
_write_upload_index()
app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    print(f"AltiMap on http://{args.host}:{args.port}/  (upload at /upload.html)")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
