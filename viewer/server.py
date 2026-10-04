"""Local app server: serves the dashboards and runs DA3 on uploaded imagery.

The static dashboards cannot do this themselves -- DA3 needs a GPU and a Python
process -- so this wraps the same pipeline modules the batch exporters use. An
uploaded file therefore gets byte-identical treatment to a batch-exported scene:
same metrics, same rg16 encoding, same georeferencing and calibration rules.

    .venv-da3/bin/python -m viewer.server                  # http://127.0.0.1:8000/
    .venv-da3/bin/python -m viewer.server --host 0.0.0.0   # reachable from other machines (a VM)

Serves the React viewer (frontend/dist) at "/" when it has been built, the older
dashboards at /dashboards/, and the API under /api/.

Bound to 127.0.0.1 by default. This accepts file uploads and runs inference on
them; do not expose it to a network you do not trust.
"""

from __future__ import annotations

import argparse
import json
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from viewer.geo import (encode_grid16, encode_rg16, fit_absolute_elevation, read_geo_meta, square,
                        square_heights)
from viewer.metrics import luminance, scene_metrics
from viewer.terrain import build_terrain, height_field

WEB_DIR = Path(__file__).resolve().parent / "web"
UPLOAD_DIR = WEB_DIR / "data-uploads"
SCENES_DIR = UPLOAD_DIR / "scenes"

MAX_UPLOAD_BYTES = 200 * 1024 * 1024
ALLOWED_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
PROCESS_RES = 504
CLASSIFY_RES = 513  # matches gamus-terrain's HEIGHT_SAMPLE_WIDTH/HEIGHT

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
_height_model = None
_dem = None

CACHE_DIR = Path(__file__).resolve().parent / "cache"
SYNRS3D_DIR = CACHE_DIR / "SynRS3D"
# Fine-tuned checkpoint when present, stock RS3DAda otherwise; override with ALTIMAP_HEIGHT_CKPT.
HEIGHT_CKPTS = (CACHE_DIR / "best.pth", SYNRS3D_DIR / "pretrain" / "RS3DAda_vitl_DPT_height.pth")
VIEW_MAX_SIDE = 1024  # texture/height previews sent to the browser; GeoTIFFs keep full resolution
CITY_MAX_SIDE = 2048  # grid the 3D city model (footprints, trees) is extracted on


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


def _height_ckpt() -> Path:
    import os

    if os.environ.get("ALTIMAP_HEIGHT_CKPT"):
        return Path(os.environ["ALTIMAP_HEIGHT_CKPT"])
    for ckpt in HEIGHT_CKPTS:
        if ckpt.exists():
            return ckpt
    raise HTTPException(503, f"no height checkpoint; expected one of {[str(c) for c in HEIGHT_CKPTS]}")


# One upload on the height models at a time: they are shared, and estimate() moves v2 and CHMv2
# onto the GPU and back in place, so two overlapping uploads pulled a model off the GPU in the
# middle of the other's run ("Input type (c10::BFloat16) and bias type (float)"). A 6 GB GPU
# can't hold two passes anyway. Uploads queue instead.
_ESTIMATE_LOCK = threading.Lock()


def _get_height_model() -> dict:
    """The height pipeline (v1 + 75%-weighted v2 on buildings + CHMv2 in forest),
    loaded on first request, same reasoning as _get_model. -> kwargs for estimate()."""
    global _height_model
    if _height_model is None:
        from viewer.estimate import load_pipeline

        _height_model = load_pipeline(_height_ckpt(), SYNRS3D_DIR)
    return _height_model


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


GRID_SIDE = 1025  # the viewer's mesh vertices per side (HEIGHT_SAMPLE_WIDTH/HEIGHT in main.jsx)


VIEW_BG = (7, 17, 30)  # the viewer's background colour, so a scene's padding reads as empty


def _png_data_uri(arr: np.ndarray) -> str:
    import base64
    import io

    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


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
    rgb_full = _read_rgb_bytes(data, suffix)
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

    return JSONResponse({
        "seconds": round(elapsed, 3),
        "width": CLASSIFY_RES,
        "height_px": CLASSIFY_RES,
        "class_pixel_counts": counts,
        "rgb": _png_data_uri(rgb),
        "height": _png_data_uri(np.stack([height_u8] * 3, axis=-1)),
        "classes": _png_data_uri(classes_to_rgb(class_map)),
    })


def _read_reference(path: Path) -> np.ndarray:
    """A reference height map (metres): GeoTIFF/PNG via rasterio, or a GAMUS *_AGL.h5."""
    if path.suffix.lower() == ".h5":
        from viewer.gamus_dataset import load_h5

        return np.squeeze(load_h5(path)).astype(np.float32)
    import rasterio

    with rasterio.open(path) as src:
        ref = src.read(1).astype(np.float32)
        if src.nodata is not None and not np.isnan(src.nodata):
            ref[ref == src.nodata] = np.nan
    return ref


def _reference_on_grid(ref_path: Path, scene_dir: Path, shape: tuple[int, int]) -> tuple[np.ndarray, bool]:
    """Reference heights on the image grid -> (array, reprojected?). Georeferenced
    reference + georeferenced image: reprojected by coordinates. Otherwise the two are
    assumed to cover the same area and the reference is resampled to the image size."""
    if ref_path.suffix.lower() in {".tif", ".tiff"}:
        import rasterio

        from viewer.geo import warp_to_grid

        with rasterio.open(ref_path) as r, rasterio.open(scene_dir / "ndsm.tif") as img:
            both = r.crs is not None and img.crs is not None
            bounds, crs = tuple(img.bounds), img.crs
        if both:
            return warp_to_grid([ref_path], bounds, crs, shape), True
    ref = _read_reference(ref_path)
    if ref.shape != tuple(shape):
        ref = np.asarray(Image.fromarray(ref.astype(np.float32), mode="F").resize(
            (shape[1], shape[0]), Image.BILINEAR))
    return ref, False


def _error_png(err: np.ndarray, size: tuple[int, int]) -> tuple[str, float]:
    """Signed error (model - reference) as blue (model low) / white / red (model high),
    clipped at a round limit near the 95th percentile of |error|; no data is grey."""
    finite = np.abs(err[np.isfinite(err)])
    p95 = float(np.percentile(finite, 95)) if finite.size else 1.0
    limit = next((v for v in (0.5, 1, 2, 3, 5, 10, 20, 50, 100, 200) if v >= p95), 500.0)
    e = np.asarray(Image.fromarray(np.nan_to_num(err, nan=np.inf).astype(np.float32), mode="F")
                   .resize(size, Image.NEAREST))
    t = np.clip(np.where(np.isfinite(e), e, 0) / limit, -1, 1)[..., None]
    blue, white, red = np.array([33, 102, 172]), np.array([247, 247, 247]), np.array([178, 24, 43])
    rgb = np.where(t < 0, white + (blue - white) * -t, white + (red - white) * t)
    rgb[~np.isfinite(e)] = (90, 90, 90)
    return _png_data_uri(square(rgb.astype(np.uint8), (90, 90, 90))), float(limit)


def _validate(out: dict, ref_path: Path, scene_dir: Path,
              reference_kind: str = "auto") -> tuple[dict, dict]:
    """Score against a reference: height above ground (vs the nDSM) or an absolute DSM
    (vs the exported DSM), explicitly selected or auto-detected for older clients; plus
    per-class errors, an error map and a scatter sample for the viewer."""
    from viewer.height_metrics import class_scores, height_scores, select_reference_kind
    from viewer.height_model import clean_height

    ndsm, dsm = out["ndsm"], out["dsm"]
    ref, reprojected = _reference_on_grid(ref_path, scene_dir, ndsm.shape)
    ref = np.where(ref < -1e4, np.nan, ref).astype(np.float32)
    kind = select_reference_kind(ndsm, dsm, ref, reference_kind)
    if kind == "ndsm":
        ref = clean_height(ref)  # height above ground: drop impossible values
    pred = dsm if kind == "dsm" else ndsm
    scores = height_scores(pred, ref, out["classes"])
    err = (pred - ref).astype(np.float32)
    finite = np.isfinite(err)
    if not finite.any():
        raise ValueError("the reference has no valid heights over the image")
    idx = np.flatnonzero(finite)
    pick = np.random.default_rng(0).choice(idx, size=min(1500, idx.size), replace=False)
    png, limit = _error_png(err, _view_size(ndsm.shape))
    validation = {k: _clean(v) if k != "n" else v for k, v in scores.items()}
    validation.update(
        reference_kind=kind, reprojected=reprojected, bias=_clean(float(np.mean(err[finite]))),
        coverage=float(finite.mean()),
        by_class=[{k: _clean(v) if isinstance(v, float) else v for k, v in r.items()}
                  for r in class_scores(pred, ref, out["classes"])],
        scatter=[[round(float(ref.flat[i]), 2), round(float(pred.flat[i]), 2)] for i in pick])
    return validation, {"png": png, "limit_m": limit}


def _view_size(shape: tuple[int, int], max_side: int = 0) -> tuple[int, int]:
    h, w = shape
    s = min(1.0, (max_side or VIEW_MAX_SIDE) / max(h, w))
    return max(1, round(w * s)), max(1, round(h * s))  # PIL (width, height)


# In-flight /api/estimate jobs: job id -> {"stage": str, "progress": 0..1}, polled by the UI.
JOBS: dict[str, dict] = {}
JOB_ID = re.compile(r"[A-Za-z0-9-]{1,64}")


@app.get("/api/progress/{job}")
def estimate_progress(job: str):
    if not JOB_ID.fullmatch(job):
        raise HTTPException(404, "no such job")
    # The UI starts polling as it starts sending the file, before the server has registered the
    # job: an unknown (well-formed) id is an upload still in transit, not an error.
    return JOBS.get(job, {"stage": "Uploading", "progress": 0.0})


@app.post("/api/estimate")
async def estimate_endpoint(file: UploadFile = File(...), gsd: float | None = Form(None),
                            reference: UploadFile | None = File(None),
                            reference_kind: Literal["auto", "ndsm", "dsm"] = Form("auto"),
                            gcps: UploadFile | None = File(None),
                            job: str | None = Form(None), tta: bool = Form(True),
                            base_dem: str = Form("srtm"),
                            building_source: Literal["hybrid", "image"] = Form("hybrid")):
    """Height model on an upload: metric nDSM always, absolute DSM for GeoTIFFs.

    Response keeps the classify-static shape (rgb/height/classes data URIs) so
    the terrain viewer consumes it unchanged, plus metres, GeoTIFF download
    paths, the 3D city model, ground relief for georeferenced input, and -- if a
    reference height map is attached -- RMSE/MAE/r against it. `reference_kind` selects
    "ndsm" (above ground), "dsm" (absolute elevation), or "auto" (legacy proximity guess).
    A `gcps` CSV (lon, lat, height) corrects a GeoTIFF's absolute DSM;
    `base_dem` ("glo30" or "srtm") picks the public DEM
    it sits on, and the response scores it against both. With a `job` id,
    progress is readable at /api/progress/<job> while this request runs.
    `building_source="hybrid"` reconstructs mapped OSM building parts and simple
    roofs where available; "image" keeps image-derived blocks. This affects the
    city model and buildings GeoJSON only, never the raw DSM/nDSM rasters.
    """
    from starlette.concurrency import run_in_threadpool

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
    if gsd is not None and not (0.01 <= gsd <= 100):
        raise HTTPException(422, "gsd must be between 0.01 and 100 metres per pixel")
    if job is not None and not JOB_ID.fullmatch(job):
        raise HTTPException(422, "bad job id")
    if base_dem not in ("glo30", "srtm"):
        raise HTTPException(422, "base_dem must be glo30 or srtm")

    scene_id = f"{_safe_stem(file.filename or 'upload')}__{uuid.uuid4().hex[:8]}"
    scene_dir = SCENES_DIR / scene_id
    staged = scene_dir / f"source{suffix}"
    scene_dir.mkdir(parents=True, exist_ok=True)
    staged.write_bytes(data)

    ref_path = None
    if reference is not None and reference.filename:
        ref_suffix = Path(reference.filename).suffix.lower()
        if ref_suffix not in {".tif", ".tiff", ".png", ".h5"}:
            raise HTTPException(415, "reference must be .tif, .png or a GAMUS _AGL.h5")
        ref_path = scene_dir / f"reference{ref_suffix}"
        ref_path.write_bytes(await reference.read())

    gcp_points = None
    if gcps is not None and gcps.filename:
        from viewer.estimate import read_gcps

        gcp_path = scene_dir / "gcps.csv"
        gcp_path.write_bytes((await gcps.read())[:1_000_000])
        try:
            gcp_points = read_gcps(gcp_path)[:1000]
        except (UnicodeDecodeError, OSError) as exc:
            raise HTTPException(422, f"could not read the control points CSV: {exc}") from exc
        finally:
            gcp_path.unlink(missing_ok=True)
        if not gcp_points:
            raise HTTPException(422, "no control points found; expected CSV rows of lon, lat, height "
                                     "(or a header naming lat/lon/height columns)")

    def report(stage: str, fraction: float) -> None:
        if job:
            JOBS[job] = {"stage": stage, "progress": round(float(fraction), 3)}

    report("Queued", 0.0)
    try:
        # Model inference is blocking: run it off the event loop so the server keeps
        # answering progress polls and static files meanwhile.
        payload = await run_in_threadpool(_run_estimate, staged, scene_dir, scene_id, gsd,
                                          ref_path, report, tta, gcp_points, base_dem, reference_kind,
                                          building_source)
    except HTTPException:
        raise
    except Exception as exc:
        detail = str(exc).replace(str(scene_dir), "the upload")  # never show server paths
        raise HTTPException(422, f"could not process this image: {detail}") from exc
    finally:
        staged.unlink(missing_ok=True)
        if ref_path is not None:
            ref_path.unlink(missing_ok=True)
        if job:
            JOBS.pop(job, None)
    return JSONResponse(payload)


def _run_estimate(staged: Path, scene_dir: Path, scene_id: str, gsd: float | None,
                  ref_path: Path | None, report, tta: bool = True, gcps: list | None = None,
                  base_dem: str = "srtm", reference_kind: str = "auto",
                  building_source: str = "hybrid") -> dict:
    from viewer.city_model import bridges, city_model
    from viewer.classify import classes_to_rgb
    from viewer.estimate import estimate
    from viewer.height_metrics import height_scores
    from viewer.height_model import GAMUS_GSD_M, clean_height

    if not _ESTIMATE_LOCK.acquire(blocking=False):
        report("Waiting for another upload to finish", 0.0)
        _ESTIMATE_LOCK.acquire()
    try:
        report("Loading height model", 0.01)
        models = _get_height_model()
        started = time.perf_counter()
        out = estimate(staged, out_dir=scene_dir, gsd_m=gsd, tta=tta, report=report, gcps=gcps,
                       base_dem=base_dem, **models)
    finally:
        _ESTIMATE_LOCK.release()
    validation, error_view = None, None
    if ref_path is not None:
        report("Scoring against reference", 0.88)
        validation, error_view = _validate(out, ref_path, scene_dir, reference_kind)

    report("Building 3D city model", 0.92)
    record = out["record"]
    max_m = max(record["ndsm_max_m"], 1e-3)
    size = _view_size(out["ndsm"].shape)
    rgb = np.asarray(Image.fromarray(out["rgb"]).resize(size, Image.BILINEAR))
    # nodata (NaN) shows as flat ground in previews and the city model; the GeoTIFFs keep NaN
    ndsm_filled = np.nan_to_num(out["ndsm"])
    height = np.asarray(Image.fromarray(ndsm_filled, mode="F").resize(size, Image.BILINEAR))
    height_u8 = np.clip(height / max_m * 255.0, 0, 255).astype(np.uint8)
    classes = np.asarray(Image.fromarray(out["classes"]).resize(size, Image.NEAREST))
    def to_view(a: np.ndarray) -> np.ndarray:
        return np.asarray(Image.fromarray(a.astype(np.float32), mode="F").resize(size, Image.BILINEAR))

    def relief_png(a: np.ndarray) -> dict:
        """Linear 8-bit relief: elevation = min_m + pixel / 255 * relief_m."""
        lo = float(np.nanmin(a))
        span = max(float(np.nanmax(a)) - lo, 1e-3)
        u8 = np.clip((np.nan_to_num(a, nan=lo) - lo) / span * 255.0, 0, 255).astype(np.uint8)
        return {"png": _png_data_uri(np.stack([u8] * 3, axis=-1)), "min_m": lo, "relief_m": span}

    # Terrain for georeferenced input: bare earth under the city model, and the exported
    # DSM for "Exact surface" view. Both relative to their own minimum.
    # Everything the viewer draws is centred in a square (`geo.square`) at the image's own aspect;
    # surfaces continue the terrain into the border, objects (nDSM, classes) are flat there.
    terrain, ground_zero = None, None
    if out["ground"] is not None and out["dsm"] is not None:
        ground_view = to_view(out["ground"])
        ground_zero = float(np.nanmin(ground_view))
        dsm_view = square(ground_view) + square(to_view(out["dsm"]) - ground_view, 0.0)
        terrain = {"ground": relief_png(square(ground_view)), "dsm": relief_png(dsm_view)}
    grids = {"side": GRID_SIDE, "ndsm": encode_grid16(square_heights(ndsm_filled, GRID_SIDE, 0.0), GRID_SIDE)}
    if terrain is not None:
        ground_g = square_heights(out["ground"], GRID_SIDE)
        dsm_g = ground_g + square_heights(out["dsm"] - out["ground"], GRID_SIDE, 0.0)
        grids.update(ground=encode_grid16(ground_g, GRID_SIDE), dsm=encode_grid16(dsm_g, GRID_SIDE))

    # 3D city model for the viewer (display only; GeoTIFFs stay raw), built on a grid up to
    # 2x finer than the previews so footprints follow the buildings closely.
    csize = _view_size(out["ndsm"].shape, CITY_MAX_SIDE)

    def to_city(a: np.ndarray, nearest: bool = False) -> np.ndarray:
        if a.ndim == 3 or a.dtype == np.uint8:
            return np.asarray(Image.fromarray(a).resize(csize, Image.NEAREST if nearest else Image.BILINEAR))
        return np.asarray(Image.fromarray(a.astype(np.float32), mode="F").resize(csize, Image.BILINEAR))

    city_gsd = (record["gsd_m"] or GAMUS_GSD_M) * out["ndsm"].shape[1] / csize[0]
    city_ground = None
    if ground_zero is not None:
        city_ground = to_city(out["ground"]) - ground_zero  # same zero as the terrain relief PNG
    city = city_model(to_city(ndsm_filled), to_city(out["classes"], nearest=True),
                      to_city(out["rgb"]), city_gsd, ground=city_ground)
    geometry_info = {"requested": building_source, "mapped_parts": 0,
                     "image_parts": len(city["buildings"]), "heights": {}, "roof_meshes": 0}
    if building_source == "hybrid" and record["georeferenced"]:
        report("Reconstructing mapped building parts", 0.94)
        try:
            import rasterio
            from rasterio.transform import Affine
            from rasterio.warp import transform_bounds
            from collections import Counter
            from viewer import mapped_buildings

            with rasterio.open(staged) as src:
                bounds = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
                city_transform = src.transform * Affine.scale(src.width / csize[0], src.height / csize[1])
                city_crs = src.crs
            elements = mapped_buildings.fetch(bounds)
            if elements is None:
                geometry_info["note"] = "OpenStreetMap unavailable; using image-derived buildings"
            else:
                mapped = mapped_buildings.reconstruct(elements, city_transform, city_crs,
                    to_city(ndsm_filled), to_city(out["classes"], nearest=True),
                    to_city(out["rgb"]), city_gsd, ground=city_ground)
                city["buildings"] = mapped_buildings.combine(city["buildings"], mapped)
                geometry_info.update(mapped_parts=len(mapped), image_parts=len(city["buildings"]) - len(mapped),
                    heights=dict(Counter(b["height_source"] for b in mapped)),
                    roof_meshes=sum("roof" in b for b in mapped))
                if not mapped:
                    geometry_info["note"] = "No usable mapped buildings in this extent; using image-derived buildings"
        except Exception:
            geometry_info["note"] = "Mapped reconstruction unavailable; using image-derived buildings"
    elif building_source == "hybrid":
        geometry_info["note"] = "Map assistance requires a georeferenced image; using image-derived buildings"
    record["building_geometry"] = geometry_info
    _write_buildings_geojson(scene_dir, city["buildings"], out["ndsm"].shape)
    record["files"].append("buildings.geojson")
    try:
        facilities = _place_facilities(staged, out["ndsm"].shape, city, city_ground) if record["georeferenced"] else []
    except Exception:  # an odd OSM feature must not cost a finished upload its result
        facilities = None
    if facilities is not None:
        counts: dict[str, int] = {}
        for f in facilities:
            counts[f["kind"]] = counts.get(f["kind"], 0) + 1
        record["facilities"] = counts
    else:
        record["facilities_error"] = "OpenStreetMap unreachable"
        facilities = []
    # Embankments: crest lines for the 3D view, in image coordinates like the city model.
    h0, w0 = out["ndsm"].shape
    embank = []
    if ground_zero is not None:
        from shapely.geometry import LineString, box

        frame = box(0, 0, 1, 1)
        for e in out.get("embankments") or []:
            if not np.isfinite(e["crest_m"]):
                continue
            clipped = LineString([(c / w0, r / h0) for c, r in e["pts"]]).intersection(frame)
            for part in getattr(clipped, "geoms", [clipped]):
                if part.geom_type == "LineString" and len(part.coords) >= 2:
                    embank.append({"kind": e["kind"], "name": e["name"], "crest": round(e["crest_m"] - ground_zero, 2),
                                   "pts": [[round(u, 5), round(v, 5)] for u, v in part.coords]})
    if out.get("deck") is not None and ground_zero is not None:  # bridges join the 3D view only
        deck_city = np.asarray(Image.fromarray(out["deck"].astype(np.float32), mode="F").resize(csize, Image.NEAREST))
        city["buildings"] += bridges(deck_city, city_gsd, to_city(out["rgb"]), zero=ground_zero)
    # The GeoJSON keeps image coordinates; the viewer gets them inside its centred square.
    h, w = out["ndsm"].shape
    fh, fw = h / max(h, w), w / max(h, w)
    for b in city["buildings"]:
        b["rings"] = [[[round(0.5 + (u - 0.5) * fw, 5), round(0.5 + (v - 0.5) * fh, 5)] for u, v in ring]
                      for ring in b["rings"]]
        if "roof" in b:
            b["roof"]["vertices"] = [[0.5 + (u - 0.5) * fw, 0.5 + (v - 0.5) * fh, z]
                                      for u, v, z in b["roof"]["vertices"]]
    for t in city["trees"] + facilities:
        t["u"], t["v"] = round(0.5 + (t["u"] - 0.5) * fw, 5), round(0.5 + (t["v"] - 0.5) * fh, 5)
    for e in embank:
        e["pts"] = [[round(0.5 + (u - 0.5) * fw, 5), round(0.5 + (v - 0.5) * fh, 5)] for u, v in e["pts"]]

    names = ["background", "ground", "low_vegetation", "buildings", "water", "roads", "trees"]
    counts = np.bincount(classes.ravel(), minlength=len(names))
    record.update(id=scene_id, seconds=round(time.perf_counter() - started, 2),
                  checkpoint=_height_ckpt().name, validation=validation,
                  class_pixel_counts={n: int(counts[i]) for i, n in enumerate(names)})
    (scene_dir / "meta.json").write_text(json.dumps(record))
    report("Done", 1.0)
    return {
        **record,
        "max_m": round(max_m, 2),
        "width": size[0],
        "height_px": size[1],
        "view_shape": [max(h, w), max(h, w)],  # pixels the viewer's square spans
        "downloads": {f: f"/data-uploads/scenes/{scene_id}/{f}" for f in record["files"]},
        "rgb": _png_data_uri(square(rgb, VIEW_BG)),
        "height": _png_data_uri(square(np.stack([height_u8] * 3, axis=-1), 0)),
        "classes": _png_data_uri(classes_to_rgb(square(classes, 0))),
        "city": city,
        "facilities": facilities,
        "embankments": embank,
        "terrain": terrain,
        "grids": grids,
        "error": error_view,
    }


def _place_facilities(path: Path, shape: tuple[int, int], city: dict, ground) -> list[dict] | None:
    """OpenStreetMap critical facilities in the scene (viewer/facilities.py), placed on the city
    model; None if OpenStreetMap was unreachable."""
    import rasterio
    from rasterio.warp import transform_bounds

    from viewer import facilities as fac

    with rasterio.open(path) as src:
        crs, tf, bounds = src.crs, src.transform, src.bounds
    found = fac.fetch(transform_bounds(crs, "EPSG:4326", *bounds))
    return None if found is None else fac.place(found, tf, crs, shape, city["buildings"], ground)


def _write_buildings_geojson(scene_dir: Path, buildings: list[dict], shape: tuple[int, int]) -> None:
    """Footprints + heights as GeoJSON: WGS84 lon/lat for georeferenced input
    (via the nDSM's transform), pixel coordinates otherwise."""
    import rasterio
    from rasterio.warp import transform_geom

    rows, cols = shape
    with rasterio.open(scene_dir / "ndsm.tif") as src:
        transform, crs = src.transform, src.crs
    features = []
    for b in buildings:
        rings = [[[u * cols, v * rows] for u, v in ring] for ring in b["rings"]]
        if crs is not None:
            rings = [[list(transform * (x, y)) for x, y in ring] for ring in rings]
        geom = {"type": "Polygon", "coordinates": [ring + ring[:1] for ring in rings]}
        if crs is not None:
            geom = transform_geom(crs, "EPSG:4326", geom)
        properties = {"height_m": b["h"], "source": b.get("source", "Image prediction"),
                      "height_source": b.get("height_source", "Image height estimate")}
        for key in ("source", "osm_id", "name", "height_source", "roof_source", "roof_shape", "min_height_m"):
            if key in b:
                properties[key] = b[key]
        features.append({"type": "Feature", "geometry": geom, "properties": properties})
    doc = {"type": "FeatureCollection", "features": features}
    if any(b.get("source") == "OpenStreetMap" for b in buildings):
        doc["attribution"] = "© OpenStreetMap contributors; https://www.openstreetmap.org/copyright"
    if crs is None:
        doc["note"] = "no georeferencing: coordinates are image pixels (x = column, y = row)"
    (scene_dir / "buildings.geojson").write_text(json.dumps(doc))


def _read_rgb_bytes(data: bytes, suffix: str) -> np.ndarray:
    """Same normalization as _read_rgb, for bytes already read into memory
    (this endpoint never stages the upload to disk -- it's not saved
    anywhere and there is no georeferencing path that would need the file)."""
    import io

    if suffix in {".tif", ".tiff"}:
        try:
            import rasterio

            with rasterio.MemoryFile(data) as memfile, memfile.open() as src:
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
            pass
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"), dtype=np.uint8)


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
# The React viewer (frontend/dist, from `npm run build`) at "/", so one process and one URL
# serve the whole app; the older vanilla dashboards stay reachable under /dashboards/.
FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"
app.mount("/data-uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")
app.mount("/dashboards", StaticFiles(directory=str(WEB_DIR), html=True), name="dashboards")
app.mount("/", StaticFiles(directory=str(FRONTEND_DIST if FRONTEND_DIST.is_dir() else WEB_DIR),
                           html=True), name="web")


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    viewer = "the 3D viewer" if FRONTEND_DIST.is_dir() else "dashboards (run `npm run build` in frontend/ for the 3D viewer)"
    print(f"AltiMap on http://{args.host}:{args.port}/  -> {viewer}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
