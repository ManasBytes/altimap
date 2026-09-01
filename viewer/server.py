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
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from viewer.geo import encode_rg16, fit_absolute_elevation, read_geo_meta
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
_dem = None


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
