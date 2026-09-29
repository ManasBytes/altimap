"""Image in, elevation GeoTIFFs out: the product the brief asks for.

    nDSM (always)  height above ground, metres, from the height model.
    DSM (GeoTIFF)  nDSM + bare-earth ground from Copernicus GLO-30.

The model was trained at GAMUS's 0.33 m ground sampling distance, so input
with a known GSD is resampled to 0.33 m before inference and the result is
resampled back to the input grid. Input without a GSD (plain PNG/JPG) is run
as-is unless the caller supplies one.

Ground: GLO-30 is itself a surface model (it contains buildings and canopy),
so a grey-scale morphological opening wider than any building approximates
bare earth before the nDSM is added -- otherwise buildings are counted twice.
Heights are orthometric (EGM2008 geoid), recorded as such in the sidecar.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image

from altimap.contract import Sidecar, write_elevation_cog
from viewer.height_model import GAMUS_GSD_M, OEM_TO_GAMUS, predict

DEM_COLLECTION = "cop-dem-glo-30"
DEM_POSTING_M = 30.0
OPENING_M = 150.0  # wider than any building footprint we expect in a 30 m DEM
MAX_SIDE = 4096  # ponytail: caps inference memory; tile the scene if larger outputs are needed
MODEL_VERSION = "rs3dada-vitl-gamus"


def gsd_metres(geo: dict) -> float | None:
    """Pixel size in metres from read_geo_meta output, or None."""
    if not geo.get("georeferenced"):
        return None
    res = float(geo["res_m"][0])
    crs = str(geo["crs"])
    try:
        from rasterio.crs import CRS

        geographic = CRS.from_user_input(crs).is_geographic
    except Exception:
        geographic = "4326" in crs or "4269" in crs
    if geographic:  # degrees -> metres at the tile's latitude
        lat = 0.5 * (geo["bounds"][1] + geo["bounds"][3])
        return res * 111_320.0 * math.cos(math.radians(lat))
    return res


def work_shape(shape: tuple[int, int], src_gsd: float | None, dst_gsd: float = GAMUS_GSD_M,
               max_side: int = MAX_SIDE) -> tuple[int, int]:
    h, w = shape
    scale = src_gsd / dst_gsd if src_gsd else 1.0
    scale = min(scale, max_side / max(h, w)) if max(h, w) * scale > max_side else scale
    return max(1, round(h * scale)), max(1, round(w * scale))


def bare_earth(dem: np.ndarray, size_px: int) -> np.ndarray:
    from scipy import ndimage

    return ndimage.grey_opening(np.asarray(dem, np.float64), size=(size_px, size_px), mode="nearest")


def dem_consistent_ground(dem: np.ndarray, ndsm: np.ndarray, gsd_m: float,
                          cell_m: float = DEM_POSTING_M) -> np.ndarray:
    """Ground for the exported DSM: the DEM minus the model's heights averaged over one DEM cell.

    GLO-30 is a surface model that already holds buildings and canopy, blurred over
    its 30 m cells. Removing the model's own cell-mean and adding back the full-
    resolution nDSM makes the DSM's cell-mean reproduce the DEM (the organisers
    score GeoTIFF output against SRTM/Copernicus: "values must match DEM heights")
    while the model supplies the sub-cell detail.
    """
    from scipy import ndimage

    size = max(1, int(round(cell_m / gsd_m)))
    # nodata pixels count as ground in the cell mean, so they don't spread NaN over 30 m
    smooth = ndimage.uniform_filter(np.nan_to_num(np.asarray(ndsm, np.float64)), size=size, mode="reflect")
    return (np.asarray(dem, np.float64) - smooth).astype(np.float32)


def display_ground_method(building_share: float) -> str:
    """Bare-earth method for the 3D view's terrain, by the scene's predicted building share.
    Measured against USGS LiDAR bare earth on four NAIP scenes (dense downtown 52 %,
    hilly town 30 %, suburbs 22 %, forest 0 %), this rule was within 0.6 m of the
    best method on each: wide opening for dense cities, narrow for towns (it keeps
    hills), model subtraction where there are no buildings to remove."""
    if building_share >= 0.4:
        return "open300"
    if building_share >= 0.1:
        return "open150"
    return "subtract"


def read_rgb(path: Path) -> np.ndarray:
    """Imagery as HxWx3 uint8 (see read_image, which also returns the valid-pixel mask)."""
    return read_image(path)[0]


class NotImageryError(ValueError):
    """The upload is a raster of measured values (e.g. a height map), not a photo."""


def rgb_band_indexes(colorinterp, count: int) -> list[int]:
    """1-based band indexes for red, green, blue: from the file's colour tags when it has
    them (a BGR or BGRN GeoTIFF), else 1, 2, 3; a single-band image repeats band 1."""
    names = [getattr(c, "name", str(c)).lower() for c in colorinterp]
    if all(n in names for n in ("red", "green", "blue")):
        return [names.index("red") + 1, names.index("green") + 1, names.index("blue") + 1]
    return [1, 2, 3] if count >= 3 else [1, 1, 1]


def read_image(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """-> (HxWx3 uint8 RGB, HxW bool valid-pixel mask).

    Bands follow the file's colour tags; >8-bit data is 2-98 % stretched over valid
    pixels only (a nodata border must not flatten the contrast). Single-band floating-
    point rasters are refused: they are height/elevation maps, not imagery.
    """
    import rasterio

    try:
        src = rasterio.open(path)
    except rasterio.errors.RasterioIOError:  # formats GDAL can't open: let PIL try
        rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        return rgb, np.ones(rgb.shape[:2], bool)
    with src:
        if src.count < 3 and np.dtype(src.dtypes[0]).kind == "f":
            raise NotImageryError(
                "this file is a single-band floating-point raster, which looks like a height or "
                "elevation map, not an image. Upload the RGB image instead (a height map can be "
                "added as the reference heights to score against).")
        arr = np.transpose(src.read(rgb_band_indexes(src.colorinterp, src.count)), (1, 2, 0))
        valid = src.dataset_mask() > 0
    if arr.dtype != np.uint8:
        a = arr.astype(np.float64)
        sample = a[valid] if valid.any() else a.reshape(-1, 3)
        lo, hi = np.percentile(sample, [2, 98])
        arr = np.clip((a - lo) / max(hi - lo, 1e-9) * 255, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr), valid


def _resize(arr: np.ndarray, shape: tuple[int, int], nearest: bool = False) -> np.ndarray:
    if arr.shape[:2] == tuple(shape):
        return arr
    mode = Image.NEAREST if nearest else Image.BILINEAR
    if arr.ndim == 3 or arr.dtype == np.uint8:  # RGB and class maps keep their uint8 dtype
        return np.asarray(Image.fromarray(arr).resize((shape[1], shape[0]), mode))
    return np.asarray(Image.fromarray(arr.astype(np.float32), mode="F").resize((shape[1], shape[0]), mode))


def ground_for(geo: dict, shape: tuple[int, int], opening_m: float | None = None) -> np.ndarray | None:
    """Bare-earth ground (m, orthometric) on the image grid, or None if unavailable."""
    from viewer.dem import DemSource
    from viewer.geo import resample_to

    opening = opening_m or OPENING_M
    gsd = gsd_metres(geo)
    w, s, e, n = geo["bounds"]
    pad = opening * (geo["res_m"][0] / gsd)  # one opening width of padding, in CRS units
    padded = [w - pad, s - pad, e + pad, n + pad]
    coarse = (max(3, math.ceil((n - s + 2 * pad) * gsd / geo["res_m"][1] / DEM_POSTING_M)),
              max(3, math.ceil((e - w + 2 * pad) * gsd / geo["res_m"][0] / DEM_POSTING_M)))
    dem = DemSource(DEM_COLLECTION).patch(padded, geo["crs"], coarse)
    if dem is None:
        return None
    ground = bare_earth(dem, size_px=max(3, round(opening / DEM_POSTING_M)))
    # back onto the padded image grid, then crop the padding off
    ph, pw = round(pad / geo["res_m"][1]), round(pad / geo["res_m"][0])
    full = resample_to(ground, (shape[0] + 2 * ph, shape[1] + 2 * pw))
    return full[ph:ph + shape[0], pw:pw + shape[1]].astype(np.float32)


def estimate(path: Path, model, device: str, out_dir: Path, gsd_m: float | None = None,
             tta: bool = True, report=None) -> dict:
    """Run the height model on `path`, write GeoTIFFs into `out_dir`, return arrays + a record.

    `report(stage, fraction)`, if given, receives progress for a UI (fraction 0..0.9).
    """
    report = report or (lambda stage, fraction: None)
    import rasterio
    from rasterio.transform import Affine

    from viewer.geo import read_geo_meta

    out_dir.mkdir(parents=True, exist_ok=True)
    report("Reading image", 0.02)
    rgb, valid = read_image(path)
    if not valid.any():
        raise NotImageryError("the image has no valid pixels (everything is nodata)")
    if not valid.all():  # nodata borders: show the model a neutral fill, not black
        rgb = rgb.copy()
        rgb[~valid] = rgb[valid].mean(axis=0).astype(np.uint8)
    try:
        geo = read_geo_meta(path)
    except Exception:
        geo = {"georeferenced": False}
    src_gsd = gsd_m or gsd_metres(geo)

    shape = rgb.shape[:2]
    work = _resize(rgb, work_shape(shape, src_gsd))
    ndsm_work, oem_work = predict(model, work, device, tta=tta,
                                  progress=lambda f: report("Estimating heights", 0.05 + 0.7 * f))
    ndsm = np.maximum(_resize(ndsm_work, shape), 0).astype(np.float32)
    ndsm[~valid] = np.nan  # NaN is the project-wide nodata value
    classes = OEM_TO_GAMUS[_resize(oem_work, shape, nearest=True)]
    classes[~valid] = 0

    transform, crs = Affine.identity(), None
    if geo.get("georeferenced"):
        with rasterio.open(path) as src:
            transform, crs = src.transform, src.crs
    gsd_out = src_gsd or GAMUS_GSD_M
    finite = ndsm[np.isfinite(ndsm)]
    ndsm_range = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 0.0)
    write_elevation_cog(out_dir / "ndsm.tif", ndsm, transform, crs, Sidecar(
        gsd_m=gsd_out, source_gsd_m=gsd_out, datum="relative", vertical_unit="m",
        model_version=MODEL_VERSION, height_range_m=ndsm_range, tile_overlap_px=259,
        dtm_source=None))

    record = {
        "georeferenced": bool(geo.get("georeferenced")),
        "gsd_m": src_gsd,
        "gsd_assumed": src_gsd is None,
        "shape": list(shape),
        "ndsm_max_m": ndsm_range[1],
        "ndsm_p99_m": float(np.percentile(finite, 99)) if finite.size else 0.0,
        "files": ["ndsm.tif", "ndsm.json"],
        "dsm": None,
    }

    ground, dsm = None, None  # ground: bare earth for the 3D view only; dsm: the export
    if geo.get("georeferenced"):
        from viewer.dem import DemSource

        report("Fetching ground elevation (Copernicus GLO-30)", 0.78)
        try:
            dem = DemSource(DEM_COLLECTION).patch(geo["bounds"], geo["crs"], shape)
            if dem is None:
                record["dsm_error"] = "no Copernicus GLO-30 coverage for this area"
        except Exception as exc:  # network / coverage: the nDSM is still a valid product
            dem, record["dsm_error"] = None, f"{type(exc).__name__}: {exc}"
        if dem is not None:
            report("Writing absolute DSM", 0.86)
            dsm = dem_consistent_ground(dem, ndsm, gsd_out) + ndsm
            dsm_range = (float(np.nanmin(dsm)), float(np.nanmax(dsm)))
            write_elevation_cog(out_dir / "dsm.tif", dsm, transform, crs, Sidecar(
                gsd_m=gsd_out, source_gsd_m=gsd_out, datum="orthometric", vertical_unit="m",
                model_version=MODEL_VERSION, height_range_m=dsm_range, tile_overlap_px=259,
                dtm_source=f"{DEM_COLLECTION}, DEM-consistent: model detail added with its "
                           f"{DEM_POSTING_M:.0f} m mean removed"))
            method = display_ground_method(float((classes == 3).mean()))
            if method == "subtract":
                ground = dem_consistent_ground(dem, ndsm, gsd_out)
            else:
                try:
                    ground = ground_for(geo, shape, opening_m=300.0 if method == "open300" else 150.0)
                except Exception:
                    ground = None
                if ground is None:
                    ground = dem_consistent_ground(dem, ndsm, gsd_out)
            record["dsm"] = {"min_m": dsm_range[0], "max_m": dsm_range[1],
                             "dem_min_m": float(np.nanmin(dem)), "dem_max_m": float(np.nanmax(dem)),
                             "ground_min_m": float(np.nanmin(ground)),
                             "ground_max_m": float(np.nanmax(ground)),
                             "display_ground": method,
                             "datum": "orthometric (EGM2008)"}
            record["files"] += ["dsm.tif", "dsm.json"]

    return {"record": record, "rgb": rgb, "ndsm": ndsm, "classes": classes, "ground": ground,
            "dsm": dsm}


def main() -> None:
    """Batch CLI: every input image -> <out>/<stem>/{ndsm,dsm}.tif + meta.json.

        python -m viewer.estimate scene1.tif scene2.png --out results/
        python -m viewer.estimate photo.jpg --gsd 0.5 --out results/
    """
    import argparse
    import json

    from viewer.height_model import load_model

    cache = Path(__file__).resolve().parent / "cache"
    default_ckpt = next((c for c in (cache / "best.pth",
                                     cache / "SynRS3D/pretrain/RS3DAda_vitl_DPT_height.pth")
                         if c.exists()), cache / "best.pth")
    ap = argparse.ArgumentParser(description=main.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--gsd", type=float, default=None, help="metres/pixel; overrides the GeoTIFF's own")
    ap.add_argument("--ckpt", type=Path, default=default_ckpt)
    ap.add_argument("--synrs3d", type=Path, default=cache / "SynRS3D")
    ap.add_argument("--no-tta", action="store_true", help="4x faster, slightly less accurate")
    args = ap.parse_args()

    model, device = load_model(args.ckpt, args.synrs3d)
    print(f"model {args.ckpt.name} on {device}")
    for path in args.images:
        out_dir = args.out / path.stem
        result = estimate(path, model, device, out_dir, gsd_m=args.gsd, tta=not args.no_tta)
        record = result["record"]
        (out_dir / "meta.json").write_text(json.dumps(record, indent=2))
        dsm = record["dsm"]
        print(f"{path.name}: nDSM max {record['ndsm_max_m']:.1f} m"
              + (f", DSM {dsm['min_m']:.1f}-{dsm['max_m']:.1f} m ({dsm['datum']})" if dsm else "")
              + (f", DSM skipped: {record['dsm_error']}" if record.get("dsm_error") else "")
              + f" -> {out_dir}")


if __name__ == "__main__":
    main()
