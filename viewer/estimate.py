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

Ground control points (lon, lat, height CSV) correct the DSM's vertical offset when it
exceeds GLO-30's own accuracy (e.g. a datum mix-up).
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
def read_gcps(path: Path) -> list[tuple[float, float, float]]:
    """Ground control points from a CSV: lon, lat, height (m above mean sea level, the
    DEM's datum). A header row naming the columns (lon/lng/longitude/x, lat/latitude/y,
    h/height/elev/elevation/z/alt) sets their order; without one, lon,lat,height."""
    import csv

    names = {"lon": 0, "lng": 0, "longitude": 0, "x": 0, "lat": 1, "latitude": 1, "y": 1,
             "h": 2, "height": 2, "elev": 2, "elevation": 2, "z": 2, "alt": 2}
    order, points = [0, 1, 2], []
    with open(path, newline="") as f:
        for row in csv.reader(f):
            cells = [c.strip().lower() for c in row]
            if not cells or not any(cells):
                continue
            try:
                vals = [float(c) for c in cells[:max(order) + 1]]
            except ValueError:  # a header: map named columns
                found = {names[c]: i for i, c in enumerate(cells) if c in names}
                if len(found) == 3:
                    order = [found[0], found[1], found[2]]
                continue
            if len(vals) > max(order):
                points.append((vals[order[0]], vals[order[1]], vals[order[2]]))
    return points


GCP_MIN_OFFSET_M = 4.0  # Copernicus GLO-30's specified absolute vertical accuracy (LE90)


def gcp_correction(ground: np.ndarray, transform, crs, gcps) -> tuple[np.ndarray | None, dict]:
    """Vertical offset (m, a constant field to add to the DSM and ground) from ground control
    points, compared with `ground`, the bare-earth estimate the points measure.

    Applied only when it is a real offset: at least GCP_MIN_OFFSET_M, i.e. beyond GLO-30's own
    accuracy (a datum mix-up such as GPS ellipsoidal heights vs EGM2008 is tens of metres), and
    better than no correction at predicting left-out points. Measured against 3DEP LiDAR on
    three US scenes where GLO-30 has no such offset, smaller fitted corrections (and tilted
    planes) made the DSM worse, e.g. suburb 3.99 -> 6.06 m."""
    from rasterio.warp import transform as warp

    xs, ys = warp("EPSG:4326", crs, [g[0] for g in gcps], [g[1] for g in gcps])
    inv = ~transform
    res, used = [], []
    for (lon, lat, h), x, y in zip(gcps, xs, ys):
        c, r = inv * (x, y)
        r, c = int(math.floor(r)), int(math.floor(c))
        if 0 <= r < ground.shape[0] and 0 <= c < ground.shape[1] and np.isfinite(ground[r, c]):
            res.append(h - float(ground[r, c]))
            used.append({"lon": lon, "lat": lat, "height_m": h, "ground_m": float(ground[r, c])})
    n = len(res)
    info = {"n_given": len(gcps), "n_used": n}
    if n == 0:
        info["error"] = "no control point falls inside the image"
        return None, info
    rv = np.array(res)
    offset = float(rv.mean())
    rmse_none = float(np.sqrt(np.mean(rv ** 2)))
    # leave-one-out: each point predicted by the mean of the others
    left_out = float(np.sqrt(np.mean((rv - (rv.sum() - rv) / (n - 1)) ** 2))) if n > 1 else None
    info.update(offset_m=offset, rmse_before_m=rmse_none, rmse_left_out_m=left_out, points=used)
    if abs(offset) < GCP_MIN_OFFSET_M:
        note = (f"not applied: the points put the ground within {abs(offset):.1f} m of them on "
                f"average, inside GLO-30's own ±{GCP_MIN_OFFSET_M:.0f} m accuracy")
    elif left_out is not None and left_out >= rmse_none:
        note = "not applied: the points don't agree on a common offset"
    else:
        for p, r in zip(used, rv - offset):
            p["residual_after_m"] = float(r)
        info.update(model="offset", rmse_after_m=float(np.sqrt(np.mean((rv - offset) ** 2))))
        return np.full(ground.shape, offset, np.float32), info
    for p, r in zip(used, rv):
        p["residual_after_m"] = float(r)
    info.update(model="none", rmse_after_m=rmse_none, note=note)
    return None, info


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


BARE_EARTH_PAD_M = 300.0  # margin around the image: the widest bare-earth opening below


def padded_dem(geo: dict) -> tuple[np.ndarray | None, tuple[int, int]]:
    """One GLO-30 read for the whole upload: the image extent plus BARE_EARTH_PAD_M on every
    side, at the DEM's ~30 m posting -> (coarse array, (pad rows, pad cols) in image pixels).
    The export DEM and the viewer's bare-earth ground both come from it (reading twice
    doubled the upload's network time)."""
    from viewer.dem import glo30

    gsd = gsd_metres(geo)
    w, s, e, n = geo["bounds"]
    pad = BARE_EARTH_PAD_M * (geo["res_m"][0] / gsd)  # in CRS units
    padded = [w - pad, s - pad, e + pad, n + pad]
    coarse = (max(3, math.ceil((n - s + 2 * pad) * gsd / geo["res_m"][1] / DEM_POSTING_M)),
              max(3, math.ceil((e - w + 2 * pad) * gsd / geo["res_m"][0] / DEM_POSTING_M)))
    dem = glo30(padded, geo["crs"], coarse)
    return dem, (round(pad / geo["res_m"][1]), round(pad / geo["res_m"][0]))


def dem_on_image(coarse: np.ndarray, pad_px: tuple[int, int], shape: tuple[int, int]) -> np.ndarray:
    """The padded coarse DEM bilinearly resampled onto the image grid, cropping the margin.
    Separable (rows, then columns), so memory is the output size even for huge images."""
    ph, pw = pad_px
    a = np.asarray(coarse, np.float64)

    def axis_weights(n_out: int, pad: int, n_coarse: int):
        x = (np.arange(n_out) + pad + 0.5) * n_coarse / (n_out + 2 * pad) - 0.5
        x = np.clip(x, 0, n_coarse - 1)
        i0 = np.floor(x).astype(int)
        i1 = np.minimum(i0 + 1, n_coarse - 1)
        return i0, i1, x - i0

    r0, r1, tr = axis_weights(shape[0], ph, a.shape[0])
    c0, c1, tc = axis_weights(shape[1], pw, a.shape[1])
    rows = a[r0] * (1 - tr)[:, None] + a[r1] * tr[:, None]
    return (rows[:, c0] * (1 - tc) + rows[:, c1] * tc).astype(np.float32)


def ground_for(coarse: np.ndarray, pad_px: tuple[int, int], shape: tuple[int, int],
               opening_m: float = OPENING_M) -> np.ndarray:
    """Bare-earth ground (m, orthometric) on the image grid from the padded coarse DEM."""
    bare = bare_earth(coarse, size_px=max(3, round(opening_m / DEM_POSTING_M)))
    return dem_on_image(bare, pad_px, shape)


def estimate(path: Path, model, device: str, out_dir: Path, gsd_m: float | None = None,
             tta: bool = True, report=None, gcps: list | None = None) -> dict:
    """Run the height model on `path`, write GeoTIFFs into `out_dir`, return arrays + a record.

    `gcps`: (lon, lat, height) control points that correct the DSM (georeferenced input only).

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
        report("Fetching ground elevation (Copernicus GLO-30)", 0.78)
        try:
            coarse, pad_px = padded_dem(geo)
            dem = None if coarse is None else dem_on_image(coarse, pad_px, shape)
            if dem is None:
                record["dsm_error"] = "no Copernicus GLO-30 coverage for this area"
        except Exception as exc:  # network / coverage: the nDSM is still a valid product
            dem, record["dsm_error"] = None, f"{type(exc).__name__}: {exc}"
        if dem is not None:
            report("Writing absolute DSM", 0.86)
            dsm = dem_consistent_ground(dem, ndsm, gsd_out) + ndsm
            method = display_ground_method(float((classes == 3).mean()))
            if method == "subtract":
                ground = dem_consistent_ground(dem, ndsm, gsd_out)
            else:
                ground = ground_for(coarse, pad_px, shape, 300.0 if method == "open300" else 150.0)
            fix, gcp_info = (None, None)
            if gcps:
                # Control points measure the ground, so they are compared with the bare-earth
                # estimate, not the DSM: along streets the DEM-consistent DSM carries the
                # street/roof balancing of each 30 m cell, and fitting that cost accuracy on roofs.
                fix, gcp_info = gcp_correction(ground, transform, crs, gcps)
                record["gcp"] = gcp_info
                if fix is not None:
                    dsm = dsm + fix
                    ground = ground + fix  # the 3D view's terrain follows the corrected DSM
            dem_note = (f"Copernicus GLO-30, DEM-consistent: model detail added with its "
                        f"{DEM_POSTING_M:.0f} m mean removed")
            if fix is not None:
                dem_note += f"; GCP offset {gcp_info['offset_m']:+.2f} m ({gcp_info['n_used']} points)"
            dsm_range = (float(np.nanmin(dsm)), float(np.nanmax(dsm)))
            write_elevation_cog(out_dir / "dsm.tif", dsm, transform, crs, Sidecar(
                gsd_m=gsd_out, source_gsd_m=gsd_out, datum="orthometric", vertical_unit="m",
                model_version=MODEL_VERSION, height_range_m=dsm_range, tile_overlap_px=259,
                dtm_source=dem_note))
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
    ap.add_argument("--gcps", type=Path, default=None,
                    help="CSV of ground control points (lon, lat, height m) to correct the DSM")
    args = ap.parse_args()
    gcps = read_gcps(args.gcps) if args.gcps else None

    model, device = load_model(args.ckpt, args.synrs3d)
    print(f"model {args.ckpt.name} on {device}")
    for path in args.images:
        out_dir = args.out / path.stem
        result = estimate(path, model, device, out_dir, gsd_m=args.gsd, tta=not args.no_tta, gcps=gcps)
        record = result["record"]
        (out_dir / "meta.json").write_text(json.dumps(record, indent=2))
        dsm = record["dsm"]
        print(f"{path.name}: nDSM max {record['ndsm_max_m']:.1f} m"
              + (f", DSM {dsm['min_m']:.1f}-{dsm['max_m']:.1f} m ({dsm['datum']})" if dsm else "")
              + (f", DSM skipped: {record['dsm_error']}" if record.get("dsm_error") else "")
              + (f", GCPs {record['gcp']['n_used']}/{record['gcp']['n_given']} used, RMSE "
                 f"{record['gcp']['rmse_before_m']:.2f} -> {record['gcp']['rmse_after_m']:.2f} m"
                 if record.get("gcp", {}).get("n_used") else "")
              + f" -> {out_dir}")


if __name__ == "__main__":
    main()
