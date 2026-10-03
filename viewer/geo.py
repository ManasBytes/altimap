"""Georeferencing, depth encoding, and relative->absolute elevation fitting.

Scoped to the Off-nadir Scene10 (Atlanta) dataset, where 620 of 5200 tiles carry
EPSG:32616 with ~0.52 m/px and the remaining 4580 are plain rasters in a .tif
container. Both paths have to work, so every georeferenced feature here degrades
to None rather than raising.

numpy + scipy + rasterio only. No torch -- see viewer/metrics.py for why.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy import ndimage

NAN = float("nan")

# 16 bits of depth split across the R and G channels of an 8-bit PNG. Chosen
# over a raw float32 .bin because PNG compresses losslessly and the browser
# decodes it natively -- 5200 scenes at float32 would be ~5 GB of raw fetches.
# B is left at zero; it is spare capacity, not padding to be repurposed
# casually, since a nonzero B would not survive the decode below.
RG16_MAX = 65535


def encode_rg16(depth: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Quantise depth to 16 bits packed into R (high byte) and G (low byte).

    Returns (uint8 HxWx3, lo, hi). Constant input maps to all zeros with
    lo == hi, which decode_rg16 handles without dividing by zero.
    """
    arr = np.asarray(depth, dtype=np.float64)
    lo, hi = float(arr.min()), float(arr.max())
    span = hi - lo
    codes = np.zeros(arr.shape, dtype=np.uint32) if span == 0.0 else np.rint(
        (arr - lo) / span * RG16_MAX
    ).astype(np.uint32)
    codes = np.clip(codes, 0, RG16_MAX)

    out = np.zeros((*arr.shape, 3), dtype=np.uint8)
    out[..., 0] = (codes >> 8).astype(np.uint8)
    out[..., 1] = (codes & 0xFF).astype(np.uint8)
    return out, lo, hi


def encode_grid16(a: np.ndarray, side: int) -> dict:
    """Heights (metres) resampled to a side x side grid (pixel centres aligned, bilinear) as
    16-bit little-endian base64: value = lo + u16 / 65535 * span. NaN (nodata) takes the lowest
    valid value. The viewer decodes this for its mesh, probe and slope, since its 8-bit preview
    PNGs step 0.1-0.3 m, which quantised the slope readout to ~13 degree steps."""
    import base64

    a = np.asarray(a, np.float64)
    finite = np.isfinite(a)
    a = np.where(finite, a, a[finite].min() if finite.any() else 0.0)
    rows = (np.arange(side) + 0.5) * a.shape[0] / side - 0.5
    cols = (np.arange(side) + 0.5) * a.shape[1] / side - 0.5
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    g = ndimage.map_coordinates(a, [rr, cc], order=1, mode="nearest")
    lo = float(g.min())
    span = max(float(g.max()) - lo, 1e-3)
    u16 = np.rint((g - lo) / span * 65535).astype("<u2")
    return {"b64": base64.b64encode(u16.tobytes()).decode(), "lo": lo, "span": span}


def decode_rg16(rgb: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Inverse of encode_rg16. Mirrors the GLSL decode in shaders.js."""
    codes = rgb[..., 0].astype(np.float64) * 256.0 + rgb[..., 1].astype(np.float64)
    if hi == lo:
        return np.full(codes.shape, lo, dtype=np.float64)
    return lo + codes / RG16_MAX * (hi - lo)


def resample_to(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Bilinear resample to `shape`. Used to bring a coarse DEM patch onto the
    depth grid -- the DEM is ~10 m posting against 0.52 m imagery, so this is
    upsampling by ~20x and carries no detail it did not already have."""
    src = np.asarray(array, dtype=np.float64)
    if src.shape == tuple(shape):
        return src
    zoom = (shape[0] / src.shape[0], shape[1] / src.shape[1])
    # order=1 bilinear; NaNs would smear, so callers pass NaN-free or accept it.
    return ndimage.zoom(src, zoom, order=1, mode="nearest")


def warp_to_grid(sources, bounds, crs, shape: tuple[int, int]) -> np.ndarray:
    """Rasters (paths or URLs, any CRS; tiles, a mosaic or one file) resampled bilinearly
    onto the grid `bounds` (w, s, e, n in `crs`) x `shape`. The first source to cover a
    pixel wins; nodata and uncovered pixels are NaN. Used for local and uploaded DEMs,
    georeferenced reference heights and LiDAR benchmarks, so all land on the image grid
    through the same reprojection."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import from_bounds
    from rasterio.vrt import WarpedVRT
    from rasterio.warp import transform_bounds

    h, w = shape
    dst = from_bounds(*bounds, width=w, height=h)
    out = np.full(shape, np.nan, np.float32)
    for source in sources:
        with rasterio.open(source) as src:
            if src.crs is None:
                continue
            sw, ss, se, sn = transform_bounds(crs, src.crs, *bounds)
            b = src.bounds
            if sw > b.right or se < b.left or ss > b.top or sn < b.bottom:
                continue  # tile doesn't touch the grid: skip without reading it
            nodata = src.nodata if src.nodata is not None else np.nan
            with WarpedVRT(src, crs=crs, transform=dst, width=w, height=h, src_nodata=nodata,
                           nodata=np.nan, dtype="float32", resampling=Resampling.bilinear) as vrt:
                a = vrt.read(1)
        a[a < -1e4] = np.nan  # large negative sentinels even when nodata is unset
        fill = np.isnan(out) & np.isfinite(a)
        out[fill] = a[fill]
        if np.isfinite(out).all():
            break
    return out


def fit_absolute_elevation(
    height01: np.ndarray,
    dem: np.ndarray,
    min_valid: int = 16,
) -> tuple[float, float, float]:
    """Least-squares fit `elevation_m = scale * height01 + offset`.

    This is the brief's scale-calibration step: map a scale-agnostic relative
    height onto metric elevation using a lower-resolution DEM.

    Returns (scale_m, offset_m, r2). All three are nan when the fit is
    unidentifiable -- too few valid DEM pixels, or a DEM with no relief. A flat
    DEM would otherwise fit scale=0 and silently flatten the terrain, which
    reads as a successful calibration.

    Honest limits, since the numbers this returns look authoritative:
      * 3DEP seamless is BARE EARTH at ~10 m. It constrains terrain level, not
        building height. A commercial-area tile's buildings are residuals to
        this fit, not part of it.
      * scale therefore carries the terrain's metric range, and applying it to
        building-scale residuals is an extrapolation, not a measurement.
    """
    h = np.asarray(height01, dtype=np.float64).ravel()
    d = np.asarray(dem, dtype=np.float64).ravel()
    keep = np.isfinite(h) & np.isfinite(d)
    h, d = h[keep], d[keep]

    if h.size < min_valid:
        return NAN, NAN, NAN
    if d.std() == 0.0:
        # No relief to fit against: report the level, refuse the scale.
        return NAN, float(d.mean()), NAN
    if h.std() == 0.0:
        return NAN, float(d.mean()), NAN

    design = np.column_stack([h, np.ones(h.size)])
    coef, *_ = np.linalg.lstsq(design, d, rcond=None)
    residual = d - design @ coef
    ss_tot = float(((d - d.mean()) ** 2).sum())
    r2 = NAN if ss_tot == 0.0 else 1.0 - float((residual**2).sum()) / ss_tot
    return float(coef[0]), float(coef[1]), r2


def ground_size_m(width: int, height: int, res: tuple[float, float]) -> tuple[float, float]:
    """Ground footprint in metres from pixel count and pixel size."""
    return width * abs(res[0]), height * abs(res[1])


def read_geo_meta(path: Path) -> dict:
    """Georeferencing for one raster, or `georeferenced: False` if it has none.

    4580 of the 5200 tiles in this dataset are plain rasters that merely use the
    .tif container -- no CRS, no geotransform, no GCPs, no RPCs -- so rasterio
    hands back an identity transform. Detect that rather than trusting the
    extension.
    """
    import rasterio

    with rasterio.open(path) as src:
        return geo_meta_from_dataset(src)


def geo_meta_from_dataset(src) -> dict:
    """Same shape as read_geo_meta, from an already-open rasterio dataset.

    Split out so callers reading through GDAL's /vsizip/ (imagery packaged
    inside a .zip, never extracted to a real path) get an identical dict --
    downstream code (refine_scenes.py, validate.py, the dashboard JS) must not
    care which one produced it.
    """
    has_crs = src.crs is not None
    has_transform = not src.transform.is_identity
    has_gcps = bool(src.gcps and src.gcps[0])
    has_rpcs = bool(src.rpcs)
    base = {"width": src.width, "height": src.height, "bands": src.count}

    if not (has_crs and has_transform):
        return {
            **base,
            "georeferenced": False,
            "crs": None,
            "bounds": None,
            "res_m": None,
            "ground_m": None,
            "partial_hints": {"gcps": has_gcps, "rpcs": has_rpcs},
        }

    # Measure the two pixel vectors in metres. For geographic CRSs this uses
    # geodesic distances at the raster centre instead of treating degrees as m.
    col = (float(src.transform.a), float(src.transform.d))
    row = (float(src.transform.b), float(src.transform.e))
    x0, y0 = src.transform * (src.width / 2.0, src.height / 2.0)
    x1, y1 = src.transform * (src.width / 2.0 + 1.0, src.height / 2.0)
    x2, y2 = src.transform * (src.width / 2.0, src.height / 2.0 + 1.0)
    try:
        from rasterio.crs import CRS

        geographic = CRS.from_user_input(src.crs).is_geographic
    except Exception:
        geographic = False
    if geographic:
        try:
            from pyproj import Geod

            geod = Geod(ellps="WGS84")
            _, _, col_m = geod.inv(x0, y0, x1, y1)
            _, _, row_m = geod.inv(x0, y0, x2, y2)
        except Exception:
            lat = float(y0)
            col_m = abs(x1 - x0) * 111_320.0 * math.cos(math.radians(lat))
            row_m = abs(y2 - y0) * 111_320.0
    else:
        col_m = float(np.hypot(*col))
        row_m = float(np.hypot(*row))
    metric_res = (abs(float(col_m)), abs(float(row_m)))
    # Rotation/shear coefficients are expressed in the CRS' native units, so
    # compare them with the affine diagonal before converting to metres.
    transform_scale = max(abs(float(src.transform.a)), abs(float(src.transform.e)), 1e-12)
    rotated = (abs(float(src.transform.b)) > transform_scale * 1e-7 or
               abs(float(src.transform.d)) > transform_scale * 1e-7)
    anisotropic = min(metric_res) > 0 and max(metric_res) / min(metric_res) > 1.05
    geospatial_warnings = []
    if rotated:
        geospatial_warnings.append(
            "rotated/sheared affine transform: absolute DSM is disabled for this upload"
        )
    if anisotropic:
        geospatial_warnings.append(
            "non-square pixel size: absolute DSM is disabled for this upload"
        )
    res = (abs(src.res[0]), abs(src.res[1]))
    return {
        **base,
        "georeferenced": True,
        "crs": str(src.crs),
        "bounds": list(src.bounds),
        "res_m": list(res),
        "res_metric_m": list(metric_res),
        "ground_m": list(ground_size_m(src.width, src.height, metric_res)),
        "geospatial_warnings": geospatial_warnings,
    }
