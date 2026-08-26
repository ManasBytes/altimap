"""Footprint-constrained refinement of a relative height field.

The problem this solves, from the 2026-08-24/25 runs: DA3 emits smooth depth
with no crisp edges, so buildings render as melted mounds with sloped skirts,
and a global plane fit is dragged around by those mounds. On the 620
georeferenced Atlanta tiles the naive relative->absolute calibration was usable
on only 5.6%, with 370/620 coming back INVERTED.

Building footprints fix both halves:

  1. Estimate the ground from NON-building pixels only, so structures stop
     biasing the surface they are supposed to sit on.
  2. Collapse each footprint to a single roof level, which is what turns a
     melted mound into a flat roof with vertical walls.

And known building heights give a metric anchor that a bare-earth DEM cannot:
the DEM is blind to buildings by construction, while buildings are exactly what
the model responds to.

Pure numpy + scipy. No torch, no rasterio -- see viewer/metrics.py for why.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

NAN = float("nan")
MIN_ANCHORS = 2       # a 2-parameter fit needs at least 2 points to mean anything
MIN_GROUND_PX = 32


def fit_ground(field: np.ndarray, mask: np.ndarray, order: int = 1) -> np.ndarray:
    """Smooth ground surface fitted to the pixels NOT covered by a footprint.

    `mask` is 0 for ground and a positive building id elsewhere.

    order=1 fits a plane, order=2 a quadratic. A plane is the default because
    the measured gain from a quadratic over these scenes was ~9 points of R2,
    and the extra freedom starts absorbing real relief.

    Falls back to fitting everything when a tile is (almost) fully built over,
    since a surface of nan would poison every downstream metric silently.
    """
    h, w = field.shape
    yy, xx = np.mgrid[0:h, 0:w]
    u, v = xx / w, yy / h

    terms = [np.ones(u.size), u.ravel(), v.ravel()]
    if order >= 2:
        terms += [(u * u).ravel(), (v * v).ravel(), (u * v).ravel()]
    design = np.column_stack(terms)

    z = field.ravel().astype(np.float64)
    ground_px = (mask.ravel() == 0) & np.isfinite(z)
    if ground_px.sum() < MIN_GROUND_PX:
        ground_px = np.isfinite(z)

    coef, *_ = np.linalg.lstsq(design[ground_px], z[ground_px], rcond=None)
    return (design @ coef).reshape(h, w)


def building_heights(ndsm: np.ndarray, mask: np.ndarray,
                     percentile: float = 50.0) -> dict[int, float]:
    """One height per footprint, as a robust statistic of the nDSM inside it.

    Median rather than max: a footprint picks up overhanging trees, edge
    bleed and off-nadir layover, and a max would latch onto whichever of those
    is worst. The trade is that it under-reads genuinely pitched roofs.
    """
    ids = [int(i) for i in np.unique(mask) if i > 0]
    out: dict[int, float] = {}
    for bid in ids:
        vals = ndsm[mask == bid]
        vals = vals[np.isfinite(vals)]
        if vals.size:
            out[bid] = float(np.percentile(vals, percentile))
    return out


def flatten_roofs(ndsm: np.ndarray, mask: np.ndarray,
                  heights: dict[int, float]) -> np.ndarray:
    """Replace every footprint's interior with its single roof level.

    This is what produces vertical walls: the height field steps at the
    footprint boundary instead of ramping across it. A draped mesh can render
    a near-vertical face given a step this sharp; it cannot invent one from
    smooth depth.
    """
    out = np.array(ndsm, dtype=np.float64, copy=True)
    for bid, height in heights.items():
        out[mask == bid] = height
    out[mask == 0] = 0.0
    return out


def metric_scale_from_buildings(
    relative: dict[int, float],
    known_m: dict[int, float],
) -> tuple[float, float, float, int]:
    """Fit `metres = scale * relative + offset` using buildings of known height.

    Returns (scale, offset, r2, n_anchors). The caller decides usability --
    this deliberately reports a negative scale rather than hiding it, because
    an inverted fit is a different failure from a weak one and the two need
    different responses.
    """
    ids = sorted(set(relative) & set(known_m))
    x = np.array([relative[i] for i in ids], dtype=np.float64)
    y = np.array([known_m[i] for i in ids], dtype=np.float64)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y, n = x[keep], y[keep], int(keep.sum())

    if n < MIN_ANCHORS or x.std() == 0.0:
        return NAN, NAN, NAN, n

    design = np.column_stack([x, np.ones(n)])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ coef
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = NAN if ss_tot == 0.0 else 1.0 - float((residual**2).sum()) / ss_tot
    return float(coef[0]), float(coef[1]), r2, n


def refine(
    height01: np.ndarray,
    mask: np.ndarray,
    known_m: dict[int, float] | None = None,
    order: int = 1,
) -> dict:
    """Full refinement for one scene.

    Returns the ground surface, the nDSM, the flattened nDSM, per-building
    relative heights, and -- when enough known heights are present -- a metric
    calibration derived from buildings rather than from a bare-earth DEM.
    """
    ground = fit_ground(height01, mask, order=order)
    ndsm = height01 - ground
    heights = building_heights(ndsm, mask)
    flat = flatten_roofs(ndsm, mask, heights)

    result = {
        "ground": ground,
        "ndsm": ndsm,
        "flat": flat,
        "heights": heights,
        "n_buildings": len(heights),
        "metric": None,
    }

    if known_m:
        scale, offset, r2, n = metric_scale_from_buildings(heights, known_m)
        usable = bool(np.isfinite(scale) and scale > 0 and np.isfinite(r2)
                      and r2 >= 0.3 and n >= 3)
        result["metric"] = {
            "scale_m": scale, "offset_m": offset, "fit_r2": r2, "n_anchors": n,
            "usable": usable,
            "reject_reason": None if usable else (
                f"only {n} anchors" if n < 3
                else "inverted (negative scale)" if np.isfinite(scale) and scale <= 0
                else "weak fit" if np.isfinite(r2) else "unfittable"),
            "source": "overture-building-heights",
        }
    return result


def edge_sharpness(field: np.ndarray, mask: np.ndarray) -> float:
    """Mean gradient magnitude on footprint boundaries, normalised by the
    field's own spread.

    A direct measure of the thing being fixed: melted mounds score low because
    their sides ramp gradually, flat-roofed prisms score high because the
    height steps at the wall. Comparable before and after refinement on the
    same scene; not comparable across scenes.
    """
    arr = np.asarray(field, dtype=np.float64)
    spread = float(arr.max() - arr.min())
    if spread == 0.0:
        return NAN
    boundary = (ndimage.maximum_filter(mask > 0, size=3)
                != ndimage.minimum_filter(mask > 0, size=3))
    if not boundary.any():
        return NAN
    gx = ndimage.sobel(arr, axis=1, mode="nearest")
    gy = ndimage.sobel(arr, axis=0, mode="nearest")
    return float(np.hypot(gx, gy)[boundary].mean() / spread)
