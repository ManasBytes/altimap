"""Ramp-corrected diagnostics for monocular depth on nadir imagery.

DA3 fits a tilted plane to nadir scenes instead of reading relief (see
docs/superpowers/spikes/2026-08-24-da3-nadir-domain-gap.md). Everything here is
built around separating that plane from whatever survives underneath it.

Deliberately imports only numpy and scipy -- no torch, no cv2, no
depth_anything_3 -- so the main test venv can cover it without the 4.5 GB
inference venv. There is a test asserting this. Do not add heavy imports.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

NAN = float("nan")


def _normalized_grid(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """(u, v) in [0, 1), u along columns, v along rows."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    return xx / w, yy / h


def fit_plane(depth: np.ndarray) -> tuple[tuple[float, float, float], float]:
    """Least-squares plane `z = a + b*u + c*v` in normalized coordinates.

    Normalized rather than pixel coordinates so the coefficients are
    resolution-independent -- the shader applies them against UVs directly.

    Returns ((a, b, c), r2). r2 is nan for constant input, where the fit is
    exact but explains nothing, so a value of 1.0 would be actively misleading.
    """
    u, v = _normalized_grid(depth.shape)
    z = depth.ravel().astype(np.float64)
    design = np.column_stack([np.ones(z.size), u.ravel(), v.ravel()])
    coef, *_ = np.linalg.lstsq(design, z, rcond=None)

    residual = z - design @ coef
    ss_tot = float(((z - z.mean()) ** 2).sum())
    r2 = NAN if ss_tot == 0.0 else 1.0 - float((residual**2).sum()) / ss_tot
    return (float(coef[0]), float(coef[1]), float(coef[2])), r2


def plane_surface(plane: tuple[float, float, float], shape: tuple[int, int]) -> np.ndarray:
    a, b, c = plane
    u, v = _normalized_grid(shape)
    return a + b * u + c * v


def detrend(depth: np.ndarray, plane: tuple[float, float, float]) -> np.ndarray:
    """Depth with the fitted ramp removed -- the residual relief."""
    return depth - plane_surface(plane, depth.shape)


def luminance(rgb: np.ndarray) -> np.ndarray:
    """Rec. 601 luma. Accepts float [0,1] or uint8 [0,255]; returns input scale."""
    arr = rgb.astype(np.float64)
    return 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]


def gradient_magnitude(image: np.ndarray) -> np.ndarray:
    """Sobel gradient magnitude, edge-replicated."""
    arr = image.astype(np.float64)
    gx = ndimage.sobel(arr, axis=1, mode="nearest")
    gy = ndimage.sobel(arr, axis=0, mode="nearest")
    return np.hypot(gx, gy)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Correlation that returns nan instead of raising on degenerate input."""
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if x.size < 2 or x.std() == 0.0 or y.std() == 0.0:
        return NAN
    return float(np.corrcoef(x, y)[0, 1])


def scene_metrics(
    depth: np.ndarray,
    lum: np.ndarray,
    conf: np.ndarray | None = None,
) -> dict:
    """Full diagnostic record for one scene.

    `lum` must already be on the depth grid -- callers resize RGB before
    calling, since RGB arrives at 640 and depth at 504.
    """
    plane, r2 = fit_plane(depth)
    residual = detrend(depth, plane)
    median = float(np.median(depth))
    u, v = _normalized_grid(depth.shape)
    lum_edges = gradient_magnitude(lum)

    return {
        "plane": {"a": plane[0], "b": plane[1], "c": plane[2]},
        # Fraction of the scene DA3 explained with a ramp. 1 - plane_r2 is the
        # fraction that is something else. Replaces relief_ratio, which was
        # confounded with the artifact it was meant to detect.
        "plane_r2": r2,
        "residual_relief": float(residual.std() / median) if median != 0.0 else NAN,
        # Does residual relief land on image edges? Real structure should.
        "structure_alignment": _pearson(gradient_magnitude(residual), lum_edges),
        # Is confidence a reliability estimate or just a texture detector?
        "conf_texture_corr": _pearson(conf, lum_edges) if conf is not None else NAN,
        "row_corr": _pearson(v, depth),
        "col_corr": _pearson(u, depth),
        "depth_min": float(depth.min()),
        "depth_max": float(depth.max()),
        "depth_median": median,
        "residual_min": float(residual.min()),
        "residual_max": float(residual.max()),
    }
