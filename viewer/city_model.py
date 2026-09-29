"""Model output -> 3D city model (LoD1): extruded building parts + individual trees.

For the 3D view only. The exported nDSM/DSM GeoTIFFs stay the raw model output:
on 40 GAMUS val tiles, regularizing heights like this raised RMSE from 2.66 m to
2.86-3.38 m (real roofs are not single flat blocks, and mislabelled edge pixels
get flattened), so accuracy-graded outputs never go through here.

Buildings: each connected building component is split only into genuinely
distinct height levels (a tower on a podium), found from its *interior* pixels:
the model's heights ramp up softly over the first metres inside a wall, and
banding those ramps would stack every roof into terraces. Parts too small or too
thin to be a building on their own merge into their neighbours; each part is
extruded to the median of its interior heights. Footprints come back as polygons
in normalized image coordinates (u = column / width, v = row / height).

Trees: one crown per local height peak inside the tree class, sized from the
tree's height and the canopy extent, coloured from the photo.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

BUILDING = 3  # GAMUS class indices, see viewer/classify.py CLASS_NAMES
TREE = 6


def _nearest_fill(values: np.ndarray, known: np.ndarray) -> np.ndarray:
    """Every pixel takes the value of the nearest pixel where `known` is True."""
    idx = ndimage.distance_transform_edt(~known, return_distances=False, return_indices=True)
    return values[tuple(idx)]


def _split_levels(h: np.ndarray, min_px: int, min_gap_m: float, depth: int = 2) -> list[float]:
    """Height thresholds separating distinct roof levels (recursive 1-D Otsu).
    A split needs both sides to be building-sized and their means `min_gap_m` apart."""
    if depth == 0 or h.size < 2 * min_px or h.var() < 1e-6:
        return []
    best, best_t = 0.0, None
    for t in np.unique(np.quantile(h, np.linspace(0.05, 0.95, 37))):
        lo, hi = h[h <= t], h[h > t]
        if lo.size < min_px or hi.size < min_px or hi.mean() - lo.mean() < min_gap_m:
            continue
        between = lo.size * hi.size * (hi.mean() - lo.mean()) ** 2
        if between > best:
            best, best_t = between, float(t)
    if best_t is None:
        return []
    return sorted(_split_levels(h[h <= best_t], min_px, min_gap_m, depth - 1) + [best_t]
                  + _split_levels(h[h > best_t], min_px, min_gap_m, depth - 1))


def _component_parts(cm: np.ndarray, smooth: np.ndarray, min_px: int, edge_px: int,
                     thin_px: int, vote_px: int, min_gap_m: float) -> np.ndarray:
    """Local part labels (1..n) for one building component, 0 outside it."""
    interior = ndimage.binary_erosion(cm, iterations=edge_px)
    if interior.sum() < min_px:
        interior = cm
    thresholds = _split_levels(smooth[interior], min_px, min_gap_m)
    if not thresholds:
        return cm.astype(np.int32)

    level = np.digitize(smooth, thresholds, right=True)  # h == t is the lower level, as in _split_levels
    votes = [ndimage.uniform_filter(((level == lv) & cm).astype(np.float32), size=vote_px)
             for lv in range(len(thresholds) + 1)]
    level = np.where(cm, np.argmax(votes, axis=0), -1)  # majority vote removes speckle

    parts = np.zeros(cm.shape, np.int32)
    n_parts = 0
    for lv in range(len(thresholds) + 1):
        lab, n = ndimage.label(level == lv)
        parts[lab > 0] = lab[lab > 0] + n_parts
        n_parts += n
    # Small or thin parts (a ramp ring around a roof, a sliver) merge into their neighbours.
    good = np.zeros(cm.shape, bool)
    for pid in range(1, n_parts + 1):
        m = parts == pid
        if m.sum() >= min_px and ndimage.binary_erosion(m, iterations=thin_px).any():
            good |= m
    if not good.any():
        return cm.astype(np.int32)
    return np.where(cm, _nearest_fill(parts, good), 0).astype(np.int32)


def building_parts(ndsm: np.ndarray, classes: np.ndarray, gsd_m: float, min_gap_m: float = 5.0,
                   min_area_m2: float = 20.0, min_height_m: float = 2.0):
    """-> (parts: int32 HxW, 0 = no building; heights: {part_id: metres})."""
    min_px = max(4, int(round(min_area_m2 / gsd_m**2)))
    edge_px = max(1, int(round(3.0 / gsd_m)))  # the model's height ramp inside a wall, ~3 m
    thin_px = edge_px  # a part no wider than two ramps is a ramp ring or a sliver
    vote_px = 2 * max(1, int(round(1.5 / gsd_m))) + 1
    raw = np.asarray(ndsm, np.float32)

    mask = ndimage.binary_opening(classes == BUILDING, iterations=1)
    # Fill holes smaller than a building (label noise) but keep real courtyards.
    holes = ndimage.binary_fill_holes(mask) & ~mask
    hole_lab, n_holes = ndimage.label(holes)
    if n_holes:
        sizes = ndimage.sum(holes, hole_lab, np.arange(1, n_holes + 1))
        mask |= np.isin(hole_lab, np.flatnonzero(sizes < min_px) + 1)

    parts = np.zeros(mask.shape, np.int32)
    heights: dict[int, float] = {}
    if not mask.any():
        return parts, heights
    smooth = ndimage.median_filter(raw, size=5)
    comps, _ = ndimage.label(mask)
    next_id = 1
    for cid, sl in enumerate(ndimage.find_objects(comps), start=1):
        cm = comps[sl] == cid
        if cm.sum() < min_px:
            continue
        local = _component_parts(cm, smooth[sl], min_px, edge_px, thin_px, vote_px, min_gap_m)
        for lid in np.unique(local[local > 0]):
            m = local == lid
            inner = ndimage.binary_erosion(m, iterations=edge_px)
            h = float(np.median(raw[sl][inner if inner.sum() >= 4 else m]))
            if m.sum() < min_px or h < min_height_m:  # speckle, or pavement labelled as roof
                continue
            parts[sl][m] = next_id
            heights[next_id] = h
            next_id += 1
    return parts, heights


def footprints(parts: np.ndarray, heights: dict, shape: tuple[int, int], gsd_m: float,
               rgb: np.ndarray | None = None, simplify_m: float = 0.75,
               ground: np.ndarray | None = None) -> list[dict]:
    """Part map -> [{"h": metres, "b": base, "t": top, "c": roof [r, g, b], "rings": [...]}].

    Rings are [[u, v], ...] in [0, 1], outer first. With a `ground` grid (metres),
    a block spans from the lowest ground under it (so it never floats on a slope)
    to its median ground + h; without one, from 0 to h. The roof colour (mean photo
    colour) tints the walls in the viewer."""
    if not heights:
        return []
    pids = np.array(sorted(heights))
    base, top = {}, {}
    for pid in pids:
        pid = int(pid)
        if ground is None:
            base[pid], top[pid] = 0.0, heights[pid]
        else:
            g = ground[parts == pid]
            base[pid], top[pid] = float(g.min()), float(np.median(g)) + heights[pid]
    colours = {}
    if rgb is not None:
        means = np.stack([ndimage.mean(rgb[..., ch], labels=parts, index=pids) for ch in range(3)], 1)
        colours = {int(pid): [int(round(c)) for c in m] for pid, m in zip(pids, means)}
    from rasterio import features
    from shapely.geometry import MultiPolygon
    from shapely.geometry import shape as to_shape

    rows, cols = shape
    tolerance = max(0.5, simplify_m / gsd_m)  # pixels
    out = []
    for geom, pid in features.shapes(parts.astype(np.int32), mask=parts > 0, connectivity=4):
        pid = int(pid)
        if pid not in heights:
            continue
        poly = to_shape(geom).simplify(tolerance, preserve_topology=True)
        for p in poly.geoms if isinstance(poly, MultiPolygon) else [poly]:
            if p.is_empty or p.area < 1.0:
                continue
            rings = [p.exterior, *p.interiors]
            out.append({
                "h": round(heights[pid], 2),
                "b": round(base[pid], 2),
                "t": round(top[pid], 2),
                "c": colours.get(pid, [200, 195, 185]),
                "rings": [[[round(x / cols, 5), round(y / rows, 5)] for x, y in r.coords[:-1]]
                          for r in rings],
            })
    return out


def trees(ndsm: np.ndarray, classes: np.ndarray, rgb: np.ndarray, gsd_m: float,
          min_height_m: float = 3.0, spacing_m: float = 5.0,
          ground: np.ndarray | None = None) -> list[dict]:
    """One crown per canopy peak: [{"u", "v", "h": metres, "b": ground at trunk, "r": crown
    radius m, "c": [r, g, b]}]."""
    tree = classes == TREE
    if not tree.any():
        return []
    rows, cols = tree.shape
    heights = np.where(tree, np.asarray(ndsm, np.float32), 0)
    smooth = ndimage.gaussian_filter(heights, sigma=max(0.5, 0.75 / gsd_m))
    window = max(3, int(round(spacing_m / gsd_m)) | 1)
    peaks = (smooth == ndimage.maximum_filter(smooth, size=window)) & tree & (smooth >= min_height_m)
    lab, n = ndimage.label(peaks)  # a flat-topped crown can yield a plateau of equal maxima
    if n == 0:
        return []
    to_edge_m = ndimage.distance_transform_edt(tree) * gsd_m
    out = []
    for cy, cx in ndimage.center_of_mass(peaks, lab, np.arange(1, n + 1)):
        y, x = int(round(cy)), int(round(cx))
        h = float(heights[max(0, y - 1):y + 2, max(0, x - 1):x + 2].max())
        if h < min_height_m:
            continue
        # Crown radius: canopy extent around the peak, but never narrower than a real
        # tree of this height (~0.22 h), so dense groves don't become lollipops.
        r = float(np.clip(max(to_edge_m[y, x], 0.22 * h), 1.5, min(8.0, 0.5 * h + 1.0)))
        k = max(1, int(round(r / gsd_m / 2)))
        colour = rgb[max(0, y - k):y + k + 1, max(0, x - k):x + k + 1].reshape(-1, 3).mean(axis=0)
        out.append({"u": round((x + 0.5) / cols, 5), "v": round((y + 0.5) / rows, 5),
                    "h": round(h, 2), "b": round(float(ground[y, x]), 2) if ground is not None else 0.0,
                    "r": round(r, 2), "c": [int(round(c)) for c in colour]})
    return out


def city_model(ndsm: np.ndarray, classes: np.ndarray, rgb: np.ndarray, gsd_m: float,
               ground: np.ndarray | None = None) -> dict:
    """`ground`: terrain under the model (metres, same grid), e.g. relief above the scene's
    lowest ground; blocks and trees are placed on it."""
    parts, heights = building_parts(ndsm, classes, gsd_m)
    return {"buildings": footprints(parts, heights, np.shape(ndsm), gsd_m, rgb, ground=ground),
            "trees": trees(ndsm, classes, rgb, gsd_m, ground=ground)}
