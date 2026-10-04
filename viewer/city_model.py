"""Model output -> 3D city model (LoD1): extruded building parts + individual trees.

For the 3D view only. The exported nDSM/DSM GeoTIFFs stay the raw model output:
on 40 GAMUS val tiles, regularizing heights like this raised RMSE from 2.66 m to
2.86-3.38 m (real roofs are not single flat blocks, and mislabelled edge pixels
get flattened), so accuracy-graded outputs never go through here.

Buildings: each connected building blob is cut into roofs, one per roof plateau,
along the height step or dip between neighbours, so touching row houses and a tower
on its podium become separate blocks. Each roof is one block, extruded to the median
of its interior heights; it is never banded into height levels. The model's heights
fall off gradually at walls, across a facade seen at an angle and along its rows of
windows, and banding that slope stood every such roof as a staircase. Outlines follow
the traced shape and are squared off only where that barely moves them. Footprints come
back as polygons in normalized image coordinates (u = column / width, v = row / height).

Facades: wide slopes steeper than 45 degrees that hang below a roof are an oblique
facade; they belong to that roof but don't set its height, so a tower keeps its roof
height instead of a median diluted by its facade.

On 40 GAMUS val tiles (viewer/city_eval.py) one block per roof, against banding each
roof into levels >= 2.5 m apart: footprint IoU 0.843 -> 0.848, edge F1 0.663 -> 0.674,
height RMSE on buildings 2.87 -> 3.05 m (a podium's own level is lost when no step
separates it), separate buildings 2434 -> 2036. City-model RMSE vs 3DEP LiDAR: downtown
Philadelphia 43.75 -> 43.21 m, Pittsburgh 5.44 -> 5.75 m.

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


def _roof_segments(cm: np.ndarray, smooth: np.ndarray, gsd_m: float, min_px: int,
                   drop_m: float = 1.0, window_m: float = 6.0) -> np.ndarray:
    """Split one building blob into roofs (labels 1..n, 0 outside): one seed per roof
    plateau (pixels within `drop_m` of their local maximum over `window_m`), flooded over
    the height gradient, so each cut lands on the step or dip between neighbouring roofs.
    Touching row houses of similar height become separate blocks this way."""
    win = int(round(window_m / gsd_m)) | 1
    local_max = ndimage.maximum_filter(np.where(cm, smooth, -1e9), size=win)
    seeds = cm & (smooth >= local_max - drop_m)
    lab, n = ndimage.label(seeds)
    if n > 1:
        sizes = ndimage.sum(seeds, lab, np.arange(1, n + 1))
        keep = np.flatnonzero(sizes >= max(1, round(4.0 / gsd_m**2))) + 1  # >= 4 m^2 plateaus
        lab = np.where(np.isin(lab, keep), np.searchsorted(keep, lab) + 1, 0)
        n = len(keep)
    if n <= 1:
        return cm.astype(np.int32)
    grad = np.hypot(ndimage.sobel(smooth, 0), ndimage.sobel(smooth, 1))
    cost = np.full(cm.shape, 65535, np.uint16)  # outside the blob: impassable
    cost[cm] = np.clip(grad[cm] / max(float(grad[cm].max()), 1e-6) * 60000, 0, 60000)
    segs = np.where(cm, ndimage.watershed_ift(cost, lab.astype(np.int32)), 0)
    # Ties along the blob's edge can hand a roof stray pixels far from it: keep each roof's
    # main piece only, and only roofs big enough to be a building; the rest joins a neighbour.
    good = np.zeros(cm.shape, bool)
    for i in range(1, n + 1):
        pieces, k = ndimage.label(segs == i)
        if k:
            main = pieces == np.argmax(np.bincount(pieces.ravel())[1:]) + 1
            if main.sum() >= min_px:
                good |= main
    return np.where(cm, _nearest_fill(segs, good), 0) if good.any() else cm.astype(np.int32)


def building_parts(ndsm: np.ndarray, classes: np.ndarray, gsd_m: float, min_area_m2: float = 20.0,
                   min_height_m: float = 2.0, max_slope: float = 1.0, wall_m: float = 8.0):
    """-> (parts: int32 HxW, 0 = no building; heights: {part_id: metres})."""
    min_px = max(4, int(round(min_area_m2 / gsd_m**2)))
    edge_px = max(1, int(round(3.0 / gsd_m)))  # the model's height ramp inside a wall, ~3 m
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
    roof_smooth = ndimage.gaussian_filter(smooth, sigma=max(0.5, 0.66 / gsd_m))
    # A slope steeper than `max_slope` (1 = 45 degrees) and wider than `wall_m` is no roof but a
    # facade seen at an angle, its rows of windows averaged out over a storey: it never sets a
    # roof's height. A slope rising above the nearest roof is a building of its own and counts.
    storey = ndimage.gaussian_filter(smooth, sigma=1.5 / gsd_m)
    steep = np.hypot(*np.gradient(storey)) >= max_slope * gsd_m
    wide = ndimage.binary_opening(steep, iterations=max(1, int(round(wall_m / 2 / gsd_m))))
    roof = mask & ~wide
    flat = ~(wide & (smooth <= _nearest_fill(smooth, roof))) if roof.any() else ~wide
    comps, _ = ndimage.label(mask)
    next_id = 1
    for cid, sl in enumerate(ndimage.find_objects(comps), start=1):
        cm = comps[sl] == cid
        if cm.sum() < min_px:
            continue
        roofs = _roof_segments(cm, roof_smooth[sl], gsd_m, min_px)
        for rid in np.unique(roofs[roofs > 0]):
            m = roofs == rid
            inner = ndimage.binary_erosion(m, iterations=edge_px)
            inner = inner & flat[sl] if (inner & flat[sl]).sum() >= 4 else inner
            h = float(np.median(raw[sl][inner if inner.sum() >= 4 else m]))
            if m.sum() < min_px or h < min_height_m:  # speckle, or pavement labelled as roof
                continue
            parts[sl][m] = next_id
            heights[next_id] = h
            next_id += 1
    return parts, heights


def _orthogonal_ring(coords: list) -> list | None:
    """Snap a ring (already rotated to the building's main axis) to horizontal/vertical
    edges; corners are recomputed where consecutive snapped edges meet."""
    n = len(coords)
    edges = []  # [is_horizontal, position, length]
    for i in range(n):
        (x1, y1), (x2, y2) = coords[i], coords[(i + 1) % n]
        horizontal = abs(x2 - x1) >= abs(y2 - y1)
        edges.append([horizontal, (y1 + y2) / 2 if horizontal else (x1 + x2) / 2,
                      float(np.hypot(x2 - x1, y2 - y1))])
    merged: list = []
    for e in edges:  # consecutive edges of one orientation are one wall
        if merged and merged[-1][0] == e[0]:
            m, total = merged[-1], merged[-1][2] + e[2]
            m[1] = (m[1] * m[2] + e[1] * e[2]) / total if total > 0 else m[1]
            m[2] = total
        else:
            merged.append(list(e))
    if len(merged) > 1 and merged[0][0] == merged[-1][0]:
        last = merged.pop()
        total = merged[0][2] + last[2]
        merged[0][1] = (merged[0][1] * merged[0][2] + last[1] * last[2]) / total if total > 0 else merged[0][1]
        merged[0][2] = total
    if len(merged) < 4 or len(merged) % 2:
        return None
    pts = []
    for i, a in enumerate(merged):
        b = merged[(i + 1) % len(merged)]
        pts.append((b[1], a[1]) if a[0] else (a[1], b[1]))
    return pts


def regularize(poly, grid_px: float, rect_fill: float = 0.8, max_shift: float = 0.10):
    """Footprint -> clean LoD1 outline. Near-rectangular footprints (>= `rect_fill` of their
    minimum rotated rectangle) become that rectangle at equal area; others get right-angled
    edges along the building's main axis. The traced `poly` is kept instead whenever the clean
    version would move more than `max_shift` of the footprint's area (symmetric difference):
    squaring off every outline cost footprint IoU 0.843 -> 0.793 and edge F1 0.661 -> 0.569
    on 40 GAMUS val tiles; with this cap it is 0.841 / 0.655 (viewer/city_eval.py)."""
    import math

    from shapely import affinity
    from shapely.geometry import Polygon

    if poly.is_empty or poly.area <= 0:
        return poly
    shell_only = Polygon(poly.exterior)  # courtyards must not stop a block being a rectangle
    mrr = shell_only.minimum_rotated_rectangle
    if mrr.area <= 0:
        return poly
    if shell_only.area / mrr.area >= rect_fill:
        k = math.sqrt(shell_only.area / mrr.area)
        out = affinity.scale(mrr, k, k, origin=mrr.centroid)
        for ring in poly.interiors:  # keep courtyards, cleaned up the same way
            out = out.difference(regularize(Polygon(ring), grid_px, rect_fill))
        return out if _close(out, poly, max_shift) else poly

    corners = list(mrr.exterior.coords)
    (x1, y1), (x2, y2) = max(zip(corners[:-1], corners[1:]), key=lambda e: math.dist(*e))
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    origin = poly.centroid
    rotated = affinity.rotate(poly, -angle, origin=origin).simplify(grid_px, preserve_topology=True)
    if rotated.is_empty or rotated.geom_type != "Polygon":
        return poly
    shell = _orthogonal_ring(list(rotated.exterior.coords)[:-1])
    if shell is None:
        return poly
    holes = [h for h in (_orthogonal_ring(list(r.coords)[:-1]) for r in rotated.interiors) if h]
    out = affinity.rotate(Polygon(shell, holes), angle, origin=origin)
    return out if _close(out, poly, max_shift) else poly


def _close(out, poly, max_shift: float) -> bool:
    """A valid single polygon that moves at most `max_shift` of `poly`'s area."""
    return (out.geom_type == "Polygon" and out.is_valid and not out.is_empty
            and poly.symmetric_difference(out).area <= max_shift * poly.area)


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
            p = regularize(p, grid_px=max(1.0, 1.5 / gsd_m))  # clean LoD1 walls
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
        # tree of this height (~0.3 h), so dense groves don't become lollipops.
        r = float(np.clip(max(to_edge_m[y, x], 0.3 * h), 1.5, min(9.0, 0.5 * h + 1.0)))
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
