"""Flood embankments, levees, dams and weirs from OpenStreetMap (georeferenced input only).

Embankments are terrain, but narrow terrain: the bare-earth ground removes everything
narrower than its 150-300 m opening to take buildings out of GLO-30, and with them the levee.
On the Baton Rouge Mississippi levee (LiDAR crest 14.3 m) that ground read the crest
3.2-5.6 m low, raw GLO-30 1.95 m low. So along OSM embankments the ground keeps raw GLO-30,
and the 3D view draws each one as a flood-defence line at its crest.
"""

from __future__ import annotations

import numpy as np

QUERY = ('(way["man_made"="dyke"]({bbox});way["man_made"="embankment"]({bbox});'
         'way["embankment"="yes"]({bbox});way["waterway"~"^(dam|weir)$"]({bbox}););out geom tags;')
HALF_WIDTH_M = 25.0  # a river levee's crest plus slopes, either side of the mapped line
FEATHER_M = 20.0


def kind(tags: dict) -> str:
    """dam / weir / levee are flood works; a road or railway on an embankment is raised ground
    the opening would also erase, but not a flood defence."""
    if tags.get("waterway") in ("dam", "weir"):
        return tags["waterway"]
    if tags.get("man_made") == "dyke":
        return "levee"
    if "railway" in tags:
        return "rail embankment"
    return "road embankment" if "highway" in tags else "embankment"


def fetch(bounds_lonlat) -> list[dict] | None:
    """[{"lonlat": [(lon, lat), ...], "kind", "name"}]; None when OpenStreetMap is unreachable."""
    from viewer import osm

    els = osm.query(QUERY, bounds_lonlat)
    if els is None:
        return None
    return [{"lonlat": [(p["lon"], p["lat"]) for p in el["geometry"]], "kind": kind(el.get("tags", {})),
             "name": el.get("tags", {}).get("name", "")}
            for el in els if len(el.get("geometry", [])) >= 2]


def lines_on_grid(ways: list[dict], transform, crs) -> list[np.ndarray]:
    """Each way as an (n, 2) array of (col, row) image coordinates."""
    from rasterio.warp import transform as warp

    inv = ~transform
    out = []
    for w in ways:
        xs, ys = warp("EPSG:4326", crs, [p[0] for p in w["lonlat"]], [p[1] for p in w["lonlat"]])
        cols, rows = inv * (np.asarray(xs), np.asarray(ys))
        out.append(np.stack([cols, rows], axis=1))
    return out


def corridor(lines: list[np.ndarray], shape: tuple[int, int], gsd_m: float) -> np.ndarray:
    """Weight 1 within HALF_WIDTH_M of any embankment line, falling to 0 over FEATHER_M."""
    from rasterio.features import rasterize
    from scipy import ndimage
    from shapely.geometry import LineString

    geoms = [(LineString(pts), 1) for pts in lines if len(pts) >= 2]
    on = (rasterize(geoms, out_shape=shape, fill=0, all_touched=True, dtype="uint8") > 0) if geoms \
        else np.zeros(shape, bool)  # identity transform: lines are already in (col, row)
    if not on.any():
        return np.zeros(shape, np.float32)
    d = ndimage.distance_transform_edt(~on) * gsd_m
    return np.clip(1 - (d - HALF_WIDTH_M) / FEATHER_M, 0, 1).astype(np.float32)


def keep_in_ground(ground: np.ndarray, dem: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """The bare-earth ground with the raw DEM restored along embankments (never lowered)."""
    return (ground + weight * np.maximum(np.nan_to_num(dem - ground), 0)).astype(np.float32)
