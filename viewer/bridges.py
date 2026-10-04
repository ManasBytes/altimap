"""Bridge decks from OpenStreetMap, their height from the terrain where they reach the banks.

The height model reads bridge decks as ground, and the land-cover map calls most deck pixels
over a river "water", so neither finds them. OpenStreetMap does: bridges are ways tagged
bridge=*, mapped worldwide. A deck meets the ground at its ends (where the bridge=* tag
stops), so its elevation is interpolated between the terrain at those ends. Only for
georeferenced input (OSM needs coordinates, the ends need a DEM).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

# Public Overpass servers are volunteer-run and often busy (504s, timeouts): try several, within
# FETCH_BUDGET_S overall, and keep every answer on disk so a scene that worked once keeps its
# bridges without the network.
OVERPASS = ("https://overpass-api.de/api/interpreter",
            "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
            "https://overpass.private.coffee/api/interpreter",
            "https://overpass.kumi.systems/api/interpreter")
FETCH_BUDGET_S = 60.0
LANE_M = 3.5
WIDTH_M = {  # deck width by OSM highway/railway type when no width or lanes tag says otherwise
    "motorway": 12.0, "trunk": 12.0, "primary": 10.0, "secondary": 9.0, "tertiary": 8.0,
    "motorway_link": 6.0, "trunk_link": 6.0, "primary_link": 6.0, "secondary_link": 6.0,
    "residential": 7.0, "unclassified": 7.0, "service": 5.0, "living_street": 6.0,
    "footway": 3.0, "cycleway": 3.0, "path": 3.0, "pedestrian": 4.0, "steps": 3.0, "track": 4.0,
    "rail": 6.0, "light_rail": 6.0, "subway": 6.0, "tram": 6.0,
}


def _cache_path(query: str) -> Path:
    root = Path(os.environ.get("ALTIMAP_DEM_CACHE", Path(__file__).resolve().parent / "cache" / "dem"))
    return root / "osm" / f"{hashlib.sha256(query.encode()).hexdigest()}.json"


def fetch(bounds_lonlat) -> list[dict] | None:
    """Bridge ways crossing a (west, south, east, north) box: [{"lonlat": [(lon, lat), ...], "tags"}].
    None when no Overpass server answers in time (the upload continues without bridges)."""
    import requests

    w, s, e, n = (round(float(v), 5) for v in bounds_lonlat)
    query = f'[out:json][timeout:25];way["bridge"]["bridge"!="no"]({s},{w},{n},{e});out geom tags;'
    cache = _cache_path(query)
    data = None
    if cache.exists():
        try:
            data = json.loads(cache.read_text())
        except (OSError, ValueError):
            data = None
    deadline = time.monotonic() + FETCH_BUDGET_S
    for url in OVERPASS if data is None else ():
        left = deadline - time.monotonic()
        if left < 5:
            break
        try:
            r = requests.post(url, data={"data": query}, timeout=(min(10, left), min(30, left)),
                              headers={"User-Agent": "AltiMap (SIH 2026 PS 26175)"})
            r.raise_for_status()
            data = r.json()
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(data))
            break
        except Exception:  # busy server, rate limit, bad JSON: try the next one
            continue
    if data is None:
        return None
    return [{"lonlat": [(p["lon"], p["lat"]) for p in el["geometry"]], "tags": el.get("tags", {})}
            for el in data.get("elements", []) if el.get("geometry")]


def width_m(tags: dict) -> float:
    """Deck width: the width tag, else lanes x 3.5 m plus shoulders, else a default by type."""
    try:
        return float(str(tags["width"]).split()[0])
    except (KeyError, ValueError):
        pass
    try:
        return int(tags["lanes"]) * LANE_M + 1.5
    except (KeyError, ValueError):
        pass
    return WIDTH_M.get(tags.get("highway") or tags.get("railway"), 7.0)


def _groups(lines: list[list[tuple[float, float]]], tol: float) -> tuple[list[int], list[list[tuple]]]:
    """Connected groups of ways (sharing an end node) and each group's open ends: a bridge split
    into several ways, or a ramp joining it, is one structure whose ends are on land."""
    def key(p):
        return round(p[0] / tol), round(p[1] / tol)

    parent = list(range(len(lines)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    ends: dict = {}
    for i, line in enumerate(lines):
        for p in (line[0], line[-1]):
            ends.setdefault(key(p), []).append((i, p))
    for members in ends.values():
        for j, _ in members[1:]:
            parent[find(j)] = find(members[0][0])
    group = [find(i) for i in range(len(lines))]
    free: dict = {}
    for members in ends.values():
        if len(members) == 1:  # an end no other bridge way continues from: the deck lands here
            i, p = members[0]
            free.setdefault(group[i], []).append(p)
    return group, [free.get(g, []) for g in group]


def decks(ways: list[dict], transform, crs, shape: tuple[int, int], elevation_at) -> np.ndarray:
    """Deck elevation (m) on the image grid, NaN where there is no bridge. `elevation_at(xs, ys)`
    gives terrain elevation at CRS coordinates, also beyond the image (bridges run off it).
    Inside a group, each pixel takes the inverse-distance-squared blend of the group's land
    ends; where decks cross (ramps over ramps), the higher one is the visible surface."""
    from rasterio.features import rasterize
    from rasterio.warp import transform as warp
    from shapely.geometry import LineString

    out = np.full(shape, np.nan, np.float32)
    if not ways:
        return out
    lines = []
    for way in ways:
        xs, ys = warp("EPSG:4326", crs, [p[0] for p in way["lonlat"]], [p[1] for p in way["lonlat"]])
        lines.append(list(zip(xs, ys)))
    px = abs(transform.a)
    _, free = _groups(lines, tol=px)
    for line, way, ends in zip(lines, ways, free):
        if len(line) < 2 or not ends:
            continue
        poly = LineString(line).buffer(width_m(way["tags"]) / 2, cap_style="flat")
        mask = rasterize([(poly, 1)], out_shape=shape, transform=transform, fill=0, dtype="uint8") > 0
        if not mask.any():
            continue
        rows, cols = np.nonzero(mask)
        x, y = transform * (cols + 0.5, rows + 0.5)
        ex, ey = np.array(ends).T
        z = np.asarray(elevation_at(ex, ey), np.float64)
        if not np.isfinite(z).any():
            continue
        ok = np.isfinite(z)
        d2 = (x[:, None] - ex[ok][None]) ** 2 + (y[:, None] - ey[ok][None]) ** 2 + px * px
        deck = (z[ok][None] / d2).sum(1) / (1.0 / d2).sum(1)
        out[rows, cols] = np.fmax(out[rows, cols], deck)
    return out
