"""Critical facilities for disaster response, from OpenStreetMap (georeferenced input only).

Hospitals and clinics, fire and police stations, emergency shelters and assembly points, and
schools, colleges and universities (the usual relief shelters). The 3D view highlights the
building each one stands in and pins the ones in the open.
"""

from __future__ import annotations

KINDS = (  # (OSM key, value, label); the first match names a feature
    ("amenity", "hospital", "hospital"), ("healthcare", "hospital", "hospital"),
    ("amenity", "clinic", "clinic"), ("healthcare", "clinic", "clinic"),
    ("amenity", "fire_station", "fire station"), ("amenity", "police", "police"),
    ("social_facility", "shelter", "shelter"), ("emergency", "assembly_point", "assembly point"),
    ("amenity", "school", "school"), ("amenity", "college", "college"),
    ("amenity", "university", "university"),
)


def fetch(bounds_lonlat) -> list[dict] | None:
    """[{"kind", "name", "lon", "lat"}] in a (west, south, east, north) box; ways and relations
    are placed at their centre. None when OpenStreetMap is unreachable."""
    from viewer import osm

    body = "(" + "".join(f'nwr["{k}"="{v}"]({{bbox}});' for k, v, _ in KINDS) + ");out center tags;"
    els = osm.query(body, bounds_lonlat)
    if els is None:
        return None
    out = []
    for el in els:
        tags = el.get("tags", {})
        kind = next((label for k, v, label in KINDS if tags.get(k) == v), None)
        where = el.get("center") or (el if "lat" in el else None)
        if kind and where:
            out.append({"kind": kind, "name": tags.get("name", ""), "lon": where["lon"], "lat": where["lat"]})
    return out


def place(found: list[dict], transform, crs, shape: tuple[int, int], buildings: list[dict], ground=None) -> list[dict]:
    """Facilities in image coordinates [{"kind", "name", "u", "v", "z": metres the pin stands on}],
    off-image ones dropped. A facility inside a city-model building (normalized rings) marks it
    with "facility" and pins on its roof; otherwise it stands on `ground` (metres, any grid)."""
    import numpy as np
    from rasterio.warp import transform as warp
    from shapely.geometry import Point, Polygon

    if not found:
        return []
    xs, ys = warp("EPSG:4326", crs, [f["lon"] for f in found], [f["lat"] for f in found])
    cols, rows = ~transform * (np.asarray(xs), np.asarray(ys))  # lists fail inside affine
    h, w = shape
    blocks = [(b, Polygon(b["rings"][0])) for b in buildings if b.get("kind") != "bridge"]
    out = []
    for f, c, r in zip(found, cols, rows):
        u, v = float(c) / w, float(r) / h
        if not (0 <= u < 1 and 0 <= v < 1):
            continue
        p = Point(u, v)
        home = next((b for b, poly in blocks if poly.is_valid and poly.contains(p)), None)
        if home is not None:
            home["facility"] = {"kind": f["kind"], "name": f["name"]}
            z = home["t"]
        elif ground is not None:
            z = float(ground[int(v * ground.shape[0]), int(u * ground.shape[1])])
        else:
            z = 0.0
        out.append({"kind": f["kind"], "name": f["name"], "u": u, "v": v, "z": round(z, 2)})
    return out
