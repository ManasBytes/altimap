"""Display-only OSM building reconstruction; raw elevation rasters are never modified.

Mapped outlines/parts constrain horizontal geometry. Explicit heights take priority,
then floor-count estimates, then image heights. Roof dimensions without tags remain
estimates. OSM coverage and accuracy are not guaranteed.
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np
from shapely.geometry import GeometryCollection, LineString, MultiPoint, Polygon, box
from shapely.ops import polygonize, transform as transform_shape, triangulate, unary_union


def metres(value) -> float | None:
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(m|ft|feet|')?\s*", str("" if value is None else value).lower())
    if not match:
        return None
    height = float(match[1]) * (0.3048 if match[2] in ("ft", "feet", "'") else 1)
    return height if np.isfinite(height) and 0 <= height <= 2000 else None


def fetch(bounds_lonlat):
    from viewer import osm

    return osm.query('(way["building"]({bbox});way["building:part"]({bbox});'
                     'relation["building"]({bbox});relation["building:part"]({bbox}););out geom;',
                     bounds_lonlat)


def _geometry(element):
    def points(geometry):
        return [(p["lon"], p["lat"]) for p in geometry or [] if "lon" in p and "lat" in p]

    if element["type"] == "way":
        ring = points(element.get("geometry"))
        return Polygon(ring) if len(ring) >= 4 and ring[0] == ring[-1] else None
    outer, inner = [], []
    for member in element.get("members", []):
        ring = points(member.get("geometry"))
        if len(ring) >= 2 and member.get("role") in ("outer", "inner"):
            (inner if member["role"] == "inner" else outer).append(LineString(ring))
    if not outer:
        return None
    shell = unary_union(list(polygonize(unary_union(outer))))
    holes = unary_union(list(polygonize(unary_union(inner)))) if inner else GeometryCollection()
    return shell.difference(holes)


def roof_mesh(poly, top: float, height: float, kind: str, shape, orientation="along"):
    """Triangulated simple roof + boundary edges for gable walls, in u/v/metres.

    Reject incomplete triangulations rather than emitting holes across courtyards.
    Non-flat roof height and orientation can be approximate even when shape is mapped.
    """
    if height <= 0 or kind not in ("gabled", "hipped", "pyramidal", "dome", "cone", "skillion"):
        return None
    corners = np.asarray(poly.minimum_rotated_rectangle.exterior.coords[:-1])
    lengths = np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1)
    i = int(np.argmax(lengths))
    axis = (corners[(i + 1) % 4] - corners[i]) / lengths[i]
    if orientation == "across":
        axis = np.array([-axis[1], axis[0]])
    cross = np.array([-axis[1], axis[0]])
    center = np.array(poly.minimum_rotated_rectangle.centroid.coords[0])
    local = (corners - center) @ np.stack([axis, cross], axis=1)
    hx, hy = np.max(np.abs(local), axis=0)
    if min(hx, hy) < 1e-6:
        return None
    # Densify edges so gable intersections are explicit. Add ridge endpoints and
    # interior samples for curved roofs; no facade pixels become ground footprints.
    candidates = []
    for ring in [poly.exterior, *poly.interiors]:
        coords = list(ring.coords)
        for a, b in zip(coords[:-1], coords[1:]):
            candidates.extend([a, tuple((np.asarray(a) + b) / 2)])
    for x in (-1, -0.5, 0, 0.5, 1):
        for y in (-1, -0.5, 0, 0.5, 1):
            candidates.append(tuple(center + axis * hx * x + cross * hy * y))
    if kind == "hipped":
        candidates.extend([tuple(center + axis * max(0, hx - hy)),
                           tuple(center - axis * max(0, hx - hy))])
    from shapely.geometry import Point

    candidates = [p for p in candidates if poly.covers(Point(p))]
    faces = [t for t in triangulate(GeometryCollection([poly, MultiPoint(candidates)])) if poly.covers(t)]
    if not faces or abs(sum(t.area for t in faces) - poly.area) > max(1e-5, poly.area * 1e-6):
        return None

    def elevation(p):
        x, y = (np.asarray(p) - center) @ np.stack([axis, cross], axis=1)
        if kind == "gabled":
            fraction = 1 - abs(y) / hy
        elif kind == "hipped":
            fraction = min(hx - abs(x), hy - abs(y)) / hy
        elif kind == "pyramidal":
            fraction = 1 - max(abs(x) / hx, abs(y) / hy)
        elif kind == "skillion":
            fraction = (y / hy + 1) / 2
        else:
            radius = np.hypot(x / hx, y / hy)
            fraction = np.sqrt(max(0, 1 - radius**2)) if kind == "dome" else 1 - radius
        return top - height + height * float(np.clip(fraction, 0, 1))

    vertices, indices, lookup = [], [], {}
    for triangle in faces:
        coords = list(triangle.exterior.coords)[:3]
        a, b, c = map(np.asarray, coords)
        # Clockwise image coordinates produce upward normals in the viewer's x/z plane.
        if (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) > 0:
            coords.reverse()
        face = []
        for p in coords:
            if p not in lookup:
                lookup[p] = len(vertices)
                vertices.append([p[0] / shape[1], p[1] / shape[0], elevation(p)])
            face.append(lookup[p])
        indices.append(face)
    counts = Counter(tuple(sorted((a, b))) for face in indices
                     for a, b in zip(face, face[1:] + face[:1]))
    edges = [[a, b] for face in indices for a, b in zip(face, face[1:] + face[:1])
             if counts[tuple(sorted((a, b)))] == 1]
    return {"vertices": vertices, "faces": indices, "edges": edges, "height": height, "shape": kind}


def reconstruct(elements, transform, crs, ndsm, classes, rgb, gsd_m, ground=None):
    from rasterio.features import rasterize
    from rasterio.warp import transform as warp

    shape = ndsm.shape
    frame = box(0, 0, shape[1], shape[0])

    def project(lon, lat, z=None):
        x, y = warp("EPSG:4326", crs, lon, lat)
        col, row = ~transform * (np.asarray(x), np.asarray(y))
        return col, row

    records = []
    for element in elements:
        tags = element.get("tags", {})
        if tags.get("building", tags.get("building:part")) in (None, "no") or tags.get("location") == "underground":
            continue
        geometry = _geometry(element)
        if geometry is None or geometry.is_empty or not geometry.is_valid:
            continue
        geometry = transform_shape(project, geometry).intersection(frame)
        for poly in geometry.geoms if geometry.geom_type == "MultiPolygon" else [geometry]:
            if poly.geom_type == "Polygon" and poly.area * gsd_m**2 >= 0.25:
                records.append({"poly": poly, "tags": tags, "part": tags.get("building:part") not in (None, "no"),
                                "id": f'{element["type"]}/{element["id"]}'})
    outlines = [r for r in records if not r["part"]]
    parts = [r for r in records if r["part"]]
    # Simple 3D Buildings: outlines with mapped parts are not themselves rendered.
    # Their overall height can describe a tower, not the height of the whole body.
    drawable = list(parts)
    for outline in outlines:
        children = [r for r in parts if outline["poly"].intersection(r["poly"]).area >= 0.9 * r["poly"].area]
        if not children:
            drawable.append(outline)
    buildings = []
    for record in drawable:
        poly, tags = record["poly"], record["tags"]
        parents = [r for r in records if r is not record and
                   (r["poly"].area > poly.area * 1.01 or
                    (record["part"] and not r["part"] and r["poly"].area >= poly.area * 0.99))
                   and r["poly"].intersection(poly).area >= poly.area * 0.95]
        parents.sort(key=lambda r: r["poly"].area)
        height = metres(tags.get("height"))
        source = "OSM height tag"
        if height is None and metres(tags.get("building:levels")) is not None:
            height = metres(tags["building:levels"]) * 3.2
            source = "OSM floors × 3.2 m (estimate)"
        if height is None:
            parent = next((r for r in parents if metres(r["tags"].get("height")) is not None), None)
            if parent:
                height = metres(parent["tags"]["height"])
                source = "OSM parent height (estimate for this part)"
        mask = rasterize([(poly, 1)], out_shape=shape, dtype="uint8") > 0
        if not mask.any():
            continue
        if height is None:
            values = ndsm[mask & (classes == 3) & np.isfinite(ndsm)]
            if not values.size:
                continue  # no mapped dimension or image support: do not invent a building
            height = float(np.percentile(values, 75))
            source = "Image height estimate"
        minimum = metres(tags.get("min_height")) or 0
        if not minimum and metres(tags.get("building:min_level")) is not None:
            minimum = metres(tags["building:min_level"]) * 3.2
        if height <= minimum + 0.5:
            continue
        terrain = float(np.median(ground[mask])) if ground is not None else 0
        colour = np.median(rgb[mask], axis=0).astype(int).tolist()
        roof_kind = tags.get("roof:shape", "flat")
        roof_height = metres(tags.get("roof:height"))
        roof_source = "OSM roof height tag" if roof_height is not None else "Estimated roof dimensions"
        if roof_height is None:
            corners = np.asarray(poly.minimum_rotated_rectangle.exterior.coords[:-1])
            width = np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1).min() * gsd_m
            roof_height = min(width * 0.2, height * 0.25, 12.0) if roof_kind != "flat" else 0
        if source == "OSM floors × 3.2 m (estimate)" and roof_height > 0:
            # OSM floor counts exclude levels in the roof; explicit height tags
            # already include the roof and must never get this addition.
            height += roof_height
            source = "OSM floors × 3.2 m + roof (estimate)"
        roof_height = min(roof_height, height - minimum)
        roof = roof_mesh(poly, terrain + height, roof_height, roof_kind, shape, tags.get("roof:orientation", "along"))
        building = {"h": round(height, 2), "b": round(terrain + minimum, 2), "t": round(terrain + height, 2),
                    "min_height_m": minimum, "c": colour,
                    "rings": [[[x / shape[1], y / shape[0]] for x, y in ring.coords[:-1]]
                              for ring in [poly.exterior, *poly.interiors]],
                    "source": "OpenStreetMap", "osm_id": record["id"],
                    "name": tags.get("name") or next((r["tags"].get("name") for r in parents if r["tags"].get("name")), ""),
                    "height_source": source, "roof_source": roof_source if roof else "Flat roof approximation",
                    "roof_shape": roof_kind}
        if roof:
            building["roof"] = roof
        buildings.append(building)
    return buildings


def combine(image_buildings, mapped):
    """Retain image-derived fallback outside mapped coverage; avoid duplicate facade blocks."""
    if not mapped:
        return image_buildings
    coverage = unary_union([Polygon(b["rings"][0], b["rings"][1:]) for b in mapped])
    fallback = []
    for b in image_buildings:
        poly = Polygon(b["rings"][0], b["rings"][1:])
        if poly.is_valid and poly.area and poly.intersection(coverage).area / poly.area < 0.15:
            fallback.append(b)
    return mapped + fallback


def roof_elevation(building, u: float, v: float) -> float:
    """Actual mesh support under a facility pin, rather than every roof's peak height."""
    roof = building.get("roof")
    if roof:
        for face in roof["faces"]:
            a, b, c = (roof["vertices"][i] for i in face)
            denominator = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
            if abs(denominator) < 1e-15:
                continue
            wa = ((b[1] - c[1]) * (u - c[0]) + (c[0] - b[0]) * (v - c[1])) / denominator
            wb = ((c[1] - a[1]) * (u - c[0]) + (a[0] - c[0]) * (v - c[1])) / denominator
            wc = 1 - wa - wb
            if min(wa, wb, wc) >= -1e-8:
                return wa * a[2] + wb * b[2] + wc * c[2]
    return building["t"]
