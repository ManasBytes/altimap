import numpy as np
import pytest
from rasterio.transform import from_origin
from shapely.geometry import Polygon, box

from viewer.mapped_buildings import combine, metres, reconstruct, roof_elevation, roof_mesh


def _way(ident, bounds, tags):
    x0, y0, x1, y1 = bounds
    coords = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
    return {"type": "way", "id": ident, "tags": tags,
            "geometry": [{"lon": x, "lat": 64 - y} for x, y in coords]}


def _build(elements, ndsm=None):
    n = np.full((64, 64), 10.0, np.float32) if ndsm is None else ndsm
    return reconstruct(elements, from_origin(0, 64, 1, 1), "EPSG:4326", n,
                       np.full(n.shape, 3, np.uint8), np.full(n.shape + (3,), 120, np.uint8),
                       1.0, ground=np.full(n.shape, 7.0, np.float32))


def test_height_units_and_invalid_values():
    assert metres("20 m") == 20
    assert metres("100 ft") == pytest.approx(30.48)
    assert metres("100'") == pytest.approx(30.48)
    assert metres("0") == 0
    for value in (None, "nan", "-3", "5;10", "3000", "unknown"):
        assert metres(value) is None


def test_mapped_tower_parts_replace_parent_outline_and_keep_metric_scale():
    elements = [_way(1, (10, 10, 50, 50), {"building": "yes", "height": "90", "name": "Tower"}),
                _way(2, (10, 10, 50, 50), {"building:part": "yes", "height": "20"}),
                _way(3, (20, 20, 30, 30), {"building:part": "yes", "height": "90", "min_height": "20"})]
    buildings = _build(elements)
    assert len(buildings) == 2
    tower = next(b for b in buildings if b["osm_id"] == "way/3")
    assert tower["h"] == 90 and tower["b"] == 27 and tower["t"] == 97
    assert tower["name"] == "Tower"
    assert tower["height_source"] == "OSM height tag"
    assert Polygon(tower["rings"][0]).area * 64**2 == pytest.approx(100)


def test_floor_count_and_image_fallback_are_explicit_estimates():
    buildings = _build([_way(1, (5, 5, 15, 15), {"building": "yes", "building:levels": "4"}),
                        _way(2, (30, 30, 50, 50), {"building": "yes"})])
    assert buildings[0]["h"] == 12.8
    assert "estimate" in buildings[0]["height_source"]
    assert buildings[1]["h"] == 10
    assert buildings[1]["height_source"] == "Image height estimate"


def test_part_with_same_footprint_inherits_parent_name_and_height_as_estimate():
    elements = [_way(1, (10, 10, 50, 50), {"building": "yes", "height": "60", "name": "Hall"}),
                _way(2, (10, 10, 50, 50), {"building:part": "yes"})]
    (part,) = _build(elements)
    assert part["name"] == "Hall" and part["h"] == 60
    assert part["height_source"] == "OSM parent height (estimate for this part)"


def test_unknown_height_without_image_support_is_not_invented():
    assert _build([_way(1, (5, 5, 15, 15), {"building": "yes"})],
                  ndsm=np.full((64, 64), np.nan, np.float32)) == []


def test_floor_estimate_adds_roof_but_explicit_total_height_does_not():
    tags = {"building": "yes", "building:levels": "3", "roof:shape": "gabled", "roof:height": "4"}
    (estimated,) = _build([_way(1, (5, 5, 25, 25), tags)])
    assert estimated["h"] == 13.6
    (explicit,) = _build([_way(2, (5, 5, 25, 25), dict(tags, height="20"))])
    assert explicit["h"] == 20
    assert explicit["roof"]["height"] == 4


def test_multipolygon_courtyard_and_frame_clipping():
    outer = _way(1, (-5, 5, 55, 55), {})["geometry"]
    inner = _way(2, (20, 20, 40, 40), {})["geometry"]
    elements = [{"type": "relation", "id": 3, "tags": {"building": "yes", "height": "30"},
                 "members": [{"role": "outer", "geometry": outer}, {"role": "inner", "geometry": inner}]}]
    (b,) = _build(elements)
    assert len(b["rings"]) == 2
    poly = Polygon(b["rings"][0], b["rings"][1:])
    assert poly.area * 64**2 == pytest.approx(55 * 50 - 20 * 20)
    assert min(u for ring in b["rings"] for u, v in ring) >= 0


@pytest.mark.parametrize("kind", ["gabled", "hipped", "pyramidal", "dome", "cone", "skillion"])
def test_roof_is_closed_complete_and_points_upward(kind):
    poly = box(8, 8, 48, 28)
    roof = roof_mesh(poly, 30, 6, kind, (64, 64))
    assert roof is not None
    vertices = np.array(roof["vertices"])
    assert vertices[:, 2].max() == pytest.approx(30)
    assert vertices[:, 2].min() >= 24
    triangles = [Polygon(vertices[face, :2]) for face in roof["faces"]]
    assert sum(p.area for p in triangles) * 64**2 == pytest.approx(poly.area)
    for face in roof["faces"]:
        pts = vertices[face]
        world = pts[:, [0, 2, 1]]  # u -> x, elevation -> y, v -> z
        assert np.cross(world[1] - world[0], world[2] - world[0])[1] > 0
    assert roof["edges"]


def test_shaped_roof_does_not_cover_courtyard():
    poly = box(0, 0, 40, 40).difference(box(15, 15, 25, 25))
    roof = roof_mesh(poly, 30, 5, "gabled", (64, 64))
    assert roof is not None
    vertices = np.array(roof["vertices"])
    for face in roof["faces"]:
        assert poly.covers(Polygon(vertices[face, :2] * 64))


def test_image_fallback_is_kept_outside_mapped_coverage():
    mapped = [{"rings": [[[0.1, 0.1], [0.4, 0.1], [0.4, 0.4], [0.1, 0.4]]]}]
    duplicate = {"rings": mapped[0]["rings"], "h": 20}
    separate = {"rings": [[[0.6, 0.6], [0.9, 0.6], [0.9, 0.9], [0.6, 0.9]]], "h": 30}
    assert combine([duplicate, separate], mapped) == mapped + [separate]
    assert combine([duplicate, separate], []) == [duplicate, separate]


def test_facility_support_follows_pitched_roof_instead_of_floating_at_peak():
    roof = roof_mesh(box(0, 0, 64, 64), 30, 10, "pyramidal", (64, 64))
    building = {"roof": roof, "t": 30}
    assert roof_elevation(building, 0.5, 0.5) == pytest.approx(30)
    assert roof_elevation(building, 0.5, 0.1) == pytest.approx(22)
