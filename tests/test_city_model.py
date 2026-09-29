import numpy as np
from shapely.geometry import Polygon

from viewer.city_model import BUILDING, TREE, building_parts, city_model, footprints, trees

GSD = 0.5


def _scene(shape=(120, 120)):
    return np.zeros(shape, np.float32), np.ones(shape, np.uint8)  # ndsm, classes (1 = ground)


def _add(ndsm, classes, rows, cols, height, cls=BUILDING):
    ndsm[rows, cols] = height
    classes[rows, cols] = cls


def _area(rings):
    return Polygon(rings[0], rings[1:]).area


def test_separate_buildings_become_separate_blocks_with_their_heights():
    ndsm, classes = _scene()
    _add(ndsm, classes, slice(10, 40), slice(10, 40), 10.0)
    _add(ndsm, classes, slice(60, 100), slice(60, 90), 25.0)
    parts = footprints(*building_parts(ndsm, classes, GSD), shape=ndsm.shape, gsd_m=GSD)
    assert sorted(round(p["h"]) for p in parts) == [10, 25]
    assert all(p["c"] == [200, 195, 185] for p in parts)  # no photo -> neutral facade
    rgb = np.zeros(ndsm.shape + (3,), np.uint8)
    rgb[60:100, 60:90] = (180, 60, 40)  # red-brick roof
    coloured = footprints(*building_parts(ndsm, classes, GSD), shape=ndsm.shape, gsd_m=GSD, rgb=rgb)
    assert next(p for p in coloured if p["h"] > 20)["c"] == [180, 60, 40]
    tall = next(p for p in parts if p["h"] > 20)
    assert abs(_area(tall["rings"]) - (40 * 30) / 120**2) < 0.02  # normalized footprint area


def test_tower_on_podium_splits_into_two_heights():
    ndsm, classes = _scene()
    _add(ndsm, classes, slice(20, 100), slice(20, 100), 8.0)
    _add(ndsm, classes, slice(45, 75), slice(45, 75), 30.0)
    parts = footprints(*building_parts(ndsm, classes, GSD), shape=ndsm.shape, gsd_m=GSD)
    heights = sorted(round(p["h"]) for p in parts)
    assert heights[0] == 8 and heights[-1] == 30


def test_speckle_and_low_misclassified_regions_are_dropped():
    ndsm, classes = _scene()
    _add(ndsm, classes, slice(5, 7), slice(5, 7), 12.0)  # 1 m^2 speckle
    _add(ndsm, classes, slice(50, 90), slice(50, 90), 0.8)  # "building" at pavement height
    parts = footprints(*building_parts(ndsm, classes, GSD), shape=ndsm.shape, gsd_m=GSD)
    assert parts == []


def test_courtyard_is_kept_as_a_hole():
    ndsm, classes = _scene()
    _add(ndsm, classes, slice(20, 100), slice(20, 100), 15.0)
    _add(ndsm, classes, slice(45, 75), slice(45, 75), 0.0, cls=1)  # open courtyard
    parts = footprints(*building_parts(ndsm, classes, GSD), shape=ndsm.shape, gsd_m=GSD)
    assert len(parts) == 1 and len(parts[0]["rings"]) == 2


def test_rings_are_normalized_to_unit_square():
    ndsm, classes = _scene()
    _add(ndsm, classes, slice(0, 30), slice(90, 120), 12.0)  # touches the image edge
    parts = footprints(*building_parts(ndsm, classes, GSD), shape=ndsm.shape, gsd_m=GSD)
    pts = np.array([p for ring in parts[0]["rings"] for p in ring])
    assert pts.min() >= 0.0 and pts.max() <= 1.0


def test_soft_edge_ramps_do_not_become_terraces():
    # The model's heights ramp up over a few pixels at walls; that must stay one block.
    ndsm, classes = _scene()
    _add(ndsm, classes, slice(20, 100), slice(20, 100), 1.0)
    yy, xx = np.mgrid[20:100, 20:100]
    edge = np.minimum.reduce([yy - 20, 99 - yy, xx - 20, 99 - xx]).astype(np.float32)
    ndsm[20:100, 20:100] = np.minimum(24.0, 2.0 + edge * 4.0)  # 2 m at the wall -> 24 m in ~6 px
    parts = footprints(*building_parts(ndsm, classes, GSD), shape=ndsm.shape, gsd_m=GSD)
    assert len(parts) == 1 and 22 <= parts[0]["h"] <= 24


def test_touching_houses_of_different_height_become_separate_blocks():
    ndsm, classes = _scene()
    _add(ndsm, classes, slice(20, 60), slice(20, 50), 6.0)  # row houses sharing a wall,
    _add(ndsm, classes, slice(20, 60), slice(50, 80), 9.0)  # only 3 m apart in height
    parts = footprints(*building_parts(ndsm, classes, GSD), shape=ndsm.shape, gsd_m=GSD)
    assert sorted(round(p["h"]) for p in parts) == [6, 9]


def test_equal_roofs_split_along_the_dip_between_them():
    ndsm, classes = _scene()
    _add(ndsm, classes, slice(20, 60), slice(20, 80), 8.0)
    ndsm[20:60, 48:52] = 4.0  # the lower seam between two neighbouring roofs
    parts = footprints(*building_parts(ndsm, classes, GSD), shape=ndsm.shape, gsd_m=GSD)
    assert len(parts) == 2 and all(7 <= p["h"] <= 8.5 for p in parts)


def _dome(ndsm, classes, cy, cx, radius, height):
    yy, xx = np.mgrid[: ndsm.shape[0], : ndsm.shape[1]]
    d = np.hypot(yy - cy, xx - cx)
    crown = d < radius
    ndsm[crown] = np.maximum(ndsm[crown], height * np.sqrt(1 - (d[crown] / radius) ** 2))
    classes[crown] = TREE


def test_trees_become_individual_crowns_with_photo_colour():
    ndsm, classes = _scene()
    rgb = np.zeros(ndsm.shape + (3,), np.uint8)
    rgb[...] = (40, 110, 50)
    _dome(ndsm, classes, 30, 30, 10, 12.0)
    _dome(ndsm, classes, 30, 80, 12, 18.0)
    _add(ndsm, classes, slice(80, 110), slice(80, 110), 20.0)  # a building is not a tree
    found = sorted(trees(ndsm, classes, rgb, GSD), key=lambda t: t["h"])
    assert len(found) == 2
    small, big = found
    assert abs(small["h"] - 12.0) < 1.0 and abs(big["h"] - 18.0) < 1.0
    assert abs(big["u"] - 80 / 120) < 0.03 and abs(big["v"] - 30 / 120) < 0.03
    assert 2.0 <= small["r"] <= 6.0 and big["r"] > small["r"]
    assert small["c"] == [40, 110, 50]


def test_low_shrubs_are_not_trees():
    ndsm, classes = _scene()
    _dome(ndsm, classes, 60, 60, 8, 1.5)
    assert trees(ndsm, classes, np.zeros(ndsm.shape + (3,), np.uint8), GSD) == []


def test_empty_scene_gives_no_buildings_and_no_trees():
    ndsm, classes = _scene()
    model = city_model(ndsm, classes, np.zeros(ndsm.shape + (3,), np.uint8), GSD)
    assert model == {"buildings": [], "trees": []}


def test_buildings_and_trees_stand_on_sloping_ground():
    ndsm, classes = _scene()
    _add(ndsm, classes, slice(40, 80), slice(40, 80), 12.0)
    _dome(ndsm, classes, 20, 100, 8, 10.0)
    yy, xx = np.mgrid[:120, :120]
    ground = (0.5 * xx).astype(np.float32)  # rises 60 m across the tile
    rgb = np.zeros(ndsm.shape + (3,), np.uint8)
    model = city_model(ndsm, classes, rgb, GSD, ground=ground)
    (b,) = model["buildings"]
    assert abs(b["b"] - 20.0) < 1.0  # lowest ground under the footprint (column 40)
    assert abs(b["t"] - (30.0 + 12.0)) < 1.5  # median ground (~column 60) + height
    (t,) = model["trees"]
    assert abs(t["b"] - 50.0) < 1.0  # ground at the trunk (column 100)


def test_without_ground_blocks_start_at_zero():
    ndsm, classes = _scene()
    _add(ndsm, classes, slice(40, 80), slice(40, 80), 12.0)
    (b,) = city_model(ndsm, classes, np.zeros(ndsm.shape + (3,), np.uint8), GSD)["buildings"]
    assert b["b"] == 0.0 and b["t"] == b["h"]


def _angles_deg(ring):
    pts = np.array(ring)
    out = []
    for i in range(len(pts)):
        a, b, c = pts[i - 1], pts[i], pts[(i + 1) % len(pts)]
        v1, v2 = a - b, c - b
        cos = v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2))
        out.append(np.degrees(np.arccos(np.clip(cos, -1, 1))))
    return np.array(out)


def test_regularize_turns_a_ragged_rectangle_into_a_clean_one():
    from shapely.geometry import Polygon

    from viewer.city_model import regularize

    rng = np.random.default_rng(0)
    xs = np.r_[np.linspace(0, 40, 30), np.full(20, 40.0), np.linspace(40, 0, 30), np.zeros(20)]
    ys = np.r_[np.zeros(30), np.linspace(0, 24, 20), np.full(30, 24.0), np.linspace(24, 0, 20)]
    ragged = Polygon(np.c_[xs, ys] + rng.normal(0, 0.6, (100, 2))).buffer(0)
    clean = regularize(ragged, grid_px=2.0)
    assert len(clean.exterior.coords) - 1 == 4
    assert abs(clean.area - ragged.area) / ragged.area < 0.05
    assert np.allclose(_angles_deg(clean.exterior.coords[:-1]), 90, atol=1)


def test_regularize_keeps_an_l_shape_right_angled():
    from shapely.geometry import Polygon

    from viewer.city_model import regularize

    l_shape = Polygon([(0, 0), (40, 0), (40, 15), (15, 15), (15, 40), (0, 40)])
    wobbly = Polygon(np.array(l_shape.exterior.coords) + np.random.default_rng(1).normal(0, 0.4, (7, 2)))
    clean = regularize(wobbly.buffer(0), grid_px=2.0)
    ring = clean.exterior.coords[:-1]
    assert len(ring) == 6
    assert np.all(np.abs(_angles_deg(ring) - 90) < 2)  # inner and outer corners are both 90 degrees
    assert abs(clean.area - l_shape.area) / l_shape.area < 0.08


def test_regularize_follows_a_rotated_building():
    from shapely import affinity
    from shapely.geometry import box

    from viewer.city_model import regularize

    rotated = affinity.rotate(box(0, 0, 50, 20), 30, origin=(0, 0))
    clean = regularize(rotated.buffer(0.3).buffer(-0.3), grid_px=2.0)
    ring = np.array(clean.exterior.coords[:-1])
    assert len(ring) == 4
    edge = ring[1] - ring[0]
    angle = np.degrees(np.arctan2(edge[1], edge[0])) % 90
    assert min(abs(angle - 30), abs(angle - 60)) < 2  # edges stay at the building's own angle


def test_regularize_keeps_the_traced_shape_of_a_building_that_is_not_boxy():
    from shapely.geometry import Point

    from viewer.city_model import regularize

    rotunda = Point(0, 0).buffer(20)  # squaring this off would move far more than 10 % of it
    out = regularize(rotunda, grid_px=2.0)
    assert rotunda.symmetric_difference(out).area <= 0.10 * rotunda.area
    assert len(out.exterior.coords) > 8  # still round, not a box
