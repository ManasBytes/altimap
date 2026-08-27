import numpy as np
import pytest

from viewer.refine import (
    building_heights,
    fit_ground,
    flatten_roofs,
    metric_scale_from_buildings,
)


def _scene(h=40, w=40):
    """A tilted ground plane with two flat-roofed blocks on it.

    Height convention throughout: larger = higher, already normalised out of
    depth by the caller.
    """
    yy, xx = np.mgrid[0:h, 0:w]
    ground = 0.10 + 0.30 * (xx / w) + 0.10 * (yy / h)
    field = ground.copy()
    mask = np.zeros((h, w), dtype=np.int32)
    field[5:12, 5:12] += 0.25
    mask[5:12, 5:12] = 1
    field[22:32, 20:30] += 0.40
    mask[22:32, 20:30] = 2
    return field, ground, mask


def test_fit_ground_ignores_building_pixels():
    """Buildings dragging the plane fit is precisely why the naive global fit
    misbehaves; the ground must be estimated from non-building pixels only."""
    field, ground, mask = _scene()
    fitted = fit_ground(field, mask)
    off = mask == 0
    assert np.abs(fitted[off] - ground[off]).max() < 1e-9


def test_fit_ground_without_mask_is_a_plain_plane_fit():
    field, ground, _ = _scene()
    fitted = fit_ground(field, np.zeros_like(field, dtype=np.int32))
    # With every pixel in play the buildings pull the plane upward.
    assert fitted.mean() > ground.mean()


def test_fit_ground_falls_back_when_everything_is_masked():
    """A tile fully covered by footprints must not produce a nan surface."""
    field, _, _ = _scene()
    fitted = fit_ground(field, np.ones_like(field, dtype=np.int32))
    assert np.isfinite(fitted).all()


def test_building_heights_recovers_per_building_relief():
    field, _, mask = _scene()
    ground = fit_ground(field, mask)
    heights = building_heights(field - ground, mask)
    assert set(heights) == {1, 2}
    assert heights[1] == pytest.approx(0.25, abs=0.02)
    assert heights[2] == pytest.approx(0.40, abs=0.02)


def test_building_heights_is_robust_to_outlier_pixels():
    """A few wild pixels inside a footprint (trees, edges, layover) must not
    move the roof -- hence a median rather than a max."""
    field, _, mask = _scene()
    ground = fit_ground(field, mask)
    ndsm = field - ground
    ndsm[5, 5] = 9.0
    ndsm[6, 6] = -9.0
    heights = building_heights(ndsm, mask)
    assert heights[1] == pytest.approx(0.25, abs=0.03)


def test_flatten_roofs_replaces_each_footprint_with_one_level():
    field, _, mask = _scene()
    ground = fit_ground(field, mask)
    ndsm = field - ground
    flat = flatten_roofs(ndsm, mask, building_heights(ndsm, mask))
    assert np.ptp(flat[mask == 1]) == pytest.approx(0.0, abs=1e-12)
    assert np.ptp(flat[mask == 2]) == pytest.approx(0.0, abs=1e-12)
    # Ground is untouched.
    assert np.abs(flat[mask == 0]).max() < 1e-9


def test_metric_scale_from_buildings_recovers_metres_per_unit():
    """Known building heights are the anchor: 0.25 units == 10 m here."""
    rel = {1: 0.25, 2: 0.40, 3: 0.10}
    known = {1: 10.0, 2: 16.0, 3: 4.0}
    scale, offset, r2, n = metric_scale_from_buildings(rel, known)
    assert n == 3
    assert scale == pytest.approx(40.0, abs=1e-6)
    assert offset == pytest.approx(0.0, abs=1e-6)
    assert r2 == pytest.approx(1.0, abs=1e-9)


def test_metric_scale_needs_enough_anchors():
    scale, offset, r2, n = metric_scale_from_buildings({1: 0.2}, {1: 8.0})
    assert n == 1
    assert np.isnan(scale) and np.isnan(r2)


def test_metric_scale_rejects_an_inverted_relationship():
    """Taller buildings reading as lower is not a weak fit, it is a wrong one."""
    rel = {1: 0.10, 2: 0.20, 3: 0.30}
    known = {1: 30.0, 2: 20.0, 3: 10.0}
    scale, offset, r2, n = metric_scale_from_buildings(rel, known)
    assert scale < 0        # reported, so the caller can reject on sign
    assert n == 3


def test_metric_scale_ignores_buildings_without_a_known_height():
    rel = {1: 0.25, 2: 0.40, 9: 0.9}
    known = {1: 10.0, 2: 16.0}
    scale, offset, r2, n = metric_scale_from_buildings(rel, known)
    assert n == 2
    assert scale == pytest.approx(40.0, abs=1e-6)


def test_extrude_known_buildings_uses_real_ground_plus_real_height():
    from viewer.refine import extrude_known_buildings

    elev = np.full((20, 20), 100.0)   # flat 100 m terrain
    elev[5:15, 5:15] += 3.0           # local DEM bump under the footprint: 103 m
    mask = np.zeros((20, 20), dtype=np.int32)
    mask[5:15, 5:15] = 1              # one building footprint, id 1

    fused, n = extrude_known_buildings(elev, mask, {1: 20.0})
    assert n == 1
    # ground under the footprint is 103 m (median of elev there), + 20 m building
    assert np.allclose(fused[5:15, 5:15], 123.0)
    assert np.allclose(fused[mask == 0], 100.0)   # untouched elsewhere


def test_extrude_known_buildings_leaves_unknown_footprints_untouched():
    from viewer.refine import extrude_known_buildings

    elev = np.full((10, 10), 50.0)
    mask = np.zeros((10, 10), dtype=np.int32)
    mask[2:6, 2:6] = 1     # footprint exists...
    mask[7:9, 7:9] = 2     # ...but neither has a known Overture height

    fused, n = extrude_known_buildings(elev, mask, {})
    assert n == 0
    assert np.allclose(fused, elev)   # no fabricated height anywhere


def test_extrude_known_buildings_skips_nan_heights_without_crashing():
    from viewer.refine import extrude_known_buildings

    elev = np.full((6, 6), 10.0)
    mask = np.zeros((6, 6), dtype=np.int32)
    mask[1:3, 1:3] = 1
    fused, n = extrude_known_buildings(elev, mask, {1: float("nan")})
    assert n == 0
    assert np.allclose(fused, 10.0)
