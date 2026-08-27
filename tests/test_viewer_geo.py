import numpy as np
import pytest

from viewer.geo import (
    decode_rg16,
    encode_rg16,
    fit_absolute_elevation,
    ground_size_m,
    resample_to,
)


def test_encode_rg16_roundtrip_within_quantisation_step():
    rng = np.random.default_rng(0)
    depth = rng.uniform(0.7, 1.3, size=(64, 48)).astype(np.float32)
    rgb, lo, hi = encode_rg16(depth)
    assert rgb.dtype == np.uint8 and rgb.shape == (64, 48, 3)
    back = decode_rg16(rgb, lo, hi)
    # One 16-bit step across the range is the theoretical floor on error.
    assert np.abs(back - depth).max() <= (hi - lo) / 65535.0


def test_encode_rg16_uses_the_full_code_range():
    depth = np.linspace(2.0, 3.0, 256, dtype=np.float32).reshape(16, 16)
    rgb, lo, hi = encode_rg16(depth)
    assert lo == pytest.approx(2.0)
    assert hi == pytest.approx(3.0)
    codes = rgb[..., 0].astype(int) * 256 + rgb[..., 1].astype(int)
    assert codes.min() == 0
    assert codes.max() == 65535


def test_encode_rg16_of_constant_depth_does_not_divide_by_zero():
    rgb, lo, hi = encode_rg16(np.full((8, 8), 1.5, dtype=np.float32))
    assert np.isfinite([lo, hi]).all()
    back = decode_rg16(rgb, lo, hi)
    assert np.allclose(back, 1.5)


def test_fit_absolute_elevation_recovers_known_scale_and_offset():
    """height01 in [0,1] mapped onto real metres by elev = scale*h + offset."""
    h = np.linspace(0, 1, 64).reshape(8, 8)
    dem = 3.5 * h + 100.0
    scale, offset, r2 = fit_absolute_elevation(h, dem)
    assert scale == pytest.approx(3.5, abs=1e-9)
    assert offset == pytest.approx(100.0, abs=1e-9)
    assert r2 == pytest.approx(1.0, abs=1e-12)


def test_fit_absolute_elevation_ignores_non_finite_dem_pixels():
    h = np.linspace(0, 1, 64).reshape(8, 8)
    dem = 2.0 * h + 50.0
    dem[0, 0] = np.nan
    dem[3, 3] = np.nan
    scale, offset, r2 = fit_absolute_elevation(h, dem)
    assert scale == pytest.approx(2.0, abs=1e-9)
    assert offset == pytest.approx(50.0, abs=1e-9)


def test_fit_absolute_elevation_returns_nan_when_dem_is_flat():
    """A flat DEM carries no scale information -- the fit is unidentifiable.
    Returning 0.0 scale would silently flatten the terrain instead of saying so."""
    h = np.linspace(0, 1, 64).reshape(8, 8)
    scale, offset, r2 = fit_absolute_elevation(h, np.full((8, 8), 200.0))
    assert np.isnan(r2)
    assert offset == pytest.approx(200.0, abs=1e-6)


def test_fit_absolute_elevation_returns_nan_with_too_few_valid_pixels():
    h = np.linspace(0, 1, 64).reshape(8, 8)
    dem = np.full((8, 8), np.nan)
    dem[0, 0] = 10.0
    scale, offset, r2 = fit_absolute_elevation(h, dem)
    assert np.isnan(scale) and np.isnan(r2)


def test_resample_to_changes_shape_and_preserves_range():
    src = np.linspace(0.0, 1.0, 40 * 40).reshape(40, 40)
    out = resample_to(src, (10, 10))
    assert out.shape == (10, 10)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_resample_to_is_a_noop_when_shapes_match():
    src = np.arange(25, dtype=np.float64).reshape(5, 5)
    assert np.array_equal(resample_to(src, (5, 5)), src)


def test_ground_size_m_multiplies_pixel_count_by_resolution():
    assert ground_size_m(826, 826, (0.5173, 0.5173)) == pytest.approx((427.3, 427.3), abs=0.1)


def test_bboxes_intersect_detects_overlap_and_disjoint():
    """Regression: DemSource silently reused one city's raster handle for the
    next, reading all-NaN windows and reporting 'no coverage' instead of
    reopening. This is the check that should have caught it."""
    from viewer.dem import bboxes_intersect

    austin = (-97.9, 30.1, -97.6, 30.3)
    vienna = (16.2, 48.1, 16.5, 48.3)
    overlapping = (-97.8, 30.2, -97.5, 30.4)

    assert bboxes_intersect(austin, overlapping) is True
    assert bboxes_intersect(austin, vienna) is False
    assert bboxes_intersect(austin, austin) is True
