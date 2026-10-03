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


def test_encode_grid16_keeps_millimetre_heights_on_the_mesh_grid():
    import base64

    from viewer.geo import encode_grid16

    yy, xx = np.mgrid[0:1024, 0:768]
    heights = (0.01 * xx + 0.02 * yy).astype(np.float32)  # a gentle slope, 0 to ~28 m
    heights[:5, :5] = np.nan  # nodata corner
    g = encode_grid16(heights, 513)
    u16 = np.frombuffer(base64.b64decode(g["b64"]), "<u2").reshape(513, 513)
    decoded = g["lo"] + u16 / 65535 * g["span"]  # what the viewer does
    assert np.isfinite(decoded).all()
    # centre sample of the grid vs the true slope there (pixel-centre aligned resampling)
    r, c = (256 + 0.5) * 1024 / 513 - 0.5, (256 + 0.5) * 768 / 513 - 0.5
    assert abs(decoded[256, 256] - (0.01 * c + 0.02 * r)) < 0.005
    # neighbours differ by the real slope, not by an 8-bit step
    step = np.diff(decoded[256, 250:260])
    assert np.allclose(step, 0.01 * 768 / 513, atol=0.002)


def test_warp_to_grid_mosaics_tiles_in_another_crs(tmp_path):
    import rasterio
    from rasterio.transform import from_origin

    from viewer.geo import warp_to_grid

    # two lon/lat DEM tiles side by side (west one = 100 m, east one = 200 m, 0.001 deg posting)
    for name, lon0, value in (("w.tif", 77.0, 100.0), ("e.tif", 77.1, 200.0)):
        with rasterio.open(tmp_path / name, "w", driver="GTiff", width=100, height=100, count=1,
                           dtype="float32", crs="EPSG:4326", nodata=-32767,
                           transform=from_origin(lon0, 28.1, 0.001, 0.001)) as dst:
            dst.write(np.full((1, 100, 100), value, np.float32))
    # a UTM 43N grid straddling the tile seam at lon 77.1 (x ~ 709.8 km at lat 28.05)
    from rasterio.warp import transform
    xs, ys = transform("EPSG:4326", "EPSG:32643", [77.08, 77.12], [28.04, 28.06])
    grid = warp_to_grid([tmp_path / "w.tif", tmp_path / "e.tif"],
                        (xs[0], ys[0], xs[1], ys[1]), "EPSG:32643", (50, 80))
    assert np.nanmin(grid) >= 99 and np.nanmax(grid) <= 201
    assert np.isclose(grid[25, 2], 100, atol=1) and np.isclose(grid[25, -3], 200, atol=1)
    # outside every tile -> NaN, never a sentinel
    far = warp_to_grid([tmp_path / "w.tif"], (xs[0] + 5e4, ys[0], xs[1] + 5e4, ys[1]), "EPSG:32643", (10, 10))
    assert np.isnan(far).all()


def test_requests_without_a_timeout_get_one_after_importing_dem(monkeypatch):
    import requests

    import viewer.dem  # noqa: F401  (installs the default)

    seen = {}

    def fake_send(self, request, **kwargs):
        seen["timeout"] = kwargs.get("timeout")
        raise requests.ConnectionError("offline in tests")

    monkeypatch.setattr(requests.Session, "send", fake_send)
    try:
        requests.Session().get("https://example.invalid/")
    except requests.ConnectionError:
        pass
    assert seen["timeout"] == (20, viewer.dem.HTTP_TIMEOUT_S)


def test_glo30_tile_names_follow_the_south_west_corner():
    from viewer.dem import glo30_tile_urls

    (url,) = glo30_tile_urls(-79.99, 40.42, -79.97, 40.44)  # Pittsburgh
    assert url.endswith("Copernicus_DSM_COG_10_N40_00_W080_00_DEM/Copernicus_DSM_COG_10_N40_00_W080_00_DEM.tif")
    names = [u.rsplit("/", 1)[1] for u in glo30_tile_urls(77.9, 12.9, 78.1, 13.1)]  # straddles 78 E, 13 N
    assert names == ["Copernicus_DSM_COG_10_N12_00_E077_00_DEM.tif", "Copernicus_DSM_COG_10_N12_00_E078_00_DEM.tif",
                     "Copernicus_DSM_COG_10_N13_00_E077_00_DEM.tif", "Copernicus_DSM_COG_10_N13_00_E078_00_DEM.tif"]
    assert glo30_tile_urls(-0.5, -0.5, -0.4, -0.4)[0].rsplit("/", 1)[1] == "Copernicus_DSM_COG_10_S01_00_W001_00_DEM.tif"


def test_glo30_does_not_turn_an_unresolved_tile_into_zero(monkeypatch, tmp_path):
    import rasterio
    import viewer.dem as dem
    import viewer.geo as geo

    class OpenSource:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_open(url, *args, **kwargs):
        if url == "missing":
            raise rasterio.errors.RasterioIOError("unavailable")
        return OpenSource()

    class NoFallback:
        def __init__(self, *args, **kwargs):
            pass

        def patch(self, *args, **kwargs):
            return None

    monkeypatch.setenv("ALTIMAP_DEM_CACHE", str(tmp_path / "dem-cache"))
    monkeypatch.setattr(dem, "glo30_tile_urls", lambda *args: ["available", "missing"])
    monkeypatch.setattr(rasterio, "open", fake_open)
    monkeypatch.setattr(geo, "warp_to_grid", lambda *args: np.array([[100.0, np.nan]], np.float32))
    monkeypatch.setattr(dem, "DemSource", NoFallback)
    assert dem.glo30((0.0, 0.0, 1.0, 1.0), "EPSG:4326", (1, 2)) is None
