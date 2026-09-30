import math

import numpy as np

from viewer.estimate import _resize, bare_earth, gsd_metres, work_shape


def test_work_shape_resamples_to_model_gsd() -> None:
    # 0.5 m imagery -> 0.33 m model grid: 1.5x more pixels per side
    assert work_shape((1000, 600), 0.495, 0.33) == (1500, 900)


def test_work_shape_without_gsd_keeps_shape() -> None:
    assert work_shape((700, 500), None, 0.33) == (700, 500)


def test_work_shape_caps_longest_side() -> None:
    h, w = work_shape((10000, 5000), 0.33, 0.33, max_side=4096)
    assert max(h, w) == 4096 and math.isclose(h / w, 2.0, rel_tol=0.01)


def test_gsd_metres_projected_and_geographic() -> None:
    assert gsd_metres({"georeferenced": True, "crs": "EPSG:32643", "res_m": [0.5, 0.5],
                       "bounds": [0, 0, 1, 1]}) == 0.5
    # 1e-5 degrees at the equator is ~1.11 m
    g = gsd_metres({"georeferenced": True, "crs": "EPSG:4326", "res_m": [1e-5, 1e-5],
                    "bounds": [77.0, -0.01, 77.01, 0.01]})
    assert 1.0 < g < 1.2


def test_gsd_metres_none_when_not_georeferenced() -> None:
    assert gsd_metres({"georeferenced": False}) is None


def test_resize_keeps_class_maps_integer() -> None:
    classes = np.array([[0, 7], [3, 5]], np.uint8)
    out = _resize(classes, (4, 4), nearest=True)
    assert out.dtype == np.uint8 and set(np.unique(out)) == {0, 3, 5, 7}


def test_resize_heights_stay_float() -> None:
    out = _resize(np.array([[0.0, 10.0]], np.float32), (1, 4))
    assert out.dtype == np.float32 and out.max() <= 10.0


def test_bare_earth_removes_a_building_sized_bump() -> None:
    dem = np.full((20, 20), 100.0)
    dem[8:10, 8:10] = 130.0  # a 60 m wide "building" in a 30 m DEM
    ground = bare_earth(dem, size_px=5)
    assert np.allclose(ground, 100.0)


def test_bare_earth_keeps_a_broad_hill() -> None:
    yy, xx = np.mgrid[0:40, 0:40]
    dem = 100.0 + 0.5 * xx  # smooth slope
    ground = bare_earth(dem, size_px=5)
    assert np.abs(ground - dem)[5:-5, 5:-5].max() < 1.5


def test_dem_consistent_ground_keeps_dem_mean_and_adds_detail():
    from viewer.estimate import dem_consistent_ground

    from scipy import ndimage

    yy, xx = np.mgrid[0:300, 0:300]
    ndsm = np.zeros((300, 300), np.float32)
    ndsm[100:160, 100:160] = 20.0  # one 20 m building (0.5 m pixels -> 30 m square)
    # Copernicus is a surface model: terrain plus the building, blurred over its 30 m cells.
    terrain = 500.0 + 0.2 * xx
    glo = terrain + ndimage.uniform_filter(ndsm, size=60)
    dsm = dem_consistent_ground(glo, ndsm, gsd_m=0.5) + ndsm
    assert abs(dsm.mean() - glo.mean()) < 0.5  # absolute level = the DEM
    roof, street = dsm[130, 130] - terrain[130, 130], dsm[130, 30] - terrain[130, 30]
    assert abs((roof - street) - 20.0) < 2.0  # the building stands ~20 m above its street


def test_display_ground_method_by_building_share():
    from viewer.estimate import display_ground_method

    assert display_ground_method(0.52) == "open300"  # dense downtown
    assert display_ground_method(0.30) == "open150"  # hilly or suburban town
    assert display_ground_method(0.02) == "subtract"  # forest, farmland


def _tif(path, arr, **kw):
    import rasterio

    arr = arr if arr.ndim == 3 else arr[None]
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[1], width=arr.shape[2],
                       count=arr.shape[0], dtype=arr.dtype, **kw) as dst:
        dst.write(arr)
    return path


def test_height_map_upload_is_rejected_with_a_clear_message(tmp_path):
    import pytest

    from viewer.estimate import NotImageryError, read_image

    path = _tif(tmp_path / "agl.tif", np.random.default_rng(0).random((32, 32)).astype(np.float32) * 30)
    with pytest.raises(NotImageryError, match="height"):
        read_image(path)


def test_band_order_follows_the_files_colour_tags(tmp_path):
    import rasterio
    from rasterio.enums import ColorInterp

    from viewer.estimate import read_image

    bgr = np.stack([np.full((8, 8), v, np.uint8) for v in (10, 20, 30)])  # stored blue, green, red
    path = tmp_path / "bgr.tif"
    with rasterio.open(path, "w", driver="GTiff", height=8, width=8, count=3, dtype="uint8") as dst:
        dst.write(bgr)
        dst.colorinterp = [ColorInterp.blue, ColorInterp.green, ColorInterp.red]
    rgb, _ = read_image(path)
    assert rgb[0, 0].tolist() == [30, 20, 10]


def test_untagged_bands_default_to_1_2_3_and_single_band_to_grey(tmp_path):
    from viewer.estimate import read_image

    three = np.stack([np.full((8, 8), v, np.uint8) for v in (10, 20, 30)])
    assert read_image(_tif(tmp_path / "rgb.tif", three))[0][0, 0].tolist() == [10, 20, 30]
    pan = np.full((8, 8), 77, np.uint8)  # e.g. a panchromatic band: allowed, as grey
    assert read_image(_tif(tmp_path / "pan.tif", pan))[0][0, 0].tolist() == [77, 77, 77]


def test_nodata_border_is_masked_and_ignored_by_the_stretch(tmp_path):
    from viewer.estimate import read_image

    img = np.full((3, 20, 20), 1000, np.uint16)
    img[:, 5:15, 5:15] = 3000
    img[:, :, :2] = 0  # nodata strip
    rgb, valid = read_image(_tif(tmp_path / "scene.tif", img, nodata=0))
    assert not valid[:, :2].any() and valid[:, 2:].all()
    assert rgb[10, 10, 0] > rgb[3, 3, 0]  # stretch spans the real data, not the zeros


def test_gsd_metres_for_any_geographic_crs():
    g = gsd_metres({"georeferenced": True, "crs": "EPSG:4674", "res_m": [1e-5, 1e-5],
                    "bounds": [-47.0, -0.01, -46.99, 0.01]})  # SIRGAS 2000, geographic
    assert 1.0 < g < 1.2


def test_read_gcps_follows_the_header_and_skips_junk(tmp_path):
    from viewer.estimate import read_gcps

    f = tmp_path / "gcps.csv"
    f.write_text("name,lat,lon,elevation\nA,28.61,77.20,215.5\n\nB,28.62,77.21,219\n")
    assert read_gcps(f) == []  # a non-numeric first column: rows don't parse as numbers
    f.write_text("lat,lon,elevation\n28.61,77.20,215.5\n\n28.62,77.21,219\n")
    assert read_gcps(f) == [(77.20, 28.61, 215.5), (77.21, 28.62, 219.0)]  # -> lon, lat, h
    f.write_text("77.20,28.61,215.5\n")  # no header: lon, lat, height
    assert read_gcps(f) == [(77.20, 28.61, 215.5)]


def _utm_grid():
    from rasterio.transform import from_origin

    return from_origin(700_000.0, 3_100_000.0, 1.0, 1.0), "EPSG:32643"  # 1 m pixels, UTM 43N


def _lonlat(transform, crs, rows, cols):
    from rasterio.warp import transform as warp

    xs, ys = zip(*[transform * (c + 0.5, r + 0.5) for r, c in zip(rows, cols)])
    lons, lats = warp(crs, "EPSG:4326", list(xs), list(ys))
    return lons, lats


def test_gcps_remove_a_datum_sized_offset_and_ignore_outside_points():
    from viewer.estimate import gcp_correction

    transform, crs = _utm_grid()
    true = np.full((200, 200), 250.0, np.float32)
    ground = true - 30.0  # e.g. GPS ellipsoidal points vs an EGM2008 DEM: tens of metres
    lons, lats = _lonlat(transform, crs, [20, 100, 180], [30, 90, 150])
    gcps = [(lo, la, 250.0 + e) for lo, la, e in zip(lons, lats, (0.3, -0.2, 0.1))]
    gcps.append((0.0, 0.0, 10.0))  # far outside the image
    fix, info = gcp_correction(ground, transform, crs, gcps)
    assert info["n_given"] == 4 and info["n_used"] == 3 and info["model"] == "offset"
    assert abs(float(fix[0, 0]) - 30.0) < 0.2
    assert info["rmse_after_m"] < 0.3 and abs(info["rmse_before_m"] - 30.0) < 0.3


def test_gcps_leave_small_or_inconsistent_offsets_alone():
    from viewer.estimate import GCP_MIN_OFFSET_M, gcp_correction

    transform, crs = _utm_grid()
    ground = np.full((200, 200), 300.0, np.float32)
    rows, cols = [10, 10, 190, 190, 100, 50], [10, 190, 10, 190, 100, 150]
    lons, lats = _lonlat(transform, crs, rows, cols)
    # within GLO-30's own accuracy: not a datum problem, leave it
    small = [(lo, la, 300.0 + GCP_MIN_OFFSET_M / 2) for lo, la in zip(lons, lats)]
    fix, info = gcp_correction(ground, transform, crs, small)
    assert fix is None and info["model"] == "none" and "accuracy" in info["note"]
    # large on average but the points disagree wildly: no common offset to apply
    wild = [(lo, la, 300.0 + e) for lo, la, e in zip(lons, lats, (40, -35, 30, -25, 45, -20))]
    fix, info = gcp_correction(ground, transform, crs, wild)
    assert fix is None and info["model"] == "none"


def test_one_padded_dem_read_serves_the_export_and_the_bare_earth_ground():
    from viewer.estimate import dem_on_image, ground_for

    # a 30 m-posted DEM over the image plus a 300 m margin: a plane z = 100 + 0.01 x + 0.02 y
    # (x, y in image pixels of 1 m), so any correct resampling onto the image reproduces it
    pad, shape = 300, (1020, 780)  # padded extents 1620 x 1380 m: whole 30 m cells, as padded_dem asks
    rows_c, cols_c = (shape[0] + 2 * pad) // 30, (shape[1] + 2 * pad) // 30
    yc = (np.arange(rows_c) + 0.5) * 30 - 0.5 - pad
    xc = (np.arange(cols_c) + 0.5) * 30 - 0.5 - pad
    coarse = 100 + 0.01 * xc[None, :] + 0.02 * yc[:, None]
    dem = dem_on_image(coarse, (pad, pad), shape)
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    assert dem.shape == shape and np.abs(dem - (100 + 0.01 * xx + 0.02 * yy)).max() < 0.05
    # the opening removes a 60 m "building" block but keeps the terrain plane
    bumpy = coarse.copy()
    bumpy[20:22, 20:22] += 30
    ground = ground_for(bumpy, (pad, pad), shape, 150.0)
    assert np.abs(ground - dem).max() < 1.5
