import math

import numpy as np

from viewer.estimate import _resize, bare_earth, gsd_metres


def test_working_scale_resamples_to_model_gsd() -> None:
    from viewer.estimate import working_scale

    assert abs(working_scale((1000, 600), 0.495) - 1.5) < 1e-9  # 0.495 m -> 0.33 m
    assert working_scale((700, 500), None) == 1.0  # unknown GSD: run as-is


def test_working_scale_bounds_the_work_for_huge_coarse_scenes() -> None:
    from viewer.estimate import MAX_WORK_PIXELS, working_scale

    s = working_scale((5000, 5000), 10.0)  # 50 km at 10 m would be 150k px a side at 0.33 m
    assert abs((5000 * s) ** 2 - MAX_WORK_PIXELS) / MAX_WORK_PIXELS < 1e-6


def _fake_run(work, progress=None):
    """A per-pixel "model": height from red, class from green, so tiling can't change it."""
    if progress:
        progress(1.0)
    return work[..., 0].astype(np.float32) / 10.0 + 1.0, (work[..., 1] > 127).astype(np.uint8)


def test_tiled_prediction_matches_one_pass_and_fills_every_pixel(monkeypatch) -> None:
    import viewer.estimate as est

    yy, xx = np.mgrid[:900, :700]
    rgb = np.stack([(xx * 255 / 700), (yy * 255 / 900), np.full(xx.shape, 90)], -1).astype(np.uint8)
    one_h, one_c = est.predict_scene(rgb, 1.5, _fake_run)  # fits in one tile
    monkeypatch.setattr(est, "TILE_WORK_PX", 300)  # force ~4 x 5 tiles
    monkeypatch.setattr(est, "TILE_MARGIN_WORK_PX", 40)
    seen = []
    h, c = est.predict_scene(rgb, 1.5, _fake_run, progress=seen.append)
    assert h.shape == c.shape == rgb.shape[:2]
    assert h.min() >= 1.0  # every pixel written (the fake model never returns < 1)
    assert np.abs(h - one_h).max() < 0.2 and (c != one_c).mean() < 0.01
    assert seen and seen[-1] == 1.0 and seen == sorted(seen)


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


def test_geospatial_metadata_warns_for_rotated_and_non_square_grids(tmp_path):
    import rasterio
    from affine import Affine
    from rasterio.crs import CRS

    from viewer.geo import read_geo_meta

    path = tmp_path / "rotated.tif"
    with rasterio.open(path, "w", driver="GTiff", width=8, height=8, count=3,
                       dtype="uint8", crs=CRS.from_epsg(32643),
                       transform=Affine(1.0, 0.1, 500000.0, 0.0, -2.0, 3000000.0)) as dst:
        dst.write(np.zeros((3, 8, 8), np.uint8))
    meta = read_geo_meta(path)
    assert meta["georeferenced"]
    assert any("rotated" in w for w in meta["geospatial_warnings"])
    assert any("non-square" in w for w in meta["geospatial_warnings"])


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


def test_fusion_uses_validated_v2_weight_on_buildings_only():
    from viewer.estimate import fuse_heights

    h1 = np.full((4, 4), 10.0, np.float32)
    h2 = np.full((4, 4), 30.0, np.float32)
    classes = np.ones((4, 4), np.uint8)
    classes[:2] = 3  # top half buildings
    out = fuse_heights(h1, h2, classes)
    assert np.all(out[:2] == 25.0) and np.all(out[2:] == 10.0)
    assert fuse_heights(h1, None, classes) is h1  # no second model: unchanged


def test_forest_is_extensive_canopy_not_a_tree_cluster():
    from viewer.estimate import forest_mask

    gsd = 1.0
    cls = np.ones((600, 600), np.uint8)
    cls[:, :400] = 6  # a 400 m wide forest
    cls[500:540, 480:520] = 6  # a 40 m cluster of garden trees
    forest = forest_mask(cls, gsd)
    assert forest[300, 100] and forest[300, 300]  # inside the forest
    assert not forest[520, 500]  # the cluster isn't forest
    assert not forest[300, 450]  # open ground beside the forest
    assert not forest[cls != 6].any()


def test_dem_agreement_scores_each_dem_in_30m_cells_and_survives_a_failed_read(monkeypatch):
    import viewer.estimate as est

    dem = np.full((120, 120), 500.0, np.float32)
    dsm = dem.copy()
    dsm[::2] += 4.0  # +-2 m detail around the DEM: every 30 m cell mean is DEM + 2

    def no_network(geo, name):
        raise OSError("SRTM unreachable")

    monkeypatch.setattr(est, "padded_dem", no_network)
    out = est.dem_agreement(dsm, {}, dsm.shape, gsd_m=1.0, dems={"glo30": dem})
    assert out["srtm"] is None  # an unreadable DEM is reported, not raised
    assert out["glo30"]["cell_rmse"] == 2.0 and out["glo30"]["cell_bias"] == 2.0


def test_dsm_never_drops_below_bare_earth_where_the_dem_under_reads_a_tower():
    from viewer.estimate import compose_dsm, dem_consistent_ground

    ndsm = np.zeros((120, 120), np.float32)
    ndsm[40:80, 40:80] = 100.0  # a 20 m tower in its 30 m DEM cell (0.5 m pixels)
    bare = np.full(ndsm.shape, 10.0, np.float32)
    dem = np.full(ndsm.shape, 30.0, np.float32)  # the radar DEM saw only part of the tower
    unfloored = dem_consistent_ground(dem, ndsm, 0.5) + ndsm
    assert unfloored[60, 35] < 10.0  # the cell mean alone pushes the street beside it underground
    dsm = compose_dsm(dem, ndsm, 0.5, bare)
    assert dsm.min() >= 10.0 and dsm[60, 35] == 10.0  # streets stay on the ground
    assert dsm[60, 60] == unfloored[60, 60] > 80.0  # the tower keeps its height


def test_an_unreadable_upload_gets_a_plain_message_not_a_server_path(tmp_path):
    import pytest

    from viewer.estimate import NotImageryError, read_image

    bad = tmp_path / "upload.png"
    bad.write_bytes(b"\x00not an image\x00" * 100)
    with pytest.raises(NotImageryError) as e:
        read_image(bad)
    assert str(tmp_path) not in str(e.value) and "PNG, JPG or GeoTIFF" in str(e.value)
