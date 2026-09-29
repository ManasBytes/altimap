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
