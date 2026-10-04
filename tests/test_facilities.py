import numpy as np
from rasterio.transform import from_origin
from rasterio.warp import transform as warp

from viewer import facilities


def test_facilities_land_in_their_building_or_on_the_ground():
    crs, t = "EPSG:32617", from_origin(500000, 4480000, 1.0, 1.0)  # 100 x 100 m, 1 m pixels
    (lon_a, lon_b, lon_out), (lat_a, lat_b, lat_out) = warp(crs, "EPSG:4326", [500025, 500075, 500300],
                                                            [4479975, 4479925, 4479950])
    found = [{"kind": "hospital", "name": "General", "lon": lon_a, "lat": lat_a},
             {"kind": "school", "name": "", "lon": lon_b, "lat": lat_b},
             {"kind": "police", "name": "far", "lon": lon_out, "lat": lat_out}]
    building = {"rings": [[[0.1, 0.1], [0.4, 0.1], [0.4, 0.4], [0.1, 0.4]]], "t": 30.0}
    ground = np.full((50, 50), 7.0)
    out = facilities.place(found, t, crs, (100, 100), [building], ground)
    assert [f["kind"] for f in out] == ["hospital", "school"]  # the off-image one is dropped
    assert building["facility"] == {"kind": "hospital", "name": "General"}
    assert out[0]["z"] == 30.0 and out[1]["z"] == 7.0  # on the roof, on the ground
    assert abs(out[0]["u"] - 0.25) < 0.01 and abs(out[1]["v"] - 0.75) < 0.01
