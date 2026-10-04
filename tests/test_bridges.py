import numpy as np
from rasterio.transform import from_origin
from rasterio.warp import transform as warp

from viewer import bridges


def test_deck_width_follows_tags_then_lanes_then_type():
    assert bridges.width_m({"width": "11.5 m", "lanes": "2"}) == 11.5
    assert bridges.width_m({"lanes": "4", "highway": "secondary"}) == 4 * 3.5 + 1.5
    assert bridges.width_m({"highway": "footway"}) == 3.0
    assert bridges.width_m({}) == 7.0


def test_a_bridge_split_into_ways_is_one_deck_between_its_land_ends():
    # One 400 m bridge across a river, mapped as two ways meeting mid-span (over the water).
    # The banks are at 230 m (west) and 240 m (east); the river below is at 217 m.
    crs, t = "EPSG:32617", from_origin(500000, 4480000, 1.0, 1.0)  # 1 m pixels, 600 x 100 m

    def to_lonlat(xs, ys):
        return list(zip(*warp(crs, "EPSG:4326", xs, ys)))

    y = 4480000 - 50
    ways = [{"lonlat": to_lonlat([500100, 500300], [y, y]), "tags": {"highway": "primary"}},
            {"lonlat": to_lonlat([500300, 500500], [y, y]), "tags": {"highway": "primary"}}]
    terrain = lambda xs, ys: np.where(np.asarray(xs) < 500200, 230.0, np.where(np.asarray(xs) > 500400, 240.0, 217.0))
    deck = bridges.decks(ways, t, crs, (100, 600), terrain)
    assert np.isfinite(deck[50, 110:490]).all() and np.isnan(deck[10, 300])  # a deck strip, nothing beside it
    assert abs(deck[50, 105] - 230) < 1.5 and abs(deck[50, 495] - 240) < 1.5  # lands on each bank
    assert 233 < deck[50, 300] < 237  # mid-span between the banks, not down at the water (217 m)


def test_osm_answers_are_cached_so_a_scene_keeps_its_bridges_offline(monkeypatch, tmp_path):
    import requests

    monkeypatch.setenv("ALTIMAP_DEM_CACHE", str(tmp_path))
    answer = {"elements": [{"geometry": [{"lon": 1.0, "lat": 2.0}, {"lon": 1.1, "lat": 2.0}], "tags": {"bridge": "yes"}}]}

    class Reply:
        def raise_for_status(self):
            pass

        def json(self):
            return answer

    monkeypatch.setattr(requests, "post", lambda *a, **k: Reply())
    assert len(bridges.fetch((1.0, 1.9, 1.2, 2.1))) == 1

    def offline(*a, **k):
        raise requests.ConnectionError("no network")

    monkeypatch.setattr(requests, "post", offline)
    assert bridges.fetch((1.0, 1.9, 1.2, 2.1))[0]["tags"] == {"bridge": "yes"}  # from the cache
    assert bridges.fetch((5.0, 5.0, 5.1, 5.1)) is None  # an uncached area without network: no bridges
