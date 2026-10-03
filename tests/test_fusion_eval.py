from types import SimpleNamespace

import numpy as np

from viewer.fusion_eval import _cache_matches, report, route, soft_route, write_csv


def test_building_route_only_changes_predicted_buildings():
    h1 = np.array([[1, 2], [3, 4]], dtype=np.float32)
    h2 = h1 + 10
    classes = np.array([[3, 1], [6, 3]], dtype=np.uint8)
    np.testing.assert_allclose(route(h1, h2, classes, 0.5), [[6, 2], [3, 9]])


def test_soft_route_is_v1_below_threshold_and_v2_at_high_probability():
    h1 = np.zeros((1, 3), dtype=np.float32)
    h2 = np.full((1, 3), 10.0, dtype=np.float32)
    p = np.array([[0.1, 0.5, 0.9]], dtype=np.float32)
    np.testing.assert_allclose(soft_route(h1, h2, p, 0.5, 0.1, 0.9), [[0, 2.5, 5]])


def test_fusion_csv_contains_overall_and_height_bin_rows(tmp_path):
    result = {
        "split": "val",
        "variants": {
            "hard_w0.50": {
                "overall": {"ALL": {"n": 2, "rmse": 1, "mae": 1, "bias": 0,
                                     "pearson": 1, "building_rmse": 1, "building_mae": 1,
                                     "building_bias": 0}},
                "height_bins": {"ALL": [{"bin": "0-2m", "n": 2, "rmse": 1, "mae": 1,
                                          "bias": 0, "pearson": 1, "building_rmse": 1,
                                          "building_mae": 1, "building_bias": 0}]},
            }
        },
    }
    path = tmp_path / "fusion.csv"
    write_csv(result, path)
    rows = path.read_text().splitlines()
    assert len(rows) == 3
    assert "overall" in rows[1] and "height_bin" in rows[2]


def test_report_excludes_invalid_reference_sentinel(tmp_path):
    tile = SimpleNamespace(split="val", scene_id="DC_sample")
    np.savez(tmp_path / "val__DC_sample.npz",
             h1=np.array([[1.0, 9.0]], np.float32),
             h2=np.array([[3.0, 9.0]], np.float32),
             p_building=np.array([[1.0, 0.0]], np.float32),
             predicted_classes=np.array([[3, 1]], np.uint8),
             truth_classes=np.array([[3, 1]], np.uint8),
             reference=np.array([[2.0, -5.0]], np.float32),
             tta=np.bool_(False))
    args = SimpleNamespace(cache=tmp_path, split="val", tta=False)

    result = report([tile], args, ["hard_w0.50"])

    scores = result["variants"]["hard_w0.50"]["overall"]["ALL"]
    assert scores["n"] == 1
    assert scores["rmse"] == 0.0


def test_tta_cache_provenance_is_enforced(tmp_path):
    old = tmp_path / "old.npz"
    np.savez(old, h1=np.zeros(1))
    assert _cache_matches(old, {"h1"}, False)
    assert not _cache_matches(old, {"h1"}, True)

    current = tmp_path / "current.npz"
    np.savez(current, h1=np.zeros(1), tta=np.bool_(True))
    assert _cache_matches(current, {"h1"}, True)
    assert not _cache_matches(current, {"h1"}, False)
