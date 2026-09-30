import math

import numpy as np

from tests.conftest import make_synthetic_ndsm
from viewer.height_metrics import ScoreAccumulator, height_scores


def test_perfect_prediction_scores_zero_error() -> None:
    ref = make_synthetic_ndsm()
    s = height_scores(ref.copy(), ref)
    assert s["rmse"] == 0.0 and s["mae"] == 0.0
    assert math.isclose(s["pearson"], 1.0)
    assert s["n"] == ref.size


def test_constant_prediction_has_undefined_correlation() -> None:
    ref = make_synthetic_ndsm()
    s = height_scores(np.full_like(ref, 3.0), ref)
    assert math.isnan(s["pearson"])
    assert s["mae"] > 0


def test_nan_pixels_are_excluded() -> None:
    ref = make_synthetic_ndsm()
    pred = ref.copy()
    pred[0, 0] = np.nan
    ref2 = ref.copy()
    ref2[1, 1] = np.nan
    s = height_scores(pred, ref2)
    assert s["n"] == ref.size - 2
    assert s["rmse"] == 0.0


def test_building_rmse_uses_only_building_pixels() -> None:
    ref = np.zeros((4, 4), np.float32)
    pred = np.zeros((4, 4), np.float32)
    classes = np.zeros((4, 4), np.uint8)
    classes[0, :] = 3
    pred[0, :] = 2.0  # 2 m error on buildings only
    s = height_scores(pred, ref, classes)
    assert math.isclose(s["building_rmse"], 2.0)
    assert math.isclose(s["rmse"], 1.0)  # sqrt(4 * 4 / 16)


def test_building_rmse_is_nan_without_buildings() -> None:
    ref = make_synthetic_ndsm()
    s = height_scores(ref.copy(), ref, np.zeros(ref.shape, np.uint8))
    assert math.isnan(s["building_rmse"])


def test_accumulator_pooled_over_tiles_matches_one_shot() -> None:
    ref = make_synthetic_ndsm()
    pred = ref * 0.8 + 1.0
    classes = (ref > 10).astype(np.uint8) * 3
    acc = ScoreAccumulator()
    acc.add(pred[:32], ref[:32], classes[:32])
    acc.add(pred[32:], ref[32:], classes[32:])
    pooled, whole = acc.result(), height_scores(pred, ref, classes)
    for key in ("rmse", "mae", "pearson", "building_rmse"):
        assert math.isclose(pooled[key], whole[key], rel_tol=1e-9)
    assert pooled["n"] == whole["n"]


def test_empty_accumulator_is_nan() -> None:
    assert math.isnan(ScoreAccumulator().result()["rmse"])


def test_all_nan_input_returns_nan_not_raise() -> None:
    ref = np.full((3, 3), np.nan, np.float32)
    s = height_scores(ref, ref)
    assert s["n"] == 0 and math.isnan(s["rmse"])


def test_class_scores_locate_the_error():
    from viewer.height_metrics import class_scores

    ref = np.zeros((10, 10))
    ref[:5] = 20.0  # top half buildings
    classes = np.ones((10, 10), np.uint8)
    classes[:5] = 3
    pred = ref.copy()
    pred[:5] -= 4.0  # buildings read 4 m low, ground exact
    pred[9, 9] = np.nan
    rows = {r["class"]: r for r in class_scores(pred, ref, classes)}
    assert set(rows) == {"ground", "buildings"}
    assert rows["buildings"]["bias"] == -4.0 and rows["buildings"]["rmse"] == 4.0
    assert rows["ground"]["rmse"] == 0.0 and abs(rows["buildings"]["share"] - 50 / 99) < 1e-9
