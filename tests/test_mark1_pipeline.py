from pathlib import Path

import numpy as np
import pytest

from mark1.artifacts import export_scene
from mark1.calibration import calibration_gate, fit_affine
from mark1.codec import encode_rg16
from mark1.gamus import GamusScene, GamusTriplet, MissingH5DependencyError
from mark1.metrics import evaluate_height
from mark1.models import DAV2Predictor, MissingModelDependencyError, RDAHPredictor
from mark1.semantic_prior import fit_class_priors, predict_semantic_height


def test_metrics_mask_units_and_class_mae() -> None:
    pred = np.array([[1, 4], [np.nan, 10]], dtype=np.float32)
    ref = np.array([[1, 2], [-1, 8]], dtype=np.float32)
    classes = np.array([[3, 3], [6, 6]])
    result = evaluate_height(pred, ref, classes)
    assert result.valid_pixels == 3
    assert result.mae_m == pytest.approx(4 / 3)
    assert result.mse_m2 == pytest.approx(8 / 3)
    assert result.rmse_m == pytest.approx(np.sqrt(8 / 3))
    assert result.bias_m == pytest.approx(4 / 3)
    assert result.class_mae_m == {"building": 1.0, "tree": 2.0}


def test_affine_calibration_and_validation_gate() -> None:
    raw = np.array([[1, 2, 3]], dtype=np.float32)
    reference = 2 * raw + 1
    calibration = fit_affine(raw, reference)
    np.testing.assert_allclose(calibration.apply(raw), reference)
    assert calibration_gate(raw, calibration.apply(raw), reference)


def test_model_interfaces_fail_clearly_without_runtime() -> None:
    with pytest.raises(MissingModelDependencyError, match="DAV2"):
        DAV2Predictor()
    with pytest.raises(MissingModelDependencyError, match="RDAH"):
        RDAHPredictor()


def test_codec_and_exporter_write_scientific_and_browser_assets(tmp_path: Path) -> None:
    shape = (8, 8)
    rgb = np.zeros((*shape, 3), dtype=np.uint8)
    depth = np.tile(np.linspace(0, 1, shape[1], dtype=np.float32), (shape[0], 1))
    reference = np.ones(shape, dtype=np.float32) * 5
    prediction = reference + 1
    classes = np.zeros(shape, dtype=np.int16)
    manifest = export_scene(tmp_path, "DC_TEST", "train", rgb, depth, prediction, reference, classes)
    assert manifest["metrics"]["mae_m"] == pytest.approx(1)
    for name in ("predicted.npy", "reference_agl.npy", "classes.npy", "dav2-depth.png", "predicted-height.png", "reference-height.png", "error-heatmap.png", "scene.json"):
        assert (tmp_path / name).is_file()
    codec = encode_rg16(reference, tmp_path / "codec.png", 0, 10)
    assert codec["encoding"] == "rg16-linear"


def test_semantic_prior_uses_training_medians_and_depth_shape() -> None:
    reference = np.array([[0, 0, 8, 12], [1, 2, 16, 20]], dtype=np.float32)
    classes = np.array([[1, 1, 3, 3], [2, 2, 6, 6]], dtype=np.int16)
    priors = fit_class_priors([(reference, classes)])
    assert priors[1] == 0
    assert priors[3] == 10
    assert priors[6] == 18
    depth = np.arange(8, dtype=np.float32).reshape(2, 4)
    prediction = predict_semantic_height(depth, classes, priors)
    assert prediction.shape == reference.shape
    assert np.all(prediction[classes == 1] == 0)
    assert prediction[1, 3] > prediction[0, 3]


def test_local_gamus_triplet_is_wired_when_h5py_is_available() -> None:
    directory = Path(r"C:\Users\Rishabh\Downloads\SIH\data\gamus_mark1\raw\DC_03_26")
    if not directory.exists():
        pytest.skip("local GAMUS triplet is not available")
    try:
        triplet = GamusTriplet.from_directory(directory, "DC_03_26")
        scene = GamusScene(triplet)
        rgb = scene.load_rgb()
    except MissingH5DependencyError:
        pytest.skip("h5py is not installed in this environment")
    assert rgb.shape == (1024, 1024, 3)
    agl, classes = scene.load_targets()
    assert agl.shape == classes.shape == (1024, 1024)
