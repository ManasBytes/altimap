"""Optional train-only global affine scale calibration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .metrics import evaluate_height


@dataclass(frozen=True)
class AffineCalibration:
    scale: float
    offset: float

    def apply(self, prediction: np.ndarray) -> np.ndarray:
        return self.scale * np.asarray(prediction, dtype=np.float32) + self.offset


def fit_affine(prediction: np.ndarray, reference_agl: np.ndarray) -> AffineCalibration:
    """Fit one global ``reference = scale * prediction + offset`` transform."""
    pred = np.asarray(prediction, dtype=np.float64)
    ref = np.asarray(reference_agl, dtype=np.float64)
    if pred.shape != ref.shape:
        raise ValueError("prediction and reference must have matching shapes")
    mask = np.isfinite(pred) & np.isfinite(ref) & (ref >= 0)
    if np.count_nonzero(mask) < 2:
        raise ValueError("at least two valid pixels are required for calibration")
    matrix = np.column_stack((pred[mask], np.ones(np.count_nonzero(mask))))
    scale, offset = np.linalg.lstsq(matrix, ref[mask], rcond=None)[0]
    return AffineCalibration(float(scale), float(offset))


def calibration_gate(
    baseline_prediction: np.ndarray,
    calibrated_prediction: np.ndarray,
    reference_agl: np.ndarray,
    minimum_improvement: float = 0.05,
) -> bool:
    """Accept calibration only when validation MAE improves by the threshold."""
    if minimum_improvement < 0 or not np.all(np.isfinite(calibrated_prediction)):
        return False
    baseline = evaluate_height(baseline_prediction, reference_agl).mae_m
    calibrated = evaluate_height(calibrated_prediction, reference_agl).mae_m
    return calibrated <= baseline * (1.0 - minimum_improvement)
