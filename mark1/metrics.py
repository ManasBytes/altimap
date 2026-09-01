"""Prediction-versus-AGL metrics in metres (and square metres for MSE)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class HeightMetrics:
    mae_m: float
    mse_m2: float
    rmse_m: float
    bias_m: float
    valid_pixels: int
    class_mae_m: dict[str, float]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def evaluate_height(
    prediction: np.ndarray,
    reference_agl: np.ndarray,
    classes: np.ndarray | None = None,
    class_names: Mapping[int, str] | None = None,
) -> HeightMetrics:
    """Evaluate with finite prediction/reference and ``reference >= 0`` mask."""
    pred = np.asarray(prediction, dtype=np.float64)
    ref = np.asarray(reference_agl, dtype=np.float64)
    if pred.shape != ref.shape or pred.ndim != 2:
        raise ValueError(f"prediction and reference must be matching 2D arrays, got {pred.shape}, {ref.shape}")
    if classes is not None and np.asarray(classes).shape != ref.shape:
        raise ValueError("classes must have the same shape as prediction")
    valid = np.isfinite(pred) & np.isfinite(ref) & (ref >= 0)
    if not np.any(valid):
        raise ValueError("no valid AGL pixels remain after masking")
    error = pred[valid] - ref[valid]
    class_mae: dict[str, float] = {}
    if classes is not None:
        labels = np.asarray(classes)
        names = class_names or {3: "building", 6: "tree"}
        for label, name in names.items():
            mask = valid & (labels == label)
            if np.any(mask):
                class_mae[name] = float(np.mean(np.abs(pred[mask] - ref[mask])))
    mse = float(np.mean(error * error))
    return HeightMetrics(
        mae_m=float(np.mean(np.abs(error))),
        mse_m2=mse,
        rmse_m=float(np.sqrt(mse)),
        bias_m=float(np.mean(error)),
        valid_pixels=int(error.size),
        class_mae_m=class_mae,
    )
