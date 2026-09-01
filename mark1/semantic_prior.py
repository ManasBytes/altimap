"""CLS-assisted semantic height fallback for the Mark-1 GAMUS prototype."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


CLASS_NAMES = {
    0: "other", 1: "ground", 2: "low vegetation", 3: "building",
    4: "water", 5: "road", 6: "tree",
}


def fit_class_priors(samples: Iterable[tuple[np.ndarray, np.ndarray]]) -> dict[int, float]:
    """Fit one global median AGL per CLS class from training scenes only."""
    buckets: dict[int, list[np.ndarray]] = {label: [] for label in CLASS_NAMES}
    for reference, classes in samples:
        reference = np.asarray(reference, dtype=np.float32)
        classes = np.asarray(classes)
        valid = np.isfinite(reference) & (reference >= 0)
        for label in CLASS_NAMES:
            values = reference[valid & (classes == label)]
            if values.size:
                buckets[label].append(values)
    return {
        label: float(np.median(np.concatenate(values))) if values else 0.0
        for label, values in buckets.items()
    }


def predict_semantic_height(
    relative_depth: np.ndarray,
    classes: np.ndarray,
    priors: dict[int, float],
) -> np.ndarray:
    """Modulate global class priors with the scene's robust DAV2 geometry cue."""
    depth = np.asarray(relative_depth, dtype=np.float32)
    labels = np.asarray(classes)
    if depth.shape != labels.shape or depth.ndim != 2:
        raise ValueError("relative_depth and classes must be matching 2D arrays")
    finite = depth[np.isfinite(depth)]
    if finite.size == 0:
        raise ValueError("relative_depth contains no finite values")
    low, high = np.percentile(finite, (2, 98))
    normalized = np.clip((np.nan_to_num(depth, nan=low) - low) / max(high - low, 1e-6), 0, 1)
    prediction = np.zeros_like(depth, dtype=np.float32)
    for label in CLASS_NAMES:
        mask = labels == label
        prediction[mask] = float(priors.get(label, 0.0)) * (0.65 + 0.70 * normalized[mask])
    return np.maximum(prediction, 0.0)
