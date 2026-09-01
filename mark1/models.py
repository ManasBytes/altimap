"""Model boundaries for frozen DAV2 and RDAH inference.

Mark-1 does not train weights.  These interfaces accept injected callables so
the offline runner can bind the installed DAV2/RDAH implementations without
making this package import a particular research repository at runtime.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


class MissingModelDependencyError(ImportError):
    """Raised when a requested pretrained model runtime is unavailable."""


class DAV2Predictor:
    """Frozen Depth Anything V2 adapter around an installed inference callable."""

    def __init__(self, infer: Callable[[np.ndarray], np.ndarray] | None = None) -> None:
        self._infer = infer
        if infer is None:
            raise MissingModelDependencyError(
                "DAV2 is not configured. Install the DAV2 runtime/checkpoint and "
                "pass its frozen inference callable to DAV2Predictor."
            )

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        result = np.asarray(self._infer(rgb), dtype=np.float32)
        if result.shape != np.asarray(rgb).shape[:2]:
            raise ValueError(f"DAV2 must return (H, W), got {result.shape}")
        return result


class RDAHPredictor:
    """Frozen RDAH-Net adapter consuming RGB and the DAV2 prior."""

    def __init__(self, infer: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None) -> None:
        self._infer = infer
        if infer is None:
            raise MissingModelDependencyError(
                "RDAH-Net is not configured. Install the RDAH runtime/checkpoint "
                "and pass its frozen inference callable to RDAHPredictor."
            )

    def predict(self, rgb: np.ndarray, relative_depth: np.ndarray) -> np.ndarray:
        if np.asarray(rgb).shape[:2] != np.asarray(relative_depth).shape:
            raise ValueError("RGB and relative depth must share a spatial shape")
        result = np.asarray(self._infer(rgb, relative_depth), dtype=np.float32)
        if result.shape != np.asarray(relative_depth).shape:
            raise ValueError(f"RDAH must return (H, W), got {result.shape}")
        return result
