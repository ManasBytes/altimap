"""Offline GAMUS Mark-1 building blocks.

The package deliberately keeps model inference, data targets, and evaluation
as separate interfaces so reference AGL/CLS data cannot accidentally become a
prediction input.
"""

from .calibration import AffineCalibration, calibration_gate, fit_affine
from .gamus import GamusScene, GamusTriplet, MissingH5DependencyError
from .metrics import HeightMetrics, evaluate_height

__all__ = [
    "AffineCalibration",
    "GamusScene",
    "GamusTriplet",
    "HeightMetrics",
    "MissingH5DependencyError",
    "calibration_gate",
    "evaluate_height",
    "fit_affine",
]
