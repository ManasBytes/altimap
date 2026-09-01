"""Portable GAMUS HDF5 triplet adapter.

GAMUS distributes one RGB, AGL, and CLS HDF5 file per scene.  Files normally
contain a dataset named ``image``; the loader also accepts a single unnamed
dataset, making it portable across the published variants.  ``load_rgb`` is
the prediction boundary.  ``load_targets`` is a separate, explicit
evaluation boundary and is never called by model code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


class MissingH5DependencyError(ImportError):
    """Raised when HDF5 support is requested without the optional h5py extra."""


def _h5py() -> Any:
    try:
        import h5py  # type: ignore
    except ImportError as exc:
        raise MissingH5DependencyError(
            "GAMUS HDF5 support requires h5py. Install it with "
            "`pip install h5py` (or add h5py to the offline environment)."
        ) from exc
    return h5py


def _read_h5(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    with _h5py().File(path, "r") as handle:
        if "image" in handle:
            dataset = handle["image"]
        else:
            datasets = []
            handle.visititems(lambda _, obj: datasets.append(obj) if hasattr(obj, "shape") else None)
            if len(datasets) != 1:
                raise ValueError(f"{path} must contain dataset 'image' or one dataset; found {len(datasets)}")
            dataset = datasets[0]
        return np.asarray(dataset)


@dataclass(frozen=True)
class GamusTriplet:
    """Paths for one scene; filenames are validated against the scene ID."""

    scene_id: str
    rgb: Path
    agl: Path
    cls: Path

    @classmethod
    def from_directory(cls, directory: str | Path, scene_id: str) -> "GamusTriplet":
        root = Path(directory)
        paths = {kind: root / f"{scene_id}_{kind}.h5" for kind in ("RGB", "AGL", "CLS")}
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing GAMUS triplet file(s): " + ", ".join(missing))
        return cls(scene_id, paths["RGB"], paths["AGL"], paths["CLS"])


@dataclass(frozen=True)
class GamusScene:
    """Validated scene arrays with an intentionally split prediction API."""

    triplet: GamusTriplet

    def load_rgb(self) -> np.ndarray:
        """Return only RGB for inference, never AGL or CLS."""
        rgb = _read_h5(self.triplet.rgb)
        if rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError(f"RGB must have shape (H, W, 3), got {rgb.shape}")
        if not np.issubdtype(rgb.dtype, np.number):
            raise ValueError(f"RGB must be numeric, got {rgb.dtype}")
        return rgb.astype(np.uint8, copy=False)

    def load_targets(self) -> tuple[np.ndarray, np.ndarray]:
        """Return AGL and CLS for evaluation after prediction is complete."""
        rgb = self.load_rgb()
        agl = _read_h5(self.triplet.agl)
        classes = _read_h5(self.triplet.cls)
        if agl.shape != rgb.shape[:2] or classes.shape != rgb.shape[:2]:
            raise ValueError(
                f"RGB/AGL/CLS grids must match: {rgb.shape}, {agl.shape}, {classes.shape}"
            )
        if agl.ndim != 2 or classes.ndim != 2:
            raise ValueError("AGL and CLS must be two-dimensional")
        if not np.issubdtype(agl.dtype, np.number) or not np.issubdtype(classes.dtype, np.number):
            raise ValueError("AGL and CLS must be numeric")
        if not np.all(np.isfinite(agl[np.isfinite(agl)])):
            raise ValueError("AGL contains invalid values")
        if np.any((classes < 0) | (classes > 6)):
            raise ValueError("CLS contains values outside the GAMUS class range 0..6")
        return agl.astype(np.float32, copy=False), classes.astype(np.int16, copy=False)

    def validate(self) -> tuple[int, int]:
        """Validate all three files and return ``(height, width)``."""
        rgb = self.load_rgb()
        agl, classes = self.load_targets()
        return rgb.shape[:2] if agl.shape == classes.shape else (0, 0)
