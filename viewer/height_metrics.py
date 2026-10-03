"""Scores a predicted height map against a reference one (both metres).

Pure numpy, so it runs in the torch-free .venv. NaN in either input marks a
pixel as unscored. Degenerate input gives nan for the affected metric rather
than raising or silently reporting 0.0.
"""

from __future__ import annotations

import numpy as np

BUILDING_CLASS = 3  # GAMUS class index, see viewer/classify.py CLASS_NAMES
HEIGHT_BINS = ((0.0, 2.0), (2.0, 5.0), (5.0, 10.0), (10.0, 20.0),
               (20.0, 50.0), (50.0, float("inf")))
HEIGHT_BIN_NAMES = ("0-2m", "2-5m", "5-10m", "10-20m", "20-50m", ">50m")


def _summary(pred: np.ndarray, ref: np.ndarray) -> dict:
    """Metrics for already-filtered, finite arrays."""
    if pred.size == 0:
        return {"n": 0, "rmse": float("nan"), "mae": float("nan"),
                "bias": float("nan")}
    e = pred - ref
    return {"n": int(pred.size), "rmse": float(np.sqrt(np.mean(e ** 2))),
            "mae": float(np.mean(np.abs(e))), "bias": float(np.mean(e))}


def height_scores(pred: np.ndarray, ref: np.ndarray, classes: np.ndarray | None = None,
                  building_class: int = BUILDING_CLASS) -> dict:
    pred = np.asarray(pred, np.float64)
    ref = np.asarray(ref, np.float64)
    valid = np.isfinite(pred) & np.isfinite(ref)
    n = int(valid.sum())
    out = {"n": n, "rmse": float("nan"), "mae": float("nan"),
           "bias": float("nan"), "pearson": float("nan"),
           "building_rmse": float("nan"), "building_mae": float("nan"),
           "building_bias": float("nan")}
    if n == 0:
        return out

    err = pred[valid] - ref[valid]
    out["rmse"] = float(np.sqrt(np.mean(err ** 2)))
    out["mae"] = float(np.mean(np.abs(err)))
    out["bias"] = float(np.mean(err))

    p, r = pred[valid], ref[valid]
    if p.std() > 0 and r.std() > 0:
        out["pearson"] = float(np.corrcoef(p, r)[0, 1])

    if classes is not None:
        bmask = valid & (np.asarray(classes) == building_class)
        if bmask.any():
            be = pred[bmask] - ref[bmask]
            out["building_rmse"] = float(np.sqrt(np.mean(be ** 2)))
            out["building_mae"] = float(np.mean(np.abs(be)))
            out["building_bias"] = float(np.mean(be))
    return out


def height_bin_scores(pred: np.ndarray, ref: np.ndarray,
                      classes: np.ndarray | None = None,
                      building_class: int = BUILDING_CLASS) -> list[dict]:
    """Return pooled metrics by true/reference height bin.

    Bins are half-open except for the final open-ended bin. Invalid pixels are
    excluded before binning, so nodata cannot become a low-height observation.
    """
    pred = np.asarray(pred, np.float64)
    ref = np.asarray(ref, np.float64)
    valid = np.isfinite(pred) & np.isfinite(ref)
    rows = []
    for name, (lo, hi) in zip(HEIGHT_BIN_NAMES, HEIGHT_BINS):
        mask = valid & (ref >= lo) & (ref < hi)
        row = {"bin": name, **_summary(pred[mask], ref[mask])}
        if classes is not None:
            bmask = mask & (np.asarray(classes) == building_class)
            row["building"] = _summary(pred[bmask], ref[bmask])
        rows.append(row)
    return rows


CLASS_NAMES = {1: "ground", 2: "low vegetation", 3: "buildings", 4: "water", 5: "roads", 6: "trees"}


def class_scores(pred: np.ndarray, ref: np.ndarray, classes: np.ndarray) -> list[dict]:
    """RMSE / MAE / bias (mean of pred - ref) and pixel share per land-cover class, so a
    validation can say where the error is (e.g. tall buildings, not roads)."""
    pred = np.asarray(pred, np.float64)
    ref = np.asarray(ref, np.float64)
    valid = np.isfinite(pred) & np.isfinite(ref)
    total = max(int(valid.sum()), 1)
    rows = []
    for c, name in CLASS_NAMES.items():
        m = valid & (np.asarray(classes) == c)
        if m.sum() == 0:
            continue
        e = pred[m] - ref[m]
        rows.append({"class": name, "share": float(m.sum() / total), "rmse": float(np.sqrt(np.mean(e ** 2))),
                     "mae": float(np.mean(np.abs(e))), "bias": float(np.mean(e))})
    return rows


class ScoreAccumulator:
    """Pixel-pooled scores over many tiles, from running sums. Averaging
    per-tile RMSEs instead would over-weight near-empty tiles."""

    def __init__(self, building_class: int = BUILDING_CLASS, pearson: bool = True) -> None:
        self.building_class = building_class
        self.pearson = pearson
        self.n = self.b_n = 0
        self.se = self.ae = self.bias_sum = self.b_se = self.b_ae = self.b_bias_sum = 0.0
        self.sp = self.sr = self.spp = self.srr = self.spr = 0.0

    def add(self, pred: np.ndarray, ref: np.ndarray, classes: np.ndarray | None = None) -> None:
        pred = np.asarray(pred, np.float64)
        ref = np.asarray(ref, np.float64)
        valid = np.isfinite(pred) & np.isfinite(ref)
        self._add_valid(pred[valid], ref[valid],
                        None if classes is None else np.asarray(classes)[valid])

    def _add_valid(self, p: np.ndarray, r: np.ndarray,
                   classes: np.ndarray | None = None) -> None:
        """Add arrays already filtered to finite values.

        The large evaluation harness uses this to avoid converting and
        rechecking the same 1024x1024 tile once for every height bin.
        """
        p = np.asarray(p, np.float64)
        r = np.asarray(r, np.float64)
        e = p - r
        self.n += p.size
        self.se += float(e @ e)
        self.ae += float(np.abs(e).sum())
        self.bias_sum += float(e.sum())
        if self.pearson:
            self.sp += float(p.sum())
            self.sr += float(r.sum())
            self.spp += float(p @ p)
            self.srr += float(r @ r)
            self.spr += float(p @ r)
        if classes is not None:
            b = np.asarray(classes) == self.building_class
            self.b_n += int(b.sum())
            self.b_se += float(e[b] @ e[b])
            self.b_ae += float(np.abs(e[b]).sum())
            self.b_bias_sum += float(e[b].sum())

    def result(self) -> dict:
        nan = float("nan")
        if self.n == 0:
            return {"n": 0, "rmse": nan, "mae": nan, "bias": nan,
                    "pearson": nan, "building_rmse": nan, "building_mae": nan,
                    "building_bias": nan}
        n = self.n
        if self.pearson:
            cov = self.spr / n - (self.sp / n) * (self.sr / n)
            vp = self.spp / n - (self.sp / n) ** 2
            vr = self.srr / n - (self.sr / n) ** 2
            pearson = float(cov / np.sqrt(vp * vr)) if vp > 0 and vr > 0 else nan
        else:
            pearson = nan
        return {
            "n": n,
            "rmse": float(np.sqrt(self.se / n)),
            "mae": self.ae / n,
            "bias": self.bias_sum / n,
            "pearson": pearson,
            "building_rmse": float(np.sqrt(self.b_se / self.b_n)) if self.b_n else nan,
            "building_mae": self.b_ae / self.b_n if self.b_n else nan,
            "building_bias": self.b_bias_sum / self.b_n if self.b_n else nan,
        }
