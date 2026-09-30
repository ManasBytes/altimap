"""Scores a predicted height map against a reference one (both metres).

Pure numpy, so it runs in the torch-free .venv. NaN in either input marks a
pixel as unscored. Degenerate input gives nan for the affected metric rather
than raising or silently reporting 0.0.
"""

from __future__ import annotations

import numpy as np

BUILDING_CLASS = 3  # GAMUS class index, see viewer/classify.py CLASS_NAMES


def height_scores(pred: np.ndarray, ref: np.ndarray, classes: np.ndarray | None = None,
                  building_class: int = BUILDING_CLASS) -> dict:
    pred = np.asarray(pred, np.float64)
    ref = np.asarray(ref, np.float64)
    valid = np.isfinite(pred) & np.isfinite(ref)
    n = int(valid.sum())
    out = {"n": n, "rmse": float("nan"), "mae": float("nan"),
           "pearson": float("nan"), "building_rmse": float("nan")}
    if n == 0:
        return out

    err = pred[valid] - ref[valid]
    out["rmse"] = float(np.sqrt(np.mean(err ** 2)))
    out["mae"] = float(np.mean(np.abs(err)))

    p, r = pred[valid], ref[valid]
    if p.std() > 0 and r.std() > 0:
        out["pearson"] = float(np.corrcoef(p, r)[0, 1])

    if classes is not None:
        bmask = valid & (np.asarray(classes) == building_class)
        if bmask.any():
            out["building_rmse"] = float(np.sqrt(np.mean((pred[bmask] - ref[bmask]) ** 2)))
    return out


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

    def __init__(self, building_class: int = BUILDING_CLASS) -> None:
        self.building_class = building_class
        self.n = self.b_n = 0
        self.se = self.ae = self.b_se = 0.0
        self.sp = self.sr = self.spp = self.srr = self.spr = 0.0

    def add(self, pred: np.ndarray, ref: np.ndarray, classes: np.ndarray | None = None) -> None:
        pred = np.asarray(pred, np.float64)
        ref = np.asarray(ref, np.float64)
        valid = np.isfinite(pred) & np.isfinite(ref)
        p, r = pred[valid], ref[valid]
        e = p - r
        self.n += p.size
        self.se += float(e @ e)
        self.ae += float(np.abs(e).sum())
        self.sp += float(p.sum())
        self.sr += float(r.sum())
        self.spp += float(p @ p)
        self.srr += float(r @ r)
        self.spr += float(p @ r)
        if classes is not None:
            b = (np.asarray(classes) == self.building_class)[valid]
            self.b_n += int(b.sum())
            self.b_se += float(e[b] @ e[b])

    def result(self) -> dict:
        nan = float("nan")
        if self.n == 0:
            return {"n": 0, "rmse": nan, "mae": nan, "pearson": nan, "building_rmse": nan}
        n = self.n
        cov = self.spr / n - (self.sp / n) * (self.sr / n)
        vp = self.spp / n - (self.sp / n) ** 2
        vr = self.srr / n - (self.sr / n) ** 2
        return {
            "n": n,
            "rmse": float(np.sqrt(self.se / n)),
            "mae": self.ae / n,
            "pearson": float(cov / np.sqrt(vp * vr)) if vp > 0 and vr > 0 else nan,
            "building_rmse": float(np.sqrt(self.b_se / self.b_n)) if self.b_n else nan,
        }
