"""Check exported DSMs against the public DEMs the organisers score GeoTIFFs against.

    .venv-da3/bin/python -m viewer.dem_check demo/india/*.tif           # Sikkim test scenes
    .venv-da3/bin/python -m viewer.dem_check scene.tif --ckpt other.pth

The FAQ scores GeoTIFF output "as an absolute DSM against SRTM/Copernicus" ("values must match
DEM heights"). For each scene this runs the app's pipeline and compares the DSM with Copernicus
GLO-30 and with SRTM GL1, both per pixel and as approximate 30 m block means (not native DEM cells;
the same `estimate.dem_agreement` numbers the app shows), and reports the model's object
heights by land-cover class as a sanity check where no LiDAR exists (e.g. India). Heights:
GLO-30 is EGM2008, SRTM EGM96; the two geoids differ by up to ~2 m, which is left in the
numbers.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np

import viewer.dem  # noqa: F401  (GDAL / requests timeouts before any remote read)

CLASSES = {3: "buildings", 6: "trees", 2: "low vegetation", 1: "ground", 5: "roads"}


def cell_means(a: np.ndarray, px: int) -> np.ndarray:
    """Means over image-aligned px x px blocks, excluding missing pixels."""
    h, w = (a.shape[0] // px) * px, (a.shape[1] // px) * px
    b = a[:h, :w].reshape(h // px, px, w // px, px)
    valid = np.isfinite(b)
    count = valid.sum(axis=(1, 3))
    return np.divide(np.where(valid, b, 0).sum(axis=(1, 3)), count,
                     out=np.full(count.shape, np.nan), where=count > 0)


def compare(dsm: np.ndarray, ref: np.ndarray, cell_px: int) -> dict:
    """JSON-safe agreement metrics; undefined values are None, never zero or NaN."""
    dsm, ref = np.asarray(dsm, np.float64), np.asarray(ref, np.float64)
    ok = np.isfinite(dsm) & np.isfinite(ref)
    e = (dsm - ref)[ok]
    # Both block means must use the same pixels, otherwise nodata changes the comparison.
    cd, cr = cell_means(np.where(ok, dsm, np.nan), cell_px), cell_means(np.where(ok, ref, np.nan), cell_px)
    ce = (cd - cr)[np.isfinite(cd) & np.isfinite(cr)]
    r = None
    if e.size > 1 and np.ptp(dsm[ok]) > 0 and np.ptp(ref[ok]) > 0:
        r = float(np.corrcoef(dsm[ok], ref[ok])[0, 1])
    result = {"pixel_rmse": float(np.sqrt(np.mean(e ** 2))) if e.size else None,
              "pixel_bias": float(np.mean(e)) if e.size else None,
              "cell_rmse": float(np.sqrt(np.mean(ce ** 2))) if ce.size else None,
              "cell_bias": float(np.mean(ce)) if ce.size else None, "r": r}
    return {k: v if v is not None and np.isfinite(v) else None for k, v in result.items()}


def main() -> None:
    from viewer.dem import BASE_DEMS
    from viewer.estimate import estimate, load_pipeline

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+", type=Path)
    ap.add_argument("--ckpt", type=Path, default=Path("viewer/cache/best.pth"))
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--base-dem", choices=tuple(BASE_DEMS), default="glo30")
    args = ap.parse_args()
    models = load_pipeline(args.ckpt)
    with tempfile.TemporaryDirectory() as tmp:
        for path in args.images:
            out = estimate(path, out_dir=Path(tmp) / path.stem, tta=args.tta,
                           base_dem=args.base_dem, compare_dems=True, **models)
            rec = out["record"]
            if out["dsm"] is None:
                print(f"{path.name}: no DSM ({rec.get('dsm_error')})")
                continue
            print(f"\n{path.name}: {rec['gsd_m']:.2f} m px, DSM on {BASE_DEMS[args.base_dem][1]} "
                  f"{rec['dsm']['min_m']:.0f}-{rec['dsm']['max_m']:.0f} m, model saw {rec['work_gsd_m']:.2f} m")
            for name, c in rec["dem_agreement"].items():
                label = BASE_DEMS[name][1]
                if c is None:
                    print(f"  vs {label:18s} no coverage")
                    continue
                text = {k: "unavailable" if v is None else format(v, ".4f" if k == "r" else
                        "+.2f" if "bias" in k else ".2f") for k, v in c.items()}
                print(f"  vs {label:18s} approximate 30 m blocks: RMSE {text['cell_rmse']} "
                      f"bias {text['cell_bias']} | per pixel: RMSE {text['pixel_rmse']} "
                      f"bias {text['pixel_bias']} r {text['r']}")
            ndsm, cls = out["ndsm"], out["classes"]
            parts = []
            for k, label in CLASSES.items():
                m = (cls == k) & np.isfinite(ndsm)
                if m.mean() > 0.005:
                    parts.append(f"{label} {m.mean():.0%}: median {np.median(ndsm[m]):.1f} m, "
                                 f"p95 {np.percentile(ndsm[m], 95):.1f} m")
            print("  model heights above ground: " + " | ".join(parts))


if __name__ == "__main__":
    main()
