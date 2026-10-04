"""Check exported DSMs against the public DEMs the organisers score GeoTIFFs against.

    .venv-da3/bin/python -m viewer.dem_check demo/india/*.tif           # Sikkim test scenes
    .venv-da3/bin/python -m viewer.dem_check scene.tif --ckpt other.pth

The FAQ scores GeoTIFF output "as an absolute DSM against SRTM/Copernicus" ("values must match
DEM heights"). For each scene this runs the app's pipeline and compares the DSM with Copernicus
GLO-30 and with SRTM GL1, both per pixel and as 30 m cell means (the DEMs' own resolution;
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
    """Means over px x px blocks (the DEM's 30 m cells), ignoring NaN."""
    h, w = (a.shape[0] // px) * px, (a.shape[1] // px) * px
    b = a[:h, :w].reshape(h // px, px, w // px, px)
    return np.nanmean(b, axis=(1, 3))


def compare(dsm: np.ndarray, ref: np.ndarray, cell_px: int) -> dict:
    ok = np.isfinite(dsm) & np.isfinite(ref)
    e = (dsm - ref)[ok]
    cd, cr = cell_means(dsm, cell_px), cell_means(ref, cell_px)
    ce = (cd - cr)[np.isfinite(cd) & np.isfinite(cr)]
    return {"pixel_rmse": float(np.sqrt(np.mean(e ** 2))), "pixel_bias": float(np.mean(e)),
            "cell_rmse": float(np.sqrt(np.mean(ce ** 2))), "cell_bias": float(np.mean(ce)),
            "r": float(np.corrcoef(dsm[ok], ref[ok])[0, 1])}


def main() -> None:
    from viewer.dem import BASE_DEMS
    from viewer.estimate import estimate, load_pipeline

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+", type=Path)
    ap.add_argument("--ckpt", type=Path, default=Path("viewer/cache/best.pth"))
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--base-dem", choices=tuple(BASE_DEMS), default="srtm")
    args = ap.parse_args()
    models = load_pipeline(args.ckpt)
    with tempfile.TemporaryDirectory() as tmp:
        for path in args.images:
            out = estimate(path, out_dir=Path(tmp) / path.stem, tta=args.tta, base_dem=args.base_dem, **models)
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
                print(f"  vs {label:18s} 30 m cells: RMSE {c['cell_rmse']:6.2f} bias {c['cell_bias']:+6.2f} | "
                      f"per pixel: RMSE {c['pixel_rmse']:6.2f} bias {c['pixel_bias']:+6.2f} r {c['r']:.4f}")
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
