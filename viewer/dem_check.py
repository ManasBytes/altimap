"""Check exported DSMs against the public DEMs the organisers score GeoTIFFs against.

    .venv-da3/bin/python -m viewer.dem_check demo/india/*.tif           # Sikkim test scenes
    .venv-da3/bin/python -m viewer.dem_check scene.tif --ckpt other.pth

The FAQ scores GeoTIFF output "as an absolute DSM against SRTM/Copernicus" ("values must match
DEM heights"). For each scene this runs the app's pipeline and compares the DSM with Copernicus
GLO-30 and with NASADEM (NASA's reprocessed SRTM), both per pixel and as 30 m cell means (the
DEMs' own resolution), and reports the model's object heights by land-cover class as a sanity
check where no LiDAR exists (e.g. India). Heights: GLO-30 is EGM2008, NASADEM EGM96; the two
geoids differ by up to ~2 m, which is left in the numbers.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np

import viewer.dem  # noqa: F401  (GDAL / requests timeouts before any remote read)

CLASSES = {3: "buildings", 6: "trees", 2: "low vegetation", 1: "ground", 5: "roads"}


def nasadem(bounds, crs, shape) -> np.ndarray | None:
    """NASADEM (SRTM reprocessed, 1 arcsec, EGM96) on the grid, via Planetary Computer."""
    import planetary_computer
    import pystac_client
    from rasterio.warp import transform_bounds

    from viewer.geo import warp_to_grid

    cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                                    modifier=planetary_computer.sign_inplace, timeout=60)
    items = list(cat.search(collections=["nasadem"], bbox=transform_bounds(crs, "EPSG:4326", *bounds),
                            max_items=10).items())
    if not items:
        return None
    return warp_to_grid([i.assets["elevation"].href for i in items], bounds, crs, shape)


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
    import rasterio

    from viewer.dem import glo30
    from viewer.estimate import estimate, load_pipeline

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+", type=Path)
    ap.add_argument("--ckpt", type=Path, default=Path("viewer/cache/best.pth"))
    ap.add_argument("--tta", action="store_true")
    args = ap.parse_args()
    models = load_pipeline(args.ckpt)
    with tempfile.TemporaryDirectory() as tmp:
        for path in args.images:
            out = estimate(path, out_dir=Path(tmp) / path.stem, tta=args.tta, **models)
            rec, dsm = out["record"], out["dsm"]
            if dsm is None:
                print(f"{path.name}: no DSM ({rec.get('dsm_error')})")
                continue
            with rasterio.open(path) as src:
                bounds, crs, shape = tuple(src.bounds), src.crs, src.shape
            cell_px = max(1, round(30.0 / rec["gsd_m"]))
            print(f"\n{path.name}: {rec['gsd_m']:.2f} m px, DSM {rec['dsm']['min_m']:.0f}-{rec['dsm']['max_m']:.0f} m, "
                  f"model saw {rec['work_gsd_m']:.2f} m")
            for name, ref in (("Copernicus GLO-30", glo30(bounds, crs, shape)),
                              ("NASADEM (SRTM)", nasadem(bounds, crs, shape))):
                if ref is None:
                    print(f"  vs {name:18s} no coverage")
                    continue
                c = compare(dsm, ref, cell_px)
                print(f"  vs {name:18s} 30 m cells: RMSE {c['cell_rmse']:6.2f} bias {c['cell_bias']:+6.2f} | "
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
