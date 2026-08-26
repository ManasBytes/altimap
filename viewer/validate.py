"""Validate model-estimated building heights against Overture reference heights.

This is the brief's 50% criterion -- "RMSE, MAE, and correlation against LiDAR or
reference data" -- which was blocked until now: 3DEP lidar has NO coverage over
Atlanta (verified against a Provo control that returns 13 items), so there was
nothing to score against. Overture supplies a height for 78% of buildings here,
which is a weaker reference than lidar but a real one.

Two numbers are reported, and the distinction matters:

  * Pearson correlation is SCALE-FREE. Relative depth has no metric unit, so
    correlation is the only thing measurable without first choosing a scale. It
    is the honest core result.

  * RMSE/MAE are computed after fitting the best possible per-scene scale and
    offset by least squares against the reference itself. That is an ORACLE
    calibration -- it uses the answers to set the scale -- so the errors are a
    LOWER BOUND on what any real calibration could achieve. Reported as such.

Reference caveats, stated because the numbers look authoritative:
  * Overture heights are themselves largely ML-derived, not surveyed.
  * These tiles are 19-31 deg off-nadir, so a roof is laterally displaced from
    its map-coordinate footprint by height*tan(angle); footprint sampling mixes
    roof, facade and ground.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from viewer.footprints import DEFAULT_CACHE, BuildingIndex
from viewer.geo import decode_rg16
from viewer.refine import building_heights, fit_ground
from viewer.terrain import height_field

DEFAULT_DATA = Path("viewer/web/data-offnadir")


def scene_pairs(index: BuildingIndex, scene_dir: Path, meta: dict, percentile: float):
    """(model_relative, reference_metres) per building for one scene."""
    arr = np.asarray(Image.open(scene_dir / "depth.png").convert("RGB"))
    depth = decode_rg16(arr, meta["depth_lo"], meta["depth_hi"])
    h01 = height_field(depth)
    shape = (meta["height"], meta["width"])
    mask, known, stats = index.rasterize(meta["geo"]["bounds"], meta["geo"]["crs"], shape)
    if stats["n_with_height"] == 0:
        return np.array([]), np.array([])
    ndsm = h01 - fit_ground(h01, mask)
    rel = building_heights(ndsm, mask, percentile=percentile)
    ids = sorted(set(rel) & set(known))
    return (np.array([rel[i] for i in ids], dtype=float),
            np.array([known[i] for i in ids], dtype=float))


def oracle_errors(x: np.ndarray, y: np.ndarray):
    """RMSE/MAE after the best-possible least-squares scale+offset (an oracle)."""
    if x.size < 2 or x.std() == 0:
        return float("nan"), float("nan")
    design = np.column_stack([x, np.ones(x.size)])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    pred = design @ coef
    return float(np.sqrt(((pred - y) ** 2).mean())), float(np.abs(pred - y).mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--buildings", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--percentile", type=float, default=75.0,
                        help="roof statistic inside each footprint")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    index = BuildingIndex(args.buildings)
    metas = []
    for p in sorted((args.data / "scenes").glob("*/meta.json")):
        m = json.loads(p.read_text())
        if m.get("geo", {}).get("georeferenced"):
            metas.append((p.parent, m))
    if args.limit:
        metas = metas[: args.limit]
    print(f"validating {len(metas)} georeferenced scenes against Overture heights")

    all_x, all_y = [], []
    per_scene, by_class, by_angle = [], defaultdict(list), defaultdict(list)

    for i, (scene_dir, meta) in enumerate(metas, 1):
        x, y = scene_pairs(index, scene_dir, meta, args.percentile)
        if x.size < 3:
            continue
        corr = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else float("nan")
        rmse, mae = oracle_errors(x, y)
        per_scene.append({"id": meta["id"], "n": int(x.size), "corr": corr,
                          "rmse_m": rmse, "mae_m": mae})
        by_class[meta["class"]].append(corr)
        by_angle[meta["oa"]].append(corr)
        all_x.append(x)
        all_y.append(y)
        if i % 100 == 0:
            print(f"  [{i}/{len(metas)}]")

    if not per_scene:
        raise SystemExit("no scenes had enough reference heights")

    corrs = np.array([r["corr"] for r in per_scene], dtype=float)
    rmses = np.array([r["rmse_m"] for r in per_scene], dtype=float)
    maes = np.array([r["mae_m"] for r in per_scene], dtype=float)
    fin = np.isfinite(corrs)

    print(f"\nscenes scored: {len(per_scene)} | buildings: {sum(r['n'] for r in per_scene)}")
    print("=" * 62)
    print("PER-BUILDING HEIGHT, model vs Overture reference")
    print(f"  correlation (scale-free)  median {np.median(corrs[fin]):+.3f}"
          f"   p25 {np.percentile(corrs[fin],25):+.3f}"
          f"   p75 {np.percentile(corrs[fin],75):+.3f}")
    print(f"  scenes with corr > 0.5:   {int((corrs[fin] > 0.5).sum())}/{int(fin.sum())}")
    print(f"  scenes with corr < 0:     {int((corrs[fin] < 0).sum())}/{int(fin.sum())}")
    print(f"\n  RMSE (ORACLE scale)       median {np.nanmedian(rmses):.2f} m"
          f"   -- LOWER BOUND, uses the answers to set scale")
    print(f"  MAE  (ORACLE scale)       median {np.nanmedian(maes):.2f} m")

    # Pooled, to avoid small-n per-scene correlations dominating.
    X, Y = np.concatenate(all_x), np.concatenate(all_y)
    pooled = float(np.corrcoef(X, Y)[0, 1])
    prmse, pmae = oracle_errors(X, Y)
    print(f"\n  pooled over all buildings: corr {pooled:+.3f}, "
          f"oracle RMSE {prmse:.2f} m, MAE {pmae:.2f} m")
    print(f"  reference height range: {Y.min():.1f}-{Y.max():.1f} m, median {np.median(Y):.1f} m")

    print(f"\n{'class':<20} {'n':>5} {'median corr':>12}")
    print("-" * 40)
    for cls in sorted(by_class):
        v = np.array(by_class[cls], dtype=float); v = v[np.isfinite(v)]
        if v.size:
            print(f"{cls:<20} {v.size:>5} {np.median(v):>+12.3f}")

    print(f"\n{'off-nadir':<12} {'n':>5} {'median corr':>12}")
    print("-" * 32)
    for oa in sorted(by_angle):
        v = np.array(by_angle[oa], dtype=float); v = v[np.isfinite(v)]
        if v.size:
            print(f"OA{oa:<10} {v.size:>5} {np.median(v):>+12.3f}")

    out = {
        "reference": "overture-building-heights",
        "reference_is_ml_derived": True,
        "roof_percentile": args.percentile,
        "scenes_scored": len(per_scene),
        "buildings_scored": int(sum(r["n"] for r in per_scene)),
        "median_corr": float(np.median(corrs[fin])),
        "pooled_corr": pooled,
        "median_rmse_m_oracle": float(np.nanmedian(rmses)),
        "median_mae_m_oracle": float(np.nanmedian(maes)),
        "pooled_rmse_m_oracle": prmse,
        "pooled_mae_m_oracle": pmae,
        "rmse_is_oracle_lower_bound": True,
        "per_scene": per_scene,
    }
    (args.data / "validation_summary.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.data / 'validation_summary.json'}")


if __name__ == "__main__":
    main()
