"""Apply footprint-constrained refinement to the exported georeferenced scenes.

Reads depth back out of the exported rg16 PNGs rather than re-running DA3 --
the model is deterministic and inference was the expensive part, so there is
nothing to gain from repeating it.

Writes a second height field per scene (`refined.png`) plus refinement metadata,
and reports two things that decide whether this was worth doing:

  * edge sharpness on footprint boundaries, before vs after
  * metric calibration success rate from BUILDING heights, against the 5.6%
    the bare-earth DEM managed over the same tiles
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from viewer.footprints import DEFAULT_CACHE, BuildingIndex
from viewer.geo import decode_rg16, encode_rg16
from viewer.refine import edge_sharpness, refine
from viewer.terrain import height_field

DEFAULT_DATA = Path("viewer/web/data-offnadir")


def load_depth(scene_dir: Path, meta: dict) -> np.ndarray:
    if meta.get("encoding") == "rg16-png":
        arr = np.asarray(Image.open(scene_dir / "depth.png").convert("RGB"))
        return decode_rg16(arr, meta["depth_lo"], meta["depth_hi"])
    return np.fromfile(scene_dir / "depth.bin",
                       dtype="<f4").reshape(meta["height"], meta["width"]).astype(np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--buildings", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--order", type=int, default=1, choices=[1, 2],
                        help="ground surface order: 1 plane, 2 quadratic")
    args = parser.parse_args()

    index = BuildingIndex(args.buildings)
    print(f"loaded {len(index)} Overture footprints "
          f"({index.meta.get('with_height', '?')} with height, "
          f"release {index.meta.get('release', '?')})")

    scenes_dir = args.data / "scenes"
    metas = sorted(scenes_dir.glob("*/meta.json"))
    todo = []
    for p in metas:
        m = json.loads(p.read_text())
        if m.get("geo", {}).get("georeferenced"):
            todo.append((p.parent, m))
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(todo)} georeferenced scenes to refine")

    rows = []
    for i, (scene_dir, meta) in enumerate(todo, 1):
        depth = load_depth(scene_dir, meta)
        h01 = height_field(depth)
        shape = (meta["height"], meta["width"])

        mask, known, stats = index.rasterize(meta["geo"]["bounds"], meta["geo"]["crs"], shape)
        if stats["n_footprints"] == 0:
            continue

        result = refine(h01, mask, known_m=known, order=args.order)

        # Rebase the flattened nDSM into a [0,1] height field so it can ride the
        # same texture path and exaggeration slider as the unrefined surface.
        flat = result["flat"]
        lo, hi = float(flat.min()), float(flat.max())
        rebased = np.zeros_like(flat) if hi == lo else (flat - lo) / (hi - lo)
        # encode_rg16 stores "depth"; the viewer inverts via depth_max - d, so
        # invert here to keep tall buildings tall after that inversion.
        encoded, elo, ehi = encode_rg16((1.0 - rebased).astype(np.float32))
        Image.fromarray(encoded).save(scene_dir / "refined.png", compress_level=6)

        before = edge_sharpness(h01, mask)
        after = edge_sharpness(rebased, mask)

        meta["refined"] = {
            "depth_lo": elo, "depth_hi": ehi,
            "depth_min": elo, "depth_max": ehi,
            "n_footprints": stats["n_footprints"],
            "n_with_height": stats["n_with_height"],
            "coverage": round(stats["coverage"], 4),
            "edge_sharpness_before": _c(before),
            "edge_sharpness_after": _c(after),
            "ground_order": args.order,
            "metric": _clean_metric(result["metric"]),
        }
        (scene_dir / "meta.json").write_text(json.dumps(meta))

        rows.append({
            # "oa" (off-nadir angle) only exists in the Atlanta export; the
            # Inria set is single-view nadir and carries "num" (tile number)
            # instead. Report whichever is present rather than assuming one.
            "id": meta["id"], "class": meta["class"],
            "oa": meta.get("oa", meta.get("num")),
            "n": stats["n_footprints"], "nh": stats["n_with_height"],
            "cov": stats["coverage"], "before": before, "after": after,
            "metric": result["metric"],
            "dem_usable": bool((meta.get("absolute") or {}).get("usable")),
        })
        if i % 50 == 0 or i == len(todo):
            print(f"  [{i}/{len(todo)}]")

    _report(rows, args.data)


def _c(v):
    if v is None:
        return None
    f = float(v)
    return None if not np.isfinite(f) else round(f, 5)


def _clean_metric(m):
    if not m:
        return None
    return {k: (_c(v) if isinstance(v, float) else v) for k, v in m.items()}


def _report(rows: list[dict], out: Path) -> None:
    if not rows:
        print("\nno scenes refined")
        return

    before = np.array([r["before"] for r in rows], dtype=float)
    after = np.array([r["after"] for r in rows], dtype=float)
    ok = np.isfinite(before) & np.isfinite(after)

    print(f"\n{'':<34}{'median':>10}{'p25':>9}{'p75':>9}")
    print("-" * 62)
    print(f"{'edge sharpness BEFORE refine':<34}{np.median(before[ok]):>10.4f}"
          f"{np.percentile(before[ok],25):>9.4f}{np.percentile(before[ok],75):>9.4f}")
    print(f"{'edge sharpness AFTER refine':<34}{np.median(after[ok]):>10.4f}"
          f"{np.percentile(after[ok],25):>9.4f}{np.percentile(after[ok],75):>9.4f}")
    improved = int((after[ok] > before[ok]).sum())
    print(f"\nsharper after refinement: {improved}/{int(ok.sum())} "
          f"({100*improved/max(int(ok.sum()),1):.1f}%)")

    fp = np.array([r["n"] for r in rows])
    nh = np.array([r["nh"] for r in rows])
    cov = np.array([r["cov"] for r in rows])
    print(f"\nfootprints/tile: median {np.median(fp):.0f} (max {fp.max()})"
          f" | with height: median {np.median(nh):.0f}"
          f" | built coverage: median {100*np.median(cov):.1f}%")

    mets = [r["metric"] for r in rows if r["metric"]]
    usable = [m for m in mets if m["usable"]]
    anchors = np.array([m["n_anchors"] for m in mets], dtype=float)
    r2 = np.array([m["fit_r2"] for m in mets], dtype=float)
    scale = np.array([m["scale_m"] for m in mets], dtype=float)
    fin = np.isfinite(r2)

    print("\nmetric calibration from BUILDING heights (Overture)")
    print("-" * 62)
    print(f"  scenes attempted:        {len(mets)}")
    print(f"  anchors/scene:           median {np.median(anchors):.0f}")
    if fin.any():
        q = np.percentile(r2[fin], [25, 50, 75])
        print(f"  fit_r2 quartiles:        {q[0]:.3f}  {q[1]:.3f}  {q[2]:.3f}")
    neg = int((np.isfinite(scale) & (scale <= 0)).sum())
    print(f"  inverted (scale <= 0):   {neg}/{len(mets)}")
    print(f"  USABLE:                  {len(usable)}/{len(mets)} "
          f"({100*len(usable)/max(len(mets),1):.1f}%)")
    dem_ok = sum(1 for r in rows if r["dem_usable"])
    print(f"  for comparison, bare-earth DEM on the same tiles: "
          f"{dem_ok}/{len(rows)} ({100*dem_ok/max(len(rows),1):.1f}%)")

    summary = {
        "n_scenes": len(rows),
        "edge_sharpness_median_before": float(np.median(before[ok])) if ok.any() else None,
        "edge_sharpness_median_after": float(np.median(after[ok])) if ok.any() else None,
        "sharper_fraction": improved / max(int(ok.sum()), 1),
        "building_calibration_usable": len(usable),
        "building_calibration_attempted": len(mets),
        "dem_calibration_usable": dem_ok,
    }
    (out / "refinement_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out / 'refinement_summary.json'}")


if __name__ == "__main__":
    main()
