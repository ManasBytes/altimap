"""Export the Off-nadir Scene10 (Atlanta) dataset as browsable 3D scenes.

Dataset shape: 10 classes x 20 scenes x 26 off-nadir angles (OA8-OA54) = 5200
GeoTIFFs. Two facts drive the design:

  * Only 620 of 5200 carry a CRS (EPSG:32616, ~0.52 m/px). The other 4580 are
    plain uint8 rasters in a .tif container. Both must export.
  * Georeferencing exists only at OA19/24/27 (+OA31 for 20 scenes). The
    NEAR-NADIR images have none. So the metric path and the nadir path cover
    disjoint subsets, and no scene gets both.

Depth ships as 16-bit packed into the R/G channels of a PNG (~360 KB) rather
than raw float32 (~1 MB): 5200 scenes of float32 would be ~5 GB of fetches.

Run under .venv-da3 (torch + rasterio). Resumable via --resume.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from viewer.geo import encode_rg16, fit_absolute_elevation, read_geo_meta
from viewer.metrics import luminance, scene_metrics
from viewer.terrain import build_terrain, height_field

DEFAULT_DATASET = Path("/home/biplab-dev/Downloads/Off-nadir-Scene10 dataset")
DEFAULT_MODEL = "depth-anything/DA3-SMALL"
NAME_RE = re.compile(r"^(?P<cls>.+)_(?P<scene>\d+)_OA(?P<oa>\d+)\.tif$", re.IGNORECASE)

# Below this, the relative->absolute fit explains too little of the DEM to be
# worth showing as metres. Not tuned -- chosen as an obviously-weak threshold,
# and the observed distribution is reported alongside it.
CALIBRATION_MIN_R2 = 0.3


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def scan_dataset(root: Path) -> list[dict]:
    """Every tile with its class, scene number and off-nadir angle."""
    out = []
    for path in sorted(root.glob("*/*.tif")):
        m = NAME_RE.match(path.name)
        if not m:
            continue
        cls = path.parent.name
        out.append({
            "path": path,
            "cls": cls,
            "scene": int(m.group("scene")),
            "oa": int(m.group("oa")),
            "id": f"{slug(cls)}__s{int(m.group('scene')):02d}__oa{int(m.group('oa')):02d}",
        })
    return out


def read_rgb(path: Path) -> np.ndarray:
    import rasterio

    with rasterio.open(path) as src:
        bands = min(3, src.count)
        arr = src.read(list(range(1, bands + 1)))
    arr = np.transpose(arr, (1, 2, 0))
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    return np.ascontiguousarray(arr[:, :, :3].astype(np.uint8))


def _load_model(model_name: str):
    import torch
    from depth_anything_3.api import DepthAnything3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"loading {model_name} onto {device}...")
    return DepthAnything3.from_pretrained(model_name).to(device=device), str(device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true", help="skip scenes already written")
    parser.add_argument("--no-dem", action="store_true", help="skip absolute-elevation calibration")
    parser.add_argument("--glb-for", type=int, default=0,
                        help="also write terrain.glb for the N most-structured scenes")
    parser.add_argument("--glb-res", type=int, default=256)
    parser.add_argument("--out", type=Path, default=Path("viewer/web/data-offnadir"))
    args = parser.parse_args()

    tiles = scan_dataset(args.dataset)
    if not tiles:
        raise SystemExit(f"no tiles matched under {args.dataset}")
    if args.limit:
        tiles = tiles[: args.limit]
    classes = sorted({t["cls"] for t in tiles})
    angles = sorted({t["oa"] for t in tiles})
    print(f"{len(tiles)} tiles | {len(classes)} classes | angles OA{angles[0]}-OA{angles[-1]} "
          f"({len(angles)} distinct)")

    scenes_dir = args.out / "scenes"
    scenes_dir.mkdir(parents=True, exist_ok=True)

    dem = None
    if not args.no_dem:
        from viewer.dem import DemSource
        dem = DemSource()
    dem_cache: dict[tuple, np.ndarray | None] = {}

    model, device = _load_model(args.model)

    records: list[dict] = []
    started = time.perf_counter()
    skipped = 0

    for i, tile in enumerate(tiles, 1):
        scene_dir = scenes_dir / tile["id"]
        meta_path = scene_dir / "meta.json"
        if args.resume and meta_path.exists():
            try:
                records.append(json.loads(meta_path.read_text()))
                skipped += 1
                continue
            except json.JSONDecodeError:
                pass  # corrupt partial write -- redo it

        rgb = read_rgb(tile["path"])
        prediction = model.inference([rgb], process_res=args.process_res)
        depth = np.asarray(prediction.depth[0], dtype=np.float32)
        if not np.isfinite(depth).all():
            raise ValueError(f"non-finite depth for {tile['path'].name}")
        conf = None
        if prediction.conf is not None:
            conf = np.asarray(prediction.conf[0], dtype=np.float32)

        h, w = depth.shape
        rgb_small = np.asarray(
            Image.fromarray(rgb).resize((w, h), Image.BILINEAR), dtype=np.uint8)
        lum = luminance(rgb_small.astype(np.float64) / 255.0)

        record = {
            "id": tile["id"],
            "class": tile["cls"],
            "scene": tile["scene"],
            "oa": tile["oa"],
            "source_image": tile["path"].name,
            "width": w,
            "height": h,
            # is_metric arrives as an empty addict Dict for non-metric models;
            # int() raises, truthiness reads correctly.
            "is_metric": bool(prediction.is_metric),
            "encoding": "rg16-png",
            **scene_metrics(depth, lum, conf),
        }

        geo = read_geo_meta(tile["path"])
        record["geo"] = geo
        record["absolute"] = None

        if geo["georeferenced"] and dem is not None:
            # The three angles of one scene share a footprint to within a metre,
            # so cache on rounded bounds -- 200 unique reads instead of 620.
            key = (tuple(round(b) for b in geo["bounds"]), geo["crs"], h, w)
            if key not in dem_cache:
                dem_cache[key] = dem.patch(geo["bounds"], geo["crs"], (h, w))
            patch = dem_cache[key]
            if patch is not None:
                hf = height_field(depth)
                scale, offset, r2 = fit_absolute_elevation(hf, patch)
                valid = patch[np.isfinite(patch)]
                # A negative scale means DA3's "higher" anti-correlates with real
                # terrain -- hills read as valleys. That is not a weak
                # calibration, it is an inverted one, and metres derived from it
                # would be worse than useless. Gate on sign AND fit quality so
                # the viewer never offers a metric readout it cannot defend.
                ok = bool(np.isfinite(r2) and r2 >= CALIBRATION_MIN_R2
                          and np.isfinite(scale) and scale > 0)
                record["absolute"] = {
                    "source": "3dep-seamless",
                    "dem_item": dem.item_id,
                    "scale_m": scale,
                    "offset_m": offset,
                    "fit_r2": r2,
                    "usable": ok,
                    "reject_reason": None if ok else (
                        "inverted (negative scale)" if np.isfinite(scale) and scale <= 0
                        else "weak fit" if np.isfinite(r2) else "unfittable"),
                    "dem_min_m": float(valid.min()) if valid.size else None,
                    "dem_max_m": float(valid.max()) if valid.size else None,
                    # Recorded so nothing downstream mistakes this for a
                    # building-height reference: 3DEP seamless is BARE EARTH at
                    # ~10 m posting against 0.52 m imagery.
                    "reference_is_bare_earth": True,
                    "reference_posting_m": 10.0,
                }

        scene_dir.mkdir(parents=True, exist_ok=True)
        encoded, lo, hi = encode_rg16(depth)
        record["depth_lo"] = lo
        record["depth_hi"] = hi
        Image.fromarray(encoded).save(scene_dir / "depth.png", compress_level=6)
        Image.fromarray(rgb_small).save(scene_dir / "rgb.jpg", quality=88)
        meta_path.write_text(json.dumps(record))
        records.append(record)

        if i % 100 == 0 or i == len(tiles):
            rate = (i - skipped) / max(time.perf_counter() - started, 1e-9)
            eta = (len(tiles) - i) / max(rate, 1e-9) / 60
            print(f"  [{i}/{len(tiles)}] {rate:.1f} img/s  eta {eta:.1f} min"
                  + (f"  (skipped {skipped})" if skipped else ""))

    if dem is not None:
        dem.close()

    # --- optional GLB meshes for the most-structured scenes -----------------
    if args.glb_for:
        ranked = sorted(
            (r for r in records if np.isfinite(r.get("structure_alignment", np.nan))),
            key=lambda r: -r["structure_alignment"])[: args.glb_for]
        print(f"\nwriting {len(ranked)} terrain.glb meshes")
        by_id = {t["id"]: t for t in tiles}
        for r in ranked:
            depth = _decode_depth(scenes_dir / r["id"])
            plane = (r["plane"]["a"], r["plane"]["b"], r["plane"]["c"])
            build_terrain(height_field(depth, plane),
                          scenes_dir / r["id"] / "rgb.jpg",
                          scenes_dir / r["id"] / "terrain.glb",
                          res=args.glb_res)
            r["has_glb"] = True
            (scenes_dir / r["id"] / "meta.json").write_text(json.dumps(r))
        _ = by_id

    _write_index(args, records, classes, angles, device)


def _decode_depth(scene_dir: Path) -> np.ndarray:
    from viewer.geo import decode_rg16

    meta = json.loads((scene_dir / "meta.json").read_text())
    arr = np.asarray(Image.open(scene_dir / "depth.png").convert("RGB"))
    return decode_rg16(arr, meta["depth_lo"], meta["depth_hi"])


def _write_index(args, records, classes, angles, device) -> None:
    (args.out / "metrics_full.json").write_text(json.dumps(records))

    groups = defaultdict(dict)
    for r in records:
        groups[(r["class"], r["scene"])][str(r["oa"])] = r["id"]

    def agg(rows, key):
        vals = np.array([r[key] for r in rows], dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return None
        return {"median": float(np.median(vals)),
                "q1": float(np.percentile(vals, 25)),
                "q3": float(np.percentile(vals, 75))}

    by_class = {c: {"n": sum(1 for r in records if r["class"] == c),
                    **{k: agg([r for r in records if r["class"] == c], k)
                       for k in ("plane_r2", "residual_relief",
                                 "structure_alignment", "conf_texture_corr")}}
                for c in classes}

    by_angle = {str(a): {"n": sum(1 for r in records if r["oa"] == a),
                         **{k: agg([r for r in records if r["oa"] == a], k)
                            for k in ("plane_r2", "structure_alignment")}}
                for a in angles}

    georef = sum(1 for r in records if r["geo"]["georeferenced"])
    calibrated = sum(1 for r in records if r.get("absolute"))
    usable = sum(1 for r in records if (r.get("absolute") or {}).get("usable"))

    index = {
        "dataset": "off-nadir-scene10",
        "model": args.model,
        "process_res": args.process_res,
        "device": device,
        "total": len(records),
        "georeferenced": georef,
        "calibrated": calibrated,
        "calibration_usable": usable,
        "calibration_min_r2": CALIBRATION_MIN_R2,
        "classes": classes,
        "angles": angles,
        "class_aggregates": by_class,
        "angle_aggregates": by_angle,
        "groups": [{"class": c, "scene": s, "by_angle": v}
                   for (c, s), v in sorted(groups.items())],
        # Compact per-scene rows: 5200 full records would be a ~6 MB fetch.
        "scenes": [{
            "id": r["id"], "c": r["class"], "s": r["scene"], "a": r["oa"],
            "r2": _r(r["plane_r2"]), "st": _r(r["structure_alignment"]),
            "rr": _r(r["residual_relief"]),
            "g": 1 if r["geo"]["georeferenced"] else 0,
            "gm": round(r["geo"]["ground_m"][0], 1) if r["geo"]["ground_m"] else None,
            "es": _r((r["absolute"] or {}).get("scale_m")),
            "er": _r((r["absolute"] or {}).get("fit_r2")),
            "ok": 1 if (r["absolute"] or {}).get("usable") else 0,
            "glb": 1 if r.get("has_glb") else 0,
        } for r in records],
    }
    (args.out / "index.json").write_text(json.dumps(index))

    print(f"\n{'class':<20} {'n':>5} {'plane_r2':>9} {'struct':>8}")
    print("-" * 45)
    for c in classes:
        e = by_class[c]
        r2 = e["plane_r2"]["median"] if e["plane_r2"] else float("nan")
        st = e["structure_alignment"]["median"] if e["structure_alignment"] else float("nan")
        print(f"{c:<20} {e['n']:>5} {r2:>9.3f} {st:>8.3f}")

    print(f"\n{'off-nadir':<10} {'n':>5} {'plane_r2':>9} {'struct':>8}")
    print("-" * 35)
    for a in angles:
        e = by_angle[str(a)]
        r2 = e["plane_r2"]["median"] if e["plane_r2"] else float("nan")
        st = e["structure_alignment"]["median"] if e["structure_alignment"] else float("nan")
        print(f"OA{a:<8} {e['n']:>5} {r2:>9.3f} {st:>8.3f}")

    print(f"\n{len(records)} scenes | {georef} georeferenced | {calibrated} calibration attempted "
          f"| {usable} usable (fit_r2 >= {CALIBRATION_MIN_R2} and scale > 0)")
    _report_calibration(records)
    print(f"wrote {args.out / 'index.json'}")


def _report_calibration(records: list[dict]) -> None:
    """Distribution of the relative->absolute fit, printed so the threshold is
    read alongside the evidence rather than taken on trust."""
    cal = [r["absolute"] for r in records if r.get("absolute")]
    if not cal:
        return
    r2 = np.array([c["fit_r2"] for c in cal], dtype=float)
    scale = np.array([c["scale_m"] for c in cal], dtype=float)
    finite = np.isfinite(r2)
    print("\nscale calibration vs 3DEP seamless (10 m, bare earth)")
    print("-" * 56)
    if finite.any():
        q = np.percentile(r2[finite], [10, 25, 50, 75, 90])
        print("  fit_r2 deciles: " + "  ".join(f"{v:.3f}" for v in q)
              + "   (p10 p25 p50 p75 p90)")
    neg = int((np.isfinite(scale) & (scale <= 0)).sum())
    print(f"  inverted fits (scale <= 0): {neg}/{len(cal)}"
          f"  -- model height anti-correlates with real terrain")
    weak = int((finite & (r2 < CALIBRATION_MIN_R2)).sum())
    print(f"  weak fits (r2 < {CALIBRATION_MIN_R2}): {weak}/{len(cal)}")
    ok = sum(1 for c in cal if c.get("usable"))
    print(f"  usable: {ok}/{len(cal)} ({100*ok/len(cal):.1f}%)")


def _r(value, digits: int = 4):
    """Round for the compact index; JSON has no NaN, so unmeasurable -> None."""
    if value is None:
        return None
    v = float(value)
    return None if not np.isfinite(v) else round(v, digits)


if __name__ == "__main__":
    main()
