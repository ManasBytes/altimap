"""Export the Inria Aerial Image Labeling set (HR_0.5m.zip) as browsable scenes.

360 tiles, 10 cities (5 US, 5 Austrian), 0.5 m/px, TRUE nadir orthophoto,
100% georeferenced -- a strict upgrade over the Off-nadir Scene10 Atlanta set
(620/5200, 19-31deg off-nadir) for exactly the two things that set struggled
with: domain-gap measurement without an oblique confound, and elevation
calibration coverage.

See docs/superpowers/spikes/2026-08-2X-inria-nadir-domain-gap.md: on a 40-tile
sample the ramp was if anything WORSE on true nadir imagery than on the
confounded set (38/40 dominant-axis |corr| > 0.5 vs 25/28), so the off-nadir
angle was never the explanation.

Reads tiles directly out of the zip via GDAL's /vsizip/ -- the zip is 8 GB and
nothing needs extracting to disk. Cities span the US and Austria, so elevation
calibration uses Copernicus GLO-30 (global) rather than 3DEP (US-only, and the
source that failed on the Atlanta set anyway).

Everything downstream of this exporter is UNCHANGED: viewer/refine_scenes.py
and viewer/validate.py already take --data/--buildings as CLI args and work
against this dataset's output with no code changes.

Run under .venv-da3 (torch + rasterio).
"""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from viewer.dem import DemSource
from viewer.geo import encode_rg16, fit_absolute_elevation, geo_meta_from_dataset
from viewer.metrics import luminance, scene_metrics
from viewer.terrain import build_terrain, height_field

DEFAULT_ZIP = Path("/home/biplab-dev/Downloads/HR_0.5m.zip")
DEFAULT_MODEL = "depth-anything/DA3-SMALL"
DEM_COLLECTION = "cop-dem-glo-30"
NAME_RE = re.compile(r"^HR_0\.5m/(?P<city>[a-z-]+)(?P<num>\d+)\.tif$")

CALIBRATION_MIN_R2 = 0.3


def list_tiles(zip_path: Path) -> list[dict]:
    names = zipfile.ZipFile(zip_path).namelist()
    out = []
    for n in sorted(names):
        m = NAME_RE.match(n)
        if m:
            out.append({"entry": n, "city": m.group("city"), "num": int(m.group("num")),
                       "id": f"{m.group('city')}__t{int(m.group('num')):02d}"})
    return out


def read_tile(zip_path: Path, entry: str):
    """-> (rgb uint8 HxWx3, geo dict, rasterio bounds/crs for the DEM call)."""
    import rasterio

    vsi = f"/vsizip/{zip_path}/{entry}"
    with rasterio.open(vsi) as src:
        rgb = np.transpose(src.read([1, 2, 3]), (1, 2, 0)).astype(np.uint8)
        geo = geo_meta_from_dataset(src)
    return rgb, geo


def _load_model(model_name: str):
    import torch
    from depth_anything_3.api import DepthAnything3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"loading {model_name} onto {device}...")
    return DepthAnything3.from_pretrained(model_name).to(device=device), str(device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-dem", action="store_true")
    parser.add_argument("--glb-for", type=int, default=40,
                        help="write terrain.glb for the N most-structured scenes")
    parser.add_argument("--glb-res", type=int, default=256)
    parser.add_argument("--out", type=Path, default=Path("viewer/web/data-inria"))
    args = parser.parse_args()

    tiles = list_tiles(args.zip)
    if args.limit:
        tiles = tiles[: args.limit]
    if not tiles:
        raise SystemExit(f"no tiles matched inside {args.zip}")
    cities = sorted({t["city"] for t in tiles})
    print(f"{len(tiles)} tiles across {len(cities)} cities: {cities}")

    scenes_dir = args.out / "scenes"
    scenes_dir.mkdir(parents=True, exist_ok=True)

    dem = None if args.no_dem else DemSource(collection=DEM_COLLECTION)

    model, device = _load_model(args.model)

    records: list[dict] = []
    import time
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
                pass

        rgb, geo = read_tile(args.zip, tile["entry"])
        prediction = model.inference([rgb], process_res=args.process_res)
        depth = np.asarray(prediction.depth[0], dtype=np.float32)
        if not np.isfinite(depth).all():
            raise ValueError(f"non-finite depth for {tile['entry']}")
        conf = np.asarray(prediction.conf[0], dtype=np.float32) if prediction.conf is not None else None

        h, w = depth.shape
        rgb_small = np.asarray(Image.fromarray(rgb).resize((w, h), Image.BILINEAR), dtype=np.uint8)
        lum = luminance(rgb_small.astype(np.float64) / 255.0)

        record = {
            "id": tile["id"], "class": tile["city"], "num": tile["num"],
            "source_image": tile["entry"].rsplit("/", 1)[-1],
            "width": w, "height": h,
            "is_metric": bool(prediction.is_metric),
            "encoding": "rg16-png",
            "geo": geo,
            "absolute": None,
            **scene_metrics(depth, lum, conf),
        }

        if geo["georeferenced"] and dem is not None:
            patch = dem.patch(geo["bounds"], geo["crs"], (h, w))
            if patch is not None:
                hf = height_field(depth)
                scale, offset, r2 = fit_absolute_elevation(hf, patch)
                valid = patch[np.isfinite(patch)]
                ok = bool(np.isfinite(r2) and r2 >= CALIBRATION_MIN_R2
                          and np.isfinite(scale) and scale > 0)
                record["absolute"] = {
                    "source": DEM_COLLECTION, "dem_item": dem.item_id,
                    "scale_m": scale, "offset_m": offset, "fit_r2": r2, "usable": ok,
                    "reject_reason": None if ok else (
                        "inverted (negative scale)" if np.isfinite(scale) and scale <= 0
                        else "weak fit" if np.isfinite(r2) else "unfittable"),
                    "dem_min_m": float(valid.min()) if valid.size else None,
                    "dem_max_m": float(valid.max()) if valid.size else None,
                    "reference_is_bare_earth": False,  # Copernicus GLO-30 is a DSM, not bare earth
                    "reference_posting_m": 30.0,
                }

        scene_dir.mkdir(parents=True, exist_ok=True)
        encoded, lo, hi = encode_rg16(depth)
        record["depth_lo"], record["depth_hi"] = lo, hi
        Image.fromarray(encoded).save(scene_dir / "depth.png", compress_level=6)
        Image.fromarray(rgb_small).save(scene_dir / "rgb.jpg", quality=88)
        meta_path.write_text(json.dumps(record))
        records.append(record)

        if i % 40 == 0 or i == len(tiles):
            rate = (i - skipped) / max(time.perf_counter() - started, 1e-9)
            eta = (len(tiles) - i) / max(rate, 1e-9) / 60
            print(f"  [{i}/{len(tiles)}] {rate:.1f} img/s  eta {eta:.1f} min")

    if dem is not None:
        dem.close()

    if args.glb_for:
        ranked = sorted(
            (r for r in records if np.isfinite(r.get("structure_alignment", np.nan))),
            key=lambda r: -r["structure_alignment"])[: args.glb_for]
        print(f"\nwriting {len(ranked)} terrain.glb meshes")
        for r in ranked:
            sd = scenes_dir / r["id"]
            arr = np.asarray(Image.open(sd / "depth.png").convert("RGB"))
            from viewer.geo import decode_rg16
            depth = decode_rg16(arr, r["depth_lo"], r["depth_hi"])
            plane = (r["plane"]["a"], r["plane"]["b"], r["plane"]["c"])
            build_terrain(height_field(depth, plane), sd / "rgb.jpg", sd / "terrain.glb",
                          res=args.glb_res)
            r["has_glb"] = True
            (sd / "meta.json").write_text(json.dumps(r))

    _write_index(args, records, cities, device)


def _write_index(args, records, cities, device) -> None:
    (args.out / "metrics_full.json").write_text(json.dumps(records))

    def agg(rows, key):
        vals = np.array([r[key] for r in rows], dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return {"median": None, "q1": None, "q3": None}
        return {"median": float(np.median(vals)), "q1": float(np.percentile(vals, 25)),
                "q3": float(np.percentile(vals, 75))}

    by_city = {c: {"n": sum(1 for r in records if r["class"] == c),
                   **{k: agg([r for r in records if r["class"] == c], k)
                      for k in ("plane_r2", "residual_relief", "structure_alignment")}}
               for c in cities}

    georef = sum(1 for r in records if r["geo"]["georeferenced"])
    calibrated = sum(1 for r in records if r.get("absolute"))
    usable = sum(1 for r in records if (r.get("absolute") or {}).get("usable"))

    index = {
        "dataset": "inria-hr-0.5m", "model": args.model, "process_res": args.process_res,
        "device": device, "total": len(records), "georeferenced": georef,
        "calibration_attempted": calibrated, "calibration_usable": usable,
        "classes": cities, "class_aggregates": by_city,
        "scenes": [{
            "id": r["id"], "c": r["class"], "n": r["num"],
            "r2": _r(r["plane_r2"]), "st": _r(r["structure_alignment"]),
            "rr": _r(r["residual_relief"]), "g": 1,
            "gm": round(r["geo"]["ground_m"][0], 1) if r["geo"]["ground_m"] else None,
            "es": _r((r["absolute"] or {}).get("scale_m")),
            "er": _r((r["absolute"] or {}).get("fit_r2")),
            "glb": 1 if r.get("has_glb") else 0,
        } for r in sorted(records, key=lambda r: r["plane_r2"])],
    }
    (args.out / "index.json").write_text(json.dumps(index))

    print(f"\n{'city':<14} {'n':>4} {'plane_r2':>9} {'struct':>8}")
    print("-" * 40)
    for c in cities:
        e = by_city[c]
        r2 = e["plane_r2"]["median"] if e["plane_r2"]["median"] is not None else float("nan")
        st = e["structure_alignment"]["median"] if e["structure_alignment"]["median"] is not None else float("nan")
        print(f"{c:<14} {e['n']:>4} {r2:>9.3f} {st:>8.3f}")
    print(f"\n{len(records)} scenes | {georef} georeferenced | {calibrated} calibration attempted "
          f"| {usable} usable")
    print(f"wrote {args.out / 'index.json'}")


def _r(value, digits: int = 4):
    if value is None:
        return None
    v = float(value)
    return None if not np.isfinite(v) else round(v, digits)


if __name__ == "__main__":
    main()
