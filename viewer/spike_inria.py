"""Spike: does DA3-SMALL behave differently on true-nadir 0.5 m aerial imagery?

Every prior domain-gap measurement in this project used data with a confound:
the Roboflow set mixes nadir and oblique shots, and the Off-nadir Scene10
Atlanta set is by construction 19-31deg off-nadir. This dataset (Inria Aerial
Image Labeling, repackaged at 0.5 m/px) is genuinely nadir orthophoto and
100% georeferenced (620/5200 for the Atlanta set), so it isolates the
off-nadir-angle variable for the first time.

Reads tiles directly from the zip via GDAL's /vsizip/ -- nothing is extracted
to disk. Metrics-only: this is a spike to decide whether the full export+
refine+validate pipeline is worth building for this dataset, not the pipeline
itself.

Run under .venv-da3 (imports torch + rasterio).
"""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from viewer.metrics import luminance, scene_metrics

ZIP_PATH = Path("/home/biplab-dev/Downloads/HR_0.5m.zip")
NAME_RE = re.compile(r"^HR_0\.5m/(?P<city>[a-z-]+)(?P<num>\d+)\.tif$")


def list_tiles(zip_path: Path) -> list[dict]:
    names = zipfile.ZipFile(zip_path).namelist()
    out = []
    for n in sorted(names):
        m = NAME_RE.match(n)
        if m:
            out.append({"entry": n, "city": m.group("city"), "num": int(m.group("num"))})
    return out


def read_tile(zip_path: Path, entry: str):
    import rasterio

    vsi = f"/vsizip/{zip_path}/{entry}"
    with rasterio.open(vsi) as src:
        rgb = np.transpose(src.read([1, 2, 3]), (1, 2, 0)).astype(np.uint8)
        geo = {
            "crs": str(src.crs), "bounds": list(src.bounds),
            "res_m": [abs(src.res[0]), abs(src.res[1])],
            "width": src.width, "height": src.height,
        }
    return rgb, geo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=ZIP_PATH)
    parser.add_argument("--per-city", type=int, default=4)
    parser.add_argument("--model", default="depth-anything/DA3-SMALL")
    parser.add_argument("--process-res", type=int, default=504)
    args = parser.parse_args()

    tiles = list_tiles(args.zip)
    by_city = defaultdict(list)
    for t in tiles:
        by_city[t["city"]].append(t)
    sample = [t for city in sorted(by_city) for t in by_city[city][: args.per_city]]
    print(f"{len(tiles)} tiles across {len(by_city)} cities; sampling {len(sample)} "
          f"({args.per_city}/city)")

    import torch
    from depth_anything_3.api import DepthAnything3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"loading {args.model} onto {device}...")
    model = DepthAnything3.from_pretrained(args.model).to(device=device)

    records = []
    for i, t in enumerate(sample, 1):
        rgb, geo = read_tile(args.zip, t["entry"])
        prediction = model.inference([rgb], process_res=args.process_res)
        depth = np.asarray(prediction.depth[0], dtype=np.float32)
        h, w = depth.shape
        from PIL import Image
        rgb_small = np.asarray(Image.fromarray(rgb).resize((w, h), Image.BILINEAR))
        lum = luminance(rgb_small.astype(np.float64) / 255.0)
        m = scene_metrics(depth, lum, None)
        records.append({"city": t["city"], "num": t["num"], "geo": geo, **m})
        if i % 20 == 0 or i == len(sample):
            print(f"  [{i}/{len(sample)}]")

    r2 = np.array([r["plane_r2"] for r in records])
    struct = np.array([r["structure_alignment"] for r in records])
    row = np.abs([r["row_corr"] for r in records])
    col = np.abs([r["col_corr"] for r in records])
    dom = np.maximum(row, col)

    print(f"\n{'':<26}{'median':>9}{'p25':>8}{'p75':>8}")
    print("-" * 51)
    for label, arr in [("plane_r2", r2), ("structure_alignment", struct),
                       ("dominant-axis |corr|", dom)]:
        fin = arr[np.isfinite(arr)]
        print(f"{label:<26}{np.median(fin):>9.3f}{np.percentile(fin,25):>8.3f}"
              f"{np.percentile(fin,75):>8.3f}")
    print(f"\ndominant-axis |corr| > 0.5: {int((dom>0.5).sum())}/{len(records)}"
          f"  (compare: 25/28 on the confounded Roboflow spike, 08-24)")
    print(f"scenes with negative row_corr: "
          f"{sum(1 for r in records if r['row_corr']<0)}/{len(records)}")

    print(f"\n{'city':<14}{'n':>4}{'plane_r2':>10}{'struct':>9}{'dom|corr|':>11}")
    print("-" * 48)
    for city in sorted(by_city):
        rows = [r for r in records if r["city"] == city]
        rr2 = np.array([r["plane_r2"] for r in rows])
        rst = np.array([r["structure_alignment"] for r in rows])
        rdom = np.maximum(np.abs([r["row_corr"] for r in rows]),
                          np.abs([r["col_corr"] for r in rows]))
        print(f"{city:<14}{len(rows):>4}{np.median(rr2):>10.3f}"
              f"{np.median(rst):>9.3f}{np.median(rdom):>11.3f}")

    out = Path("/tmp/inria_spike_records.json")
    out.write_text(json.dumps(records, default=float))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
