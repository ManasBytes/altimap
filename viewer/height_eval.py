"""Score a height checkpoint on GAMUS, per city, against trivial baselines.

    python -m viewer.height_eval --split test --limit 300          # zero-shot RS3DAda
    python -m viewer.height_eval --split test --ckpt runs/ft/best.pth --out ft.json

Baselines are in the same table on purpose: `zero` (no objects anywhere) and
`train-mean` (one constant height everywhere). A model that does not beat
train-mean has learned nothing worth reporting.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from viewer.gamus_dataset import find_tiles, load_tile
from viewer.height_metrics import ScoreAccumulator
from viewer.height_model import OEM_TO_GAMUS, clean_height, load_model, predict

DATA = Path(os.environ.get("ALTIMAP_DATA", Path.home() / "altimap-data"))


def city_of(scene_id: str) -> str:
    return scene_id.split("_")[0]


def train_mean_height(tiles, n: int = 100) -> float:
    train = [t for t in tiles if t.split == "train"][:n]
    if not train:
        return 0.0
    return float(np.mean([np.nanmean(clean_height(load_tile(t)[2])) for t in train]))


def evaluate(model, device, tiles, mean_h: float, tta: bool = False) -> dict:
    accs = defaultdict(lambda: {k: ScoreAccumulator() for k in ("model", "zero", "train-mean")})
    seg = {"labelled": 0, "correct": 0, "buildings": 0, "buildings_hit": 0}
    started = time.time()
    for i, tile in enumerate(tiles):
        rgb, cls, ref = load_tile(tile)
        ref = clean_height(ref)
        pred, oem = predict(model, rgb, device, tta=tta)
        for group in (city_of(tile.scene_id), "ALL"):
            accs[group]["model"].add(pred, ref, cls)
            accs[group]["zero"].add(np.zeros_like(ref), ref, cls)
            accs[group]["train-mean"].add(np.full_like(ref, mean_h), ref, cls)
        labelled = cls > 0
        seg["labelled"] += int(labelled.sum())
        seg["correct"] += int((OEM_TO_GAMUS[oem] == cls)[labelled].sum())
        seg["buildings"] += int((cls == 3).sum())
        seg["buildings_hit"] += int(((cls == 3) & (oem == 7)).sum())
        if (i + 1) % 25 == 0:
            rate = (i + 1) / (time.time() - started)
            print(f"  {i + 1}/{len(tiles)} tiles, {rate:.2f} tiles/s", flush=True)
    return {
        "scores": {g: {k: a.result() for k, a in d.items()} for g, d in accs.items()},
        "segmentation": {
            "pixel_accuracy": seg["correct"] / max(seg["labelled"], 1),
            "building_recall": seg["buildings_hit"] / max(seg["buildings"], 1),
        },
    }


def print_table(result: dict) -> None:
    print(f"{'group':<8}{'method':<12}{'RMSE':>8}{'MAE':>8}{'r':>8}{'bldRMSE':>9}{'pixels':>12}")
    for group, methods in sorted(result["scores"].items(), key=lambda kv: kv[0] == "ALL"):
        for name, s in methods.items():
            print(f"{group:<8}{name:<12}{s['rmse']:8.3f}{s['mae']:8.3f}{s['pearson']:8.3f}"
                  f"{s['building_rmse']:9.3f}{s['n']:12d}")
    seg = result["segmentation"]
    print(f"segmentation: pixel accuracy {seg['pixel_accuracy']:.3f}, "
          f"building recall {seg['building_recall']:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gamus", type=Path, default=DATA / "gamus")
    ap.add_argument("--synrs3d", type=Path, default=DATA / "SynRS3D")
    ap.add_argument("--ckpt", type=Path, default=DATA / "SynRS3D/pretrain/RS3DAda_vitl_DPT_height.pth")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0, help="0 = all tiles in the split")
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("eval.json"))
    args = ap.parse_args()

    tiles = find_tiles((args.gamus,))
    split = sorted((t for t in tiles if t.split == args.split), key=lambda t: t.scene_id)
    if args.limit:
        split = split[:: max(1, len(split) // args.limit)][: args.limit]  # spread across cities
    mean_h = train_mean_height(tiles)
    print(f"{len(split)} {args.split} tiles, train-mean height {mean_h:.2f} m, ckpt {args.ckpt.name}")

    model, device = load_model(args.ckpt, args.synrs3d)
    result = evaluate(model, device, split, mean_h, tta=args.tta)
    result.update(ckpt=str(args.ckpt), split=args.split, tiles=len(split), tta=args.tta,
                  train_mean_m=mean_h)
    print_table(result)
    args.out.write_text(json.dumps(result, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
