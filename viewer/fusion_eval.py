"""Measure v1/v2 building routing without changing the production pipeline.

Examples:
    python -m viewer.fusion_eval --prepare --gamus DATA --synrs3d viewer/cache/SynRS3D \
        --v1 viewer/cache/best.pth --v2 viewer/cache/best_v2.pth
    python -m viewer.fusion_eval --report --cache runs/fusion-cache --out runs/fusion.json

Preparation runs each model once and stores predictions. Reporting is then CPU-only,
so the five fusion weights and soft-routing thresholds can be compared cheaply.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from viewer.gamus_dataset import find_tiles, load_tile
from viewer.height_metrics import HEIGHT_BINS, HEIGHT_BIN_NAMES, ScoreAccumulator
from viewer.height_model import OEM_TO_GAMUS, clean_height, load_model, predict


def route(h1: np.ndarray, h2: np.ndarray, classes: np.ndarray, weight: float) -> np.ndarray:
    """Apply a fixed v2 weight only to model-predicted building pixels."""
    return np.where(classes == 3, (1.0 - weight) * h1 + weight * h2, h1).astype(np.float32)


def soft_route(h1: np.ndarray, h2: np.ndarray, p_building: np.ndarray,
               weight: float, low: float, high: float) -> np.ndarray:
    """Blend v2 continuously as the model's building probability rises."""
    alpha = np.clip((p_building - low) / max(high - low, 1e-6), 0.0, 1.0)
    return (h1 + alpha * weight * (h2 - h1)).astype(np.float32)


def _tile_name(tile) -> str:
    return f"{tile.split}__{tile.scene_id}".replace("/", "_").replace("\\", "_")


def _selected(tiles, limit: int, names: list[str] | None = None):
    tiles = sorted(tiles, key=lambda t: t.scene_id)
    if names:
        by_name = {tile.scene_id: tile for tile in tiles}
        missing = [name for name in names if name not in by_name]
        if missing:
            raise ValueError(f"unknown tile ids: {', '.join(missing)}")
        return [by_name[name] for name in names]
    if not limit or len(tiles) <= limit:
        return tiles
    return tiles[:: max(1, len(tiles) // limit)][:limit]


def _write(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        # Evaluation cache only: compression makes model inference wait on a
        # CPU-bound zlib pass for every 1024x1024 tile. The arrays are unchanged
        # and np.load reads both compressed and uncompressed cache files.
        np.savez(handle, **arrays)
    temporary.replace(path)


def _cache_matches(path: Path, keys: set[str], tta: bool) -> bool:
    if not path.exists():
        return False
    with np.load(path) as data:
        if not keys.issubset(data.files):
            return False
        # Old caches predate provenance metadata and were all generated without TTA.
        cached_tta = bool(data["tta"]) if "tta" in data.files else False
        return cached_tta == tta


def prepare(tiles, args) -> None:
    args.cache.mkdir(parents=True, exist_ok=True)
    model, device = load_model(args.v1, args.synrs3d)
    print(f"v1: {args.v1} on {device}", flush=True)
    for i, tile in enumerate(tiles, 1):
        path = args.cache / f"{_tile_name(tile)}.npz"
        if _cache_matches(path, {"h1", "p_building", "predicted_oem", "predicted_classes",
                                 "truth_classes", "reference"}, args.tta):
            print(f"v1 resume {i}/{len(tiles)} {tile.scene_id}", flush=True)
            continue
        rgb, truth_classes, reference = load_tile(tile)
        height, oem, p_building = predict(
            model, rgb, device, batch_size=args.batch_size, tta=args.tta,
            return_building_probability=True
        )
        _write(path, h1=height, tta=np.bool_(args.tta),
               p_building=p_building, predicted_oem=oem,
               predicted_classes=OEM_TO_GAMUS[oem],
               truth_classes=truth_classes, reference=reference)
        print(f"v1 {i}/{len(tiles)} {tile.scene_id}", flush=True)
    _release(model)

    model, device = load_model(args.v2, args.synrs3d)
    print(f"v2: {args.v2} on {device}", flush=True)
    for i, tile in enumerate(tiles, 1):
        path = args.cache / f"{_tile_name(tile)}.npz"
        if _cache_matches(path, {"h2"}, args.tta):
            print(f"v2 resume {i}/{len(tiles)} {tile.scene_id}", flush=True)
            continue
        with np.load(path) as data:
            arrays = {key: data[key] for key in data.files}
        rgb, _, _ = load_tile(tile)
        arrays["h2"] = predict(model, rgb, device, batch_size=args.batch_size, tta=args.tta)[0]
        _write(path, **arrays)
        print(f"v2 {i}/{len(tiles)} {tile.scene_id}", flush=True)
    _release(model)


def _release(model) -> None:
    model.to("cpu")
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _variants() -> dict[str, tuple]:
    variants = {f"hard_w{weight:.2f}": ("hard", weight)
                for weight in (0.0, 0.25, 0.5, 0.75, 1.0)}
    for low, high in ((0.1, 0.5), (0.2, 0.6), (0.3, 0.7), (0.4, 0.8), (0.5, 0.9)):
        variants[f"soft_{low:.1f}_{high:.1f}"] = ("soft", 0.5, low, high)
    return variants


def city_of(scene_id: str) -> str:
    return scene_id.split("_")[0]


def _add_group_metrics(overall: ScoreAccumulator, bins: dict[str, ScoreAccumulator],
                       prediction: np.ndarray, reference: np.ndarray,
                       classes: np.ndarray) -> None:
    """Pool one tile into overall and height-bin totals with one finite-mask pass."""
    prediction = np.asarray(prediction, np.float64)
    reference = np.asarray(reference, np.float64)
    valid = np.isfinite(prediction) & np.isfinite(reference)
    p, r = prediction[valid], reference[valid]
    overall._add_valid(p, r, np.asarray(classes)[valid])
    for bin_name, (lo, hi) in zip(HEIGHT_BIN_NAMES, HEIGHT_BINS):
        mask = (r >= lo) & (r < hi)
        bins[bin_name]._add_valid(p[mask], r[mask])


def report(tiles, args, names: list[str] | None = None) -> dict:
    variants = _variants()
    names = names or list(variants)
    unknown = sorted(set(names) - set(variants))
    if unknown:
        raise ValueError(f"unknown routing variants: {', '.join(unknown)}")
    groups = ["ALL", *sorted({city_of(tile.scene_id) for tile in tiles})]
    accumulators = {name: {group: ScoreAccumulator() for group in groups} for name in names}
    bins = {name: {group: {bin_name: ScoreAccumulator(pearson=False) for bin_name in HEIGHT_BIN_NAMES}
                   for group in groups} for name in names}
    for i, tile in enumerate(tiles, 1):
        path = args.cache / f"{_tile_name(tile)}.npz"
        required = {"h1", "h2", "p_building", "predicted_classes",
                    "truth_classes", "reference"}
        if not _cache_matches(path, required, args.tta):
            raise ValueError(f"cache does not match tta={args.tta}: {path}")
        with np.load(path) as data:
            h1, h2 = data["h1"], data["h2"]
            p_building, predicted = data["p_building"], data["predicted_classes"]
            truth_classes = data["truth_classes"]
            reference = clean_height(data["reference"])
        tile_groups = ("ALL", city_of(tile.scene_id))
        for name in names:
            spec = variants[name]
            if spec[0] == "hard":
                prediction = route(h1, h2, predicted, spec[1])
            else:
                prediction = soft_route(h1, h2, p_building, spec[1], spec[2], spec[3])
            for group in tile_groups:
                _add_group_metrics(accumulators[name][group], bins[name][group],
                                   prediction, reference, truth_classes)
        if i == 1 or i % 100 == 0 or i == len(tiles):
            print(f"report {i}/{len(tiles)} {tile.scene_id}", flush=True)
    result = {"split": args.split, "tiles": len(tiles), "tta": args.tta,
              "cities": groups[1:], "variants": {}}
    for name in names:
        result["variants"][name] = {
            "overall": {group: accumulators[name][group].result() for group in groups},
            "height_bins": {
                group: [{"bin": bin_name, **bins[name][group][bin_name].result()}
                        for bin_name in HEIGHT_BIN_NAMES]
                for group in groups
            },
        }
    return result


def write_csv(result: dict, path: Path) -> None:
    fields = ("split", "variant", "group", "scope", "bin", "n", "rmse", "mae", "bias",
              "pearson", "building_rmse", "building_mae", "building_bias")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for variant, payload in result["variants"].items():
            for group, scores in payload["overall"].items():
                writer.writerow({"split": result["split"], "variant": variant, "group": group,
                                 "scope": "overall", "bin": "", **scores})
            for group, rows in payload["height_bins"].items():
                for row in rows:
                    writer.writerow({"split": result["split"], "variant": variant, "group": group,
                                     "scope": "height_bin", **row})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gamus", type=Path, required=True)
    ap.add_argument("--synrs3d", type=Path, required=True)
    ap.add_argument("--v1", type=Path, required=True)
    ap.add_argument("--v2", type=Path, required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tiles", nargs="*", default=None,
                    help="exact scene ids to evaluate, preserving the supplied order")
    ap.add_argument("--cache", type=Path, default=Path("runs/fusion-cache"))
    ap.add_argument("--out", type=Path, default=Path("runs/fusion.json"))
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--only", nargs="*", default=None,
                    help="variant names to report; default is every validation variant")
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--batch-size", type=int, default=2,
                    help="sliding-window inference batch size; raise until VRAM is full")
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if not args.prepare and not args.report:
        ap.error("choose --prepare and/or --report")
    tiles = _selected([t for t in find_tiles((args.gamus,)) if t.split == args.split],
                      args.limit, args.tiles)
    if args.prepare:
        prepare(tiles, args)
    if args.report:
        result = report(tiles, args, args.only)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))
        if args.csv:
            write_csv(result, args.csv)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
