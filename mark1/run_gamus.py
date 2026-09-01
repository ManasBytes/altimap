"""Offline Mark-1 GAMUS inference and asset generation.

This command is deliberately offline: it reads a GAMUS RGB HDF5 tile, runs the
frozen DAV2 + RDAH path, then opens AGL/CLS only for evaluation and export.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .artifacts import export_scene
from .gamus import GamusScene, GamusTriplet
from .rdah_runtime import DepthAnythingV2Prior, RDAHNetPredictor


SCENES = {
    "train": ("DC_01_25", "DC_10_17", "DC_10_18", "DC_10_19", "DC_10_21", "DC_10_27"),
    "val": ("DC_02_26", "DC_20_13", "DC_20_14", "DC_20_18", "DC_20_19", "DC_20_29"),
    "test": ("DC_03_26", "DC_20_12", "DC_20_15", "DC_20_20", "DC_20_23", "DC_20_25"),
}


def _triplet(data_root: Path, split: str, scene_id: str) -> GamusTriplet:
    """Resolve either a flat scene directory or official split directories."""
    official = {
        "RGB": data_root / "images" / split / f"{scene_id}_RGB.h5",
        "AGL": data_root / "heights" / split / f"{scene_id}_AGL.h5",
        "CLS": data_root / "classes" / split / f"{scene_id}_CLS.h5",
    }
    if all(path.is_file() for path in official.values()):
        return GamusTriplet(scene_id, official["RGB"], official["AGL"], official["CLS"])
    candidates = [
        data_root / split,
        data_root / split / scene_id,
        data_root / "raw" / scene_id,
        data_root / scene_id,
    ]
    for directory in candidates:
        try:
            return GamusTriplet.from_directory(directory, scene_id)
        except FileNotFoundError:
            continue
    raise FileNotFoundError(f"No complete GAMUS triplet found for {split}/{scene_id} under {data_root}")


def process_scene(scene: GamusScene, split: str, output_root: Path, predictor: RDAHNetPredictor) -> dict[str, object]:
    started = time.perf_counter()
    rgb = scene.load_rgb()
    relative_depth = predictor.depth_prior.predict_depth(rgb)
    prediction, _ = predictor.predict_ndsm(rgb, relative_depth)
    agl, classes = scene.load_targets()
    output = output_root / split / scene.triplet.scene_id
    manifest = export_scene(output, scene.triplet.scene_id, split, rgb, relative_depth, prediction, agl, classes)
    manifest["inference_seconds"] = round(time.perf_counter() - started, 3)
    (output / "scene.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("runs/gamus"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--scene", action="append", help="Process a scene ID; repeatable")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    splits = ("train", "val", "test") if args.split == "all" else (args.split,)
    prior = DepthAnythingV2Prior(device=args.device)
    predictor = RDAHNetPredictor(args.checkpoint, prior, device=args.device)
    manifests: list[dict[str, object]] = []
    for split in splits:
        scene_ids = tuple(args.scene or SCENES[split])
        for index, scene_id in enumerate(scene_ids, 1):
            print(f"[{split}] {index}/{len(scene_ids)} {scene_id}", flush=True)
            triplet = _triplet(args.data_root, split, scene_id)
            manifests.append(process_scene(GamusScene(triplet), split, args.output_root, predictor))
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "scenes.json").write_text(json.dumps(manifests, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
