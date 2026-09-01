"""Apply the Mark-1 train/validation calibration gate to offline runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .artifacts import export_scene
from .calibration import calibration_gate, fit_affine
from .gamus import GamusScene
from .run_gamus import SCENES, _triplet


def _load_arrays(root: Path, split: str, scene_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    scene = root / split / scene_id
    return (
        np.load(scene / "relative_depth.npy"),
        np.load(scene / "predicted.npy"),
        np.load(scene / "reference_agl.npy"),
        np.load(scene / "classes.npy"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("runs/gamus"))
    args = parser.parse_args()

    train = [_load_arrays(args.output_root, "train", sid) for sid in SCENES["train"]]
    val = [_load_arrays(args.output_root, "val", sid) for sid in SCENES["val"]]
    raw_train = np.concatenate([p.reshape(-1) for _, p, _, _ in train])
    ref_train = np.concatenate([r.reshape(-1) for _, _, r, _ in train])
    calibration = fit_affine(raw_train, ref_train)
    baseline_val = np.concatenate([p.reshape(-1) for _, p, _, _ in val])
    calibrated_val = calibration.apply(baseline_val)
    ref_val = np.concatenate([r.reshape(-1) for _, _, r, _ in val])
    accepted = calibration.scale > 0 and calibration_gate(
        baseline_val.reshape(-1, 1), calibrated_val.reshape(-1, 1), ref_val.reshape(-1, 1)
    )
    if accepted:
        all_fit = train + val
        calibration = fit_affine(
            np.concatenate([p.reshape(-1) for _, p, _, _ in all_fit]),
            np.concatenate([r.reshape(-1) for _, _, r, _ in all_fit]),
        )

    summary = {
        "accepted": bool(accepted),
        "scale": calibration.scale,
        "offset": calibration.offset,
        "minimum_validation_improvement": 0.05,
        "fit": "train+validation" if accepted else "train (rejected on validation)",
    }
    (args.output_root / "calibration.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    for split, ids in SCENES.items():
        for scene_id in ids:
            relative, raw, reference, classes = _load_arrays(args.output_root, split, scene_id)
            prediction = calibration.apply(raw) if accepted else raw
            triplet = _triplet(args.data_root, split, scene_id)
            rgb = GamusScene(triplet).load_rgb()
            manifest = export_scene(
                args.output_root / split / scene_id,
                scene_id, split, rgb, relative, prediction, reference, classes,
            )
            manifest["calibration"] = summary
            path = args.output_root / split / scene_id / "scene.json"
            path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            print(f"[{split}] calibrated {scene_id}", flush=True)


if __name__ == "__main__":
    main()
