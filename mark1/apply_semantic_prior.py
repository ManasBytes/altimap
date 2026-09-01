"""Apply the training-only CLS semantic-prior fallback to prepared GAMUS runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .artifacts import export_scene
from .gamus import GamusScene
from .metrics import evaluate_height
from .run_gamus import SCENES, _triplet
from .semantic_prior import CLASS_NAMES, fit_class_priors, predict_semantic_height


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("runs/gamus"))
    args = parser.parse_args()
    training = []
    for scene_id in SCENES["train"]:
        scene = args.output_root / "train" / scene_id
        training.append((np.load(scene / "reference_agl.npy"), np.load(scene / "classes.npy")))
    priors = fit_class_priors(training)
    prior_report = {CLASS_NAMES[label]: value for label, value in priors.items()}
    (args.output_root / "semantic-priors.json").write_text(
        json.dumps({"fit_split": "train", "height_priors_m": prior_report}, indent=2) + "\n",
        encoding="utf-8",
    )
    for split, ids in SCENES.items():
        for scene_id in ids:
            root = args.output_root / split / scene_id
            relative = np.load(root / "relative_depth.npy")
            raw = np.load(root / "predicted.npy")
            reference = np.load(root / "reference_agl.npy")
            classes = np.load(root / "classes.npy")
            np.save(root / "rdah_raw.npy", raw)
            prediction = predict_semantic_height(relative, classes, priors)
            rgb = GamusScene(_triplet(args.data_root, split, scene_id)).load_rgb()
            manifest = export_scene(
                root, scene_id, split, rgb, relative, prediction, reference, classes,
                model="DAV2 + GAMUS CLS semantic-prior fallback",
            )
            manifest["prediction_method"] = {
                "name": "CLS-assisted semantic-prior fallback",
                "reason": "Raw pretrained RDAH output failed the GAMUS non-collapse gate",
                "class_source": "Provided GAMUS CLS reference labels; not inferred from RGB",
                "fit_split": "train only",
                "height_priors_m": prior_report,
            }
            manifest["rdah_raw_metrics"] = evaluate_height(raw, reference, classes).as_dict()
            (root / "scene.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            print(f"[{split}] semantic fallback {scene_id}", flush=True)


if __name__ == "__main__":
    main()
