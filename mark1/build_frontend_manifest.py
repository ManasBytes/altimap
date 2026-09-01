"""Build the committed browser manifest from offline Mark-1 scene outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .run_gamus import SCENES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=Path("runs/gamus"))
    parser.add_argument("--public", type=Path, default=Path("frontend/public/gamus"))
    args = parser.parse_args()
    scenes = []
    for split, ids in SCENES.items():
        for scene_id in ids:
            source = args.runs / split / scene_id / "scene.json"
            item = json.loads(source.read_text(encoding="utf-8"))
            base = f"/gamus/{split}/{scene_id}"
            metrics = item["metrics"]
            scenes.append({
                "id": scene_id,
                "split": split,
                "label": f"{split} scene {scene_id}",
                "previewOnly": False,
                "localGrid": True,
                "crs": None,
                "pixelSpacingM": item.get("pixel_spacing_m"),
                "model": item.get("model"),
                "predictionMethod": item.get("prediction_method"),
                "calibration": item.get("calibration"),
                "heightRangeM": item.get("height_range_m"),
                "surfaceRangesM": {
                    "predicted": item.get("predicted_range_m"),
                    "reference": item.get("reference_range_m"),
                    "error": item.get("predicted_range_m"),
                },
                "depthRange": item.get("depth_range"),
                "errorRangeM": item.get("error_range_m"),
                "metrics": {
                    "mae": metrics["mae_m"], "mse": metrics["mse_m2"],
                    "rmse": metrics["rmse_m"], "bias": metrics["bias_m"],
                    "validPixels": metrics["valid_pixels"],
                },
                "classMetrics": metrics.get("class_mae_m", {}),
                "heightEncoding": "rg16-linear",
                "layers": {
                    "rgb": f"{base}/rgb.jpg",
                    "depth": f"{base}/dav2-depth.png",
                    "predictedHeight": f"{base}/predicted-height.png",
                    "referenceHeight": f"{base}/reference-height.png",
                    "errorHeatmap": f"{base}/error-heatmap.png",
                    "classes": f"{base}/classes-preview.png",
                    "predictedSurface": f"{base}/predicted-height-preview.png",
                    "referenceSurface": f"{base}/reference-height-preview.png",
                },
                "geometry": {
                    "predicted": f"{base}/predicted-height.png",
                    "reference": f"{base}/reference-height.png",
                    # Error mode is predicted geometry with an error texture.
                    "error": f"{base}/predicted-height.png",
                },
            })
    payload = {
        "version": 2,
        "source": "earthflow/GAMUS",
        "scope": "Mark-1 selected 18 scenes",
        "notes": "DAV2 + train-fitted CLS semantic-prior fallback after raw RDAH failed the GAMUS non-collapse gate; metrics use GAMUS AGL after prediction.",
        "scenes": scenes,
    }
    args.public.mkdir(parents=True, exist_ok=True)
    (args.public / "scenes.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(scenes)} scenes to {args.public / 'scenes.json'}")


if __name__ == "__main__":
    main()
