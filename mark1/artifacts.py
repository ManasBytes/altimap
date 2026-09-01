"""Export scientific arrays and browser-ready Mark-1 scene assets."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .codec import encode_rg16
from .metrics import HeightMetrics, evaluate_height


def _preview(values: np.ndarray, path: Path, low: float, high: float) -> None:
    scaled = np.clip((np.nan_to_num(values, nan=low) - low) / max(high - low, 1e-6), 0, 1)
    rgb = np.rint(scaled * 255).astype(np.uint8)
    from .codec import _write_png
    _write_png(path, rgb, 0)


def export_scene(
    output_dir: str | Path,
    scene_id: str,
    split: str,
    rgb: np.ndarray,
    relative_depth: np.ndarray,
    predicted: np.ndarray,
    reference_agl: np.ndarray,
    classes: np.ndarray,
    model: str = "DAV2 + RDAH-Net (frozen)",
) -> dict[str, object]:
    """Write full precision arrays plus previews and a frontend manifest."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    arrays = {"relative_depth": relative_depth, "predicted": predicted, "reference_agl": reference_agl, "classes": classes}
    for name, array in arrays.items():
        np.save(root / f"{name}.npy", np.asarray(array))
    from PIL import Image
    Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB").save(root / "rgb.jpg", quality=92, optimize=True)
    metrics: HeightMetrics = evaluate_height(predicted, reference_agl, classes)
    valid_values = np.concatenate([np.asarray(predicted)[np.isfinite(predicted)], np.asarray(reference_agl)[np.isfinite(reference_agl)]])
    lo, hi = float(max(0, np.min(valid_values))), float(max(np.max(valid_values), np.min(valid_values) + 1e-6))
    encode_rg16(predicted, root / "predicted-height.png", lo, hi)
    encode_rg16(reference_agl, root / "reference-height.png", lo, hi)
    error = np.abs(np.asarray(predicted, dtype=np.float32) - np.asarray(reference_agl, dtype=np.float32))
    encode_rg16(np.nan_to_num(error, nan=0.0), root / "error-height.png", 0.0, max(1e-6, float(np.nanmax(error))))
    _preview(relative_depth, root / "dav2-depth.png", float(np.nanmin(relative_depth)), float(np.nanmax(relative_depth)))
    _preview(predicted, root / "predicted-height-preview.png", lo, hi)
    _preview(reference_agl, root / "reference-height-preview.png", lo, hi)
    _preview(np.nan_to_num(error, nan=0.0), root / "error-heatmap.png", 0.0, max(1e-6, float(np.nanmax(error))))
    class_preview = np.zeros((*classes.shape, 3), dtype=np.uint8)
    class_colors = {
        0: (40, 48, 58), 1: (123, 157, 112), 2: (88, 176, 106),
        3: (218, 126, 76), 4: (65, 146, 201), 5: (172, 148, 92),
        6: (54, 116, 72),
    }
    for label, color in class_colors.items():
        class_preview[np.asarray(classes) == label] = color
    Image.fromarray(class_preview, mode="RGB").save(root / "classes-preview.png", quality=92)
    manifest = {
        "scene_id": scene_id,
        "id": scene_id,
        "label": f"{split} scene {scene_id}",
        "split": split,
        "model": model,
        "previewOnly": False,
        "localGrid": True,
        "shape": list(np.asarray(predicted).shape),
        "pixel_spacing_m": 0.33,
        "crs": None,
        "height_range_m": [lo, hi],
        "metrics": metrics.as_dict(),
        "classMetrics": metrics.class_mae_m,
        "assets": {
            "rgb": "rgb.jpg",
            "depth": "dav2-depth.png",
            "predicted_height": "predicted-height.png",
            "reference_height": "reference-height.png",
            "error": "error-heatmap.png",
            "classes": "classes-preview.png",
            "predicted_geometry": "predicted-height.png",
            "reference_geometry": "reference-height.png",
            "error_geometry": "error-height.png",
        },
        "codec": {"predicted": "predicted-height.png", "reference": "reference-height.png", "error": "error-height.png"},
        "heightEncoding": "rg16-linear",
        "errorEncoding": "rg16-linear",
    }
    (root / "scene.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
