"""Export scientific arrays and browser-ready Mark-1 scene assets."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .codec import encode_rg16
from .metrics import HeightMetrics, evaluate_height


def _preview(values: np.ndarray, path: Path, low: float, high: float) -> None:
    scaled = np.clip((np.nan_to_num(values, nan=low) - low) / max(high - low, 1e-6), 0, 1)
    rgb = np.rint(scaled * 255).astype(np.uint8)
    from .codec import _write_png
    _write_png(path, rgb, 0)


def _color_preview(values: np.ndarray, path: Path, low: float, high: float, error: bool = False) -> None:
    scaled = np.clip((np.nan_to_num(values, nan=low) - low) / max(high - low, 1e-6), 0, 1)
    stops = np.asarray(
        [(35, 69, 188), (232, 210, 72), (204, 51, 69)] if error else
        [(9, 45, 74), (42, 119, 142), (112, 193, 157), (247, 207, 88)],
        dtype=np.float32,
    )
    positions = np.linspace(0, 1, len(stops))
    channels = [np.interp(scaled, positions, stops[:, channel]) for channel in range(3)]
    Image.fromarray(np.stack(channels, axis=-1).astype(np.uint8), mode="RGB").save(path)


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
    Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB").save(root / "rgb.jpg", quality=92, optimize=True)
    metrics: HeightMetrics = evaluate_height(predicted, reference_agl, classes)
    valid_values = np.concatenate([np.asarray(predicted)[np.isfinite(predicted)], np.asarray(reference_agl)[np.isfinite(reference_agl) & (reference_agl >= 0)]])
    lo = 0.0
    hi = float(max(np.percentile(valid_values, 99.5), 1e-6))
    pred_hi = float(max(np.percentile(np.asarray(predicted)[np.isfinite(predicted)], 99.5), 1e-6))
    ref_valid = np.asarray(reference_agl)[np.isfinite(reference_agl) & (reference_agl >= 0)]
    ref_hi = float(max(np.percentile(ref_valid, 99.5), 1e-6))
    encode_rg16(predicted, root / "predicted-height.png", lo, hi)
    encode_rg16(reference_agl, root / "reference-height.png", lo, hi)
    error = np.abs(np.asarray(predicted, dtype=np.float32) - np.asarray(reference_agl, dtype=np.float32))
    encode_rg16(np.nan_to_num(error, nan=0.0), root / "error-height.png", 0.0, max(1e-6, float(np.nanmax(error))))
    depth_lo, depth_hi = [float(v) for v in np.nanpercentile(relative_depth, (2, 98))]
    error_hi = float(max(1e-6, np.nanpercentile(error, 99.5)))
    _color_preview(relative_depth, root / "dav2-depth.png", depth_lo, depth_hi)
    _color_preview(predicted, root / "predicted-height-preview.png", 0.0, pred_hi)
    _color_preview(reference_agl, root / "reference-height-preview.png", 0.0, ref_hi)
    _color_preview(np.nan_to_num(error, nan=0.0), root / "error-heatmap.png", 0.0, error_hi, error=True)
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
        "predicted_range_m": [0.0, pred_hi],
        "reference_range_m": [0.0, ref_hi],
        "depth_range": [depth_lo, depth_hi],
        "error_range_m": [0.0, error_hi],
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
