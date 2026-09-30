"""Forest canopy heights from Meta's CHMv2 (DINOv3 satellite backbone + DPT head).

Used only for tree pixels inside extensive forest (`estimate.forest_mask`): there our height
model reads canopy ~16 m low. Against 3DEP LiDAR on a forest scene this took the nDSM from
18.5 to 15.7 m RMSE, with no change on the city, suburb and hill scenes. Outside forests
CHMv2 is worse than our model (suburban trees 5.5 -> 8.3 m RMSE), which is why it is confined.

Weights (1.35 GB): Meta's `facebook/dinov3-vitl16-chmv2-dpt-head` (gated) or the public mirror
`WEO-SAS/chm-meta-v2` (byte-identical model.safetensors). DINOv3 licence: include it with
redistributions and show "Built with DINOv3". Optional: without it the pipeline skips this step.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

CHM_GSD_M = 0.6  # CHMv2 was trained on ~0.5-0.6 m satellite imagery
PATCH, STRIDE = 384, 192
REPOS = ("facebook/dinov3-vitl16-chmv2-dpt-head", "WEO-SAS/chm-meta-v2")
# sha256 of model.safetensors, the same in both repos
WEIGHTS_SHA256 = "57c705ba12745ee1b7b53b36602c1bd4fa0781d5ef308fbec96dfdd08b8188a4"


def local_dir() -> Path | None:
    """A downloaded copy of the weights in the Hugging Face cache, or None."""
    for repo in REPOS:
        snaps = sorted(Path.home().glob(f".cache/huggingface/hub/models--{repo.replace('/', '--')}/snapshots/*"))
        for snap in reversed(snaps):
            if (snap / "model.safetensors").exists() and (snap / "config.json").exists():
                return snap
    return None


def load(device: str = "cpu"):
    """-> (model, processor), or None if the weights or `transformers` aren't installed."""
    snap = local_dir()
    if snap is None:
        return None
    try:
        from transformers import CHMv2ForDepthEstimation, CHMv2ImageProcessor
    except ImportError:
        return None
    return CHMv2ForDepthEstimation.from_pretrained(snap).to(device).eval(), CHMv2ImageProcessor.from_pretrained(snap)


def predict(canopy, rgb: np.ndarray, gsd_m: float, device: str) -> np.ndarray:
    """Canopy height (m) on the input grid: the image is resampled to CHMv2's 0.6 m and run in
    384 px windows at half-window stride, averaged where they overlap."""
    import torch

    model, proc = canopy
    k = CHM_GSD_M / gsd_m
    small = rgb if abs(k - 1) < 0.01 else np.asarray(
        Image.fromarray(rgb).resize((max(1, round(rgb.shape[1] / k)), max(1, round(rgb.shape[0] / k))), Image.BOX))
    h, w = small.shape[:2]
    acc = np.zeros((h, w), np.float32)
    cnt = np.zeros((h, w), np.float32)
    ys = sorted(set(list(range(0, max(1, h - PATCH), STRIDE)) + [max(0, h - PATCH)]))
    xs = sorted(set(list(range(0, max(1, w - PATCH), STRIDE)) + [max(0, w - PATCH)]))
    for y in ys:
        for x in xs:
            tile = small[y:y + PATCH, x:x + PATCH]
            inputs = {n: v.to(device) for n, v in proc(images=Image.fromarray(tile), return_tensors="pt").items()}
            with torch.no_grad():
                out = model(**inputs)
            d = proc.post_process_depth_estimation(out, target_sizes=[tile.shape[:2]])[0]["predicted_depth"]
            acc[y:y + PATCH, x:x + PATCH] += d.float().cpu().numpy()
            cnt[y:y + PATCH, x:x + PATCH] += 1
    chm = np.maximum(acc / np.maximum(cnt, 1), 0)
    return np.asarray(Image.fromarray(chm, "F").resize((rgb.shape[1], rgb.shape[0]), Image.BILINEAR))
