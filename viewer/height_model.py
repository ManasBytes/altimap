"""RS3DAda height model: load a checkpoint and predict metric nDSM from RGB.

RS3DAda (SynRS3D, NeurIPS 2024) is a DINOv2 ViT-L encoder with a DPT decoder
and two heads: `regression` (height above ground, metres) and `segmentation`
(8 OpenEarthMap land-cover classes). The model class lives in the cloned
SynRS3D repo (`models/dpt.py`), so callers pass that directory in.

torch is imported lazily so the pure helpers here stay testable in the
torch-free .venv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HEADS = [{"name": "regression", "nclass": 1}, {"name": "segmentation", "nclass": 8}]
MEAN = np.array([123.675, 116.28, 103.53], np.float32)
STD = np.array([58.395, 57.12, 57.375], np.float32)
PATCH = 518  # ViT patch 14 -> every input side must be a multiple of 14
GAMUS_GSD_M = 0.33  # GAMUS paper, section 1

# OpenEarthMap order: bareland, rangeland, developed, road, tree, water, agriculture, building.
# GAMUS order (viewer/classify.py): background, ground, low_veg, buildings, water, roads, trees.
# 256-entry lookup so any stray label value (e.g. a 255 nodata) maps to "ignore" instead of crashing.
GAMUS_TO_OEM = np.full(256, 255, np.uint8)  # 255 = ignore in the loss
GAMUS_TO_OEM[:7] = [255, 0, 1, 7, 5, 3, 4]
OEM_TO_GAMUS = np.array([1, 2, 1, 5, 6, 4, 2, 3], np.uint8)  # developed->ground, agriculture->low_veg


def clean_height(agl: np.ndarray) -> np.ndarray:
    """Reference heights with sentinel/outlier values (e.g. -9999) as NaN, so
    they are excluded from the loss and the metrics instead of dominating them."""
    agl = np.asarray(agl, np.float32)
    return np.where((agl > -5.0) & (agl < 1000.0), agl, np.nan).astype(np.float32)


def window_starts(length: int, patch: int, stride: int) -> list[int]:
    """Window offsets covering [0, length), the last one flush with the end.
    A length shorter than `patch` gives [0]; the caller pads up to `patch`."""
    if length <= patch:
        return [0]
    starts = list(range(0, length - patch + 1, stride))
    if starts[-1] + patch < length:
        starts.append(length - patch)
    return starts


def feather(patch: int) -> np.ndarray:
    """Raised-cosine blend weights: positive everywhere, largest at the centre,
    so overlapping windows fade into each other instead of leaving seams."""
    t = 0.5 - 0.5 * np.cos(2 * np.pi * (np.arange(patch) + 0.5) / patch)
    return np.outer(t, t).astype(np.float32)


def normalize(rgb: np.ndarray) -> np.ndarray:
    """HxWx3 uint8 -> 3xHxW float32, RS3DAda's ImageNet-style normalisation."""
    return ((rgb.astype(np.float32) - MEAN) / STD).transpose(2, 0, 1)


def build_model(synrs3d_dir: Path):
    import torch

    sys.path.insert(0, str(synrs3d_dir))
    from models.dpt import DPT_DINOv2  # noqa: E402 -- from the cloned SynRS3D repo

    # The encoder code comes via torch.hub('facebookresearch/dinov2'), which asks GitHub for
    # the default branch on *every* load (urlopen, no timeout): on a slow network that hung
    # model loading. Once the code is cached, load it from disk instead; the first run still
    # downloads it into ~/.cache/torch/hub.
    cached = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
    if not (cached / "hubconf.py").exists():
        return DPT_DINOv2(encoder="vitl", head_configs=HEADS, pretrained=False)
    hub_load = torch.hub.load

    def load_cached(repo_or_dir, model, *args, **kwargs):
        if repo_or_dir == "facebookresearch/dinov2":
            kwargs.pop("trust_repo", None)
            kwargs["source"] = "local"
            return hub_load(str(cached), model, *args, **kwargs)
        return hub_load(repo_or_dir, model, *args, **kwargs)

    torch.hub.load = load_cached
    try:
        return DPT_DINOv2(encoder="vitl", head_configs=HEADS, pretrained=False)
    finally:
        torch.hub.load = hub_load


def load_model(ckpt: Path, synrs3d_dir: Path, device: str | None = None):
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(synrs3d_dir)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:  # our fine-tune checkpoints
        state = state["model"]
    model.load_state_dict(state)
    return model.eval().to(device), device


def _forward(model, batch, device):
    """batch: Bx3xhxw float32 tensor -> (height Bxhxw, logits Bx8xhxw), float32."""
    import torch
    import torch.nn.functional as F

    use_amp = device.startswith("cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
        out = model(batch.to(device))
    h, w = batch.shape[-2:]
    height, seg = out["regression"].float(), out["segmentation"].float()
    if height.shape[-2:] != (h, w):
        height = F.interpolate(height, (h, w), mode="bilinear", align_corners=False)
        seg = F.interpolate(seg, (h, w), mode="bilinear", align_corners=False)
    return height[:, 0], seg


def predict(model, rgb: np.ndarray, device: str, patch: int = PATCH, tta: bool = False,
            batch_size: int = 2, progress=None) -> tuple[np.ndarray, np.ndarray]:
    """HxWx3 uint8 -> (height metres float32 HxW, OEM class uint8 HxW).

    `progress`, if given, is called with the fraction of windows done (0..1].

    Sliding windows at half-patch stride, feather-blended. Height is clamped
    at 0: height above ground is non-negative by definition.
    """
    import torch

    h0, w0 = rgb.shape[:2]
    pad_h, pad_w = max(0, patch - h0), max(0, patch - w0)
    if pad_h or pad_w:
        rgb = np.pad(rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    h, w = rgb.shape[:2]
    x = torch.from_numpy(normalize(rgb))

    stride = patch // 2
    boxes = [(r, c) for r in window_starts(h, patch, stride) for c in window_starts(w, patch, stride)]
    weight = feather(patch)
    # ponytail: full-res float32 accumulators; fine up to ~5k px a side, tile the scene if uploads get larger
    acc_h = np.zeros((h, w), np.float32)
    acc_s = np.zeros((8, h, w), np.float32)
    acc_w = np.zeros((h, w), np.float32)

    flips = [(), (-1,), (-2,), (-2, -1)] if tta else [()]
    for i in range(0, len(boxes), batch_size):
        chunk = boxes[i:i + batch_size]
        batch = torch.stack([x[:, r:r + patch, c:c + patch] for r, c in chunk])
        hs, ss = 0, 0
        for dims in flips:
            b = batch.flip(dims) if dims else batch
            height, seg = _forward(model, b, device)
            if dims:  # negative dims index H/W the same way on BxHxW and BxCxHxW
                height, seg = height.flip(dims), seg.flip(dims)
            hs, ss = hs + height, ss + seg
        hs = (hs / len(flips)).cpu().numpy()
        ss = (ss / len(flips)).cpu().numpy()
        for k, (r, c) in enumerate(chunk):
            acc_h[r:r + patch, c:c + patch] += hs[k] * weight
            acc_s[:, r:r + patch, c:c + patch] += ss[k] * weight
            acc_w[r:r + patch, c:c + patch] += weight
        if progress:
            progress(min(1.0, (i + len(chunk)) / len(boxes)))

    height = np.maximum(acc_h / acc_w, 0.0)[:h0, :w0]
    classes = np.argmax(acc_s, axis=0).astype(np.uint8)[:h0, :w0]
    return height.astype(np.float32), classes
