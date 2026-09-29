"""Supervised RGB land cover + AGL estimation, tiled at native pixel spacing.

Heights are estimates learned from GAMUS, not georeferenced measurements.
The training split and evaluation provenance travel with the checkpoint.
"""
from pathlib import Path
import numpy as np

from viewer.classify import CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD, build_model

CHECKPOINT = Path(__file__).resolve().parent / "cache/surface_best.pt"


def load_surface(checkpoint=CHECKPOINT, device=None):
    import torch
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = build_model(pretrained=False, with_height=True)
    model.load_state_dict(state["model"])
    return model.to(device).eval(), device, state.get("metrics", {})


def origins(size, tile, stride):
    return sorted(set([*range(0, max(1, size - tile + 1), stride), max(0, size - tile)]))


def infer_tiled(rgb, model, device, tile=512, overlap=128, with_height=True):
    """Blend probabilities (not class IDs) with a raised cosine window.

    Only one patch resides on the GPU. Accumulators are CPU float32. Native
    resolution and rectangular aspect ratios are retained, including tiny inputs.
    """
    import torch
    if tile < 32 or tile % 32 or overlap < 0 or overlap >= tile:
        raise ValueError("tile must be a multiple of 32; 0 <= overlap < tile")
    if rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) == 0:
        raise ValueError("expected a nonempty H×W×3 RGB image")
    h, w = rgb.shape[:2]
    padded = np.pad(rgb, ((0, max(0, tile-h)), (0, max(0, tile-w)), (0, 0)), mode="edge")
    hh, ww = padded.shape[:2]
    probs = np.zeros((len(CLASS_NAMES), hh, ww), np.float32)
    heights = np.zeros((hh, ww), np.float32)
    weights = np.zeros((hh, ww), np.float32)
    window = np.maximum(np.outer(np.hanning(tile), np.hanning(tile)), 0.01).astype(np.float32)
    model.eval()
    with torch.inference_mode():
        for y in origins(hh, tile, tile-overlap):
            for x in origins(ww, tile, tile-overlap):
                patch = (padded[y:y+tile, x:x+tile].astype(np.float32)/255 - IMAGENET_MEAN)/IMAGENET_STD
                tensor = torch.from_numpy(patch.astype(np.float32)).permute(2,0,1)[None].to(device)
                with torch.autocast(device_type="cuda", enabled=str(device).startswith("cuda")):
                    if with_height:
                        logits, log_height = model.surface(tensor)
                    else:
                        logits = model(tensor)
                probs[:, y:y+tile, x:x+tile] += logits.float().softmax(1)[0].cpu().numpy() * window
                if with_height:
                    heights[y:y+tile, x:x+tile] += torch.expm1(log_height.float().clamp(0, np.log1p(150)))[0].cpu().numpy() * window
                weights[y:y+tile, x:x+tile] += window
    probs = probs[:, :h, :w] / weights[None, :h, :w]
    return probs.argmax(0).astype(np.uint8), heights[:h,:w]/weights[:h,:w], probs.max(0)


def regularize_surface(height, labels):
    """Suppress isolated height noise; stabilize each connected roof locally.

    This visualization product retains the raw prediction separately. Class
    labels never become fixed elevation offsets. No filtering is applied to
    measured GeoTIFFs through this function.
    """
    from scipy.ndimage import median_filter, label, find_objects
    out = median_filter(np.clip(np.nan_to_num(height, nan=0, posinf=150, neginf=0), 0, 150), size=3)
    components, _ = label(labels == 3)
    for index, bounds in enumerate(find_objects(components), 1):
        if bounds is None:
            continue
        mask = components[bounds] == index
        if mask.sum() < 16:
            continue
        region = out[bounds]
        roof = region[mask]
        lo, mid, hi = np.percentile(roof, [10, 50, 90])
        region[mask] = mid + 0.35 * (np.clip(roof, lo, hi) - mid)
    return out.astype(np.float32)
