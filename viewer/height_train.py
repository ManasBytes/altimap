"""Fine-tune RS3DAda for metric height + land cover.

    python -m viewer.height_train --hours 8 --out runs/ft            # v1: GAMUS, BitFit
    python -m viewer.height_train --ckpt runs/ft/best.pth --out runs/v2 --hours 7.5 \\
        --batch 8 --unfreeze-blocks 8 --lr 5e-5 --encoder-lr 1e-5 --height-weight 10 \\
        --syn-data synrs3d --syn-fraction 0.3 --degrade 0.5          # v2 overnight

Encoder (DINOv2 ViT-L, `pretrained.*`): biases always train (BitFit, best of five
PEFT strategies for height in arXiv:2505.06905); `--unfreeze-blocks N` also trains
the last N transformer blocks at `--encoder-lr`. Decoder and both heads train fully.

Data: GAMUS (real 0.33 m aerial, LiDAR nDSM) optionally mixed with SynRS3D
(synthetic, many high-rises and hilly terrain, 0.3-1 m) at `--syn-fraction` of
each epoch. SynRS3D tiles are resampled to GAMUS's 0.33 m from their folder's
GSD range, so both sources match what inference feeds the model. `--degrade`
adds satellite-style degradation (coarser sensor, blur, haze, noise) to that
fraction of samples, for robustness on imagery that isn't 0.33 m aerial.

The LR schedule follows wall-clock time (cosine to the --hours deadline), so it
always completes inside the budget whatever the step speed. Keeps
runs/.../best.pth by validation RMSE on real GAMUS val tiles.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np

from viewer.gamus_dataset import find_tiles, load_tile
from viewer.height_eval import DATA, evaluate, train_mean_height
from viewer.height_model import GAMUS_GSD_M, GAMUS_TO_OEM, PATCH, clean_height, load_model, normalize

# SynRS3D folder name -> ground sampling distance range (dataset README).
SYN_GSD = {"_g005_": (0.05, 0.3), "_g05_": (0.3, 0.6), "_g1_": (0.6, 1.0)}


def _resize(a: np.ndarray, size: int, nearest: bool = False) -> np.ndarray:
    from PIL import Image

    mode = Image.NEAREST if nearest else Image.BILINEAR
    if a.ndim == 3 or a.dtype == np.uint8:
        return np.asarray(Image.fromarray(a).resize((size, size), mode))
    return np.asarray(Image.fromarray(a.astype(np.float32), mode="F").resize((size, size), mode))


def degrade(rgb: np.ndarray, rng) -> np.ndarray:
    """Satellite-style degradation of an HxWx3 uint8 crop (size unchanged)."""
    from scipy import ndimage

    out = rgb.astype(np.float32)
    size = rgb.shape[0]
    if rng.random() < 0.5:  # coarser sensor (0.43-2 m, log-uniform), resampled back up to 0.33 m;
        f = float(np.exp(rng.uniform(np.log(1.3), np.log(6.0))))  # eval spans 0.35-10 m, main case 0.6 m
        small = _resize(np.clip(out, 0, 255).astype(np.uint8), max(8, int(size / f)))
        out = _resize(small, size).astype(np.float32)
    if rng.random() < 0.3:
        out = ndimage.gaussian_filter(out, sigma=(rng.uniform(0.4, 1.2),) * 2 + (0,))
    if rng.random() < 0.3:  # atmospheric haze
        a = rng.uniform(0.7, 0.95)
        out = out * a + (1 - a) * rng.uniform(170, 225, size=3)
    if rng.random() < 0.3:
        out = 255.0 * (np.clip(out, 0, 255) / 255.0) ** rng.uniform(0.7, 1.4)
    if rng.random() < 0.4:  # saturation
        grey = out.mean(axis=2, keepdims=True)
        out = grey + (out - grey) * rng.uniform(0.6, 1.3)
    if rng.random() < 0.3:
        out = out + rng.normal(0, rng.uniform(2, 6), out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def augment(rgb: np.ndarray, labels: np.ndarray, agl: np.ndarray, rng, degrade_p: float):
    """Random PATCH crop (inputs must be >= PATCH), rot90/flip, colour, optional degradation.
    Heights are rotation/flip invariant, so geometric augmentation is free."""
    h, w = agl.shape
    r, c = rng.integers(0, h - PATCH + 1), rng.integers(0, w - PATCH + 1)
    rgb, labels, agl = (a[r:r + PATCH, c:c + PATCH] for a in (rgb, labels, agl))
    k = int(rng.integers(0, 4))
    rgb, labels, agl = (np.rot90(a, k) for a in (rgb, labels, agl))
    if rng.random() < 0.5:
        rgb, labels, agl = (a[:, ::-1] for a in (rgb, labels, agl))
    gain = rng.uniform(0.9, 1.1) * rng.uniform(0.95, 1.05, size=3)
    rgb = np.clip(rgb.astype(np.float32) * gain + rng.uniform(-10, 10), 0, 255).astype(np.uint8)
    if rng.random() < degrade_p:
        rgb = degrade(np.ascontiguousarray(rgb), rng)
    return (np.ascontiguousarray(normalize(rgb)), np.ascontiguousarray(agl, dtype=np.float32),
            np.ascontiguousarray(labels).astype(np.int64))


class GamusCrops:
    def __init__(self, tiles, degrade_p: float = 0.0):
        self.tiles = tiles
        self.degrade_p = degrade_p

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, idx: int):
        rng = np.random.default_rng()
        rgb, cls, agl = load_tile(self.tiles[idx])
        return augment(rgb, GAMUS_TO_OEM[cls], clean_height(agl), rng, self.degrade_p)


def find_synrs3d(root: Path) -> list[tuple]:
    """[(rgb, ndsm, ss_mask, (gsd_lo, gsd_hi))] for every SynRS3D tile under `root`,
    whatever the zip layout (it searches for the `opt` image folders)."""
    items = []
    for opt in sorted(p for p in Path(root).rglob("opt") if p.is_dir()):
        base = opt.parent
        gsd = next((v for k, v in SYN_GSD.items() if k in f"_{base.name}_".replace("__", "_")),
                   (0.3, 0.6))
        for img in sorted(opt.glob("*.tif")):
            h, c = base / "gt_nDSM" / img.name, base / "gt_ss_mask" / img.name
            if h.exists() and c.exists():
                items.append((img, h, c, gsd))
    return items


def _read(path: Path, bands) -> np.ndarray:
    import rasterio

    with rasterio.open(path) as src:
        return src.read(bands)


class SynCrops:
    """SynRS3D tiles resampled to 0.33 m (from their folder's GSD range) and padded to PATCH."""

    def __init__(self, items, degrade_p: float = 0.0):
        self.items = items
        self.degrade_p = degrade_p

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        rng = np.random.default_rng()
        img, hp, cp, (g_lo, g_hi) = self.items[idx]
        rgb = np.transpose(_read(img, [1, 2, 3]), (1, 2, 0))
        if rgb.dtype != np.uint8:
            a = rgb.astype(np.float32)
            lo, hi = np.percentile(a, [1, 99])
            rgb = np.clip((a - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
        agl = clean_height(_read(hp, 1))
        cls = _read(cp, 1)
        labels = np.where((cls >= 1) & (cls <= 8), cls.astype(np.int64) - 1, 255).astype(np.uint8)

        size = int(round(rgb.shape[0] * rng.uniform(g_lo, g_hi) / GAMUS_GSD_M))
        size = max(64, min(size, 2048))
        rgb, labels, agl = _resize(rgb, size), _resize(labels, size, nearest=True), _resize(agl, size)
        if size < PATCH:
            pad = PATCH - size
            rgb = np.pad(rgb, ((0, pad), (0, pad), (0, 0)), mode="reflect")
            labels = np.pad(labels, ((0, pad), (0, pad)), constant_values=255)
            agl = np.pad(agl, ((0, pad), (0, pad)), constant_values=np.nan)
        return augment(rgb, labels, agl, rng, self.degrade_p)


def losses(height, logits, ref, labels, height_weight_m: float = 0.0):
    import torch
    import torch.nn.functional as F

    def masked_mean(x, m):  # an all-masked batch gives 0, never NaN (NaN grads would poison the weights)
        m = m.float()
        return (x * m).sum() / m.sum().clamp(min=1.0)

    valid = torch.isfinite(ref)
    d = height - torch.nan_to_num(ref)
    if height_weight_m > 0:  # long tail: a pixel at h metres counts (1 + h/height_weight_m) times
        w = valid.float() * (1.0 + torch.nan_to_num(ref).clamp(min=0) / height_weight_m)
        l1 = (d.abs() * w).sum() / w.sum().clamp(min=1.0)
    else:
        l1 = masked_mean(d.abs(), valid)
    gx = masked_mean((d[:, :, 1:] - d[:, :, :-1]).abs(), valid[:, :, 1:] & valid[:, :, :-1])
    gy = masked_mean((d[:, 1:, :] - d[:, :-1, :]).abs(), valid[:, 1:, :] & valid[:, :-1, :])
    if (labels != 255).any():
        ce = F.cross_entropy(logits, labels, ignore_index=255)
    else:
        ce = logits.sum() * 0.0
    return l1, gx + gy, ce


def block_index(name: str) -> int | None:
    """Transformer block number of an encoder parameter ('pretrained.blocks.5.attn...' or,
    with chunked blocks, 'pretrained.blocks.0.5.attn...'), else None."""
    parts = name.split(".")
    if len(parts) < 4 or parts[:2] != ["pretrained", "blocks"]:
        return None
    return int(parts[3]) if parts[3].isdigit() else int(parts[2])


def trainable_groups(model, unfreeze_blocks: int = 0):
    """Freeze the encoder except biases and the last `unfreeze_blocks` blocks (+ final norm).
    -> (encoder params, decoder/head params)."""
    names = [n for n, _ in model.named_parameters()]
    if not any(n.startswith("pretrained.") for n in names):
        children = [n for n, _ in model.named_children()]
        raise RuntimeError(f"no 'pretrained.' encoder params; top-level modules are {children}")
    indices = [i for i in map(block_index, names) if i is not None]
    first_open = (max(indices) + 1 - unfreeze_blocks) if indices and unfreeze_blocks > 0 else 10**9
    encoder, rest = [], []
    for n, p in model.named_parameters():
        if not n.startswith("pretrained."):
            p.requires_grad = True
            rest.append(p)
            continue
        idx = block_index(n)
        p.requires_grad = (n.endswith(".bias")
                           or (idx is not None and idx >= first_open)
                           or (unfreeze_blocks > 0 and n.startswith("pretrained.norm.")))
        if p.requires_grad:
            encoder.append(p)
    return encoder, rest


def bitfit(model) -> int:
    encoder, rest = trainable_groups(model, 0)
    return sum(p.numel() for p in encoder + rest)


def main() -> None:
    import torch
    from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

    ap = argparse.ArgumentParser()
    ap.add_argument("--gamus", type=Path, default=DATA / "gamus")
    ap.add_argument("--synrs3d", type=Path, default=DATA / "SynRS3D", help="SynRS3D code repo (model class)")
    ap.add_argument("--ckpt", type=Path, default=DATA / "SynRS3D/pretrain/RS3DAda_vitl_DPT_height.pth")
    ap.add_argument("--out", type=Path, default=Path("runs/ft"))
    ap.add_argument("--hours", type=float, default=8.0)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4, help="decoder, heads and encoder biases")
    ap.add_argument("--encoder-lr", type=float, default=1e-5, help="unfrozen encoder blocks")
    ap.add_argument("--unfreeze-blocks", type=int, default=0)
    ap.add_argument("--height-weight", type=float, default=0.0,
                    help="metres; >0 weights each pixel's L1 by 1 + height/this (0 = plain L1)")
    ap.add_argument("--syn-data", type=Path, default=None, help="extracted SynRS3D tiles")
    ap.add_argument("--syn-fraction", type=float, default=0.3)
    ap.add_argument("--degrade", type=float, default=0.0, help="fraction of samples degraded")
    ap.add_argument("--val-every", type=int, default=1500)
    ap.add_argument("--val-tiles", type=int, default=150)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    deadline = t_start + args.hours * 3600

    tiles = find_tiles((args.gamus,))
    train = [t for t in tiles if t.split == "train"]
    val = sorted((t for t in tiles if t.split == "val"), key=lambda t: t.scene_id)
    val = random.Random(0).sample(val, min(args.val_tiles, len(val)))
    mean_h = train_mean_height(tiles)
    syn = find_synrs3d(args.syn_data) if args.syn_data else []
    print(f"{len(train)} GAMUS train tiles, {len(syn)} SynRS3D tiles, {len(val)} val tiles", flush=True)
    if not train or not val:
        raise SystemExit(f"need train and val tiles under {args.gamus}")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    model, device = load_model(args.ckpt, args.synrs3d)
    encoder, rest = trainable_groups(model, args.unfreeze_blocks)
    print(f"trainable params: encoder {sum(p.numel() for p in encoder):,} "
          f"(last {args.unfreeze_blocks} blocks + biases), decoder/heads {sum(p.numel() for p in rest):,}",
          flush=True)
    enc_bias = [p for p in encoder if p.ndim == 1]
    enc_weights = [p for p in encoder if p.ndim > 1]
    groups = [{"params": rest + enc_bias, "lr": args.lr}]
    if enc_weights:
        groups.append({"params": enc_weights, "lr": args.encoder_lr})
    params = rest + encoder
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    warmup = 500

    def lr_at(step: int) -> float:  # warmup by steps, then cosine over wall-clock time
        if step < warmup:
            return (step + 1) / warmup
        t = min(1.0, (time.time() - t_start) / (0.97 * (deadline - t_start)))
        return 0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * t))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    datasets = [GamusCrops(train, args.degrade)]
    weights = [np.full(len(train), 1.0 / len(train))]
    if syn:
        f = args.syn_fraction
        datasets.append(SynCrops(syn, args.degrade))
        weights = [weights[0] * (1 - f), np.full(len(syn), f / len(syn))]
    data = ConcatDataset(datasets)
    epoch = len(train) + (len(syn) if syn else 0)
    sampler = WeightedRandomSampler(np.concatenate(weights), num_samples=epoch, replacement=True)
    loader = DataLoader(data, batch_size=args.batch, sampler=sampler, drop_last=True,
                        num_workers=args.workers, pin_memory=True,
                        persistent_workers=args.workers > 0)

    log = (args.out / "log.jsonl").open("a")
    # Survives restarts: rerunning with --ckpt runs/.../last.pth must not let a
    # worse first validation overwrite an earlier, better best.pth.
    best_file = args.out / "best.json"
    best = json.loads(best_file.read_text())["rmse"] if best_file.exists() else float("inf")
    step, t0 = 0, time.time()
    model.train()
    while time.time() < deadline:
        for x, ref, labels in loader:
            x, ref, labels = (a.to(device, non_blocking=True) for a in (x, ref, labels))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(x)
            height, logits = out["regression"].float()[:, 0], out["segmentation"].float()
            l1, grad, ce = losses(height, logits, ref, labels, args.height_weight)
            loss = l1 + 0.5 * grad + 0.2 * ce
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            step += 1

            if step == 200:
                print(f"{(time.time() - t0) / 200:.3f} s/step, "
                      f"GPU peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)
            if step % 50 == 0:
                rec = {"step": step, "loss": loss.item(), "l1": l1.item(), "grad": grad.item(),
                       "ce": ce.item(), "lr": sched.get_last_lr()[0]}
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n")
            last = time.time() >= deadline
            if step % args.val_every == 0 or last:
                model.eval()
                res = evaluate(model, device, val, mean_h)["scores"]["ALL"]["model"]
                model.train()
                rec = {"step": step, "val": res}
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n")
                log.flush()
                state = {"model": model.state_dict(), "step": step, "val": res}
                torch.save(state, args.out / "last.pth")
                if res["rmse"] < best:
                    best = res["rmse"]
                    torch.save(state, args.out / "best.pth")
                    best_file.write_text(json.dumps({"rmse": best, "step": step, "val": res}))
                    print(f"new best val RMSE {best:.3f} m at step {step}", flush=True)
            if last:
                break
    print(f"done: {step} steps, best val RMSE {best:.3f} m", flush=True)


if __name__ == "__main__":
    main()
