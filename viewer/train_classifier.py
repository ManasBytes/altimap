"""Train the 7-class land-cover model used by the non-georeferenced path.

    .venv-da3/bin/python -m viewer.train_classifier

Trains directly against the raw GAMUS .h5 tiles (see viewer/gamus_dataset.py)
rather than the lossy JPEG derivatives baked into gamus-terrain/public:
exact float32 class indices instead of nearest-palette-color-decoded JPEG
labels, and native 1024x1024 resolution instead of a 2048x2048 JPEG that was
itself just an upscale of the same source with no extra real detail. Removing
that compression/upscale noise from the labels is worth more than almost any
amount of extra training time on the noisy version.

All 165 tiles (150 from GAMUS_50_each + 15 from GAMUS_extra_15) are used,
split exactly as GAMUS defines train/val/test -- this is still a small,
single-metro dataset, so: ImageNet-pretrained ResNet34 encoder (not trained
from scratch), heavy geometric + color augmentation to multiply effective
samples per epoch, label smoothing and inverse-frequency class weights
against the long-tailed class distribution (see the docstring in
viewer/gamus_dataset.py for the real per-class stats that motivated this),
and a long cosine schedule with warmup. Expect this to generalize well to
more aerial imagery at a similar nadir angle and ground sample distance, and
to need more/broader data before it can be trusted on imagery that looks
meaningfully different (oblique angle, very different GSD, rural terrain
outside the DC-metro tiles GAMUS covers).
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np

from viewer.classify import (
    CLASS_NAMES,
    DEFAULT_CHECKPOINT,
    IMAGENET_MEAN,
    IMAGENET_STD,
    TRAIN_RES,
    build_model,
)
from viewer.gamus_dataset import Tile, find_tiles, load_tile


class TileDataset:
    """Loads full tiles lazily and hands out random (train) or fixed (eval)
    crops. Kept dependency-free (no torch.utils.data.Dataset base class
    needed) since indexing + __len__ is all the DataLoader protocol needs."""

    def __init__(self, tiles: list[Tile], train: bool, res: int = TRAIN_RES):
        self.tiles = tiles
        self.train = train
        self.res = res

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, idx: int):
        rgb, label, _agl = load_tile(self.tiles[idx])
        h, w = rgb.shape[:2]
        res = self.res

        if self.train:
            top = random.randint(0, h - res)
            left = random.randint(0, w - res)
        else:
            top = (h - res) // 2
            left = (w - res) // 2
        rgb = rgb[top : top + res, left : left + res]
        label = label[top : top + res, left : left + res]

        if self.train:
            if random.random() < 0.5:
                rgb, label = rgb[:, ::-1].copy(), label[:, ::-1].copy()
            if random.random() < 0.5:
                rgb, label = rgb[::-1, :].copy(), label[::-1, :].copy()
            k = random.randint(0, 3)
            if k:
                rgb, label = np.rot90(rgb, k).copy(), np.rot90(label, k).copy()
            rgb = _color_jitter(rgb)

        x = (rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        return x.astype(np.float32), label.astype(np.int64)


def _color_jitter(rgb: np.ndarray) -> np.ndarray:
    """Brightness/contrast/saturation jitter. Aerial tiles differ in capture
    lighting/season far more than they differ in geometry, and the label map
    is invariant to all three, so this is free augmentation signal the
    geometric transforms above can't provide."""
    out = rgb.astype(np.float32)
    out *= random.uniform(0.85, 1.15)  # brightness
    mean = out.mean(axis=(0, 1), keepdims=True)
    out = mean + (out - mean) * random.uniform(0.85, 1.15)  # contrast
    gray = out.mean(axis=2, keepdims=True)
    out = gray + (out - gray) * random.uniform(0.8, 1.2)  # saturation
    return np.clip(out, 0, 255).astype(np.uint8)


def _collate(batch):
    import torch

    xs = torch.stack([torch.from_numpy(x).permute(2, 0, 1) for x, _ in batch])
    ys = torch.stack([torch.from_numpy(y) for _, y in batch])
    return xs, ys


def _class_weights(tiles: list[Tile]) -> np.ndarray:
    """Inverse-frequency weights so rare classes (water, background) aren't
    drowned out by trees/buildings in the loss -- computed from a quick
    downsampled pass over the training labels, not the full-res images."""
    from PIL import Image

    counts = np.zeros(len(CLASS_NAMES), dtype=np.float64)
    for tile in tiles:
        _, label, _agl = load_tile(tile)
        small = np.asarray(
            Image.fromarray(label).resize((256, 256), Image.NEAREST)
        )
        for c in range(len(CLASS_NAMES)):
            counts[c] += (small == c).sum()
    counts = np.maximum(counts, 1)
    weights = counts.sum() / (len(CLASS_NAMES) * counts)
    # Cap so one near-absent class (e.g. background is <0.1% of pixels)
    # doesn't dominate the loss and destabilize training on everything else.
    weights = np.clip(weights, None, 8.0)
    return weights / weights.mean()


def _confusion_update(conf: np.ndarray, pred: np.ndarray, target: np.ndarray) -> None:
    n = conf.shape[0]
    idx = target.reshape(-1) * n + pred.reshape(-1)
    conf += np.bincount(idx, minlength=n * n).reshape(n, n)


def _per_class_iou(conf: np.ndarray) -> np.ndarray:
    ious = np.full(conf.shape[0], np.nan)
    for c in range(conf.shape[0]):
        tp = conf[c, c]
        denom = conf[c, :].sum() + conf[:, c].sum() - tp
        if denom > 0:
            ious[c] = tp / denom
    return ious


def _evaluate(model, loader, device) -> np.ndarray:
    import torch

    model.eval()
    conf = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            with torch.autocast(device_type="cuda", enabled=device == "cuda"):
                pred = model(x).argmax(dim=1)
            _confusion_update(conf, pred.cpu().numpy(), y.numpy())
    return _per_class_iou(conf)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--patience", type=int, default=25, help="stop if val mIoU hasn't improved in this many epochs")
    args = parser.parse_args()

    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    checkpoint = args.checkpoint
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    all_tiles = find_tiles()
    train_tiles = [t for t in all_tiles if t.split == "train"]
    val_tiles = [t for t in all_tiles if t.split == "val"]
    test_tiles = [t for t in all_tiles if t.split == "test"]
    if not train_tiles or not val_tiles:
        raise SystemExit("No GAMUS train/val tiles found -- check viewer/gamus_dataset.py's DEFAULT_ROOTS")
    print(f"train tiles: {len(train_tiles)}, val: {len(val_tiles)}, test: {len(test_tiles)}")

    weights = _class_weights(train_tiles)
    print("class weights:", dict(zip(CLASS_NAMES, np.round(weights, 2))))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device),
        label_smoothing=0.05,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=args.warmup_epochs
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs - args.warmup_epochs)
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup, cosine], milestones=[args.warmup_epochs]
    )
    scaler = torch.amp.GradScaler(enabled=device == "cuda")

    train_loader = DataLoader(
        TileDataset(train_tiles, train=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=6,
        collate_fn=_collate,
        drop_last=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        TileDataset(val_tiles, train=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=_collate,
        persistent_workers=True,
    )

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best_miou = -1.0
    best_epoch = -1
    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=device == "cuda"):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_loader.dataset)
        scheduler.step()

        ious = _evaluate(model, val_loader, device)
        miou = float(np.nanmean(ious))
        dt = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"epoch {epoch + 1}/{args.epochs}  train_loss={train_loss:.4f}  "
            f"val_mIoU={miou:.4f}  lr={lr_now:.2e}  ({dt:.1f}s)"
        )

        if miou > best_miou:
            best_miou = miou
            best_epoch = epoch
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "val_miou": miou},
                checkpoint,
            )
            print(f"  -> saved new best checkpoint ({checkpoint}, mIoU={miou:.4f})")
        elif epoch - best_epoch >= args.patience:
            print(f"no val improvement in {args.patience} epochs, stopping early")
            break

    print(f"\ndone. best val mIoU={best_miou:.4f} (epoch {best_epoch + 1}) -> {checkpoint}")

    if test_tiles:
        test_loader = DataLoader(
            TileDataset(test_tiles, train=False),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            collate_fn=_collate,
        )
        state = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state["model"])
        test_ious = _evaluate(model, test_loader, device)
        print(f"\nheld-out test set ({len(test_tiles)} tiles), per-class IoU:")
        for name, iou in zip(CLASS_NAMES, test_ious):
            print(f"  {name:16s} {iou:.4f}" if np.isfinite(iou) else f"  {name:16s} (absent)")
        print(f"  {'mean':16s} {np.nanmean(test_ious):.4f}")


if __name__ == "__main__":
    main()
