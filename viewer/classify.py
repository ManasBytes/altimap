"""RGB -> 7-class land-cover map, and a static per-class height field.

For non-georeferenced imagery there is no DEM or Ground Control Point to
calibrate a depth model against, so this path skips depth estimation
entirely: a small U-Net predicts one of 7 land-cover classes per pixel, and
each class maps to a *fixed* relative height. The result is deliberately a
flat-topped "layer cake" relief, not a real surface -- it exists to give a
non-georeferenced upload an immediate, structurally-plausible 3D preview
before a depth model is wired in. Once DA3 (or similar) is added for this
path, its per-pixel output can be calibrated against these same static
values class-by-class (e.g. "DA3 says the median building pixel sits at
relative depth X; static tables say buildings are ~0.62 of the scene's
height range" -> that ratio becomes the exaggeration factor) rather than
trusting DA3's absolute scale directly, which is exactly the domain-gap risk
the project brief calls out for monocular depth on remote-sensing imagery.

The class palette mirrors CLASS_PALETTE in gamus-terrain's src/main.jsx
exactly, so a model trained here decodes the same *-classes.jpg labels the
frontend already renders, and its predictions are visually interchangeable
with them -- swap one for the other and nothing downstream needs to change.

Heavy imports (torch, torchvision) are lazy so this module stays importable
-- and CLASS_PALETTE/rgb_to_classes/classes_to_static_height stay testable --
without a GPU or ML deps installed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

CLASS_NAMES = [
    "background",
    "ground",
    "low_vegetation",
    "buildings",
    "water",
    "roads",
    "trees",
]

# Order and values match CLASS_PALETTE in gamus-terrain/src/main.jsx.
CLASS_PALETTE = np.array(
    [
        [31, 43, 61],  # background
        [145, 116, 76],  # ground
        [139, 205, 91],  # low vegetation
        [235, 143, 57],  # buildings
        [50, 157, 214],  # water
        [142, 151, 163],  # roads
        [47, 116, 81],  # trees
    ],
    dtype=np.float64,
)

BUILDING_CLASS_INDEX = CLASS_NAMES.index("buildings")

# Static relative height per class, in the same normalized units as the
# frontend's height field (see TERRAIN_BASELINE / HEIGHT_WORLD_SCALE in
# main.jsx) -- NOT metres. These are placeholders standing in for a depth
# model until one is calibrated for this path; buildings and trees sit well
# above ground/roads, water sits at the floor, matching the class ordering
# CLASS_HEIGHT_BASE/CAP already use for the georeferenced/DA3 path.
STATIC_CLASS_HEIGHT = {
    "background": 0.05,
    "ground": 0.08,
    "low_vegetation": 0.18,
    "buildings": 0.62,
    "water": 0.0,
    "roads": 0.06,
    "trees": 0.42,
}
STATIC_HEIGHT_TABLE = np.array(
    [STATIC_CLASS_HEIGHT[name] for name in CLASS_NAMES], dtype=np.float64
)

DEFAULT_CHECKPOINT = Path("viewer/cache/classifier_resnet34unet.pt")
TRAIN_RES = 512  # matches the frontend's 513-sample interactive grid


def rgb_to_classes(rgb: np.ndarray) -> np.ndarray:
    """Nearest-palette-color decode of a *-classes.jpg label image.

    JPEG compression drifts a color by a few values per channel, so nearest
    match (not exact equality) is required -- same reasoning as the decoder
    in main.jsx's buildClassField.
    """
    flat = rgb.reshape(-1, 3).astype(np.float64)
    dists = ((flat[:, None, :] - CLASS_PALETTE[None, :, :]) ** 2).sum(axis=2)
    return dists.argmin(axis=1).reshape(rgb.shape[:2]).astype(np.uint8)


def classes_to_rgb(class_map: np.ndarray) -> np.ndarray:
    """Class index grid -> palette RGB, for visualizing a prediction."""
    return CLASS_PALETTE[class_map].astype(np.uint8)


def classes_to_static_height(class_map: np.ndarray) -> np.ndarray:
    """Class index grid -> static per-class height field. No depth model."""
    return STATIC_HEIGHT_TABLE[class_map]


def build_model():
    """ResNet34-encoder U-Net, 7-class output.

    Written by hand rather than pulling in segmentation-models-pytorch: the
    encoder is just torchvision's resnet34 split at its four stride points,
    and the decoder is four matching upsample+concat+conv blocks -- small
    enough to keep as one file, and one fewer dependency to pin. ResNet34
    uses the same per-stage channel widths as ResNet18 (64/128/256/512), just
    more BasicBlocks per stage, so the decoder below needs no changes to use
    the deeper encoder -- more capacity for the fuller, cleaner (exact-label,
    no-JPEG-noise) training set without touching the U-Net's shape.
    """
    import torch
    from torch import nn
    from torchvision.models import resnet34, ResNet34_Weights

    class DecoderBlock(nn.Module):
        def __init__(self, in_ch, skip_ch, out_ch):
            super().__init__()
            self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
            self.conv = nn.Sequential(
                nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        def forward(self, x, skip):
            x = self.up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="nearest")
            return self.conv(torch.cat([x, skip], dim=1))

    class ResNet34UNet(nn.Module):
        def __init__(self, num_classes: int):
            super().__init__()
            backbone = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
            self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
            self.pool = backbone.maxpool
            self.layer1 = backbone.layer1  # 64ch,  /4
            self.layer2 = backbone.layer2  # 128ch, /8
            self.layer3 = backbone.layer3  # 256ch, /16
            self.layer4 = backbone.layer4  # 512ch, /32
            self.dec4 = DecoderBlock(512, 256, 256)
            self.dec3 = DecoderBlock(256, 128, 128)
            self.dec2 = DecoderBlock(128, 64, 64)
            self.dec1 = DecoderBlock(64, 64, 32)
            self.head = nn.Sequential(
                nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),
                nn.Conv2d(32, num_classes, 1),
            )

        def forward(self, x):
            s0 = self.stem(x)  # /2, 64ch
            p0 = self.pool(s0)  # /4
            s1 = self.layer1(p0)  # /4, 64ch
            s2 = self.layer2(s1)  # /8, 128ch
            s3 = self.layer3(s2)  # /16, 256ch
            s4 = self.layer4(s3)  # /32, 512ch
            d4 = self.dec4(s4, s3)
            d3 = self.dec3(d4, s2)
            d2 = self.dec2(d3, s1)
            d1 = self.dec1(d2, s0)
            return self.head(d1)

    return ResNet34UNet(len(CLASS_NAMES))


def load_model(checkpoint: Path = DEFAULT_CHECKPOINT, device: str | None = None):
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, device


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def predict_classes(rgb: np.ndarray, model, device: str) -> np.ndarray:
    """Uint8 HxWx3 RGB (any size) -> class-index HxW array at that same size."""
    import torch
    import torch.nn.functional as F

    h, w = rgb.shape[:2]
    x = rgb.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    x = torch.from_numpy(x.astype(np.float32)).permute(2, 0, 1)[None].to(device)
    # Encoder halves resolution 5 times (stem, pool, layer2, layer3, layer4),
    # so both dims must be multiples of 32 or the decoder's skip-concats at
    # each stage land on mismatched shapes.
    pad_h = (32 - h % 32) % 32
    pad_w = (32 - w % 32) % 32
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    with torch.no_grad():
        logits = model(x)[:, :, :h, :w]
    return logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
