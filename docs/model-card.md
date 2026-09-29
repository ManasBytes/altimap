---
license: mit
base_model: JTRNEO/RS3DAda
datasets:
  - earthflow/GAMUS
  - JTRNEO/SynRS3D
tags:
  - remote-sensing
  - height-estimation
  - ndsm
  - dsm
  - monocular-depth
  - aerial-imagery
  - satellite-imagery
pipeline_tag: depth-estimation
---

# AltiMap height model

Single-image height estimation for aerial and satellite RGB imagery: for every pixel, the
**height above ground in metres (nDSM)** plus an 8-class land-cover map. Built for Smart India
Hackathon 2026, problem statement 26175 (ISRO, "DepthWizard"), as the model behind
[AltiMap](https://github.com/ManasBytes/altimap), which turns one image into metric elevation
GeoTIFFs and a 3D city model. Setup and usage: the repo's
[SETUP.md](https://github.com/ManasBytes/altimap/blob/dilavesh-new/SETUP.md).

## Model

- **Architecture:** RS3DAda: DINOv2 ViT-L/14 encoder + DPT decoder with a height-regression head
  and a segmentation head ([SynRS3D, NeurIPS 2024](https://arxiv.org/abs/2406.18151)). Needs the
  model code from [JTRNEO/SynRS3D](https://github.com/JTRNEO/SynRS3D).
- **Starting point:** the authors' [RS3DAda height weights](https://huggingface.co/JTRNEO/RS3DAda).
- **Fine-tuning (`best.pth`, v1):** on [GAMUS](https://huggingface.co/datasets/earthflow/GAMUS)
  (0.33 m aerial RGB with airborne-LiDAR nDSM and land-cover labels over Washington DC,
  Philadelphia and New York), BitFit (encoder biases) + decoder and heads, loss L1 + 0.5 x gradient
  L1 + 0.2 x cross-entropy, 518 px crops, bf16.
- **v2 (`v2/`, when present):** continues from v1 with the last 8 encoder blocks unfrozen, a
  height-weighted L1 (to fight tall-building underestimation), 30 % SynRS3D samples (high-rise and
  hilly scenes) and blur/haze/downsampling augmentation for coarser satellite imagery.

## Results (GAMUS test split, 500 tiles, 4-flip test-time averaging)

| City | RMSE (m) fine-tuned | RMSE (m) zero-shot RS3DAda | Pearson r | Building RMSE (m) |
|---|---|---|---|---|
| Washington DC | 3.95 | 10.08 | 0.915 | 4.51 |
| New York | 3.95 | 7.17 | 0.855 | 3.20 |
| Philadelphia | 5.90 | 6.28 | 0.822 | 10.04 |
| **All** | **5.02** | **7.25** | **0.833** | **8.28** |

## Limitations (measured, not guessed)

- **Tall objects are underestimated**, e.g. ~72 m predicted vs 167 m real at Philadelphia City
  Hall; the tallest object in a GAMUS tile at 28 m vs 38 m.
- **Coarser imagery reads low:** on 0.6 m NAIP scenes (resampled to the 0.33 m training
  resolution) the nDSM bias against USGS 3DEP LiDAR was -4 m (town), -6 m (leafy suburb) and
  -16 m (forest canopy). ISRO's evaluation imagery is 0.6 m Cartosat-2S, outside the training
  domain (US cities, aerial).
- Heights are **above ground**. Absolute elevation needs a terrain model; AltiMap adds Copernicus
  GLO-30 ground for georeferenced inputs.

## Usage

```python
from huggingface_hub import hf_hub_download
ckpt = hf_hub_download("Dilavesh/altimap-height", "best.pth")  # public, no login needed

# with the AltiMap repo checked out and SynRS3D cloned into viewer/cache/SynRS3D:
from pathlib import Path
from viewer.height_model import load_model, predict
model, device = load_model(Path(ckpt), Path("viewer/cache/SynRS3D"))
ndsm_m, oem_classes = predict(model, rgb_uint8_hxwx3, device, tta=True)  # input at ~0.33 m/px
```

`code/` holds the training and data scripts used for these weights.

## License and attribution

Weights: MIT, like the RS3DAda base model. Please credit:

- **RS3DAda / SynRS3D**: Song et al., *SynRS3D: A Synthetic Dataset for Global 3D Semantic
  Understanding from Monocular Remote Sensing Imagery*, NeurIPS 2024 (code and base weights MIT).
- **GAMUS** (CC BY 4.0): *GAMUS: A Geometry-aware Multi-modal Semantic Segmentation Benchmark for
  Remote Sensing Data*, [arXiv:2305.14914](https://arxiv.org/abs/2305.14914).
- **DINOv2** (Apache 2.0), Oquab et al., Meta AI.

The v2 weights were also trained on SynRS3D data, which is CC BY-NC 4.0: treat v2 as
non-commercial.
