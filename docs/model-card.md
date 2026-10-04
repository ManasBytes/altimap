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
[SETUP.md](../SETUP.md).

The viewer can additionally use OSM outlines, building parts and height/roof tags for its
Map-assisted city model. Those mapped dimensions and floor/roof heuristics are separate from
this learned model: they do not change nDSM/DSM GeoTIFFs or the evaluation metrics below.

## Model

- **Architecture:** RS3DAda: DINOv2 ViT-L/14 encoder + DPT decoder with a height-regression head
  and a segmentation head ([SynRS3D, NeurIPS 2024](https://arxiv.org/abs/2406.18151)). Needs the
  model code from [JTRNEO/SynRS3D](https://github.com/JTRNEO/SynRS3D).
- **Starting point:** the authors' [RS3DAda height weights](https://huggingface.co/JTRNEO/RS3DAda).
- **Fine-tuning (`best.pth`, v1):** on [GAMUS](https://huggingface.co/datasets/earthflow/GAMUS)
  (0.33 m aerial RGB with airborne-LiDAR nDSM and land-cover labels over Washington DC,
  Philadelphia and New York), BitFit (encoder biases) + decoder and heads, loss L1 + 0.5 x gradient
  L1 + 0.2 x cross-entropy, 518 px crops, bf16.
- **v2 (`v2/best.pth`):** continues from v1 with the last 8 encoder blocks unfrozen, a
  height-weighted L1 (a pixel at h metres counts 1 + h/10 times), 30 % SynRS3D samples (high-rises,
  hills) and 50 % satellite-style degradation (1.3–6× coarser, blur, haze). 6.5 h on an RTX A5000,
  best checkpoint at step 25,500.

## Results (GAMUS test split, 500 tiles, 4-flip test-time averaging)

| City | RMSE (m) v1 | RMSE (m) v2 | RMSE (m) zero-shot RS3DAda | Pearson r (v1 / v2) | Building RMSE (m) (v1 / v2) |
|---|---|---|---|---|---|
| Washington DC | 3.95 | 3.86 | 10.08 | 0.915 / 0.921 | 4.51 / 4.45 |
| New York | 3.95 | 4.09 | 7.17 | 0.855 / 0.859 | 3.20 / 2.95 |
| Philadelphia | 5.90 | 5.02 | 6.28 | 0.822 / 0.863 | 10.04 / 8.04 |
| **All** | **5.02** | **4.55** | **7.25** | **0.833 / 0.864** | **8.28 / 6.74** |

## How AltiMap uses them

v2 wins on GAMUS but generalises worse to other imagery: against USGS airborne LiDAR on NAIP
scenes it over-reads trees and ground. On buildings, though, it is right where v1 isn't. So the app
combines three models by land cover:
- **v1** gives heights everywhere.
- **Building pixels** use 25% v1 + 75% v2. This fixed blend was selected on all 859 GAMUS
  validation tiles, confirmed once on all 2,861 untouched test tiles, and retained under a focused
  24-tile four-flip TTA check. Against the former 50/50 route, corrected validation overall/building
  RMSE changed 2.7511/3.3352 → 2.7402/3.2763 m; untouched test changed
  3.7573/5.2308 → 3.6972/4.9934 m. The TTA subset changed
  3.1690/5.1284 → 3.0916/4.9140 m, with only a 0.04% ordinary-scene overall regression.
- **Tree pixels in extensive forest** (≥ 80 % canopy within 150 m) come from Meta's CHMv2 canopy
  model.


## Limitations (measured, not guessed)

- **Out of domain, by landscape** (NAIP RGB vs USGS 3DEP, single pass): the current
  25/75 building-fusion pipeline was rerun on 2026-10-04. All-valid-pixel RMSE is in metres;
  parentheses show the corresponding zero-height or DEM-alone baseline.

  | Scene | nDSM RMSE (predict 0) | SRTM DSM RMSE (DEM alone) | GLO-30 DSM RMSE (DEM alone) |
  |---|---|---|---|
  | Dense city (0.3 m) | 31.58 (42.35) | 38.87 (40.56) | 34.36 (36.26) |
  | Suburb (0.6 m) | 4.47 (6.48) | 4.28 (4.71) | 3.62 (4.02) |
  | Hilly town (0.6 m) | 3.55 (6.48) | 6.24 (7.26) | 4.75 (5.53) |
  | Forest (0.6 m) | 15.67 (24.66) | 7.44 (7.13) | 8.75 (8.53) |

  The model improves DSM RMSE over both base DEMs in city/suburb/hills; forest remains
  slightly worse than DEM alone. Canopy bias is −10.75 m. A historical tower percentile
  was 104 m against 151 m LiDAR; the current evaluator does not remeasure that percentile.
  Full MAE, correlation, scope and datum notes are in the
  [current report](evaluation/landscapes-2026-10-04.md).
- Object heights hold up to about 1–2 m pixels and fade to flat by 5–10 m in the historical
  resolution sweep. Cartosat-2S imagery over India is outside the US aerial training domain;
  performance there needs independent reference validation.
- Model heights are **above ground**. The pipeline uses SRTM by default or Copernicus GLO-30
  for georeferenced absolute elevation, using EGM96 or EGM2008 respectively.

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
