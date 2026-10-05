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

This model estimates **height above ground (nDSM)** from one aerial or satellite colour image.
It also predicts land-cover labels, such as buildings and trees, using eight internal classes.
AltiMap maps these to the seven GAMUS categories shown in the viewer. Built for Smart India
Hackathon 2026, problem statement 26175 (ISRO, "DepthWizard"), as the model behind
[AltiMap](https://github.com/ManasBytes/altimap), which turns one image into metric elevation
GeoTIFFs and a 3D city model. Setup and usage: the repo's
[SETUP.md](https://github.com/ManasBytes/altimap/blob/main/SETUP.md).

This file describes the model weights and their evaluation. The complete application also builds
the city geometry and combines GeoTIFF predictions with public elevation data.

## Model

- **Architecture:** RS3DAda uses DINOv2 to read image features and DPT to turn them into height
  and class maps ([SynRS3D, NeurIPS 2024](https://arxiv.org/abs/2406.18151)). It needs the
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

## Current application results

The current building blend is **25% v1 + 75% v2**. It was selected using all 859 validation tiles
and confirmed using 2,861 test tiles. These are one-pass above-ground-height results.

| Split | Previous 50/50 overall RMSE | Current 25/75 overall RMSE | Previous building RMSE | Current building RMSE |
|---|---:|---:|---:|---:|
| Validation | 2.7511 m | 2.7402 m | 3.3352 m | 3.2763 m |
| Test | 3.7573 m | 3.6972 m | 5.2308 m | 4.9934 m |

The test improvement is 1.60% overall and 4.54% for buildings.
See the [evaluation evidence](evidence/README.md) for the reports and a separate 24-tile High/TTA check.
MAE is the average size of the error. RMSE gives larger mistakes more weight.
Building RMSE applies the same calculation only to building pixels.

## Older model comparison: 500 GAMUS test tiles

The table below is the earlier four-flip test-time-averaging experiment.
It compares the separate v1 and v2 models, not the current combined pipeline.
The images and inference settings differ from the full-split results above.
These historical results were not rerun during the documentation update.

| City | RMSE (m) v1 | RMSE (m) v2 | RMSE (m) zero-shot RS3DAda | Pearson r (v1 / v2) | Building RMSE (m) (v1 / v2) |
|---|---|---|---|---|---|
| Washington DC | 3.95 | 3.86 | 10.08 | 0.915 / 0.921 | 4.51 / 4.45 |
| New York | 3.95 | 4.09 | 7.17 | 0.855 / 0.859 | 3.20 / 2.95 |
| Philadelphia | 5.90 | 5.02 | 6.28 | 0.822 / 0.863 | 10.04 / 8.04 |
| **All** | **5.02** | **4.55** | **7.25** | **0.833 / 0.864** | **8.28 / 6.74** |

## How AltiMap uses them

v2 performed better in the older GAMUS experiment, but can predict trees and ground too high
on other imagery. The application therefore chooses models using predicted land-cover labels:

- **v1** gives heights everywhere.
- **Building pixels** use 25% v1 + 75% v2. This fixed blend was selected on all 859 GAMUS
  validation tiles, confirmed once on all 2,861 untouched test tiles, and retained under a focused
  24-tile four-flip TTA check. Against the former 50/50 route, corrected validation overall/building
  RMSE changed 2.7511/3.3352 → 2.7402/3.2763 m; untouched test changed
  3.7573/5.2308 → 3.6972/4.9934 m. The TTA subset changed
  3.1690/5.1284 → 3.0916/4.9140 m, with only a 0.04% ordinary-scene overall regression.
- **Tree pixels in extensive forest** (at least 80% canopy within about 150 m) use Meta's CHMv2
  when it is installed. This is an optional tree-height model.

Without a specialist, the application keeps v1. The result names the models actually used.
CHMv2 was absent from the dated Windows real-upload checks.


## Limits and older measurements

- **Out of domain, by landscape** (NAIP aerial RGB vs USGS 3DEP airborne LiDAR, one pass). This
  table is the historical external run with the former 50/50 building blend; v1 alone is in
  brackets. It remains useful domain-gap evidence but has not been rerun for the current 25/75
  blend. Absolute DSM uses public elevation with predicted fine detail. Its local moving-average
  method does not guarantee exact agreement on native Copernicus cells:

  | Scene | nDSM RMSE (predict 0) | nDSM bias | DSM RMSE (GLO-30 alone) |
  |---|---|---|---|
  | Dense city (0.3 m) | 30.1 m (42.4) [29.8] | −1.2 m [−6.8] | 35.0 m (36.3) [34.7] |
  | Suburb (0.6 m) | 4.5 m (6.5) [4.5] | +1.4 m [+1.3] | 3.97 m (4.02) [3.99] |
  | Hilly town (0.6 m) | 3.5 m (6.5) [3.7] | −0.4 m [−0.7] | 5.02 m (5.53) [5.04] |
  | Forest (0.6 m) | 15.7 m (24.7) [18.5] | −10.8 m [−15.9] | 8.75 m (8.53) [9.03] |

  In that historical run, very tall towers still read low (104 m vs 151 m), and forest canopy read
  ~11 m low. The 75% route is expected to help towers but that exact external scene has not been
  rerun, so no replacement tower number is claimed. Object heights hold up to ~1–2 m pixels and
  fade to flat by 5–10 m. ISRO's evaluation imagery is 0.6 m Cartosat-2S over India, outside the
  training domain (aerial imagery of US cities).
- Heights are **above ground**. Absolute surface elevation also needs a public elevation source.
  AltiMap supports Copernicus or SRTM for suitable GeoTIFF inputs. Copernicus is a surface dataset,
  not a surveyed bare-ground DTM.
- PNG/JPG images without known pixel size use an experimental 0.33 m/pixel assumption.
  Displayed metre heights and footprint sizes are not independently verified in that case.
- A public DEM used to create a DSM is not an independent reference for building-height accuracy.

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
