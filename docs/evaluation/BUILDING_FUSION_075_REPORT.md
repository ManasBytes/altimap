# Building fusion 0.75 adoption report

In simple words, we changed how the two models are combined on building pixels.
The old result used half from v1 and half from v2. The new result uses 25% from v1 and 75% from v2.
This report records the tests behind that decision. It does not describe new model training.

Date: 2026-10-03<br>
Scope: GAMUS building-height routing only<br>
Decision: adopt `25% v1 + 75% v2` on predicted building pixels

Archive note: this is the 3 October adoption record. Public copies of the small result artifacts
are linked in [the evidence index](../evidence/README.md); the local paths below document the
original run. Later upload/API verification is reported [separately](../real-upload-verification-20261004.md).

## What changed

AltiMap previously averaged its two height estimates on building pixels:

```text
old building height = 0.50 * v1 + 0.50 * v2
```

It now uses:

```text
new building height = 0.25 * v1 + 0.75 * v2
```

Non-building pixels still use v1. Forest routing, CHMv2, DEM fusion, geospatial handling, model
weights, and inference quality settings are unchanged. This change therefore adds no model download,
training, VRAM use, or inference pass.

## Why the change was considered

v1 is the general height model, but it can predict tall buildings too low.
v2 was trained to pay more attention to tall buildings, using extra high-rise examples.
Giving v2 more weight could help those buildings. Giving it too much weight could also hurt normal
buildings or pixels wrongly labelled as buildings. That is why we tested several blends first.

The weight was therefore treated as an evaluation question rather than an assumption. Fixed weights
`0.00`, `0.25`, `0.50`, `0.75`, and `1.00` and five soft-routing alternatives were compared on the
full GAMUS validation split. Only the selected candidate was then checked against the untouched test
split.

## Evaluator correction

Before the production decision, the fusion evaluator was found to include GAMUS height sentinel
pixels with value `-5 m`. These pixels mean "no reference data" and must not participate in metrics.
The shared report path now applies the existing `clean_height()` validity rule before accumulating
any overall, building, city, or height-bin metric.

The correction excluded:

- validation: 9,272,575 invalid pixels
- test: 9,881,231 invalid pixels

It changed overall metrics slightly but did not change the selected routing direction. New corrected
files were written instead of overwriting the original artifacts.

The cache also now records whether predictions used test-time augmentation (TTA). A report refuses a
cache generated with the wrong TTA mode, preventing an accidental mixed comparison.

## Corrected full-split evidence

### Validation: 859 tiles, DC and Philadelphia

| Routing | Overall RMSE | Overall MAE | Building RMSE | Building MAE | Bias |
|---|---:|---:|---:|---:|---:|
| v1 only / 0.00 | 2.8160 | 1.3241 | 3.6272 | 2.1081 | -0.1099 |
| old production / 0.50 | 2.7511 | 1.2948 | 3.3352 | 1.9405 | -0.1054 |
| selected / 0.75 | **2.7402** | **1.2873** | **3.2763** | **1.8958** | -0.1032 |
| v2 only on buildings / 1.00 | 2.7440 | 1.2852 | 3.2807 | 1.8813 | -0.1010 |

Against 0.50, 0.75 improves validation overall RMSE by 0.0109 m (0.40%) and building RMSE by
0.0589 m (1.77%). It also has the lowest overall and building RMSE among the tested fixed weights.

### Untouched test: 2,861 tiles, DC, New York, and Philadelphia

| Routing | Overall RMSE | Overall MAE | Building RMSE | Building MAE | Bias |
|---|---:|---:|---:|---:|---:|
| v1 only / 0.00 | 3.9623 | 1.5963 | 5.9289 | 2.3608 | -0.1514 |
| old production / 0.50 | 3.7573 | 1.5513 | 5.2308 | 2.1389 | -0.1159 |
| selected / 0.75 | **3.6972** | **1.5402** | **4.9934** | **2.0792** | -0.0981 |

Against 0.50, 0.75 improves test overall RMSE by 0.0601 m (1.60%) and building RMSE by 0.2375 m
(4.54%). The test split was not used to tune the weight.

The benefit is concentrated in taller structures. On test data, RMSE improves from 6.8997 to
6.8045 m for reference heights of 20-50 m and from 46.8141 to 43.4050 m above 50 m. Lower bins
regress slightly, which is why 1.00 was not adopted and why the focused ordinary-scene check below
was required.

## Focused TTA safety check

Production High quality uses four-flip TTA, while the full routing experiment used one-pass
predictions. A predeclared 24-tile validation subset was therefore rerun with TTA: eight tiles rich
in tall buildings and sixteen ordinary tiles.

| Subset | 0.50 overall RMSE | 0.75 overall RMSE | 0.50 building RMSE | 0.75 building RMSE |
|---|---:|---:|---:|---:|
| All 24 | 3.1690 | **3.0916** | 5.1284 | **4.9140** |
| Tall 8 | 4.4430 | **4.2755** | 7.4559 | **7.1108** |
| Ordinary 16 | **2.2774** | 2.2783 | **2.0104** | 2.0181 |

The pooled TTA result improves overall RMSE by 2.44% and building RMSE by 4.18%. Tall scenes improve
materially. The ordinary subset changes by only +0.04% overall RMSE, below the predeclared 2%
regression limit. The TTA gate therefore passes.

TTA itself also helped the selected 0.75 route on these 24 tiles: overall RMSE improved from 3.1399
without TTA to 3.0916 with TTA, and building RMSE improved from 4.9773 to 4.9140.

## How this reflects in the application

- Predicted building geometry will generally be taller, especially for high-rises that v1
  underestimates.
- Ground, roads, water, vegetation, and other non-building classes are numerically unchanged by the
  fusion function.
- Fast and High quality use the same 0.75 fusion rule; High still runs four-flip TTA.
- The output manifest now identifies `height v2 (75% building blend)` for traceability.
- Runtime and memory requirements are unchanged because both models were already executed.

The change still depends on semantic routing: a building missed by the semantic head remains on v1.
Soft routing was evaluated but did not beat the fixed 0.75 rule strongly enough to justify added
production complexity.

## Files changed

- `viewer/estimate.py`: production fusion formula and model provenance label
- `tests/test_estimate.py`: regression expectation for building and non-building pixels
- `viewer/fusion_eval.py`: invalid-reference cleaning, exact tile selection, and TTA cache provenance
- `tests/test_fusion_eval.py`: sentinel-mask and TTA-provenance regression tests
- `README.md`, `SETUP.md`, `ARCHITECTURE.md`, `CHANGES.MD`: documented production behavior and evidence

## Evidence artifacts

Verification completed locally:

- focused regression tests: 39 passed
- complete suite: 122 passed, 1 skipped
- Python compilation: passed
- Fast production smoke on `DC_03_26_RGB.png`: passed on CUDA, maximum nDSM 34.2 m
- High/TTA production smoke on the same image: passed on CUDA, maximum nDSM 34.4 m
- both output manifests record `height v2 (75% building blend)`

- `E:\TEST\fusion-validation-corrected.json` and `.csv`
- `E:\TEST\fusion-test-corrected.json` and `.csv`
- `E:\TEST\fusion-notta24-all.json`, `fusion-notta24-tall.json`, `fusion-notta24-ordinary.json`
- `E:\TEST\fusion-tta24-all.json`, `fusion-tta24-tall.json`, `fusion-tta24-ordinary.json`
- `E:\TEST\fusion-cache-tta24` (24 reusable TTA prediction caches)

The original uncorrected reports remain on disk only as audit history and must not be quoted as final
metrics.
