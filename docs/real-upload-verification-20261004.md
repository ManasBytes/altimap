# Real-upload verification: 4 October 2026

Tested snapshot: upstream `665f514` plus the local JSON-safety, opt-in public-DEM
comparison and camera-repaint fixes. This report does not cover the four later
teammate commits ending at `abf6a4f`.

## Corrections and regression checks

- Flat DEM correlation and other undefined agreement metrics serialize as `null`,
  not `NaN`; a valid GeoTIFF result no longer fails with HTTP 500.
- Public-DEM comparison defaults off in the API and UI. The estimation CLI opts in
  with `--compare-dems`; the dedicated DEM-check CLI explicitly enables it.
- Missing secondary DEM coverage/service failure does not invalidate the main result.
- Agreement labels describe approximate image-aligned 30 m blocks, not native DEM
  cells or independently validated building-height accuracy.
- OrbitControls change events invalidate rendering, so wheel zoom repaints even
  when the next frame-loop update reports no camera change.
- Setup instructions identify the tested snapshot instead of the older branch.

No model retraining, weight changes or production-routing changes were made.
Building fusion remains **25% v1 + 75% v2**.

## Real-image run

RTX 3050 laptop, 6 GB, CUDA inference; full-resolution images, High/four-flip TTA.
All 10 browser uploads returned HTTP 200 without recorded JavaScript errors.
Seven inputs included matching independent height references. All 17 exports
(10 nDSM, 7 absolute DSM) passed dimension/grid, float32/NaN nodata, COG-layout/
overview, non-flat height and download checks. All ten saved results passed
wheel-zoom and orbit rendering assertions after the viewer fix.

| Input | Backend seconds | Reference/output | RMSE m | MAE m | Bias m |
|---|---:|---|---:|---:|---:|
| Sikkim Namchi town | 359.26 | No independent reference | N/A | N/A | N/A |
| Sikkim Chungthang town | 328.98 | No independent reference | N/A | N/A | N/A |
| Sikkim Chungthang forest | 171.82 | No independent reference | N/A | N/A | N/A |
| NAIP Philadelphia City Hall | 33.54 | LiDAR / absolute DSM | 35.41 | 23.01 | -9.00 |
| NAIP Chevy Chase suburb | 332.89 | LiDAR / absolute DSM | 3.96 | 2.90 | -0.79 |
| NAIP Pittsburgh hills | 331.29 | LiDAR / absolute DSM | 5.02 | 3.33 | -2.39 |
| NAIP Smoky Mountains forest | 162.16 | LiDAR / absolute DSM | 8.76 | 6.84 | +5.55 |
| GAMUS DC_04_27 PNG | 18.69 | AGL / nDSM | 2.79 | 1.65 | -0.30 |
| GAMUS DC_12_17 PNG | 18.58 | AGL / nDSM | 4.21 | 2.64 | -0.25 |
| GAMUS DC_15_17 PNG | 18.62 | AGL / nDSM | 4.11 | 2.74 | -0.49 |

RMSE/MAE/bias were independently recomputed from the exported rasters and matched
the rounded API values within 0.001 m. Backend time excludes model loading and
browser overhead. The PNGs were supplied with known GSD 0.33 m/pixel.

## Qualifications

Successful processing is not proof of accurate heights. Philadelphia has a
substantial dense-urban error and negative absolute-DSM values in this snapshot.
No independent height truth was supplied for Sikkim; agreement with the DEM used
to construct the output is only a consistency diagnostic. Reference TIFF vertical
datums and acquisition-time differences were not resolved. Building/class metrics
use predicted semantic masks. GAMUS examples are spot checks, not a new blind
holdout; their split provenance must be checked before claiming unseen accuracy.

CHMv2 weights/Transformers were unavailable in the local environment, so these
forest outputs are **not** measurements of the complete canopy-specialist stack.
Docker was not rebuilt or run locally because its daemon was unavailable.

## Tests and local evidence

- Torch-free suite: 128 passed, 2 skipped (API/GPU-loss groups).
- Separate application environment: 12 passed, including 10 upload regressions.
- Frontend production build and `git diff --check` passed. Existing bundle-size
  warnings remain; the isolated test environments must not be combined because
  two existing tests require Torch to remain unloaded.

The complete report, independently recomputed CSV/JSON, browser harnesses and
77 scene screenshots remain locally in
`test_samples/results/real_upload_20261004/`. Large raw responses, screenshots,
datasets, model weights and runtime logs are intentionally not added to Git.
