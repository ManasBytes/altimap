# AltiMap evidence index

Documentation snapshot: tested main [84136c4](https://github.com/ManasBytes/altimap/commit/84136c4013837f141624ed62a1d6cff80f52c05c).
This index was written on 5 October 2026. These small files let reviewers check our work. Datasets, model
weights, prediction caches and local runtime files remain outside Git.

## 1. Dataset inventory and metric artifacts

| Artifact | Scope |
|---|---|
| [Dataset verification JSON](gamus-verification.json) | 5,004 train / 859 validation / 2,861 test; matching triplets, loader samples, no incomplete triplets |
| [Validation CSV](fusion-validation-corrected.csv) / [JSON](fusion-validation-corrected.json) | Five hard weights and five soft threshold pairs; all 859 validation tiles |
| [Test CSV](fusion-test-corrected.csv) / [JSON](fusion-test-corrected.json) | v1-only, old 0.50 and selected 0.75; all 2,861 test tiles |
| [24-tile TTA CSV](fusion-tta24-all.csv) / [JSON](fusion-tta24-all.json) | Focused validation safety check, four-flip TTA |
| [Tall-eight TTA CSV](fusion-tta24-tall.csv) | Tall-scene subset |
| [Ordinary-sixteen TTA CSV](fusion-tta24-ordinary.csv) | Ordinary-scene subset; small regression is retained |
| [24-tile one-pass CSV](fusion-notta24-all.csv) | Same selected slice without TTA |
| [Real-upload measurements CSV](real-upload-measurements-20261004.csv) | Ten uploads; seven reference comparisons, timings, actual model provenance |

The blend reports were copied from the original E-drive test results, not rebuilt from rounded
README numbers. Invalid GAMUS −5 m reference values were excluded in the corrected runs.
These scores combine all valid pixels. Averaging each image's RMSE would give a different result.

JSON copies use **null** for undefined values so ordinary JSON readers can open them.
Known values are unchanged. Original local files are kept, and CSV copies are unchanged.
An undefined correlation does not mean zero correlation.

### Split and city coverage

| Split | DC | NYC | PHL | Total |
|---|---:|---:|---:|---:|
| Train | 1,439 | 1,167 | 2,398 | 5,004 |
| Validation | 359 | 0 | 500 | 859 |
| Test | 361 | 1,000 | 1,500 | 2,861 |

Each tile has RGB, reference above-ground height and reference class arrays. Training labels teach
the models; evaluation reference heights score predictions. Production inference does not need
the uploaded scene's reference height or supplied class array.

Routing was selected on validation, then confirmed on test. Do not reopen the test split to tune
future candidates. Height-bin reporting includes pixel counts, RMSE, MAE and bias; small groups
must not be overinterpreted.

### Interpreting the evidence

- Fusion-table building scores use **reference class labels**.
- Upload building scores use **predicted building masks**.
- GAMUS nDSM errors and LiDAR absolute-DSM errors are different targets.
- One-pass full-split evaluation and the 24-tile TTA run have different protocols.
- Public-DEM agreement is not independent ground-truth building accuracy.
- No independent height references were provided for the Sikkim screenshots.

[Detailed fusion summary](../evaluation/fusion-evaluation-summary.md) ·
[Adoption/impact report](../evaluation/BUILDING_FUSION_075_REPORT.md) ·
[Dated real-upload verification](../real-upload-verification-20261004.md)

## 2. Supplied team screenshots

Original Desktop images were copied unchanged; they remain on the user's machine.
Capture date: 4 October 2026. These illustrate UI/reconstruction, not new controlled experiments.

| Repository image | Original Desktop filename | Qualification |
|---|---|---|
| [Chungthang RGB close-up](../images/current/chungthang-rgb-close.png) | Screenshot 2026-10-04 191733.png | Chungthang GeoTIFF, 0.5× display, no independent height truth |
| [Chungthang slopes](../images/current/chungthang-slope.png) | Screenshot 2026-10-04 192039.png | Derived slope visualization |
| [Chungthang overview](../images/current/chungthang-city-overview.png) | Screenshot 2026-10-04 192448.png | Main README hero; 1× city view, no independent height truth |
| [Chungthang classes](../images/current/chungthang-classes.png) | Screenshot 2026-10-04 192634.png | Predicted categories and simplified geometry |
| [PNG city](../images/current/png-city-unknown-gsd.png) | Screenshot 2026-10-04 193331.png | Unknown GSD; experimental 0.33 m/pixel assumption |
| [Roof close-up](../images/current/city-roofs-close.png) | Screenshot 2026-10-04 203126.png | Cropped capture; input scene cannot be independently identified |
| [Class close-up](../images/current/city-classes-close.png) | Screenshot 2026-10-04 203153.png | Cropped capture; not an additional accuracy measurement |
| [Height-coloured city close-up](../images/current/city-height-close.png) | Screenshot 2026-10-04 203055.png | Extruded buildings and crowns; legend absent from crop, so colours alone are not metre values |

All eight supplied Desktop captures are directly visible in the main README's city-model gallery
(one is its opening overview). Diagnostic DSM/error views are separately labelled and collapsed.

Some embedded explanatory text predates final fixes. The documented algorithm and exact snapshot,
not older wording inside an image, describe current production.

## 3. Saved main-lineage verification images

These four selected images were copied unchanged from the actual 4 October upload-verification output.
The saved float rasters were independently checked against the matching reference rasters when
preparing this documentation; resulting metrics matched the recorded measurements.

| Image | Original local verification directory | Meaning |
|---|---|---|
| [Predicted GAMUS classes](../images/current/gamus-predicted-classes.png) | gamus_DC_04_27/classes.png | RGB-predicted semantic map |
| [GAMUS height layer](../images/current/gamus-height-map.png) | gamus_DC_12_17/height.png | Predicted nDSM, metre legend |
| [Pittsburgh error](../images/current/pittsburgh-reference-error.png) | naip_pittsburgh_hills/error.png | Absolute DSM versus LiDAR; RMSE 5.02 m |
| [Chevy Chase metrics](../images/current/chevy-chase-reference-metrics.png) | naip_dc_suburbs_chevychase/metrics.png | Absolute DSM versus LiDAR; RMSE 3.96 m |

Original directories are under local
`test_samples/results/real_upload_20261004/`. The large local directory is excluded from Git;
only curated reports/images are published. None of these photographs was synthesized or retouched.

This is a presentation selection, not the complete accuracy benchmark. The original measurement CSV
and dated test record remain unfiltered, including failure cases; no result was removed to improve
the reported numbers.

Verification used CUDA on the RTX 3050 6 GB, High/four-flip TTA and v1/v2.
CHMv2 was not installed. Acquisition/vertical-datum differences were not resolved; do not imply
LiDAR perfectly contemporaneous with the input image.

## 4. Historical and experimental replay images

Captured 5 October from rebuilt, isolated worktrees using the real browser. The process reused
existing scene assets/results; it did **not** perform new ML training or inference.

| Images | Snapshot | Actual input and check |
|---|---|---|
| [Mark-1 prediction](../images/history/mark1-predicted-dc0326.jpg), [reference](../images/history/mark1-reference-dc0326.jpg), [error](../images/history/mark1-error-dc0326.jpg) | 97c078d | Prepared DC_03_26; CLS-assisted fallback, 0.5× geometry; three modes checked |
| [Biplab terrain](../images/history/biplab-reference-workspace.jpg), [classes](../images/history/biplab-reference-classes.jpg) | 17a8016 | DC_01_25 prepared reference gallery; class switch checked |
| [Experimental Accuracy](../images/experimental/prepared-demo-accuracy.jpg), [View](../images/experimental/prepared-demo-view.jpg) | e5837d0 | Latest saved-scene loader using verified **main** DC_04_27 result; tabs and orbit checked |

The experimental preview used its existing demo-bundle CLI. It packaged the saved main response,
nDSM raster and building geometry; displayed RMSE/MAE remained those of the main result.
Inference was disabled. There is no new experimental-model accuracy result here.

Browser error logs were empty for the checked previews. Orbit movement changed the latest
canvas capture. This establishes interaction/rendering for those scenarios, not correct geospatial
units, robust uploads, all model weights or full deployment.

### Additional screenshot supplied by the team

[Experimental Pittsburgh city-context image](../images/experimental/pittsburgh-city-context-team.png)
was copied unchanged from the attached `codex-clipboard-Ezudoj.png`. It depicts the newer teammate
branch, not tested main: `naip_pittsburgh_bridges.tif`, 0.60 m/pixel, bridge/embankment context and
facility markers. Its precise source commit, capture date and height accuracy were not independently
verified. It is not part of the prepared-demo smoke-test evidence or the main accuracy benchmark.

Documentation integrity is checked separately: local Markdown targets and commit/file permalinks,
strict JSON parsing, byte-identical CSV copies and screenshot file formats. Supplied photographs
and saved verification images are compared against their originals using SHA-256 hashes.

## 5. Dated checks and reproduction boundaries

Documentation replay checks, 5 October:

| Snapshot | Non-GPU suite | Frontend |
|---|---|---|
| Mark-1 97c078d | 51 passed, 1 skipped | npm ci + npm run build passed |
| Biplab 17a8016 | 52 passed, 7 skipped | npm ci + npm run build passed |
| Latest e5837d0 | 141 passed, 1 skipped | npm ci + npm run build passed |

The three installations reused their lockfiles with `npm ci --ignore-scripts`.
Large-bundle warnings remain. No model-source changes, dependency upgrades or commits were made.
Main's API/loss regression results are the separate dated 4 October checks in the upload report;
the latest branch's green unit suite must not be substituted for them.

A fresh main browser upload was not performed during documentation capture because browser file
access was unavailable; the independently checked saved upload artifacts provide that evidence.
Docker was not revalidated on this Windows machine because its daemon was unavailable.

Existing commands for an appropriately configured environment:

```powershell
# Torch-free checks, from the selected repository snapshot
.\.venv\Scripts\python.exe -m pytest -q

# Frontend reproducibility
cd frontend
npm ci
npm run build
```

Use the existing `viewer.fusion_eval` preparation/report workflow for neural-network evaluation;
consult `--help` for the exact snapshot's options. Full evaluation requires local data and trained
weights and can be expensive. Recompute candidates with matching masks/settings, not from
screenshot colours or old paper results.

## 6. Documentation-only cleanup

The root adoption report and fusion summary were relocated into `docs/evaluation/`, with references
updated. Six untracked backend/frontend/manual-server log files were moved, not deleted, to
`E:\TEST\altimap-local-log-archive-20261005`.

Local logs and `test_samples/` are now ignored; original datasets, models, tests, source and legacy
research directories remain intact. Those older directories were not deleted because they support
reproducibility and may still be referenced. No application code or production routing changed.

[Return to README](../../README.md) · [Project journey](../PROJECT_JOURNEY.md)
