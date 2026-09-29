# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

AltiMap: single-view optical remote-sensing imagery → metric elevation models (nDSM/DSM), with an
interactive three.js 3D flythrough viewer. Built for SIH 2026 problem statement 26175 ("DepthWizard",
ISRO) — full text in `docs/problem-statement.md`, which is what the design doc calls "the brief".
Graded 50% on DSM accuracy (RMSE/MAE/correlation vs LiDAR, stability across urban, sparse, hilly
and forested scenes) and 50% on rendering quality/navigability/standalone deployment.

Hard requirements from the brief that any change must keep satisfiable:
- Inputs: PNG/JPG (non-georeferenced → **rDSM**) and GeoTIFF (georeferenced → **absolute metric
  DSM**). Output: a DSM in a standard geospatial format (COG GeoTIFF, per the contract below).
- A **pre-trained monocular depth backbone** is expected for the initial relative depth. Calibration
  to metres may use a low-res DEM (SRTM 30 m suggested), GCPs, scene statistics or semantic priors.
- The optical image draped on the mesh, first-person navigation, and height/slope analysis from any
  viewpoint. The UI must let users upload imagery and validate heights against reference data.
- The whole suite deploys as one standalone module, with source code and technical documentation.
- Reference dataset: https://github.com/IMG-PROCESS-SAC/SIH2026/. **Final evaluation runs on
  ISRO's own RGB-band optical satellite imagery**, not on the development datasets, so nothing may
  depend on dataset-specific metadata (e.g. Overture coverage, GAMUS tile IDs).
- Organiser FAQ (in `docs/problem-statement.md`): evaluation imagery is **Cartosat-2S at 0.6 m**, but
  the model must work from **0.35 m to 10 m** (don't overfit to 0.6 m); GeoTIFF output is scored as an
  **absolute DSM against SRTM/Copernicus** and "values must match DEM heights"; images are nadir with
  no sun angle or timestamp; scoring is relative to other teams.

The full design reasoning lives in `docs/superpowers/specs/` — read
`2026-08-23-single-view-dsm-design.md` before making architectural decisions; it explains *why*
(e.g. why nDSM+DTM composition instead of scale-fitting depth, why DA3 over other encoders, why
2 m output resolution). `docs/superpowers/plans/` has the task-by-task implementation plans this
code was built from, `docs/superpowers/spikes/` records empirical findings that later designs
depend on.

## Environment and commands

Separate Python environments by design — do not merge them:

- **`.venv`** (Python 3.12, pinned via `.python-version`) — numpy, scipy, rasterio, pytest. No
  torch. Runs `src/altimap` and all tests.
- **`.venv-da3`** — adds PyTorch (cu128) and runs the app (`viewer.server`) and every model
  script. Heavy (~6 GB), so it's deliberately kept out of the test loop. `altimap` is installed
  into it with `-e . --no-deps`. `depth_anything_3` is only needed for the old DA3 `/api/upload`
  path, which the React viewer no longer calls.
- **`backend/.venv`** — plays the `.venv-da3` role for the Django app (Django + DRF + the full
  `viewer/` pipeline deps incl. torch), installed from `backend/requirements.txt` by `make install`.

**`SETUP.md` is the from-scratch guide** (envs, model code + weights, frontend build, run,
troubleshooting); keep it in step when setup changes. If ROS (or anything else) is on
`PYTHONPATH`, prefix Python commands with `env -u PYTHONPATH` — ROS's pytest plugins break the
test run otherwise.

Tests (main venv only, no network, no torch):
```
.venv/bin/python -m pytest -q
.venv/bin/python -m pytest tests/test_viewer_metrics.py -v   # single file
```

Inference / export scripts (`.venv-da3` only):
```
.venv-da3/bin/python -m viewer.export_scenes --limit 20 --metrics-only   # curated Roboflow dataset
.venv-da3/bin/python -m viewer.export_offnadir --resume                  # off-nadir Atlanta dataset
.venv-da3/bin/python -m viewer.export_inria                              # Inria 0.5 m true-nadir set (reads straight from the zip via /vsizip/)
.venv-da3/bin/python -m viewer.export_dem_direct                         # GeoTIFF draped on real DEM, no depth model
.venv-da3/bin/python -m viewer.refine_scenes                             # footprint-constrained refinement
.venv-da3/bin/python -m viewer.train_classifier                          # 7-class land-cover U-Net → viewer/cache/classifier_resnet34unet.pt
.venv-da3/bin/python -m viewer.server                                    # FastAPI upload/classify server + dashboards, http://localhost:8000
.venv-da3/bin/python -m viewer.estimate scene.tif photo.png --out results/   # batch: images -> ndsm.tif / dsm.tif
.venv-da3/bin/python -m viewer.city_eval [--fetch]                       # 3D city model buildings vs GAMUS truth (40 val tiles)
.venv-da3/bin/python -m viewer.height_eval --split test --ckpt viewer/cache/best.pth --synrs3d viewer/cache/SynRS3D --tta
.venv-da3/bin/python -m viewer.height_train --hours 8 --out runs/ft      # fine-tune (24 GB GPU); see scripts/overnight_v2.sh
```

The React viewer (what users see) is `cd frontend && npm run build && npm run preview -- --host
127.0.0.1 --port 5173`, talking to `viewer.server` on :8000. `npm run dev` works too but can hit
the inotify watcher limit (ENOSPC) on machines with many watched files. On a 6 GB GPU, stop the
server before running an eval: its model and the eval's model don't both fit.

Static dashboards alone (no inference, once assets are exported) can be served with
`python -m http.server` from `viewer/web/`.

Django backend + React frontend are orchestrated by the root `Makefile` (`make help` lists all):
```
make yolo        # install both envs, migrate, seed 2 synthetic demo scenes, start both, health-check
make dev / stop / status / logs     # background servers; pids + logs in .run/
make backend     # Django runserver on 127.0.0.1:8000 (foreground)
make frontend    # Vite dev server on 127.0.0.1:5173 (foreground); or `cd frontend && npm run dev`
```
`make clean` is destructive (venv, node_modules, sqlite db, media) and requires `CONFIRM=1`.

**Port 8000 collision:** Django (`make dev`) and FastAPI (`viewer.server`) both default to :8000,
and the frontend's upload tab hardcodes `http://localhost:8000/api/estimate`, which only
FastAPI serves. To use the frontend's upload, run `viewer.server`, not Django, on :8000.

## Architecture

### The contract (`src/altimap/contract.py`)

Everything in this project is organized around one narrow interface, defined in the design doc
§3.1: a producer emits (1) a Cloud-Optimized GeoTIFF of float32 elevation in metres, (2) source RGB
on the identical grid, (3) a JSON sidecar (`Sidecar` dataclass — `gsd_m`, `datum`
`"ellipsoidal"|"orthometric"|"relative"`, `height_range_m`, `dtm_source`, etc.). Nothing else is meant to cross a
subsystem boundary: the elevation/ML side never imports viewer code and vice versa. **NaN is the
project-wide nodata sentinel** — never a magic number like -9999, since that silently corrupts
statistics if a mask is forgotten.

The core modeling decision behind the whole project: predict **nDSM** (height above ground, metric,
learned from RGB) and compose `DSM = nDSM + DTM` from a public bare-earth DEM, rather than trying to
recover absolute scale from the network itself. See design doc §2 for why this specific split (and
why Copernicus GLO-30 needs morphological-opening filtering to avoid double-counting buildings).

`src/altimap/` currently only has `contract.py`. The `eval/` CLI referenced by the `altimap-eval`
entry point in `pyproject.toml` (co-registration, metric matrix, stratification — design doc §6) is
specced in `docs/superpowers/plans/2026-08-23-eval-harness-and-data-pipeline.md` but not yet built.

### `viewer/` — depth diagnostics, refinement, and the three.js dashboards

This subsystem implements the **rDSM (relative) path** on real datasets — a curated Roboflow
remote-sensing set, the Off-nadir Scene10 (Atlanta) set, and Inria Aerial — plus a model-free
DEM-drape path, a non-georeferenced classify path, and a live FastAPI upload path. Design rationale is in
`docs/superpowers/specs/2026-08-25-depth-diagnostics-and-rdsm-viewer-design.md`.

Key finding driving this whole subsystem: DA3 (Depth Anything 3) fits a **tilted plane** to nadir
imagery rather than reading real relief (`docs/superpowers/spikes/2026-08-24-da3-nadir-domain-gap.md`).
Every module here exists to separate that plane artifact from whatever structure survives under it.

Module map:

| Module | Torch? | Responsibility |
|---|---|---|
| `viewer/metrics.py` | **No — enforced by test** | Plane fit/detrend, `structure_alignment`, `plane_r2`, degenerate-input → `nan` handling. Pure numpy/scipy so it runs in `.venv`. |
| `viewer/geo.py` | No | rg16 depth encoding (16-bit height packed across PNG R/G channels), relative→absolute elevation fitting, CRS/bounds helpers. Handles imagery both with and without a CRS in the same dataset. |
| `viewer/dem.py` | No | Windowed reads of `3dep-seamless` (bare-earth DEM) via Planetary Computer STAC, for scale calibration against a DTM. |
| `viewer/terrain.py` | trimesh/Pillow lazily | Height field → textured, displaced-grid GLB mesh (`height_field` pure part is torch-free and tested; `build_terrain` isn't). |
| `viewer/footprints.py` / `fetch_buildings.py` | rasterio | Overture building footprints + heights, used to anchor/refine relative depth (Overture gives height for 78% of Atlanta buildings vs 1.5% for OSM). |
| `viewer/refine.py` / `refine_scenes.py` | No (reads back rg16 PNGs rather than re-running DA3) | Footprint-constrained refinement: estimate ground from non-building pixels, collapse each footprint to a flat roof level. Fixes DA3's "melted mound" building artifact and raised absolute-calibration usability from 5.6% on the naive bare-earth-DEM fit. |
| `viewer/validate.py` | No | Scores model heights against Overture reference heights. Reports scale-free Pearson correlation as the honest headline number; RMSE/MAE use an *oracle* per-scene scale/offset fit, explicitly reported as a lower bound, not a real calibration. |
| `viewer/export_scenes.py` | Yes | Two-pass exporter for the curated Roboflow dataset: pass 1 computes metrics over ~1000 images (no depth retained), pass 2 re-runs DA3 on a curated ~42-scene subset to write static assets (deterministic model → reproducible). |
| `viewer/export_offnadir.py` | Yes | Exporter for the 5200-image off-nadir Atlanta dataset. Only 620/5200 tiles carry a CRS; the metric and nadir-detection paths cover disjoint subsets of the data, by construction of the dataset. |
| `viewer/export_inria.py` | Yes | Inria Aerial 0.5 m set: true nadir, 100% georeferenced, US + Austria so calibration uses Copernicus GLO-30 (3DEP is US-only). `refine_scenes`/`validate` run on its output unchanged via `--data`/`--buildings`. |
| `viewer/export_dem_direct.py` | No model | Drapes a georeferenced orthophoto over the real DEM — no depth model. Encodes `fake_depth = elev_max - elevation` so it round-trips through the unchanged rg16/shader "depth" math (proved in `tests/test_viewer_dem_direct.py`). |
| `viewer/classify.py` / `train_classifier.py` / `gamus_dataset.py` | torch lazily | **Non-georeferenced path**: ResNet34-UNet predicts 7 land-cover classes, each mapped to a fixed relative height (`STATIC_CLASS_HEIGHT`, normalized units, *not* metres) — a "layer cake" placeholder preview, no depth model. Trained on raw GAMUS `.h5` tiles (exact labels), not the lossy frontend JPEGs. GAMUS roots are hardcoded to `/home/biplab-dev/...`; the checkpoint lives in gitignored `viewer/cache/`, so it must be trained before `/api/classify-static` works. |
| `viewer/height_model.py` / `height_eval.py` / `height_train.py` | torch lazily | **Metric height path** (spec `docs/superpowers/specs/2026-09-29-24h-metric-height-design.md`). RS3DAda (DINOv2 ViT-L + DPT, from the cloned SynRS3D repo in `viewer/cache/SynRS3D`) predicts nDSM in metres plus 8 OpenEarthMap classes; `height_train` fine-tunes it on GAMUS (BitFit encoder), `height_eval` scores per city against `zero`/`train-mean` baselines. Inputs must be multiples of 14 px (518 windows). GAMUS GSD is 0.33 m; the HF release names DC images `*_RGB.h5` and PHL/NYC `*_IMG.h5`. |
| `viewer/city_model.py` | No | LoD1 city model **for display only**. Buildings: each blob of building-class pixels is cut into roofs (`_roof_segments`: one seed per roof plateau, watershed over the height gradient, so touching row houses split at the step/dip between them), then each roof into height levels ≥ 2.5 m apart (`_component_parts`, interior pixels so wall ramps don't terrace). Outlines keep the traced shape; `regularize` squares one off (rectangle or right angles) **only if that moves ≤ 10 % of its area** (`max_shift`). Plus roof colour, base/top on terrain, individual tree crowns. Never feeds the GeoTIFFs (regularizing heights costs RMSE). `/api/estimate` returns it as `city` and writes `buildings.geojson`; the frontend extrudes it (`buildCityGroup`). |
| `viewer/city_eval.py` | model only if cache is cold | Scores the city model against GAMUS truth on 40 pinned val tiles (`EVAL_TILES`, `--fetch` downloads them): footprint IoU, edge F1 within 1 m, height RMSE on buildings, building count. Height predictions are cached in `viewer/cache/city_eval/`, so re-scoring a `city_model.py` change is CPU-only. Run it before and after any change to building extraction. |
| `viewer/height_metrics.py` | No | RMSE/MAE/Pearson/building-RMSE, pooled over tiles via running sums (`ScoreAccumulator`), NaN-excluding. |
| `viewer/estimate.py` | torch via model | Image → `ndsm.tif` (always) and `dsm.tif` (GeoTIFF input, DEM-consistent: `GLO-30 − mean₃₀ₘ(nDSM) + nDSM`, orthometric, so 30 m block means match the DEM as the FAQ scores it), resampled to/from 0.33 m. The *viewer's* ground is chosen separately by building share (`display_ground_method`: ≥0.4 → 300 m opening, ≥0.1 → 150 m, else subtract), measured against 3DEP LiDAR. `read_image` follows band colour tags, masks nodata, and rejects single-band float rasters (height maps) with `NotImageryError`. Post-filtering the nDSM (guided/median) was measured on 40 val tiles and gives no gain, so exports stay raw. |
| `viewer/server.py` | Yes | FastAPI app: `POST /api/estimate` (what `frontend/` calls) runs the height model via `viewer/estimate.py`, returns preview data URIs + GeoTIFF download paths, and scores against an optional uploaded reference height map; checkpoint is `viewer/cache/best.pth` if present, else stock RS3DAda, overridable with `ALTIMAP_HEIGHT_CKPT`. `POST /api/upload` runs DA3 on an uploaded image through the *same* pipeline (`metrics.py`, `geo.py`, `terrain.py`) the batch exporters use, so uploads and pre-exported scenes are treated identically; `POST /api/classify-static` runs the older static-classifier path (no longer called by `frontend/`). Models load lazily on first request, not at import. Serves `viewer/web/` as static files, mounted after the API routes. |
| `viewer/web/` | — (browser) | Static vanilla-JS dashboards (`index.html` curated set, `offnadir.html` Atlanta set, `inria.html`, `demdirect.html`, `upload.html` live upload) sharing vendored, pinned three.js (`viewer/web/vendor/three/`, **never CDN** — standalone-deployment requirement). Depth uploads as a float32 `DataTexture`; the grid mesh is displaced in the **vertex shader**, never on the CPU, so exaggeration is a free uniform and swapping scenes is one texture upload. The `heightAt()` GLSL function is shared between vertex and fragment stages (`viewer/web/js/shaders.js`) since normals are derived by finite differences in the fragment shader. |

### `backend/` — Django + SQLite

Second caller of `viewer/` as a library (settings put the repo root on `sys.path`); `viewer/`
itself is not modified for it. `scenes/pipeline.py` is a port of `viewer/server.py`'s upload
processing and deliberately imports nothing from Django (path in → files + dict out);
`scenes/services.py` is the only place a pipeline record is mapped onto a `Scene` row, shared by
the upload view and the `seed_scenes` command. Uploads process on a background thread; clients poll
`GET /api/scenes/<id>/` for `status`. Derived assets go to `backend/media/scenes/<id>/`; the
uploaded source is always deleted. Note: the current `frontend/` does **not** call `/api/scenes/`
(the React components that did were replaced by the terrain studio).

### `frontend/` — "GAMUS Terrain Studio" (React + Vite + three.js from npm)

Almost all of it is `src/main.jsx` (single large file). Scenes are static: `src/gamusScenes.js` is
the manifest for the pre-rendered `public/{split}-{id}-{rgb,height,depth,classes}.jpg` GAMUS tiles.
Uploads come from `/api/estimate`, whose height PNG is linear (`pixel/255 * max_m`), so the probe
and profile report metres for uploads only; the catalog `*-height.jpg` previews are non-linearly
encoded, so for those the UI shows relative values, never metres. The upload tab feeds uploads into the same `sample` state as scene selection, so every
viewer tool works on both. `CLASS_PALETTE` in `main.jsx` and in `viewer/classify.py` must stay
identical (same order and RGB values) — the classifier's output is decoded/rendered as if it were a
`*-classes.jpg`.

The catalog scenes are **GAMUS reference layers (LiDAR heights, annotated classes), not model
output**, restyled for looks (`buildHeightField` / `applyClassHeightScale`: median + blur,
per-scene building stretch, trees squashed). They are labelled "LiDAR reference heights, not
model output" in the UI; keep that label, and never present them as results. Uploads take the
faithful path (`metricDisplayField`: metres at true scale, 3x3 median only). Clicking a building
opens `.building-card` (height, ~floors, footprint m², roof elevation); it sits at `right: 370px`
so it clears the inspector panel (it was hidden under it once). For uploads the **Height layer
always means height above ground** (jet over 0..nDSM max, `buildHeightColors(probeField, 0, max)`):
the city model's roofs, walls and crowns switch to it (`setHeightMode`), and the city ground is
uniform 0 m. The legend's gradient is built from the same `JET_STOPS` (`JET_CSS`), so it can't
drift from the map. Heights for the mesh, probe, profile and slope come from `/api/estimate`'s
`grids` (16-bit, `viewer/geo.py:encode_grid16`, 513x513 = `GRID_SIDE` = `HEIGHT_SAMPLE_*`; the
8-bit PNGs stepped 0.1-0.3 m and quantised slope to ~13°); the PNGs remain the fallback and the
layer textures. Slope is taken from the absolute DSM when georeferenced (terrain counts), else the
nDSM.

### Models, weights and data (all under gitignored `viewer/cache/` unless noted)

| What | Where it comes from |
|---|---|
| `viewer/cache/best.pth` — our fine-tuned RS3DAda (v1: test RMSE 5.02 m vs 7.25 m zero-shot) | HF **public** [`Dilavesh/altimap-height`](https://huggingface.co/Dilavesh/altimap-height) (`best.pth`; v2 results go to `v2/` when the overnight run uploads). Its README is `docs/model-card.md`: edit there, re-upload as `README.md` |
| `viewer/cache/SynRS3D/` — RS3DAda model code | `git clone https://github.com/JTRNEO/SynRS3D`, commit `ab5a485` |
| `SynRS3D/pretrain/RS3DAda_vitl_DPT_height.pth` — stock weights, the server's fallback | HF `JTRNEO/RS3DAda` |
| DINOv2 encoder code | `torch.hub` (`facebookresearch/dinov2`), fetched on first model load, cached in `~/.cache/torch/hub` |
| GAMUS tiles (`images/`, `heights/`, `classes/` `.h5`) | HF dataset `earthflow/GAMUS`; `city_eval --fetch` pulls the 40 eval tiles into `viewer/cache/gamus` |
| Copernicus GLO-30 ground for GeoTIFF inputs | Microsoft Planetary Computer, read live per upload (`viewer/dem.py`) |
| `demo/` test images (not in git, 57 MB) | 3 GAMUS tiles as PNG + LiDAR `reference_heights/`, 4 NAIP GeoTIFFs (city, suburb, hills, forest) |

### Measured findings (don't re-run these without a reason)

- **UI numbers are computed correctly** (checked 2026-09-30 against an independent recalculation
  and USGS 3DEP LiDAR): validation RMSE/MAE/r, coverage and surface max all match. The weak part
  is the model, not the display: it reads **low on 0.6 m imagery** (NAIP nDSM bias −4 to −16 m;
  building card MAE ~4.2 m vs 2.2 m on 0.33 m GAMUS) and **under-reads tall objects** (tallest
  object 72 m vs 167 m at Philadelphia City Hall, 28 vs 38 m on a GAMUS tile). Exported DSM vs
  LiDAR DSM RMSE: suburb 4.1 m, hills 5.1 m, forest 9.5 m, dense downtown 34.7 m. v2 training
  (height-weighted loss + blur augmentation) targets exactly this; its results weren't in yet.
- **SAM 3 does not beat our model's building masks** (40 GAMUS val tiles): raw pixels IoU 0.757 /
  edge F1 0.529 vs ours 0.854 / 0.688; as 3D footprints 0.720–0.781 vs 0.787–0.841, and it
  separates fewer buildings. It's heavy (860M params, gated) and not a dependency; don't re-add
  it without beating `city_eval`.
- **Squaring off every outline costs accuracy** (IoU 0.843 → 0.793, edge F1 0.661 → 0.569); hence
  the 10 % `max_shift` cap. Splitting blobs at roof-height steps (`_roof_segments`, 2.5 m levels)
  is what took separate buildings 1281 → 2372 and building height RMSE 3.15 → 2.99 m.
- Post-filtering the nDSM (guided/median filter) gives no gain; exports stay raw (see estimate row).

### Load-bearing invariants (violating these produces silently-wrong output, not a crash)

- **`viewer/metrics.py` must never import torch, cv2, or `depth_anything_3`.** A test asserts these
  are absent from `sys.modules` after import — this is what lets the main test suite run without
  the 4.5 GB `.venv-da3`.
- **Height is `depth_max - depth`.** Depth is distance-from-sensor; larger depth means *lower*
  ground for a nadir view. Getting this backwards renders every city as a pit and looks plausible
  until you look closely.
- **`Prediction.is_metric` (from `depth_anything_3`) is an empty `addict.Dict` for non-metric
  models, not an int or bool.** Test truthiness (`bool(prediction.is_metric)`); `int()` raises
  `TypeError`.
- **No metre values anywhere in the rDSM-only UI/export paths** (curated + off-nadir dashboards).
  Height axis is labelled "relative"; the slope readout is explicitly labelled "display slope —
  depends on exaggeration", not a physical ground slope, because the vertical axis there is
  unitless.
- **`datum` is `"ellipsoidal"`, `"orthometric"` or `"relative"`** (`Sidecar.__post_init__` enforces
  this). GLO-30/SRTM ground is geoid-based, so absolute DSMs built on it are `"orthometric"`, never
  `"ellipsoidal"`: the two differ by tens of metres over India.
- **Do not modify `spikes/04_da3_nadir_check.py`.** It is the frozen, reproducible source of the
  2026-08-24 findings doc; changing it would make that doc's numbers no longer correspond to the
  code that produced them.
- YOLO label files under the Roboflow dataset carry no trailing newline — read them individually
  per-image, never concatenated (`cat labels/*.txt` welds two rows together).

## Testing conventions

- All fixtures are synthetic (`tests/conftest.py` builds long-tailed synthetic nDSMs — mostly
  near-zero ground with a few tall rectangular "buildings" — because metrics that look fine on
  uniform noise can fail on the long-tailed distributions real elevation data has).
- Degenerate-input behavior is a first-class test case throughout: constant depth, zero median,
  single-channel images — these must return `nan` for the affected metric, never raise and never
  silently become `0.0`.
- A `network` pytest marker exists (`pyproject.toml`) for tests that would need live Planetary
  Computer access, but no test currently uses it — all current coverage is fixture-based.
