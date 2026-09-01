# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

AltiMap: single-view optical remote-sensing imagery → metric elevation models (nDSM/DSM), with an
interactive three.js 3D flythrough viewer. Hackathon project graded 50% on DSM accuracy (RMSE/MAE/
correlation vs LiDAR, stratified across landscape types) and 50% on rendering quality/navigability/
standalone deployment.

The full design reasoning lives in `docs/superpowers/specs/` — read
`2026-08-23-single-view-dsm-design.md` before making architectural decisions; it explains *why*
(e.g. why nDSM+DTM composition instead of scale-fitting depth, why DA3 over other encoders, why
2 m output resolution). `docs/superpowers/plans/` has the task-by-task implementation plans this
code was built from, `docs/superpowers/spikes/` records empirical findings that later designs
depend on.

## Environment and commands

Two separate Python environments by design — do not merge them:

- **`.venv`** (Python 3.12, pinned via `.python-version`) — numpy, scipy, rasterio, pytest. No
  torch. Runs `src/altimap` and all tests.
- **`.venv-da3`** — adds PyTorch + `depth_anything_3` (+ trimesh, Pillow). Only for scripts that
  run model inference or build glTF meshes. Heavy (~4.5 GB), so it's deliberately kept out of the
  test loop.

Neither venv exists in this checkout yet; create with `uv`:
```
uv pip install -e ".[dev]"                     # main .venv
uv pip install --python .venv-da3/bin/python torch depth_anything_3 trimesh pillow rasterio fastapi uvicorn python-multipart
```

Tests (main venv only, no network, no torch):
```
.venv/bin/python -m pytest -q
.venv/bin/python -m pytest tests/test_viewer_metrics.py -v   # single file
```

Inference / export scripts (`.venv-da3` only):
```
.venv-da3/bin/python -m viewer.export_scenes --limit 20 --metrics-only   # curated Roboflow dataset
.venv-da3/bin/python -m viewer.export_offnadir --resume                  # off-nadir Atlanta dataset
.venv-da3/bin/python -m viewer.refine_scenes                             # footprint-constrained refinement
.venv-da3/bin/python -m viewer.server                                    # FastAPI upload server + dashboards, http://localhost:8000
```

Static dashboards alone (no inference, once assets are exported) can be served with
`python -m http.server` from `viewer/web/`.

## Architecture

### The contract (`src/altimap/contract.py`)

Everything in this project is organized around one narrow interface, defined in the design doc
§3.1: a producer emits (1) a Cloud-Optimized GeoTIFF of float32 elevation in metres, (2) source RGB
on the identical grid, (3) a JSON sidecar (`Sidecar` dataclass — `gsd_m`, `datum`
`"ellipsoidal"|"relative"`, `height_range_m`, `dtm_source`, etc.). Nothing else is meant to cross a
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

This subsystem currently implements the **rDSM (relative) path** on two real datasets — a curated
Roboflow remote-sensing image classification set and the Off-nadir Scene10 (Atlanta) dataset — plus
a live FastAPI upload path. Design rationale is in
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
| `viewer/server.py` | Yes | FastAPI app: `POST /api/upload` runs DA3 on an uploaded image through the *same* pipeline (`metrics.py`, `geo.py`, `terrain.py`) the batch exporters use, so uploads and pre-exported scenes are treated identically. Model loads lazily on first request, not at import. Serves `viewer/web/` as static files, mounted after the API routes. |
| `viewer/web/` | — (browser) | Three static dashboards (`index.html` curated set, `offnadir.html` Atlanta set, `upload.html` live upload) sharing vendored, pinned three.js (`viewer/web/vendor/three/`, **never CDN** — standalone-deployment requirement). Depth uploads as a float32 `DataTexture`; the grid mesh is displaced in the **vertex shader**, never on the CPU, so exaggeration is a free uniform and swapping scenes is one texture upload. The `heightAt()` GLSL function is shared between vertex and fragment stages (`viewer/web/js/shaders.js`) since normals are derived by finite differences in the fragment shader. |

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
- **`datum` is always `"ellipsoidal"` or `"relative"`** (`Sidecar.__post_init__` enforces this).
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
