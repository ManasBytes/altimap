# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

AltiMap: single-view optical remote-sensing imagery → metric elevation models (nDSM/DSM), with an
interactive three.js 3D viewer. Built for SIH 2026 problem statement 26175 ("DepthWizard", ISRO);
the brief and the organisers' FAQ are in `docs/problem-statement.md`. Graded 50% on DSM accuracy
(RMSE/MAE/correlation vs LiDAR, stability across urban, sparse, hilly and forested scenes) and 50%
on rendering quality, navigability and standalone deployment.

Hard requirements any change must keep satisfiable:
- PNG/JPG (no georeferencing) → relative DSM; GeoTIFF → **absolute metric DSM** as a COG GeoTIFF.
- A pre-trained monocular depth backbone; calibration to metres may use a low-res DEM, GCPs,
  scene statistics or semantic priors.
- Optical image draped on the mesh, first-person navigation, height/slope analysis from any view,
  upload in the UI, validation against reference data. One standalone module with source + docs.
- **Final evaluation is ISRO's own imagery** (FAQ: Cartosat-2S, 0.6 m, nadir, no sun angle or
  timestamp; must work 0.35–10 m), so nothing may depend on dataset-specific metadata (GAMUS
  tile IDs, Overture coverage). GeoTIFF output is scored as an absolute DSM against
  SRTM/Copernicus: "values must match DEM heights".

Where the rest lives:
- **`ARCHITECTURE.md`**: the full technical description (pipeline, contracts, API, models,
  measured results, limits).
- **`SETUP.md`**: the from-scratch install guide.
- `THIRD_PARTY_NOTICES.md`: model/data licences and attribution.
- `BUILDING_FUSION_075_REPORT.md`: the evidence behind the current building fusion.
- `docs/superpowers/specs/`: design reasoning; read `2026-08-23-single-view-dsm-design.md` before
  architectural changes.
- `docs/superpowers/spikes/`: frozen empirical findings.
- `AGENTS.md`: the same repo conventions for other agents; keep the two consistent.

Keep `ARCHITECTURE.md` and `SETUP.md` in step when the pipeline or setup changes.

## Environments and commands

Three Python environments, separate by design (don't merge them):
- `.venv`: Python 3.12; numpy, scipy, rasterio, pytest. No torch. Runs `src/altimap` and the test
  suite.
- `.venv-da3`: adds PyTorch (cu128) and `transformers`; runs the app and every model script, with
  `altimap` installed `-e . --no-deps`.
- `backend/.venv`: the legacy Django app, installed by `make install`.

If ROS (or anything else) is on `PYTHONPATH`, prefix Python commands with `env -u PYTHONPATH`;
ROS's pytest plugins break the run. There is no lint script: match the surrounding style.
Prettier is installed in `frontend/`.

```bash
# Tests (.venv: no GPU, no network, ~120 tests in a few seconds)
.venv/bin/python -m pytest -q
.venv/bin/python -m pytest tests/test_estimate.py -k gcp -v          # one file / matching tests

# The app: build the viewer once, then one server serves it and the API at http://127.0.0.1:8000
(cd frontend && npm install && npm run build)
.venv-da3/bin/python -m viewer.server [--host 0.0.0.0]
# frontend dev: `npm run dev` on :5173 proxies /api and /data-uploads to :8000 (vite.config.js);
# can hit the inotify limit (ENOSPC), then use build + server

# Or the whole app as one image (weights baked in; ~35 GB free disk to build, ~12 GB image)
docker build -t altimap . && docker run --gpus all -p 8000:8000 altimap

# Batch export: images -> ndsm.tif / dsm.tif (+ sidecars)
.venv-da3/bin/python -m viewer.estimate scene.tif photo.png --out results/ [--gsd 0.5] [--gcps pts.csv]

# Evaluation (stop the server first on a 6 GB GPU: two ViT-L models don't fit)
.venv-da3/bin/python -m viewer.height_eval --split test --ckpt viewer/cache/best.pth \
    --synrs3d viewer/cache/SynRS3D --tta                              # per city + height bins
.venv-da3/bin/python -m viewer.fusion_eval --prepare ... / --report ...   # v1/v2 building-routing ablations, cached
.venv-da3/bin/python -m viewer.city_eval [--fetch]                   # 3D buildings vs GAMUS truth, 40 pinned tiles
.venv-da3/bin/python -m viewer.dsm_eval [--scenes ...] [--gsd ...]   # DSM vs USGS 3DEP LiDAR, demo/ NAIP scenes
.venv-da3/bin/python -m viewer.dem_check demo/india/*.tif            # DSM vs Copernicus and SRTM (no-LiDAR scenes)
.venv-da3/bin/python scripts/fetch_india_samples.py                   # Sikkim Maxar crops -> demo/india/
.venv-da3/bin/python scripts/fetch_naip_scene.py --lat 40.44 --lon -80.006 --name naip_pittsburgh_bridges  # NAIP test scene

# Training (24 GB GPU): data, then fine-tune; scripts/overnight_v2.sh is the unattended run
.venv-da3/bin/python scripts/fetch_training_data.py
.venv-da3/bin/python -m viewer.height_train --hours 8 --out runs/ft
```

Environment variables:
- `ALTIMAP_DATA`: dataset root for `height_eval`/`height_train` (default `~/altimap-data`).
- `ALTIMAP_HEIGHT_CKPT`: override the server's height checkpoint.
- `ALTIMAP_DEM_CACHE`: where complete DEM patches are cached (default `viewer/cache/dem/`).

Port 8000: Django (`make dev`) and `viewer.server` both default to it, and only `viewer.server`
serves `/api/estimate`, which the frontend calls.

## Architecture

### The contract (`src/altimap/contract.py`)

Elevation leaves the pipeline only as three things:
- a float32 COG GeoTIFF in metres (GDAL's COG driver);
- the source RGB on the identical grid;
- a JSON sidecar (`Sidecar`: `gsd_m`, `datum`, `height_range_m`, `dtm_source`, …).

**NaN is the project-wide nodata value**, never a sentinel like −9999: a forgotten mask then
silently corrupts statistics. `src/altimap/` holds only this module; the `altimap-eval` entry
point in `pyproject.toml` is specced but not built.

The core modelling decision: predict **nDSM** (height above ground, metres) from RGB and take the
absolute level from a public DEM, rather than recovering absolute scale from the network.

### The pipeline (`viewer/estimate.py`, called by the server and the batch CLI)

`estimate()` runs these stages in order:
1. **`read_image`**:
   - follows band colour tags;
   - stretches >8-bit data over valid pixels only;
   - masks nodata;
   - rejects single-band float rasters (height maps) with `NotImageryError`.
2. **Pixel size** comes from the GeoTIFF. For PNG/JPG with none given, 0.33 m is assumed and
   flagged as experimental in the record and the UI. Rotated or sheared, or materially
   non-square, GeoTIFF grids get a warning and **no absolute DSM**.
3. **`predict_scene`** resamples to the model's 0.33 m and predicts in 3072 px tiles with a 259 px
   context margin, so large scenes run at full resolution. Only above 400 MP at 0.33 m does it
   run coarser (`work_gsd_m`).
4. **Three models, by land cover** (`load_pipeline`):
   - v1 everywhere;
   - building pixels `0.25 · v1 + 0.75 · v2` (`fuse_heights`);
   - tree pixels inside extensive forest (≥ 80 % canopy within 150 m, `forest_mask`) from Meta
     CHMv2 (`viewer/canopy.py`).

   v2 and CHMv2 are optional, kept on the CPU and moved to the GPU only while they run (`_on`),
   so a 6 GB GPU holds one ViT-L at a time.
5. **DEM**: SRTM GL1 by default (the brief names SRTM; OpenTopography via `dem.srtm`), or
   Copernicus GLO-30 (`base_dem="glo30"`; AWS Open Data via `dem.glo30`, Planetary Computer as
   fallback), which also stands in where SRTM has no coverage (beyond 60 N / 56 S) and says so
   (`record["base_dem_fallback"]`). Read once per upload (`padded_dem`: image + 300 m margin);
   complete patches are cached, keyed by source.
   Unresolved coverage stays NaN; it never becomes 0 m.
6. **DEM-consistent DSM**: `DSM = DEM − mean₃₀ₘ(nDSM) + nDSM` on the base DEM, orthometric. Every
   30 m cell reproduces the DEM, which is how the FAQ scores, while the model supplies sub-cell
   detail. In forest, the detail is dropped (the DEM already holds the canopy surface). The DSM is
   floored at bare earth (`compose_dsm`): where the DEM under-reads tall downtowns, keeping the cell mean
   pushed streets underground (Philadelphia to −35 m). `dem_agreement` then scores the DSM in
   30 m cells against both GLO-30 and SRTM; the UI shows it.
7. **Embankments** (`viewer/embankments.py`, GeoTIFF only): levees, embankments, dams and weirs
   from OpenStreetMap. The bare-earth opening erases them (Baton Rouge levee crest 3.2–5.6 m
   low), so the ground keeps raw GLO-30 within 25 m of each line (crest error → 1.97 m, the rest
   of the scene unchanged). Crest elevations go to `record["flood_defences"]`.
   **Bridges** (`viewer/bridges.py`, GeoTIFF only): the model reads decks as ground and calls
   most of them water, so OpenStreetMap places them (Overpass, several mirrors, 60 s budget,
   cached per area under the DEM cache). Each deck is interpolated between the raw GLO-30
   elevation at its land ends (connected ways are one structure) and laid over the finished
   DSM; the nDSM gets its height above the bare earth. Deck RMSE vs 3DEP LiDAR 16.6 → 9.0 m
   (downtown Pittsburgh). A failed fetch only means no bridges (`record["bridges"]`).
8. **GCPs** (`read_gcps`, `gcp_correction`): compared with the bare-earth ground. A single
   vertical offset is applied only if it is ≥ 4 m and beats no correction on left-out points.
9. **View-only ground**: the viewer's bare earth is chosen separately by building share
   (`display_ground_method`).

`viewer/height_model.py` wraps RS3DAda (DINOv2 ViT-L + DPT from the cloned SynRS3D repo). It
predicts heights in metres plus 8 OpenEarthMap classes, mapped to GAMUS's 7, in 518 px windows,
with optional 4-flip TTA. `build_model` loads the cached `torch.hub` DINOv2 code from disk, so
model loads never wait on GitHub.

### Server (`viewer/server.py`, FastAPI)

`POST /api/estimate` runs the pipeline in a thread pool, **one upload at a time**
(`_ESTIMATE_LOCK`: the models are shared and `_on` moves v2/CHMv2 between devices in place, so
overlapping uploads crashed each other); `GET /api/progress/{job}` serves its progress. The response carries:
- the record: `models` used, warnings, DSM range, `gcp`;
- previews, plus 16-bit `grids` (1025 a side; `geo.encode_grid16`) that the viewer meshes and
  probes from;
- the city model;
- validation against an optional reference.

How `_validate` treats the reference:
- georeferenced references are reprojected by coordinates (`geo.warp_to_grid`);
- it is compared with the DSM or the nDSM, whichever it matches by median distance;
- `clean_height` is applied only to nDSM references, since it would drop real elevations above
  1000 m;
- outputs: per-class errors, a scatter sample, and a signed error-map PNG.

Models load lazily on the first request. Static mounts:
- `/`: the React build (`frontend/dist`);
- `/dashboards/`: the legacy vanilla dashboards;
- `/data-uploads/`: per-upload outputs.

Unknown well-formed job ids return "Uploading", because the UI polls before the upload lands.

`viewer/city_model.py` builds an LoD1 city **for display only**:
- roofs cut at height steps (`_roof_segments`);
- one block per roof, never banded into height levels: banding the model's soft wall and
  facade slopes stood towers as staircases. Wide slopes steeper than 45° that hang below a roof
  (oblique facades) don't set its height;
- outlines squared off only if that moves ≤ 10 % of their area;
- tree crowns;
- bridge decks as 1.5 m slabs cut 0.5 m apart in elevation (`bridges`), `kind: "bridge"`, added
  after `buildings.geojson` is written so the export stays buildings only.

All OpenStreetMap reads go through `viewer/osm.py` (Overpass mirrors, 60 s budget, cached per
area). The server also places **critical facilities** (`viewer/facilities.py`: hospitals,
clinics, fire, police, shelters, assembly points, schools/colleges/universities): a facility
inside a building marks it (`facility`), and the viewer tints its walls and floats an icon over
it. Embankment crests come back as `embankments` lines for a flood-defence band.

It never feeds the GeoTIFFs; regularising heights costs RMSE. Changes to it are scored with
`viewer.city_eval`.

### Viewer (`frontend/`, React + Vite + three.js from npm)

Almost all of it is one file, `src/main.jsx`.

**Uploads vs catalog scenes**:
- Uploads are drawn faithfully, in metres at true scale, from the 16-bit grids.
- The catalog scenes (`src/gamusScenes.js`, `public/*.jpg`) are **GAMUS LiDAR reference layers,
  restyled, not model output**. Keep their "LiDAR reference heights, not model output" label and
  never present them as results.

**Mesh detail** is chosen at load (`pickMeshSegments`): 1024² segments on capable GPUs, 512² on
software, mobile and Intel integrated graphics. `?detail=high|standard` overrides it.

**Upload-only tools**:
- the Height layer, meaning height above ground, with the legend built from the same
  `JET_STOPS`;
- the Slope and Error layers;
- shader contour lines;
- the A→B profile;
- first-person Walk (`walkGround`, building cells block);
- for PNG/JPG, a scale tool: measure A→B on something of known length, enter it, and the upload
  re-runs at the implied pixel size (heights scale with it: a 1 m screenshot read as 0.33 m lost
  half its building height; entered, building RMSE 5.1 → 3.1 m on GAMUS);
- a building card;
- GLB export.

**Constants that must match the backend**:
- `CLASS_PALETTE` must stay identical to `viewer/classify.py`;
- `ERROR_CSS` must match `viewer/server.py:_error_png`.

When CHMv2 ran, the upload panel shows "Built with DINOv3", which the DINOv3 licence requires
for distributions.

### Legacy paths (kept working, not what the app uses)

- **DA3 diagnostics**:
  - `viewer/metrics.py`, `export_*.py`, `refine*.py`, `validate.py`, `footprints.py`;
  - the vanilla dashboards in `viewer/web/` (vendored three.js, **never a CDN**);
  - `POST /api/upload`.

  The key spike finding: Depth Anything 3 fits a tilted plane to nadir imagery instead of relief
  (`docs/superpowers/spikes/2026-08-24-da3-nadir-domain-gap.md`). That's why the metric path
  uses RS3DAda.
- **Static classify** (`viewer/classify.py`, `POST /api/classify-static`): a land-cover "layer
  cake". Its GAMUS roots are hardcoded to `/home/biplab-dev/...`.
- **`backend/` (Django + SQLite)**: a second caller of `viewer/` as a library;
  `scenes/pipeline.py` ports the old upload processing. The React frontend does not call
  `/api/scenes/`.

### Models and data (gitignored `viewer/cache/` unless noted)

| What | Source |
|---|---|
| `best.pth`: fine-tuned RS3DAda v1 | public HF `Dilavesh/altimap-height`; its README is `docs/model-card.md` (edit there, re-upload) |
| `best_v2.pth`: building specialist (75 % weight on building pixels) | same repo, `v2/best.pth`; trained with SynRS3D data → non-commercial |
| CHMv2 canopy model | public mirror `WEO-SAS/chm-meta-v2` (byte-identical to Meta's gated repo), found in the HF cache; DINOv3 licence |
| `SynRS3D/` model code, stock `RS3DAda` weights (fallback) | `github.com/JTRNEO/SynRS3D` at `ab5a485`; HF `JTRNEO/RS3DAda` |
| GAMUS tiles | HF `earthflow/GAMUS`; `city_eval --fetch` pulls the 40 eval tiles |
| `demo/` (not in git) | NAIP + GAMUS test scenes, LiDAR references, sample GCPs, Sikkim crops |

## Decisions backed by measurement

The numbers are in `ARCHITECTURE.md` §5 and §9 and in `BUILDING_FUSION_075_REPORT.md`. Re-run
the named evaluator before changing any of these:

- **Building fusion 25/75** (`fusion_eval`, `height_eval`): beat 50/50 on all 859 validation
  tiles, the 2,861 untouched test tiles, and a TTA subset. **v2 alone generalises worse than v1**
  (it over-reads trees and ground on NAIP), so it stays building-only.
- **CHMv2 only inside extensive forest** (`dsm_eval`):
  - Outside forests it is worse than v1.
  - The 150 m / 80 % rule leaves city, suburb and hills unchanged; a 30 m window hurt the
    suburb.
- **No forest detail in the DSM** (`dsm_eval`): including it made the forest DSM worse than
  GLO-30 alone.
- **GCP rule** (`dsm_eval`): every looser fit made the DSM worse on the LiDAR scenes. GLO-30 had
  no real offset there, and street points don't represent roofs.
- **Outlines stay traced** (`city_eval`): squaring every outline cost IoU 0.843 → 0.793.
- **Street trees** (GAMUS truth, registered): 82 % of street-tree pixels are classed tree, median
  height 7.7 vs 8.6 m; city crowns cover ~50 % of truth crowns at 88 % precision (dense canopy
  drawn as fewer, larger crowns). Crown spacing/height tuning gained ≤ 3 points: unchanged. A
  LiDAR "roadside canopy" mask on NAIP was mostly misregistered building edges; don't reuse it.
- **One block per roof** (`city_eval`, `dsm_eval` LiDAR): dropping within-roof levels removed the
  facade staircases; IoU 0.843 → 0.848, edge F1 0.663 → 0.674, building RMSE 2.87 → 3.05 m.
- **Not adopted, all measured**:
  - SAM 3 and public building models: none beat ours on `city_eval`.
  - Post-filtering the nDSM: no gain.
  - DA3: the tilted plane above.
- **Network**: the DEM comes from AWS because Planetary Computer's URL signing stalled for
  50–120 s at a time. `viewer/dem.py` sets GDAL and `requests` timeouts at import, so import it
  before any remote read.
- **Base DEM: SRTM by default** (team decision, 2026-10-04: the brief names SRTM). GLO-30 sits
  5–14 m above SRTM in the Himalaya (`dem_check`), and the FAQ names both, so the base stays
  selectable (`base_dem`) and every GeoTIFF is scored against both.

## Load-bearing invariants (breaking these gives silently wrong output, not a crash)

- **`viewer/metrics.py` must never import torch, cv2 or `depth_anything_3`.** A test enforces
  it, which keeps the suite runnable in `.venv`.
- **Datums**: `datum` is `"ellipsoidal"`, `"orthometric"` or `"relative"` (`Sidecar` enforces
  it). GLO-30/SRTM-based DSMs are `"orthometric"`; over India the two differ by tens of metres.
- **DA3 depth**: height is `depth_max − depth`. Larger depth means lower ground for a nadir view.
- **DA3 metric flag**: `Prediction.is_metric` is an empty `addict.Dict` for non-metric models.
  Test its truthiness; `int()` raises.
- **Legacy dashboards** (curated + off-nadir): no metre values anywhere. Their axes are relative
  and their slope readout is "display slope".
- **Frozen spike**: don't modify `spikes/04_da3_nadir_check.py`; the 2026-08-24 findings doc
  depends on it.
- **Roboflow YOLO labels** have no trailing newline: read them per file, never concatenated.

## Testing conventions

- **Synthetic fixtures**: `tests/conftest.py` builds long-tailed synthetic nDSMs (mostly flat
  ground plus a few tall blocks), because metrics that pass on uniform noise can fail on real
  elevation distributions.
- **Degenerate inputs are first-class**: constant depth, zero median, single-channel images
  return `nan` for the affected metric; they never raise and never become `0.0`.
- **Model code needs no torch in tests**: test the pure functions (`fuse_heights`,
  `forest_mask`, `predict_scene` with a fake per-pixel model, `gcp_correction`, `warp_to_grid`).
- **Network**: a `network` pytest marker exists for live-network tests; none currently uses it.
- **Viewer changes**: build the frontend and check upload, navigation and export in a browser.
