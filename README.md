# AltiMap

Single-view optical RGB imagery → metric elevation (nDSM / absolute DSM) → interactive 3D flythrough.
Built for SIH 2026 problem statement 26175, *DepthWizard* (ISRO). Brief: `docs/problem-statement.md`.

| Input | Output |
|---|---|
| PNG / JPG (no coordinates) | **nDSM**: height above ground in metres, per pixel (a relative DSM: heights are metric, the ground datum is unknown) |
| GeoTIFF (with CRS) | **nDSM** + **absolute DSM** on SRTM (default, as the brief suggests) or Copernicus GLO-30, same CRS and grid as the input, scored against both DEMs |

All elevation outputs are float32 Cloud-Optimized GeoTIFFs in metres, NaN nodata, with a JSON sidecar
(GSD, vertical datum, height range, DEM source).

Docs: **[SETUP.md](SETUP.md)** (install and run), **[ARCHITECTURE.md](ARCHITECTURE.md)** (how every
part works, data formats, API, models, measured results and limits), [model card](docs/model-card.md), and [demo video script](docs/DEMO_VIDEO_SCRIPT.md).

## How it works

1. **Height model.** RS3DAda ([SynRS3D, NeurIPS 2024](https://github.com/JTRNEO/SynRS3D)): a DINOv2 ViT-L
   encoder with a DPT decoder, pre-trained for monocular height on 69k synthetic remote-sensing
   images. We fine-tune it on **GAMUS** (the organisers' reference dataset: 0.33 m aerial RGB with
   LiDAR-derived nDSM), training only the encoder's bias terms (BitFit) plus the decoder, with a joint
   land-cover head as a semantic prior. Loss: L1 on metres + L1 on height gradients + 0.2 × class
   cross-entropy.
2. **Three models, by land cover.** Each was measured against USGS airborne LiDAR
   (ARCHITECTURE.md §5):
   - **Fine-tune v1**: heights everywhere.
   - **v2** (8 encoder blocks, height-weighted loss, SynRS3D high-rises): contributes 75% of the
     height on building pixels (v1 contributes 25%), selected on full GAMUS validation and confirmed
     on the untouched test split. Alone it over-reads trees and ground on unfamiliar imagery.
   - **Meta's CHMv2 canopy model**: tree heights inside extensive forest (≥ 80 % canopy within
     150 m), where the main model reads trees ~16 m low. It is worse than ours outside forests.
   - The last two are optional downloads.
3. **Scale.** The models output metres directly. Inputs are resampled to the model's 0.33 m ground
   sampling distance (from the GeoTIFF, or a user-supplied GSD), then back to the input grid.
   Large scenes are processed in ~1 km tiles with a context margin, so every part is seen at full
   resolution.
4. **Absolute elevation (GeoTIFF input).** The organisers score GeoTIFF output against SRTM/Copernicus
   and require values that "match DEM heights". So the exported DSM is **DEM-consistent**: the base
   DEM (SRTM by default, Copernicus GLO-30 on request or where SRTM has no coverage) supplies
   absolute level and terrain, and the model adds the detail inside each 30 m cell with its own
   cell-mean removed (`DSM = DEM − mean₃₀(nDSM) + nDSM`). Averaged over 30 m, the DSM reproduces the
   DEM; at full resolution it has the buildings and trees. Heights are **orthometric** (EGM96 on
   SRTM, EGM2008 on GLO-30), and the sidecar says so. In dense downtowns the radar DEMs under-record
   building mass; keeping the cell mean would push streets underground, so the DSM is floored at bare
   earth. In extensive forest the fine detail is left out (both models place it poorly under closed
   canopy), so the DSM there is the DEM's own surface.
5. **Terrain in the 3D view.** A bare-earth estimate chosen by scene type (measured against USGS
   LiDAR on four scenes): a 300 m morphological opening of the selected base DEM for dense cities, 150 m for towns
   (keeps hills), model subtraction for forest and farmland. Display only.
6. **3D city model (display).** From the same prediction, `viewer/city_model.py` builds an LoD1
   city: each building complex cut into roofs where the roof height steps or dips (touching row
   houses become separate blocks) and into distinct height levels (a tower on a podium becomes two
   blocks), outlines kept as traced unless squaring them off barely moves them, extruded with the
   photo on the roof. On 40 GAMUS val tiles (`viewer/city_eval.py`) this scores footprint IoU 0.841
   and edge F1 0.655 against the true outlines, 2372 separate buildings (was 0.787 / 0.564 / 1281);
   SAM 3 masks scored lower (0.72–0.78 IoU) and aren't used. Trees are detected as individual crowns
   (height, width and colour from the image), and flat ground. It is also exported as
   `buildings.geojson` (footprint + height, WGS84 for GeoTIFF input). This is for the 3D view only:
   regularizing heights this way raised RMSE from 2.66 m to 2.86–3.38 m on validation tiles, so the
   GeoTIFFs stay the raw model output. The viewer's *Exact surface* toggle shows that raw DSM.
7. **Viewer.** React + three.js. The RGB image is draped on a displaced mesh: 1025 × 1025 on
   capable GPUs, 513 × 513 on integrated or software graphics, chosen automatically. Navigation:
   orbit, WASD/QE flight, first-person **Walk** and waypoint flythroughs.

   Uploads get:
   - a height-and-slope probe and an **A→B height profile** (Measure mode);
   - **Slope** and **Error** layers and **contour lines**;
   - GeoTIFF downloads;
   - validation against an optional reference height map (RMSE/MAE/correlation, bias, per-class
     errors, scatter plot).
   Use the **Image** tab to upload or choose sample scenes, **View** for layers and scale,
   **Measure** for profiles and routes, **Accuracy** for validation, and **Context** for mapped facilities.
   In **View → Ground display**, **Flat ground** removes coarse-DEM street ramps for inspecting
   buildings. It is a labelled visualization: terrain and flood-defence relief are hidden,
   building heights stay estimated, and geographic readouts and GeoTIFF downloads keep their
   original elevations. The flat GLB filename includes `flat-ground`.
   Drop an image anywhere on the view or use **Open image**; a live progress bar shows each processing stage,
   and *Quality: Fast* skips the 4-flip averaging (~4× quicker). Click a building for its height,
   estimated floors, footprint area and roof elevation; *Export* saves the current 3D model as a
   `.glb` in metres (opens in Blender and other 3D tools). For PNG/JPG, a scale tool turns one
   known length (a road lane, a car) into the pixel size and reprocesses: heights scale with it.
8. **Disaster context from OpenStreetMap (GeoTIFF input).** What a height model can't see:
   - **bridges**: decks the model reads as ground, interpolated between the terrain at their ends
     and added to the DSM (deck RMSE vs LiDAR 16.6 → 9.0 m, downtown Pittsburgh);
   - **levees, embankments, dams**: kept in the terrain, which the bare-earth step would erase
     (Mississippi levee crest error 3.3–5.6 → 2.0 m), and drawn as flood-defence lines with their
     crest elevation;
   - **critical facilities**: hospitals, clinics, fire and police stations, shelters, schools,
     highlighted in 3D with an icon each.

   Answers are cached per area, so a scene keeps them offline; without the network the DSM is
   still produced, just without these layers.

## Results (GAMUS)

Metrics are pixel-pooled over whole 1024 × 1024 tiles, in metres. *Building RMSE* is restricted to
LiDAR building pixels. It matters because all-pixel RMSE hides how badly tall structures are
underestimated. Two trivial baselines are included on purpose: a model that can't beat *predict the
average height* has learned nothing.

**Zero-shot RS3DAda (before fine-tuning), 300 val tiles:**

| City | RMSE | MAE | r | Building RMSE |
|---|---|---|---|---|
| Washington DC | 9.08 | 5.13 | 0.48 | 7.54 |
| Philadelphia | 2.53 | 1.33 | 0.72 | 3.86 |
| **All** | **6.08** | **2.86** | **0.53** | **6.06** |
| baseline: predict 0 | 8.00 | 4.00 | — | 11.16 |
| baseline: predict train mean | 8.04 | 6.94 | — | 7.18 |

**Held-out test split, 500 tiles (none used for training or checkpoint selection), 4-flip TTA:**

| City | Model | RMSE | MAE | r | Building RMSE |
|---|---|---|---|---|---|
| Washington DC | **fine-tuned** | **3.95** | **2.17** | **0.915** | **4.51** |
| | zero-shot | 10.08 | 5.80 | 0.410 | 6.80 |
| New York | **fine-tuned** | **3.95** | **2.02** | **0.855** | **3.20** |
| | zero-shot | 7.17 | 3.94 | 0.480 | 4.82 |
| Philadelphia | **fine-tuned** | **5.90** | **1.39** | **0.822** | **10.04** |
| | zero-shot | 6.28 | 1.92 | 0.777 | 10.57 |
| **All** | **fine-tuned** | **5.02** | **1.72** | **0.833** | **8.28** |
| | zero-shot | 7.25 | 3.19 | 0.638 | 9.06 |
| | predict 0 | 10.26 | 4.89 | — | 16.28 |
| | predict train mean | 9.98 | 7.66 | — | 12.66 |

Fine-tuning cuts overall RMSE by 31% and MAE by 46%, and raises correlation from 0.64 to 0.83.
Land-cover pixel accuracy is 0.88, and 95.5% of LiDAR building pixels are recognised as buildings.
The weak spot is tall buildings: Philadelphia's test tiles include high-rises, and its building
RMSE (10.0 m) against MAE (1.4 m) shows a small number of very tall pixels being underestimated
(the long-tail problem, §4.3 of the design doc). New York was in the training split; the validation
split used for checkpoint selection covered DC and Philadelphia only.

## Results by landscape (USGS 3DEP airborne LiDAR)

The current 25/75 building-fusion pipeline was rerun on 2026-10-04 on four NAIP
scenes against USGS 3DEP (`viewer.dsm_eval`, single pass, no flip averaging).
All figures below are all-valid-pixel errors in metres; parentheses show the
same base DEM alone. SRTM remains the default requested by the brief.

| Scene (pixel size) | nDSM RMSE (predict 0) | SRTM DSM RMSE (DEM alone) | GLO-30 DSM RMSE (DEM alone) |
|---|---|---|---|
| Dense city, Philadelphia (0.3 m) | 31.58 (42.35) | 38.87 (40.56) | 34.36 (36.26) |
| Suburb, Chevy Chase (0.6 m) | 4.47 (6.48) | 4.28 (4.71) | 3.62 (4.02) |
| Hilly town, Pittsburgh (0.6 m) | 3.55 (6.48) | 6.24 (7.26) | 4.75 (5.53) |
| Forest, Smoky Mountains (0.6 m) | 15.67 (24.66) | 7.44 (7.13) | 8.75 (8.53) |

The model improves absolute DSM error in the city, suburb and hills, while forest
is slightly worse than the DEM alone. Dense-city errors and a −10.75 m canopy
bias remain substantial. High forest DSM correlation reflects terrain relief,
not accurate tree reconstruction. Eight LiDAR ground points reduce SRTM city
DSM RMSE to 37.18 m on the other pixels.

See the [current report](docs/evaluation/landscapes-2026-10-04.md) for MAE,
correlation, datum caveats and reproduction commands, and its
[raw metrics](docs/evaluation/landscapes-2026-10-04.json). Historical comparisons
and resolution experiments remain in [ARCHITECTURE.md](ARCHITECTURE.md) §9.

### Current building-fusion evidence

The active building rule is **25% v1 + 75% v2**, with v1 unchanged elsewhere. After fixing invalid
GAMUS `-5 m` reference masking, it beat the former 50/50 route on all 859 validation tiles
(overall/building RMSE 2.7511/3.3352 → 2.7402/3.2763 m) and all 2,861 untouched test tiles
(3.7573/5.2308 → 3.6972/4.9934 m). A 24-tile High-quality/TTA check also improved both pooled
metrics (3.1690/5.1284 → 3.0916/4.9140 m); the ordinary-scene overall regression was only 0.04%.
Soft routing was tested and not adopted because it did not provide a meaningful enough gain.

## Run it

With Docker (weights included): `docker build -t altimap . && docker run --gpus all -p 8000:8000
altimap`, then open http://localhost:8000 (SETUP.md has the requirements).

Teammates setting up from scratch: follow **[SETUP.md](SETUP.md)** (every step, model
downloads, troubleshooting). The short version:

Needs Python 3.12, Node 20+, an NVIDIA GPU (6 GB is enough for inference; CPU works, slowly), and
internet for initial model downloads and uncached GeoTIFF context (SRTM from OpenTopography,
GLO-30 from AWS Open Data with Planetary Computer fallback, and OpenStreetMap queries).

```bash
# 1. Python environment
uv venv --python 3.12 .venv-da3
uv pip install --python .venv-da3/bin/python torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv-da3/bin/python h5py numpy scipy rasterio pillow fastapi uvicorn \
    python-multipart pystac-client planetary-computer huggingface_hub shapely
uv pip install --python .venv-da3/bin/python -e . --no-deps

# 2. Model code + weights (fine-tuned checkpoint goes to viewer/cache/best.pth; if it is
#    absent, the stock RS3DAda weights are used)
git clone https://github.com/JTRNEO/SynRS3D.git viewer/cache/SynRS3D
.venv-da3/bin/python -c "from huggingface_hub import hf_hub_download; hf_hub_download('JTRNEO/RS3DAda', 'RS3DAda_vitl_DPT_height.pth', local_dir='viewer/cache/SynRS3D/pretrain')"
# fine-tuned weights, public: https://huggingface.co/Dilavesh/altimap-height
.venv-da3/bin/python -c "from huggingface_hub import hf_hub_download; hf_hub_download('Dilavesh/altimap-height', 'best.pth', local_dir='viewer/cache')"

# 3a. Batch CLI: images in, GeoTIFFs out
.venv-da3/bin/python -m viewer.estimate scene.tif photo.png --out results/
.venv-da3/bin/python -m viewer.estimate photo.jpg --gsd 0.5 --out results/   # known pixel size

# 3a'. With ground control points (CSV lon, lat, height) correcting the DSM's vertical offset
.venv-da3/bin/python -m viewer.estimate scene.tif --gcps points.csv --out results/

# 3b. Interactive app: build the viewer once, then one server serves app + API
(cd frontend && npm ci && npm run build)
.venv-da3/bin/python -m viewer.server            # open http://127.0.0.1:8000
.venv-da3/bin/python -m viewer.server --host 0.0.0.0   # on a VM, reachable from other machines
```

Reproduce training and evaluation (24 GB GPU, ~8 h):

```bash
python -m viewer.height_eval --split val --limit 300 --out zeroshot_val.json    # zero-shot baseline
python -m viewer.height_train --hours 8 --out runs/ft                           # keeps runs/ft/best.pth
python -m viewer.height_eval --split test --ckpt runs/ft/best.pth --out test.json
```

(`ALTIMAP_DATA` points at a folder holding `gamus/` and `SynRS3D/`.)

Tests (no GPU, no network): `uv pip install -e ".[dev]"`, then `python -m pytest -q`.

## Known limitations

- **Training domain.** GAMUS is 0.33 m aerial imagery from three US cities. On imagery that differs a
  lot in resolution, off-nadir angle or landscape (hilly, forested, rural India), accuracy will drop.
  Supplying the true GSD matters most.
- **PNG/JPG without a GSD** uses an explicitly reported experimental 0.33 m/pixel assumption.
  Measured on the 0.6 m NAIP scenes
  uploaded as plain PNGs, the wrong assumption moved nDSM RMSE by −0.7 to +1.0 m (better in the
  suburb, worse in the hills and forest). Pass `--gsd` (or fill the UI field) when it's known.
- **Absolute DSM accuracy depends on the chosen 30 m base DEM** (SRTM by default,
  or Copernicus GLO-30) for the ground component. Hilly terrain relief comes from the DEM, not the model.
- **Vertical datum** is orthometric: EGM96 on SRTM, EGM2008 on GLO-30. Comparing against ellipsoidal references needs a
  geoid correction.
- **Very tall buildings and forest canopy** remained low in the historical 50/50 external run
  (towers ~104 m where LiDAR says 151 m; canopy ~11 m low). The current 25/75 run still has 31.58 m city nDSM RMSE and a −10.75 m canopy bias;
  the historical tower percentile is not remeasured by the current evaluator.
- **Large scenes** run tile by tile at full resolution (no seams: ARCHITECTURE.md §9.5). They
  take minutes on a laptop GPU.
- **Network**: GeoTIFFs need Copernicus GLO-30 (AWS Open Data, with Microsoft Planetary Computer as
  fallback) and SRTM (OpenTopography); remote reads time out after 60 s and then return the nDSM
  only.
- **Licences**: RS3DAda weights are MIT (JTRNEO/RS3DAda), and so are our fine-tuned v1 weights.
  v2 is also trained on SynRS3D data (CC BY-NC 4.0): treat it as non-commercial.
