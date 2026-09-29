# AltiMap

Single-view optical RGB imagery → metric elevation (nDSM / absolute DSM) → interactive 3D flythrough.
Built for SIH 2026 problem statement 26175, *DepthWizard* (ISRO). Brief: `docs/problem-statement.md`.

| Input | Output |
|---|---|
| PNG / JPG (no coordinates) | **nDSM**: height above ground in metres, per pixel (a relative DSM: heights are metric, the ground datum is unknown) |
| GeoTIFF (with CRS) | **nDSM** + **absolute DSM** = nDSM + bare-earth ground from Copernicus GLO-30, same CRS and grid as the input |

All elevation outputs are float32 Cloud-Optimized GeoTIFFs in metres, NaN nodata, with a JSON sidecar
(GSD, vertical datum, height range, DEM source).

## How it works

1. **Height model.** RS3DAda ([SynRS3D, NeurIPS 2024](https://github.com/JTRNEO/SynRS3D)): a DINOv2 ViT-L
   encoder with a DPT decoder, pre-trained for monocular height on 69k synthetic remote-sensing
   images. We fine-tune it on **GAMUS** (the organisers' reference dataset: 0.33 m aerial RGB with
   LiDAR-derived nDSM), training only the encoder's bias terms (BitFit) plus the decoder, with a joint
   land-cover head as a semantic prior. Loss: L1 on metres + L1 on height gradients + 0.2 × class
   cross-entropy.
2. **Scale.** The model outputs metres directly. Inputs are resampled to the model's 0.33 m ground
   sampling distance (from the GeoTIFF, or a user-supplied GSD), then back to the input grid.
3. **Absolute elevation (GeoTIFF input).** The organisers score GeoTIFF output against SRTM/Copernicus
   and require values that "match DEM heights". So the exported DSM is **DEM-consistent**: Copernicus
   GLO-30 supplies absolute level and terrain, and the model adds the detail inside each 30 m cell with
   its own cell-mean removed (`DSM = GLO-30 − mean₃₀(nDSM) + nDSM`). Averaged over 30 m, the DSM
   reproduces GLO-30; at full resolution it has the buildings and trees. Heights are **orthometric
   (EGM2008 geoid)**, and the sidecar says so. Known trade-off: in dense downtowns GLO-30's radar
   under-records building mass, so street pixels come out below their true (LiDAR) level while the
   30 m average still matches the DEM.
4. **Terrain in the 3D view.** A bare-earth estimate chosen by scene type (measured against USGS
   LiDAR on four scenes): a 300 m morphological opening of GLO-30 for dense cities, 150 m for towns
   (keeps hills), model subtraction for forest and farmland. Display only.
5. **3D city model (display).** From the same prediction, `viewer/city_model.py` builds an LoD1
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
6. **Viewer.** React + three.js. The RGB image is draped on a 513 × 513 displaced mesh, with orbit,
   WASD/QE fly controls and waypoint flythroughs. Uploads show a height-and-slope probe
   (double-click in Measure mode), a centre-line height profile, land-cover layers, GeoTIFF
   downloads, and RMSE/MAE/correlation against an optional uploaded reference height map.
   Drop an image anywhere on the view or use Import; a live progress bar shows each processing stage,
   and *Quality: Fast* skips the 4-flip averaging (~4× quicker). Click a building for its height,
   estimated floors, footprint area and roof elevation; *Export* saves the current 3D model as a
   `.glb` in metres (opens in Blender and other 3D tools).

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

## Run it

Teammates setting up from scratch: follow **[SETUP.md](SETUP.md)** (every step, model
downloads, troubleshooting). The short version:

Needs Python 3.12, Node 18+, an NVIDIA GPU (6 GB is enough for inference; CPU works, slowly), and
internet on first run (DINOv2 model code from GitHub, then cached; GLO-30 ground from Microsoft
Planetary Computer for GeoTIFF inputs).

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
# fine-tuned weights (private repo: `hf auth login` with an account that has access first)
.venv-da3/bin/python -c "from huggingface_hub import hf_hub_download; hf_hub_download('Dilavesh/altimap-height', 'best.pth', local_dir='viewer/cache')"

# 3a. Batch CLI: images in, GeoTIFFs out
.venv-da3/bin/python -m viewer.estimate scene.tif photo.png --out results/
.venv-da3/bin/python -m viewer.estimate photo.jpg --gsd 0.5 --out results/   # known pixel size

# 3b. Interactive app
.venv-da3/bin/python -m viewer.server            # API on http://127.0.0.1:8000
cd frontend && npm install && npm run build && npm run preview -- --host 127.0.0.1 --port 5173
                                                 # open http://127.0.0.1:5173 → Upload tab
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
- **PNG/JPG without a GSD** is assumed to be 0.33 m/pixel. A wrong assumption changes how large
  objects look to the model and biases the predicted heights, so pass `--gsd` (or fill the UI
  field) whenever the pixel size is known.
- **Absolute DSM accuracy is bounded by GLO-30** (30 m posting, ~2–4 m vertical accuracy) for the
  ground component. Hilly terrain relief comes from the DEM, not the model.
- **Vertical datum** is orthometric (EGM2008). Comparing against ellipsoidal references needs a
  geoid correction.
- The RS3DAda weights carry no explicit licence. Research use only.
