# Setting up AltiMap

Everything a teammate needs to go from a fresh clone to the running app: upload a satellite or
aerial image, get a metric height model, a 3D city model and GeoTIFF downloads. Tested on Ubuntu
with an RTX 3050 6 GB laptop GPU; each step says what it is for, so you can tell what to skip.

## What you need

| | Version | Why |
|---|---|---|
| Linux (Ubuntu tested) | — | macOS/Windows are untested; CPU-only works but is slow |
| NVIDIA GPU + driver for CUDA 12.x | 6 GB VRAM or more | the height model (DINOv2 ViT-L) runs here; training needs 24 GB |
| [uv](https://docs.astral.sh/uv/) | any recent | creates the Python envs and installs Python 3.12 itself |
| Node.js | 20+ (22 tested) | builds the React viewer |
| git | — | also clones the SynRS3D model code |
| Disk | ~15 GB | ~7 GB Python env (PyTorch), ~6 GB model weights (3 height models + CHMv2), the rest data/cache |
| RAM | 16 GB recommended | the server holds ~5.5 GB with all three models loaded (two wait on the CPU while one runs on the GPU) |
| Internet | first run | model downloads; GeoTIFF uploads fetch the Copernicus DEM live |

## Quick start: Docker (one command, e.g. on a VM)

The `Dockerfile` builds the whole app (viewer, API, all three models, weights baked in) into one
image, so the container starts ready. Needs Docker with the NVIDIA container toolkit on the host
(`nvidia-ctk`), a driver for CUDA 12.8, and **~35 GB free disk while building** (PyTorch with
its CUDA libraries is 7 GB, the weights 4.3 GB, and Docker keeps a compressed copy plus build
cache). The final image is ~12 GB.

```bash
git clone https://github.com/ManasBytes/altimap.git && cd altimap && git checkout dilavesh-new
docker build -t altimap .                       # ~10 min; downloads PyTorch and the weights
docker run --gpus all -p 8000:8000 altimap      # then open http://<host>:8000
```

Without `--gpus all` it runs on the CPU (a 1024 px tile took 86 s instead of ~16 s). Uploads
live inside the container; add `-v altimap-uploads:/app/viewer/web/data-uploads` to keep them.
Tested: GPU visible in the container, both DEM sources reachable, an upload through v1 + v2.

The rest of this guide is the native install, for development.

## 1. Clone

```bash
git clone https://github.com/ManasBytes/altimap.git
cd altimap
git checkout dilavesh-new   # current working branch
```

## 2. Python environments

Two envs on purpose (see `CLAUDE.md`): `.venv-da3` runs the app and the models, `.venv` runs the
tests without PyTorch. Don't merge them.

```bash
# App + models (the one you need to run anything)
uv venv --python 3.12 .venv-da3
uv pip install --python .venv-da3/bin/python torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv-da3/bin/python h5py numpy scipy rasterio pillow fastapi uvicorn \
    python-multipart pystac-client planetary-computer huggingface_hub shapely requests trimesh \
    transformers safetensors
uv pip install --python .venv-da3/bin/python -e . --no-deps

# Tests only (no GPU, no network)
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
```

`cu128` wheels need an NVIDIA driver that supports CUDA 12.8 (`nvidia-smi` shows the maximum
in its header). With an older driver use `cu126` or `cu124` in the index URL instead.

## 3. Model code and weights

All of these land in `viewer/cache/`, which is gitignored.

```bash
# RS3DAda model code (DINOv2 ViT-L + DPT); pinned to the commit we developed against
git clone https://github.com/JTRNEO/SynRS3D.git viewer/cache/SynRS3D
git -C viewer/cache/SynRS3D checkout ab5a485

# Stock RS3DAda weights (public, 1.5 GB): the fallback if the fine-tuned model is missing
.venv-da3/bin/python -c "from huggingface_hub import hf_hub_download; \
hf_hub_download('JTRNEO/RS3DAda', 'RS3DAda_vitl_DPT_height.pth', local_dir='viewer/cache/SynRS3D/pretrain')"

# Our fine-tuned weights (public, no login needed): https://huggingface.co/Dilavesh/altimap-height
#   best.pth (1.5 GB): the main height model (v1)
#   v2/best.pth (1.5 GB) -> viewer/cache/best_v2.pth: second model, used for building heights
.venv-da3/bin/python -c "from huggingface_hub import hf_hub_download; import shutil; \
hf_hub_download('Dilavesh/altimap-height', 'best.pth', local_dir='viewer/cache'); \
shutil.copy(hf_hub_download('Dilavesh/altimap-height', 'v2/best.pth'), 'viewer/cache/best_v2.pth')"

# Meta CHMv2 canopy-height model (1.35 GB), used for forest canopy. This public mirror is
# byte-identical to Meta's gated facebook/dinov3-vitl16-chmv2-dpt-head (DINOv3 licence)
.venv-da3/bin/python -c "from huggingface_hub import snapshot_download; \
snapshot_download('WEO-SAS/chm-meta-v2', allow_patterns=['*.json', 'model.safetensors', 'LICENSE.md'])"
```

What each model does (all measured against USGS LiDAR, see ARCHITECTURE.md §5):
- `best.pth` gives the heights everywhere.
- `best_v2.pth` contributes 75% of the result on building pixels; `best.pth` contributes 25%.
  This blend was selected on full GAMUS validation and confirmed on the untouched test split.
  v2 alone over-reads trees and ground on unfamiliar imagery.
- CHMv2 replaces canopy heights inside extensive forest, where the main model reads trees
  ~16 m low.
- The last two are optional: without them the app runs with `best.pth` alone and says so in
  the upload panel ("Models").

The server uses `viewer/cache/best.pth` if it exists, else the stock weights. It also honours
`ALTIMAP_HEIGHT_CKPT=/path/to/ckpt.pth`. The fine-tuned model is much better (test RMSE 5.02 m
vs 7.25 m zero-shot, see README), so don't run on the fallback. The model card at
https://huggingface.co/Dilavesh/altimap-height has the architecture, results, known limitations
and licences; its source is `docs/model-card.md`. v1 is MIT. v2 is also trained on SynRS3D data
(CC BY-NC 4.0), so treat it as non-commercial. CHMv2 is under Meta's DINOv3 licence; when CHMv2
is loaded, the viewer shows the required "Built with DINOv3" credit in the model list. Review
`THIRD_PARTY_NOTICES.md` before distributing model weights.

The DINOv2 encoder code is fetched by `torch.hub` from GitHub the first time a model loads, then
loaded from `~/.cache/torch/hub` without contacting GitHub. After that the app runs offline,
except for GeoTIFF uploads, which read the Copernicus GLO-30 DEM from AWS Open Data (Microsoft
Planetary Computer as fallback), SRTM GL1 from OpenTopography's public copy, and bridges,
embankments and critical facilities from OpenStreetMap (Overpass API; answers are cached per
area, so a scene keeps them offline). Reprojected DEM patches are cached in `viewer/cache/dem/`;
set `ALTIMAP_DEM_CACHE` to put that cache on another disk. If a DEM request is incomplete,
the unresolved pixels remain nodata and the app returns the nDSM with a clear DSM warning;
they are never silently converted to sea level.

## 4. Frontend

```bash
cd frontend
npm ci
npm run build
cd ..
```

## 5. Run the app

One command from the repo root. The server serves the built viewer (step 4) and the API on the
same port:

```bash
.venv-da3/bin/python -m viewer.server                  # http://127.0.0.1:8000
.venv-da3/bin/python -m viewer.server --host 0.0.0.0   # on a VM: http://<vm-address>:8000
```

On a VM, either open port 8000 in its firewall, or keep it private and tunnel it from your laptop
with `ssh -L 8000:127.0.0.1:8000 <user>@<vm>`, then open http://127.0.0.1:8000. The server accepts
uploads and runs the model on them, so don't expose it to networks you don't trust.
The model loads on the first upload.

For a Vercel frontend backed by prepared Hugging Face scenes and a GPU VM, see
[DEPLOYMENT.md](docs/DEPLOYMENT.md). Prepared demos work while the VM is off; live processing
availability is reported separately.

(Frontend development: `cd frontend && npm run dev` serves on :5173 and forwards `/api` to the
server on :8000, see `frontend/vite.config.js`. Rebuild with `npm run build` for step 5.)

Open the app, click **Open image** (or drop a file on the viewport). The **Image** tab
contains upload options; **View** holds layers and exaggeration, **Measure** holds
profiles and camera routes, **Accuracy** shows validation, and **Context** lists mapped facilities:

- **PNG / JPG** gives heights above ground (nDSM) and a 3D city model on flat ground. Enter the
  pixel size in metres if you know it. Without one, the app uses an explicitly reported
  experimental 0.33 m/pixel assumption for resampling, heights and footprint scale;
  it is not known physical truth.
- **GeoTIFF** also gives an absolute DSM (orthometric metres: EGM96 on SRTM, EGM2008 on GLO-30)
  on real terrain when its CRS and grid are north-up and square. Rotated/sheared or strongly non-square inputs are
  rejected for absolute DSM export with a warning instead of silently producing wrong geometry.
- **Add reference heights** (optional): a single-band height map, either height above ground
  or an absolute DSM. Select **Reference heights represent** to match your file:
  **Height above ground** for GAMUS AGL/nDSM, **Absolute elevation** for LiDAR DSM.
  Auto-detect is available, but can guess incorrectly near sea level. A georeferenced reference
  is reprojected onto the image by its coordinates; any other is assumed to cover the same area. The app scores the
  model (RMSE, MAE, correlation, bias, per land-cover class, a scatter plot) and adds an
  **Error** layer showing where the model reads high or low.
- **Add ground control points** (optional, GeoTIFF only): a CSV of `lon, lat, height` (WGS84
  degrees, metres above mean sea level, the same datum as the DEM; a header row naming the
  columns is fine). They are compared with the bare-earth ground. The DSM is shifted by their
  mean difference only if that is at least 4 m (beyond GLO-30's own accuracy) and consistent
  across the points; otherwise nothing changes, and the panel says why. That's for real vertical
  offsets: e.g. GPS (ellipsoidal) heights vs EGM2008 differ by tens of metres over India.
  The current SRTM Philadelphia check applies an offset and reduces held-out-pixel DSM RMSE from 38.87 to 37.18 m; suburb/hills and the GLO-30 checks decline correction.
- **Quality**: *High* averages 4 flipped passes (~3x slower), *Fast* runs one pass.
- **Time**: a 2000 px GeoTIFF takes about 1.5–2 minutes on *Fast* with an RTX 3050 laptop GPU
  (two height-model passes, plus CHMv2 in forests). Scenes wider than ~1 km are processed in
  tiles and take proportionally longer.

In the viewer:
- Click a building for its height, floors, footprint, volume and roof elevation.
- **Measure**, then double-click the terrain: height, elevation and slope at that point; a
  second double-click draws the height profile from A to B.
- Layers: **Surface**, **Height** (above ground), **RGB**, **Classes**, **Slope** (degrees) and
  **Error** (with a reference); **Contour lines** at an automatic interval.
- **Walk**: first person at eye height on the ground, WASD (Shift runs), drag to look,
  buildings block you.
- **Export** saves the 3D model as `.glb`. The GeoTIFFs (`ndsm.tif`, `dsm.tif`) and
  `buildings.geojson` download from the right panel.
- The 3D mesh detail follows the GPU: full detail on dedicated GPUs, half on integrated or
  software graphics. Add `?detail=high` or `?detail=standard` to the address to force either.

Test images: `demo/` is supplied locally and is not in git. It holds:
- 3 GAMUS tiles (PNG) and NAIP GeoTIFFs covering city, suburb, hills, forest, bridges and a levee;
- `reference_heights/`: their LiDAR heights. Use these in the reference box, **not** as the
  image;
- `gcps/`: sample control-point CSVs for the NAIP scenes;
- `india/`: three Sikkim satellite scenes. `scripts/fetch_india_samples.py` downloads these
  itself.

Other RGB aerial or satellite images can be uploaded; accuracy depends on their domain and GSD.
For a recording walkthrough, use [docs/DEMO_VIDEO_SCRIPT.md](docs/DEMO_VIDEO_SCRIPT.md).

Batch use without the viewer:

```bash
.venv-da3/bin/python -m viewer.estimate scene.tif photo.png --out results/
.venv-da3/bin/python -m viewer.estimate photo.jpg --gsd 0.5 --out results/   # known pixel size
```

## 6. Tests

```bash
.venv/bin/python -m pytest -q     # ~110 tests, under a second, no GPU or network
```

## 7. Evaluation and training (optional)

Stop the server first on a 6 GB GPU: the server's model and an evaluation's model don't both fit.

```bash
# 3D city model (building outlines + heights) on 40 GAMUS val tiles; --fetch downloads them once
.venv-da3/bin/python -m viewer.city_eval --fetch

# DSM accuracy by landscape (city, suburb, hills, forest) and by pixel size (native -> 10 m),
# against USGS 3DEP airborne LiDAR; needs demo/ and internet the first time (LiDAR is cached)
.venv-da3/bin/python -m viewer.dsm_eval

# India: Maxar Open Data satellite scenes of Sikkim (CC BY-NC 4.0), checked against the DEMs the
# organisers score GeoTIFFs against (Copernicus GLO-30 and SRTM GL1); --base-dem srtm puts the DSM on SRTM
.venv-da3/bin/python scripts/fetch_india_samples.py     # -> demo/india/*.tif (~40 MB each)
.venv-da3/bin/python -m viewer.dem_check demo/india/*.tif

# Height model per city against baselines (needs GAMUS under $ALTIMAP_DATA/gamus, default ~/altimap-data)
.venv-da3/bin/python -m viewer.height_eval --split test --ckpt viewer/cache/best.pth \
    --synrs3d viewer/cache/SynRS3D --tta --out test.json

# Building fusion ablation: prepare model predictions once, then compare on CPU.
.venv-da3/bin/python -m viewer.fusion_eval --prepare --gamus "$ALTIMAP_DATA/gamus" \
    --synrs3d viewer/cache/SynRS3D --v1 viewer/cache/best.pth --v2 viewer/cache/best_v2.pth
.venv-da3/bin/python -m viewer.fusion_eval --report --gamus "$ALTIMAP_DATA/gamus" \
    --synrs3d viewer/cache/SynRS3D --v1 viewer/cache/best.pth --v2 viewer/cache/best_v2.pth

# Fine-tuning (24 GB GPU, ~8 h): fetch the data, then train
.venv-da3/bin/python scripts/fetch_training_data.py
.venv-da3/bin/python -m viewer.height_train --hours 8 --out runs/ft
```

`scripts/overnight_v2.sh` is the full unattended run we used on the A5000 (data, training,
test with TTA, upload to Hugging Face).

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Couldn't reach the height backend` in the viewer | `viewer.server` isn't running, or something else holds port 8000 (Django's `make dev` also defaults to :8000 and can't serve uploads). With `npm run dev`, the server must be on :8000 |
| `no height checkpoint; expected one of ...` | Step 3 weights are missing from `viewer/cache/` |
| `could not process this image: ... looks like a height or elevation map` | You uploaded a single-band height map as the image. Upload the RGB image; put the height map in the reference box |
| pytest fails with errors from `/opt/ros/...` or `launch_testing` | ROS is on `PYTHONPATH`: run `env -u PYTHONPATH .venv/bin/python -m pytest -q` (same for the server) |
| `npm run dev` fails with `ENOSPC: System limit for number of file watchers` | Use `npm run build` and the server (step 5), or raise `fs.inotify.max_user_watches` |
| The app on a VM loads but uploads fail from your laptop | You're on an old build that called `localhost:8000`: `git pull`, `npm run build`, restart the server |
| CUDA out of memory | Close other GPU apps, stop the server before evaluations, or choose *Fast* quality |
| Choppy 3D view | Add `?detail=standard` to the address (half the mesh detail) |
| GeoTIFF upload has no absolute DSM (`dsm_error` in the panel) | The selected base DEM (SRTM by default or Copernicus GLO-30) could not be read; remote reads time out after 60 s. The nDSM still works; retry when the network is back |
| First upload is slow | The model loads on the first upload (plus the DINOv2 download on the very first run); later uploads skip both |
