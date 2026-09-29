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
| Disk | ~12 GB | ~6 GB Python env (PyTorch), ~3 GB model weights, the rest data/cache |
| Internet | first run | model downloads; GeoTIFF uploads fetch the Copernicus DEM live |

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
    python-multipart pystac-client planetary-computer huggingface_hub shapely requests trimesh
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

# Our fine-tuned weights (1.5 GB). The repo is PRIVATE: ask Dilavesh to add your Hugging Face
# account, then log in once (paste a read token from https://huggingface.co/settings/tokens)
.venv-da3/bin/hf auth login
.venv-da3/bin/python -c "from huggingface_hub import hf_hub_download; \
hf_hub_download('Dilavesh/altimap-height', 'best.pth', local_dir='viewer/cache')"
```

The server uses `viewer/cache/best.pth` if it exists, else the stock weights. It also honours
`ALTIMAP_HEIGHT_CKPT=/path/to/ckpt.pth`. The fine-tuned model is much better (test RMSE 5.02 m
vs 7.25 m zero-shot, see README), so get access rather than running on the fallback.

The DINOv2 encoder code is fetched by `torch.hub` from GitHub the first time a model loads, then
cached in `~/.cache/torch/hub`. After that the app runs offline, except for GeoTIFF uploads,
which read the Copernicus GLO-30 DEM from Microsoft Planetary Computer.

## 4. Frontend

```bash
cd frontend
npm install
npm run build
cd ..
```

## 5. Run the app

Two terminals, both from the repo root:

```bash
# Terminal 1: height API on http://127.0.0.1:8000 (the model loads on the first upload)
.venv-da3/bin/python -m viewer.server

# Terminal 2: the viewer on http://127.0.0.1:5173
cd frontend && npm run preview -- --host 127.0.0.1 --port 5173
```

Open http://127.0.0.1:5173, click **Import** (or drop a file on the viewport):

- **PNG / JPG** gives heights above ground (nDSM) and a 3D city model on flat ground. Enter the
  pixel size in metres if you know it (otherwise 0.33 m is assumed; it sets the footprint scale).
- **GeoTIFF** also gives an absolute DSM (metres above sea level, EGM2008) on real terrain.
- **Add reference heights** (optional): a single-band height-above-ground GeoTIFF on the same
  grid. The app then scores the model against it (RMSE, MAE, correlation).
- **Quality**: *High* averages 4 flipped passes (~3x slower), *Fast* runs one pass.

In the viewer: click a building for its height, floors, footprint area and roof elevation;
double-click the terrain to probe height and slope; **Export** saves the 3D model as `.glb`.
The GeoTIFFs (`ndsm.tif`, `dsm.tif`) and `buildings.geojson` download from the right panel.

Test images: `demo/` is not in git (57 MB). Ask for it. It holds 3 GAMUS tiles with their LiDAR
reference heights (`demo/reference_heights/`, use those in the reference box, **not** as the
image) and 4 NAIP GeoTIFFs (city centre, suburb, hills, forest). Any RGB aerial or satellite
image works too.

Batch use without the viewer:

```bash
.venv-da3/bin/python -m viewer.estimate scene.tif photo.png --out results/
.venv-da3/bin/python -m viewer.estimate photo.jpg --gsd 0.5 --out results/   # known pixel size
```

## 6. Tests

```bash
.venv/bin/python -m pytest -q     # ~100 tests, under a second, no GPU or network
```

## 7. Evaluation and training (optional)

Stop the server first on a 6 GB GPU: the server's model and an evaluation's model don't both fit.

```bash
# 3D city model (building outlines + heights) on 40 GAMUS val tiles; --fetch downloads them once
.venv-da3/bin/python -m viewer.city_eval --fetch

# Height model per city against baselines (needs GAMUS under $ALTIMAP_DATA/gamus, default ~/altimap-data)
.venv-da3/bin/python -m viewer.height_eval --split test --ckpt viewer/cache/best.pth \
    --synrs3d viewer/cache/SynRS3D --tta --out test.json

# Fine-tuning (24 GB GPU, ~8 h): fetch the data, then train
.venv-da3/bin/python scripts/fetch_training_data.py
.venv-da3/bin/python -m viewer.height_train --hours 8 --out runs/ft
```

`scripts/overnight_v2.sh` is the full unattended run we used on the A5000 (data, training,
test with TTA, upload to Hugging Face).

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Couldn't reach the height backend` in the viewer | Terminal 1 isn't running, or something else holds port 8000. The frontend only talks to `viewer.server` on :8000; Django (`make dev`) also defaults to :8000 and can't serve uploads |
| `no height checkpoint; expected one of ...` | Step 3 weights are missing from `viewer/cache/` |
| `could not process this image: ... looks like a height or elevation map` | You uploaded a single-band height map as the image. Upload the RGB image; put the height map in the reference box |
| pytest fails with errors from `/opt/ros/...` or `launch_testing` | ROS is on `PYTHONPATH`: run `env -u PYTHONPATH .venv/bin/python -m pytest -q` (same for the server) |
| `npm run dev` fails with `ENOSPC: System limit for number of file watchers` | Use `npm run build` + `npm run preview` (step 5), or raise `fs.inotify.max_user_watches` |
| CUDA out of memory | Close other GPU apps, stop the server before evaluations, or choose *Fast* quality |
| GeoTIFF upload has no absolute DSM (`dsm_error` in the panel) | No internet or Planetary Computer unreachable: the nDSM still works, the DEM ground doesn't |
| First upload is slow | The model loads on the first upload (plus the DINOv2 download on the very first run); later uploads skip both |
