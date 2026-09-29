# Trained surface workspace

## Run locally

The existing local environment and trained weights are ready. From the repository root:

```bash
bash scripts/run-local.sh
# http://localhost:8080
```

This builds the React/Three.js workspace and serves it with its Python API, on
one origin. Stop an existing process on that port before starting another one.
For trusted LAN access set `ALTIMAP_HOST=0.0.0.0`. Do not expose the upload API
publicly: it has no authentication. No cloud deployment is required.

For a new environment, install Python 3.12, Node, a suitable PyTorch build for
the machine, then the project dependencies:

```bash
python3.12 -m venv .venv-da3
.venv-da3/bin/python -m pip install -r requirements-training.txt
npm --prefix frontend ci
```

Weights and raw datasets are local artifacts, not bundled with Git. The active
model is `viewer/cache/surface_best.pt`, with `surface_report.json` beside it.
The original initialization weights are `viewer/cache/classifier_resnet34unet.pt`.
A fresh checkout must restore those files from this workstation or train them;
it does not magically download this custom trained model. Only load trusted
PyTorch checkpoints. Without the surface weights, uploads fall back to the
original classifier + DA3; the validation panel reports missing surface weights.

Uploads are stored in `viewer/web/data-uploads` (override `ALTIMAP_UPLOAD_DIR`),
outside `frontend/dist`, so rebuilding the UI does not erase them.

## What actually changed

PNG/JPEG now uses a **ResNet34-encoder U-Net with two learned heads**: seven-class
segmentation and log(1 + AGL) regression. It is not SkySense and is not a fake
fixed-height lookup. The existing land-cover encoder was frozen; the decoder,
segmentation head and new height head were fine-tuned on aligned RGB/CLS/AGL
HDF5 triplets. BatchNorm statistics were kept fixed on the small dataset.

Inference blends softmax probabilities and height predictions in 512-pixel
tiles with 128-pixel overlaps. A cosine window suppresses tile seams. PNG/JPEG
aspect ratio is preserved. The interactive API caps the inference image's long
side at 2048 pixels for memory/latency; smaller inputs retain native resolution.
`viewer.surface.infer_tiled` itself accepts larger arrays without that cap.

The rendering-only height product uses a small median filter and locally
regularized connected building roofs. Raw model height predictions remain
available separately. No fixed class offsets are applied again in Three.js.
This is a textured height field, not a complete building/façade reconstruction.

GeoTIFFs use declared elevation bands (or an available reference DEM), not the
learned PNG heights. Semantic predictions do not alter measured DSM pixels.
A floating-point RGB band alone is no longer treated as elevation. Select an
ambiguous elevation band explicitly in the upload panel. An RGB GeoTIFF without
an elevation band needs DEM coverage; georeferencing alone does not supply
building heights. Existing measured building-footprint refinement remains for
ground DEM inputs, not DSMs.

API outputs include lossless class IDs, confidence PNG, raw predicted AGL NPY
for model-based images, textured GLB, and geospatial masks/elevation TIFFs for
georeferenced inputs. The JSON response declares inference and render dimensions.
Raw NPY heights are learned AGL estimates in the training domain, **not calibrated
metric measurements for arbitrary uploads**. The viewer labels those as relative.
Confidence is a model probability, not a measured accuracy score.

## Training and evaluation

The local experiment ran 60 epochs on 55 training tiles, selected on 55 validation
tiles, then evaluated on 55 test tiles. All original tiles are 1024 × 1024.
Class IDs: `0 background`, `1 ground`, `2 low_vegetation`, `3 buildings`,
`4 water`, `5 roads`, `6 trees`. It does not claim separate sidewalk, vehicle,
or parking predictions because these labels are absent from this training set.

```bash
.venv-da3/bin/python -m viewer.train_surface \
  --epochs 60 --batch-size 4 --seed 42 \
  --data-roots /home/biplab-dev/GAMUS_50_each /home/biplab-dev/GAMUS_extra_15 \
  --output viewer/cache/surface_balanced
```

For a new run use a new output directory. `--resume <run>/last.pt` restores model,
optimizer and AMP scaler state; use the same output directory and set `--epochs`
to the desired total. Training uses random 256-pixel crops, flips, right-angle
rotations, color jitter, class-balanced cross entropy + Dice loss, masked log-AGL
Huber loss, AdamW and gradient clipping. Heights outside 0–150 m or nonfinite
are excluded from the regression loss and evaluation. CUDA training uses AMP.
The loader caches the small local dataset in RAM; it is not a streaming loader
for arbitrarily large collections.

Best checkpoint selection uses `validation mIoU - 0.01 × AGL MAE` on four fixed
validation crops per tile. Final metrics use all validation/test pixels with
overlapping inference. The 60-epoch run selected epoch 59. Promotion of the
segmentation head requires full-tile validation mIoU to beat the original
classifier. Test results do not determine checkpoint selection or promotion.
An earlier 16-epoch trial was not promoted; this is a development holdout, not
an independent external benchmark.

| Metric | Original classifier | Trained surface model |
|---|---:|---:|
| Full validation mIoU | 39.87% | 43.66% |
| Full test mIoU | not re-evaluated | 44.46% |
| Test building IoU | — | 75.39% |
| Test tree IoU | — | 75.60% |
| Test AGL MAE / RMSE | no height head | 3.71 m / 5.60 m |

See [surface-evaluation.json](surface-evaluation.json) for split IDs and all class
scores. Background and low vegetation remain weak (test IoU 0.32% and 18.66%);
this is an improvement, not survey-grade or general-purpose recognition.
Unseen regions, image scales, seasons and off-nadir viewpoints require more
representative labeled training data and independent testing. Do not interpret
interpolating a mesh to more vertices as creating additional image information.

Active checkpoint SHA-256:
`d4bcfb06281ea96b9d7f38fa1a098c5f89afb7e330a128ea9e78c8cd70b8b23b`.

To promote a future validated run, back up the active checkpoint/report, copy
that run's `best.pt` and `report.json` to the active filenames, then restart the
server. Loading is lazy and cached per process.

Custom data can use the existing triplet adapter in `viewer/gamus_dataset.py`:
`images/<split>/<id>_RGB.h5`, `classes/<split>/<id>_CLS.h5`,
`heights/<split>/<id>_AGL.h5`, with dataset key `image` in each file. Supply its
root through `--data-roots`. Map labels into the seven IDs above; expanding the
taxonomy requires changing the head, class schema, weights and UI palette.

## Workspace controls

- Full-viewport Three.js scene, dark/light inspector, wireframe/grid toggles,
  perspective/top/front views and hide/show inspector.
- WASD horizontal flight, Q/E altitude; adjustable speed with time-based easing.
  Typing in XYZ fields does not move the camera; losing focus releases keys.
- Place nodes by clicking the actual surface, select a numbered node, move its
  XYZ handles or edit numeric coordinates, delete nodes, save/load scene-specific
  JSON routes. Playback uses arc-length travel with pause/resume and adjustable
  speed. Drag to change viewing direction without changing the route position.
  Routes do not perform obstacle avoidance; set adequate clearance manually.
- Measure by double-clicking two points. Metric elevation/distance appears only
  with geographic scale; PNG previews show relative height and scene units.
- The surface transect and statistics come from the actual displayed mesh,
  replacing the old static chart. Metric readout ignores visual exaggeration.
- RGB and 0.5× exaggeration remain the image/catalog defaults; GeoTIFF opens at
  true-scale 1×. The slider changes mesh scale without rebuilding its geometry.
  Idle views do not continuously render frames. Interactive grids use a maximum
  long side of 513 samples for image previews, 1025 for GeoTIFF, with rectangular
  aspect preserved. Full inference outputs are separate from the render grid.

Code is split into `frontend/src/App.jsx`, `components/`, `terrain/`, `data/`
and `workspace.css`. Training lives in `viewer/train_surface.py`, inference in
`viewer/surface.py`, and API orchestration in `viewer/server.py`.

## Checks

```bash
.venv-da3/bin/python -m pytest tests/test_viewer_surface.py tests/test_viewer_terrain.py tests/test_viewer_geo.py -q
npm --prefix frontend run build
curl http://localhost:8080/api/health
curl http://localhost:8080/api/model-status
```

Browser checks cover route creation/editing, pause/resume, route downloads,
scene rendering and light mode. API checks use rectangular PNG and a synthetic
GeoTIFF DSM; exported DSM pixels, CRS, transform, dimensions and custom metadata
must match the supplied input exactly. GPU training metrics do not substitute
for testing on your own target imagery.
