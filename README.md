# AltiMap Mark‑1

AltiMap Mark‑1 is a prepared, browser-based demonstration of single-view remote-sensing height estimation. It keeps the Biplab Three.js terrain experience and is packaged as a static site. The intended offline research pipeline is **Depth Anything V2 (DAV2) → pretrained RDAH-Net → predicted above-ground height**, followed by comparison with GAMUS reference data.

The current container serves the prepared frontend assets. It does not run PyTorch inference, accept live GAMUS HDF5 uploads, or fine-tune a model at runtime.

## Run the demo

Requirements: Docker Desktop with Compose.

From the repository root:

```bash
docker compose up --build
```

Open <http://localhost:8080>. Stop it with `Ctrl+C`; remove the stopped container with `docker compose down`.

The image builds the Vite app from `frontend/`, then serves `frontend/dist/` with nginx. No Node, Python, npm or model installation is required on the host.

## Mark‑1 scope

Mark‑1 is an offline-prepared evaluation and visualization demo:

```text
GAMUS RGB HDF5
      ↓
DAV2 Small relative-depth prior
      ↓
pretrained RDAH-Net checkpoint
      ↓
predicted nDSM / above-ground height
      ↓
compare with GAMUS AGL and CLS
      ↓
metrics, maps and prepared 3D assets
      ↓
Biplab Three.js viewer
```

DAV2 and RDAH weights remain frozen. Mark‑1 does not train or fine-tune either network. A later experiment may fit a single global scale calibration, but it must not fit one scale per scene.

The current repository packaging is deliberately separate from that GPU preprocessing step. The browser should receive generated assets and a scene manifest; it should not need the HDF5 files or multi-gigabyte model weights merely to launch the demo.

## GAMUS data roles

Each selected GAMUS scene is an aligned triplet:

| File | Role |
| --- | --- |
| `<ID>_RGB.h5` | The only scene input to DAV2 and RDAH. It supplies the visible aerial image. |
| `<ID>_AGL.h5` | Ground-truth above-ground height, used only after prediction for validation and the reference surface. |
| `<ID>_CLS.h5` | Semantic class labels used for class-wise diagnostics, especially building, vegetation/tree and ground errors. |

AGL is not fed into the prediction path. CLS must not replace the model prediction with the answer; it is used for reporting, confidence/uncertainty presentation and carefully documented class-aware analysis. The `biplab-new` branch did not already implement a GAMUS CLS loader or CLS-based error correction; that remains part of the Mark‑1 integration work.

## Curated 18-scene evaluation contract

The Mark‑1 offline batch contains exactly six scenes from each original GAMUS split:

```text
Train: DC_01_25, DC_10_17, DC_10_18, DC_10_19, DC_10_21, DC_10_27
Val:   DC_02_26, DC_20_13, DC_20_14, DC_20_18, DC_20_19, DC_20_29
Test:  DC_03_26, DC_20_12, DC_20_15, DC_20_20, DC_20_23, DC_20_25
```

Train scenes are support data, validation scenes select/check the optional global calibration, and test scenes are held out until final reporting. The frontend currently contains additional historical preview assets; those are useful for the gallery but are not extra Mark‑1 evaluation evidence.

GAMUS scenes are local image grids. They do not provide a GeoTIFF CRS/affine transform in this demo, so the viewer must not invent latitude/longitude, global coverage or a vertical datum. Pixel spacing and any local-grid assumption must come from the generated manifest.

## Viewer modes and active layers

The viewer should make the source of every visual explicit:

- **Predicted 3D:** geometry is generated from the final RDAH predicted nDSM and textured with the original RGB image.
- **Reference 3D:** geometry is generated from GAMUS AGL and textured with the same RGB image. This is the comparison target, not a model result.
- **Error 3D:** predicted geometry is coloured by absolute prediction error in metres.
- **RGB:** the original image used for prediction and terrain texture.
- **DAV2 depth:** the frozen model's relative-depth prior. It is not metric elevation by itself.
- **Predicted height:** floating-point RDAH output rendered as a height map.
- **Reference height:** GAMUS AGL rendered as a height map.
- **Error heat map:** `abs(predicted_height - reference_AGL)` with a metre-labelled legend.
- **Surface/slope/profile/probe:** viewer diagnostics derived from the selected surface; scientific metrics always use unsmoothed floating-point arrays.

For browser geometry, numeric heights should be packed into a lossless format such as RG16 PNG. JPG is appropriate for RGB textures and thumbnails only; its brightness must not be treated as precise metres.

## Metrics

For valid, finite pixels:

```text
error = prediction - reference
MAE    = mean(abs(error))       # metres
MSE    = mean(error²)            # square metres
RMSE   = sqrt(MSE)               # metres
bias   = mean(error)             # metres
```

The manifest should also record valid-pixel count and class-wise MAE for building, vegetation/tree and ground pixels from CLS. Viewer smoothing, clipping and vertical exaggeration are presentation operations and must never modify the arrays used for these metrics.

## Offline model and asset setup

The full preprocessing workflow is run separately on a machine with the required Python/geospatial/ML dependencies and model weights. Keep weights outside Git and outside the demo image. A conventional local layout is:

```text
models/
├── depth_anything_v2_small/       # DAV2 Small Hugging Face weights
└── rdah-track1-104best_model.pth  # RDAH-Net Track-1 checkpoint
```

The exact RDAH checkpoint must match its model architecture and preprocessing contract. Record model identifiers, checkpoint paths or hashes, preprocessing version, calibration status and source scene in each `provenance.json`. Do not present the current DA3 backend as DAV2 + RDAH.

For each scene, retain scientific arrays outside the frontend bundle:

```text
predicted_raw.npy
predicted_final.npy
reference_agl.npy
classes.npy
absolute_error.npy
metrics.json
provenance.json
```

The browser-ready directory should contain at least:

```text
rgb.jpg
dav2-depth.png
predicted-height.png
reference-height.png
error-heatmap.png
predicted-rg16.png
reference-rg16.png
scene.json
```

`scene.json` is the frontend contract. It must identify the scene and split, list every asset URL, specify height/error ranges and pixel spacing, include MAE/MSE/RMSE/bias/class metrics, and identify the DAV2/RDAH models and calibration status. Missing assets should fail visibly rather than silently showing a different layer.

## Development checks

Build the frontend without Docker:

```bash
cd frontend
npm ci
npm run build
```

Run the existing Python test suite from the repository root if the project Python environment is available:

```bash
python -m pytest -q
```

The static Mark‑1 container does not require the Django backend. The backend and older GeoTIFF/DA3 experiments remain separate and are not evidence that the DAV2 + RDAH GAMUS pipeline is complete.

## Known limitations

- Mark‑1 uses prepared scenes; live upload and runtime inference are out of scope.
- GAMUS provides local-grid height/AGL references here, not globally georeferenced absolute elevation.
- AGL is above-ground height (nDSM-like), not terrain elevation (DTM) or absolute sea-level DSM.
- DAV2 supplies a relative-depth prior; RDAH and any explicitly reported calibration supply the height estimate.
- Existing Biplab preview values should not be treated as scientific metrics until they are generated from the manifest and verified against the HDF5 arrays.
- The browser mesh may be downsampled for smooth navigation while evaluation remains full resolution.
