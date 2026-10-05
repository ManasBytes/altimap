# AltiMap browser viewer

This is the React and Three.js viewer for the tested main version.
It shows the **3D city model**, including building blocks, roof levels, tree crowns and terrain.

The early static GAMUS Terrain Studio was a prototype.
This folder now contains the upload interface and the interactive viewer.
Do not use the old prototype's file paths or scene counts as current setup instructions.

[Project README](../README.md) | [How the system works](../ARCHITECTURE.md) |
[Full setup](../SETUP.md)

## Build and run

The frontend alone does not predict heights. It needs the Python height server for uploads.
Follow the full setup guide first to install model code and download the weights.

From this folder:

```bash
npm ci
npm run build
```

Then run the server from the repository root:

```powershell
# Windows, in an already configured environment
.\.venv-da3\Scripts\python.exe -m viewer.server
```

On Linux, use `.venv-da3/bin/python -m viewer.server`.
Open http://127.0.0.1:8000. The server serves both the built viewer and the upload API.

For frontend development:

```bash
npm run dev
```

The development viewer runs on port 5173. Its configuration forwards API and saved-file requests
to the Python server on port 8000. The Python server must still be running.

Use `npm ci` to keep the versions recorded in the lockfile.
A build can show a large-bundle warning; a successful build alone does not prove upload inference works.

## Uploads and the reference gallery are different

- **Uploaded RGB images:** the backend predicts object heights and classes, then builds city geometry.
- **Built-in GAMUS scenes:** prepared reference-height images for viewing. Opening a gallery scene
  is not running the trained height model.

Set the pixel size, quality, elevation source and optional reference files before choosing an image.
A PNG without known pixel size uses a labelled experimental 0.33 m/pixel assumption.
Its displayed metre scale is not independently verified.

## City model and Exact DSM

**City model** shows separate buildings and tree crowns on estimated ground.
The original image is placed on roofs and terrain.

**Exact DSM** shows a continuous height surface. It is useful for checking the height output.
The word "exact" does not mean that its predicted heights are ground truth.

Changing the display mode or vertical exaggeration does not change the downloaded scientific rasters.
Use 1x vertical exaggeration to discuss physical proportions.

## Layers and controls

| Layer or control | Meaning |
|---|---|
| RGB | Original colour image |
| Height | Estimated height above ground, with a colour scale |
| Classes | Predicted land-cover categories |
| Surface | Shaded surface |
| Slope | Steepness of the displayed surface |
| Error | Difference from an attached reference |
| Orbit and zoom | Rotate around and inspect the scene |
| WASD / Q / E | Move horizontally / lower / raise the camera |
| Walk | Move near ground level |
| Play route | Follow an edited camera path |
| Measure | Inspect height and slope or draw a two-point height profile |
| Export | Download the displayed 3D scene as GLB |

High quality averages four flipped model predictions. Fast uses one prediction.
These settings change backend work, not only how detailed the mesh looks.

## Main files

| File | Role |
|---|---|
| `src/main.jsx` | Scene, upload interface, layers and controls |
| `src/styles.css`, `src/blender.css` | Viewer layout and appearance |
| `src/gamusScenes.js`, `public/` | Prepared gallery information and reference images |
| `vite.config.js` | Build and development-server settings |
| `package-lock.json` | Exact frontend dependency versions |

Inspector tabs and mapped bridges/facilities from the newer teammate branch are not part
of tested main. Screenshots of that work are labelled experimental in the project README.
