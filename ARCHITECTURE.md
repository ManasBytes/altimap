# How AltiMap works

This guide describes the height/API baseline tested at commit
[84136c4](https://github.com/ManasBytes/altimap/commit/84136c4013837f141624ed62a1d6cff80f52c05c).
It explains the system in simple language and gives file names for developers.

AltiMap turns one aerial or satellite colour image into an interactive 3D city model.
The model contains building blocks, roof levels, tree crowns and terrain.
It also saves height maps so we can check the prediction against a known reference.

Main also includes viewer update
[5e21266](https://github.com/ManasBytes/altimap/commit/5e21266f5a8bdef71d5a93c3b77c2058d6a4ebca).
It adds a studio inspector and prepared-result hosting without changing the Python height/API code.
Other newer-branch model and DSM changes remain separate. The full upload/accuracy checks were
not repeated for the new viewer during this documentation update. The
[5 October local check](docs/local-studio-verification-20261005.md) records saved-scene UI tests
and one fresh GeoTIFF API request, including its high error and the incomplete browser checks.

[README and screenshots](README.md) | [Setup](SETUP.md) |
[Project history](docs/PROJECT_JOURNEY.md) | [Test evidence](docs/evidence/README.md)

## 1. Words used in this guide

| Term | Simple meaning |
|---|---|
| RGB | The red, green and blue channels of a colour image |
| GeoTIFF | An image file that can include coordinates, a map system and pixel placement |
| Pixel size, or GSD | How much ground one image pixel covers, usually in metres |
| nDSM | Height above the local ground, such as a building's height |
| DSM | Surface elevation, including roofs and treetops, relative to a stated height reference |
| DTM | Ground elevation without buildings or trees |
| DEM | An elevation dataset; check its description to see which surface it measures |
| CRS | The coordinate system used to place the image on a map |
| Inference | Using a trained model to make a prediction |
| Semantic classes | Labels such as buildings, trees, roads and water |
| COG | A GeoTIFF layout designed to let software read parts of the file efficiently |
| LiDAR | Laser measurements used to make reference height maps |

A GeoTIFF colour image does not automatically contain building heights.
Its map information tells us where the image belongs. The model still has to estimate
object heights from the colour pixels.

For example, a building that is 20 m tall on ground at 100 m elevation has an nDSM height
of 20 m and a roof elevation of about 120 m. The two numbers describe different things.

## 2. The main flow

```text
Colour image
    |
    +-- Read valid pixels, pixel size and any map information
    |
    +-- v1 model predicts height above ground and land-cover classes
    |       |
    |       +-- Building pixels: combine 25% v1 with 75% v2
    |       +-- Dense forest trees: use CHMv2 if it is installed
    |       +-- Other pixels: keep v1
    |
    +-- PNG/JPG: use a local ground plane
    |   GeoTIFF: fetch and align the chosen public elevation dataset
    |
    +-- Save height maps for measurements and reference checks
    |
    +-- Build separate building blocks and tree crowns
    |
    +-- Place the original image on the roofs and terrain
    |
    +-- Show the city model in the browser
```

v2 and CHMv2 are optional. The result records which models were actually used.
The Windows upload checks used v1 and v2. CHMv2 was not installed in those checks.

## 3. Reading the input

Files: `viewer/estimate.py`, `viewer/geo.py`.

### Colour and missing pixels

`read_image` reads the image bands and follows colour tags where available.
It uses the first three bands when suitable colour tags are absent.
Higher-bit-depth and floating-point colour images are adjusted for model input using valid pixels.

Missing image pixels are excluded from output measurements. They become NaN, meaning
"no known value", in the saved height maps. A single-band floating-point height raster is
rejected as the main image. It belongs in the reference-height field instead.

### Pixel size and map information

For a supported GeoTIFF, the CRS and transform tell us the pixel size and location.
Degree-based coordinates are converted using the scene's latitude.

A PNG or JPG usually has no reliable pixel size. Users can enter it.
Without it, the app uses an explicitly labelled experimental assumption of 0.33 m per pixel.
Neither the horizontal scale nor displayed metre heights are independently verified in that case.

Rotated, sheared or strongly non-square GeoTIFF grids are not supported for absolute DSM output.
The app warns about these cases instead of silently claiming correct map geometry.
A full reproject-to-a-safe-grid workflow is future work.

## 4. Predicting object heights

Files: `viewer/height_model.py`, `viewer/height_train.py`,
`viewer/gamus_dataset.py`, `viewer/estimate.py`.

### The main model

RS3DAda uses a DINOv2 ViT-L image encoder and a DPT decoder.
In simple terms, the encoder reads image patterns. The decoder turns them into two maps:
height above ground and land-cover labels.

We adapted v1 using GAMUS colour images, measured above-ground heights and class labels.
Its training loss combines:

```text
height error
+ 0.5 x height-edge error
+ 0.2 x class-label error
```

The technical names are L1, gradient L1 and semantic cross-entropy.
Training changes model weights. Uploading an image only runs the trained model.

The original brief suggested a depth model. The later organiser FAQ permits other methods.
DINOv2 is an image-feature model, not a claim that this pipeline uses a generic
depth-pretrained model as its final height estimator.

### Building and forest specialists

v2 gives more attention to tall buildings during training.
It can also overestimate unfamiliar trees and ground, so we use it only on pixels that
the semantic model predicts as buildings.

```text
building height = 0.25 x v1 height + 0.75 x v2 height
```

If v2 is missing, v1 remains the available estimator.

CHMv2 is an optional tree-height model. It replaces predicted tree heights inside areas
with at least 80% canopy in an approximately 150 m neighbourhood.
It does not replace every tree prediction in a suburb.
The function `forest_mask` applies this rule.

### Large images and quality settings

The model expects imagery near its training pixel size of 0.33 m.
`predict_scene` adjusts the image for that working scale and processes large scenes in pieces.
Overlapping windows reduce seams. Heights are then returned to the input grid.

Normal model windows are 518 pixels wide. Larger processing tiles use 3072 pixels,
with 259 pixels of extra context around each tile.
Very large scenes can use a coarser working scale, which the result records.

Fast quality makes one prediction per model. High quality averages four flipped versions.
This is called test-time augmentation, or TTA. High uses more computation.
Total upload time also depends on scene size, model loading and elevation downloads.

Upsampling a coarse image does not restore missing roof detail.
Accepting a 10 m image is not a guarantee of accurate individual buildings at that resolution.

## 5. Giving GeoTIFF results an absolute elevation

Files: `viewer/dem.py`, `viewer/estimate.py`.

### Copernicus and SRTM

These are public elevation datasets, not different AI models.
The user selects one as the base for the absolute DSM.
Tested main defaults to Copernicus GLO-30; SRTM is another option.

Both have roughly 30 m spacing, so they do not provide a detailed measurement of every roof.
Copernicus includes surface effects from buildings and trees.
It should not be described as a surveyed, bare-ground DTM.

Copernicus uses the EGM2008 height reference. SRTM uses EGM96.
A height reference, also called a vertical datum, defines what the elevation number is measured from.
Reference data must use a compatible datum for a fair comparison.

### Adding model detail without simply counting it twice

The pipeline approximately does this:

```text
absolute DSM =
chosen public elevation
- local average of predicted object detail
+ predicted object detail
```

Removing the local average reduces double counting because a coarse surface dataset may
already include some building and tree height.

The local average comes from a moving filter on the image grid.
It is **not** calculated on the exact native Copernicus cells.
The method does not guarantee that every 30 m cell has exactly the original DEM average.

Fine detail is suppressed in extensive forest where it was unreliable.
Ground used to display the city is estimated separately.
Depending on building coverage, the app uses a ground filter or subtracts a local height estimate.
This ground is not an independently measured high-resolution DTM.

### Downloads, cache and failure handling

The app checks its cache before fetching elevation again.
Copernicus is read from AWS Open Data, with Planetary Computer as a fallback.
SRTM uses the configured OpenTopography source.

Complete aligned elevation patches are cached in `viewer/cache/dem/`.
`ALTIMAP_DEM_CACHE` can put that cache on another drive.
Unknown or unreadable pixels remain missing. Network failure must not become 0 m terrain.

If the elevation source cannot be used, the app can still return the nDSM with a DSM warning.
The optional public-DEM comparison is off by default.
Turning it on can add another network read and increase upload time.

Agreement with the elevation source used to build the DSM is a consistency check.
It is not independent proof of accurate building heights.
Missing or undefined JSON measurements are returned as `null`, not `NaN`.

### Ground control points

A user can attach a CSV of known ground elevations:

```text
lon,lat,height
```

Coordinates are WGS84 longitude and latitude. Heights must use a compatible vertical datum.
Roof heights are not ground control points.

The app compares these points with estimated ground. It applies one vertical shift only
when the difference is at least 4 m and the correction performs better on left-out points.
Otherwise it leaves the DSM unchanged and explains why.
This does not fix a model's incorrect building shapes or object-height errors.

## 6. Building the city model

File: `viewer/city_model.py`. This is the main presentation output.

1. Find regions predicted as buildings.
2. Separate touching buildings using changes in their estimated roof heights.
3. Split roof levels where the height difference is large enough.
4. Remove or merge tiny fragments and simplify the outlines.
5. Raise each building part using heights from inside its roof region.
6. Add tree crowns around local height peaks in predicted tree regions.
7. Place the objects on the estimated ground and apply image textures.

For sloping building walls seen in angled images, the code tries to join them to their roof.
This reduces some staircase-like blocks, but does not solve every difficult facade.

Tree crowns are simplified shapes. Building floors, volume and tree counts are estimates.
They are not surveyed records or a reconstruction of every real branch and window.

The city model is for viewing. It does not overwrite the scientific height rasters.
A cleaner-looking block can be less accurate than the raw prediction for an individual pixel.

## 7. What the browser shows

Files: `frontend/src/main.jsx`, `frontend/src/studio.css`, `frontend/src/hosting.js`,
`frontend/src/cityDisplay.js`, and the older `styles.css`/`blender.css` renderer styles.

The studio has Scenes, View, Measure, Accuracy and Context tabs.
Scenes offers saved model demos, live upload options and the separate reference gallery.
Saved demos restore existing textures, heights, geometry and reference scores.
They are replayed predictions, not fresh inference. `hosting.js` resolves saved-asset URLs.
`cityDisplay.js` provides a display-only flat-ground option without changing the source heights.

The current public deployment hosts the viewer and saved scenes only. It cannot process new
TIFF, PNG or JPG uploads. The local app needs its Python inference backend for new images.

The Context tab can display saved OpenStreetMap names and features. This main backend does not
yet fetch school, hospital, bridge or flood-defence context for new uploads. These names come
from map records, not a model that recognizes a building's use from the image.

| View or control | What it does |
|---|---|
| City model | Shows separate building blocks, roof levels and tree crowns |
| Exact DSM | Shows the continuous height surface; "exact" does not mean correct ground truth |
| RGB | Uses the original image as a texture |
| Height | Colours objects by estimated height above ground |
| Classes | Shows predicted land-cover labels |
| Surface | Shows surface shading |
| Slope | Colours steepness of the shown surface |
| Error | Shows model minus reference, when a reference is attached |
| Vertical exaggeration | Changes displayed height only; use 1x for physical proportions |
| Measure | Shows height/elevation/slope and a two-point profile |
| Walk, fly and orbit | Give different ways to explore the scene |
| Export | Saves the displayed 3D scene as a GLB file |

The built-in GAMUS reference gallery shows measured reference heights, not model inference.
Uploads show fresh predictions. Model demos show saved predictions.
Keep all three sources clearly labelled.

The browser uses compact height grids for rendering. Float32 GeoTIFFs are the scientific downloads.
The mesh uses fewer segments on lower-powered graphics devices.
A finer mesh does not create more measured detail in a coarse source dataset.

## 8. API and saved files

File: `viewer/server.py`.

| Endpoint | Purpose |
|---|---|
| `POST /api/estimate` | Upload an image and create a height/city result |
| `GET /api/progress/{job}` | Read the current processing step |
| `GET /api/health` | Check that the server responds |
| `GET /api/uploads` | List saved uploads |
| `DELETE /api/uploads/{scene_id}` | Delete a selected saved upload |

The main upload accepts `file`, optional `gsd`, `reference`, `gcps`, `job`,
`tta`, `base_dem` and `compare_dems`.
Images have a 200 MB upload limit. Jobs use a worker thread so the browser can poll progress.
Model inference uploads run one at a time because they share model instances and GPU memory.

The response includes preview images, compact height grids, city geometry, model names,
timing, download links and optional reference scores.
The health endpoint's old `model_loaded` field is not a reliable indicator that the current
height stack is loaded. Models load on the first inference request.

| Saved file | Meaning |
|---|---|
| `ndsm.tif` | Estimated height above ground |
| `dsm.tif` | Absolute surface elevation when georeferencing and elevation coverage allow it |
| Matching `.json` sidecars | Pixel size, height range, units, model and height-reference information |
| `buildings.geojson` | Building outlines and height estimates |
| `meta.json` | Saved upload result information |

Files are saved below `viewer/web/data-uploads/scenes/<id>/`.
Height files use float32, NaN for missing data and the COG writer in `src/altimap/contract.py`.
Outputs retain the supported source grid and CRS.
Building outlines use WGS84 coordinates for georeferenced results and image coordinates otherwise.

## 9. How accuracy is checked

An RGB image alone has no correct height answers. To measure error, upload an independent height
raster for the same area. A coloured screenshot is not a valid height reference.

The server aligns a georeferenced reference to the image.
For unreferenced arrays it assumes the same area and resamples them.
Main chooses between nDSM and absolute DSM comparison using the reference values.
Always check the displayed comparison label.

| Metric | Meaning |
|---|---|
| MAE | Average size of the height error |
| RMSE | An error measure that gives larger mistakes more weight |
| Bias | Average signed error; a negative value means predictions are too low |
| Building RMSE | RMSE using only building pixels |
| Correlation | How well the height pattern follows the reference pattern |
| Coverage | The share of pixels with valid values for comparison |

Building metrics in the fusion experiment use reference class labels.
The upload UI uses predicted class labels. These masks are different.

The scatter plot compares reference height on the horizontal axis with predicted height
on the vertical axis. The diagonal means equal values.
Scores use valid overlapping pixels, not only the dots selected for the plot.

### Current measured building blend

The complete GAMUS inventory has 5,004 training, 859 validation and 2,861 test tiles.
The blend was selected on validation and confirmed on test.

| Evaluation | Previous 50/50 overall RMSE | Current 25/75 overall RMSE | Previous building RMSE | Current building RMSE |
|---|---:|---:|---:|---:|
| Validation, 859 tiles | 2.7511 m | 2.7402 m | 3.3352 m | 3.2763 m |
| Test, 2,861 tiles | 3.7573 m | 3.6972 m | 5.2308 m | 4.9934 m |
| Separate 24-tile High/TTA check | 3.1690 m | 3.0916 m | 5.1284 m | 4.9140 m |

Full validation/test metrics use one-pass inference. The 24-tile check uses four-flip TTA.
The test improvement is 1.60% overall and 4.54% on buildings.
This is a measured improvement, not a guarantee for every landscape.

Older 500-tile model comparisons and 40-tile city experiments use different settings.
They must not be presented as a fresh evaluation of the current blend.
See [the evidence index](docs/evidence/README.md) and
[the dated upload checks](docs/real-upload-verification-20261004.md).

## 10. Running and checking the system

One FastAPI process serves the built frontend and the inference API.
The normal address is http://127.0.0.1:8000.
Use [SETUP.md](SETUP.md) for install commands, model downloads and Docker instructions.

The frontend can also load saved scenes without the inference API.
`VITE_DEMO_CATALOG` sets their catalog URL. `VITE_API_BASE` sets the live API address;
`disabled` allows only prepared scenes. New uploads still require the Python server.
See [the deployment guide](docs/DEPLOYMENT.md). No hosted deployment was performed here.

There are two separate Python environments:

- `.venv` runs the torch-free tests. Some model/API tests need the app environment.
- `.venv-da3` runs the app and model tools.

The Dockerfile exists. Earlier Docker checks were reported by the teammate.
The Windows verification did not rebuild or run Docker because its daemon was unavailable.

Inference was checked on an RTX 3050 with 6 GB GPU memory.
Large High-quality uploads can take several minutes.
The app loads model weights on the first request, so the first upload can take longer.
Training was done on a larger GPU; it does not happen during normal uploads.

The first model load may need internet for model code.
GeoTIFF uploads also need uncached public elevation data.
A local cache helps repeated scenes, but it does not provide coverage for every new location.

## 11. Limits and next steps

Current limits include tall buildings, forest canopy, coarse imagery, unknown PNG scale,
unsupported image grids, older reference data and changes in vertical datum.
Good-looking geometry does not prove low measurement error.

The current models learned mainly from aerial imagery in US cities.
Indian satellite imagery can differ. The Sikkim examples demonstrate processing and display;
without an independent reference, their height accuracy is not measured.

The next model comparisons will test RDAH-Net, Depth2Elevation and satellite-pretrained DINOv3
features against this fixed baseline. All candidates need the same images, pixel sizes,
reference masks, metrics and final-check set. We will compare raw object heights first,
then absolute DSMs and city display. No candidate is adopted only because its paper reports
a smaller error on a different dataset.

## 12. Code map

| File or folder | Role |
|---|---|
| `viewer/estimate.py` | Image reading, height combination, DSM calculation and batch command |
| `viewer/height_model.py` | Model loading and window-based prediction |
| `viewer/height_train.py`, `viewer/gamus_dataset.py` | Model training and GAMUS loading |
| `viewer/height_metrics.py`, `viewer/height_eval.py`, `viewer/fusion_eval.py` | Accuracy measures and model/blend comparisons |
| `viewer/dem.py`, `viewer/geo.py` | Public elevation data and map alignment |
| `viewer/city_model.py`, `viewer/city_eval.py` | Building/tree geometry and city checks |
| `viewer/canopy.py` | Optional CHMv2 model |
| `viewer/server.py` | Upload API and frontend hosting |
| `src/altimap/contract.py` | Scientific raster and metadata writing |
| `frontend/` | React and Three.js viewer |
| `tests/` | Unit and regression checks |
| `scripts/` | Data/model preparation helpers |
| `docs/evidence/`, `docs/evaluation/` | Small public test artifacts and decision records |
| `backend/`, old dashboards and exporters | Earlier work, not the main launch path |

[Return to README](README.md)
