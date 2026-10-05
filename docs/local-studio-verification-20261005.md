# Local studio check: 5 October 2026

## Scope

Viewer: main commit `5e21266`, with no application code edits.
Python backend: tested baseline `84136c4`, unchanged by the viewer commit.
The documentation draft and new pictures are local. They were later brought into
`C:\Users\Rishabh\Downloads\SIH\altimap-dilavesh` with the viewer update, without a new commit or push.
Nothing was pushed or deployed during this check.

The existing backend and old UI were kept on `http://127.0.0.1:8000/`.
The new Vite UI was opened at `http://127.0.0.1:5173/`, with its normal proxy to that backend.
Dependencies and test artifacts were kept on E: rather than C:.

## Completed checks

| Check | Result |
|---|---|
| Locked frontend dependency installation | `npm ci` succeeded |
| Production frontend build | `npm run build` passed; large-bundle warning remains |
| New helper tests | Four tests passed for saved asset URLs and flat-ground geometry |
| Local backend status | The UI reported live uploads available |
| Public saved collection | Seven entries appeared; Chungthang, Pittsburgh bridges and Washington loaded |
| Light and dark themes | Switched successfully; screenshots saved |
| Viewer panels | Scenes, View, Accuracy and Context exercised |
| Height and error layers | Height scale and signed error scale appeared |
| Class layer | Switched and captured; not a new semantic-accuracy evaluation |
| Flat ground | Switched on and back; display warning appeared |
| Wheel zoom | View changed after scrolling; screenshot hashes differed |
| Fresh GeoTIFF processing | HTTP 200 through the new frontend proxy |
| Fresh raster and building downloads | HEAD requests returned HTTP 200 for nDSM, DSM and buildings GeoJSON |
| Raster grid checks | Both height exports matched the input CRS, transform and dimensions |
| Raster storage checks | Float32, COG layout tag and 2x/4x overviews were present; not a separate full COG-validator run |
| Fresh error calculation | Export-based recomputation matched the API's rounded RMSE, MAE and bias |

Three.js logged deprecation warnings for Clock and shadow-map selection. No JavaScript error
was recorded in the checked saved-scene flows. Two automated clicks timed out while GPU inference
was busy; the following visible-state checks showed that the requested terrain switches happened.
This is not a stress-test pass or a guarantee that every interaction is responsive under load.

## Fresh Denver GeoTIFF request

Input: `E:\TEST\external_geotiff_tests\denver_urban\rgb.tif`.
Reference: matching `reference_dsm.tif`, derived from public USGS 3DEP surface data.
Imagery is NAIP from 2013, 1 m/pixel. The reference metadata lists acquisition dates extending
into 2014. Dates and vertical datums were not newly reconciled in this test.

Request: Fast quality, Copernicus terrain, public-DEM comparison off.
No model was retrained and no routing setting was changed.
GPU monitoring during inference showed the RTX 3050 using about 5.5 GB and 92% GPU utilization
at one sampled moment. These are spot readings, not run averages.

| Measurement | Result |
|---|---|
| HTTP result | 200 |
| End-to-end request time | 159.12 seconds |
| Backend reported time | 158.16 seconds |
| Image / output grid | 600 x 600, EPSG:26913, 1 m/pixel |
| Models reported | v1 and v2, 75% building blend; no CHMv2 reported |
| Building / tree geometry | 428 buildings, 41 trees |
| Predicted above-ground height range | 0 to 155.81 m |
| Absolute DSM range | 1518.52 to 1677.87 m |
| Valid reference pixels | 360,000 |
| Absolute DSM RMSE | 30.7943 m |
| Absolute DSM MAE | 16.8148 m |
| Bias | -9.5957 m |
| Building RMSE | 36.8675 m, using predicted building labels |

The result is non-flat and the software completes the calculation. The height error is large.
This scene is a reliability check and a recorded limitation, not proof of strong generalization.
The error was not caused by a new height algorithm in the viewer commit, because that commit
does not change the Python estimation code. The cause of this scene's error was not isolated.

Large source images are processed in model windows and may be resampled before inference.
Fast quality therefore does not mean that every image finishes in a few seconds.

The API response contained no facilities, bridges, embankments or flood-defence context fields.
That agrees with the absence of their fetching modules in this main backend.

Raw response: `E:\TEST\altimap-new-ui-review-20261005\denver-live-result.json`.
Generated rasters and GeoJSON remain in the backend's ignored upload scene
`denver_urban_rgb__0f7826e7`. They are not included in Git.

## Checks that remain incomplete

- The browser extension refused automated file selection with a file-access error. The fresh
  request was sent using curl through the same Vite proxy instead. This proves API processing,
  not the complete user file-picker/progress/result workflow.
- Browser GLB export was clicked, but no completed download event was observed within 20 seconds.
  No export error was observed either. The GLB flow is unconfirmed, not marked passed or broken.
- GCP correction, public-DEM comparison, SRTM uploads, full navigation routes, mobile layout,
  CHMv2 inference and the hosted deployment were not newly tested.
- A saved scene's visible reference scores were not independently recomputed in this UI check.
  They must stay labelled as stored scores rather than new benchmark results.

For Opera file-upload testing, enable the ChatGPT extension's "Allow access to file URLs"
setting from `opera://extensions`, then repeat a real browser upload and check its progress,
result view, Accuracy panel and downloads.

## What to show in the README and presentation

Keep the older black-UI captures as dated prototype evidence. Add light and dark studio captures
as the current viewer's saved-scene display. Changing the theme does not improve height accuracy.

Show the Pittsburgh Context view only under **saved demo / experimental map context**.
Its school, clinic and university names come from cached OpenStreetMap data. They are not
predicted from RGB, and this main backend does not fetch those fields for new uploads.
No independent height reference is attached to that demo.

The default public demo collection can be viewed without a GPU server. The public deployment
does not currently process new uploads. Fresh prediction requires the local Python inference API.
Vercel hosts the frontend in this setup, not the local models.
An external inference server could enable hosted uploads later, but was not deployed in this check.

Do not claim full end-to-end verification until manual upload and GLB export are checked.

[README](../README.md) | [Evidence index](evidence/README.md) | [Hosting guide](DEPLOYMENT.md)
