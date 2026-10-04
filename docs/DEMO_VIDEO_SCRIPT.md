# AltiMap demo video script

Target: approximately **5 minutes**, English narration, for the SIH DepthWizard submission.
Record the running application and actual uploaded predictions. Allow a few seconds for each
interaction; shorten processing waits in editing and label them **“Processing time shortened”**.

## Before recording

1. Build and start the unified app from the repository root:

   ```bash
   (cd frontend && npm ci && npm run build)
   env -u PYTHONPATH .venv-da3/bin/python -m viewer.server
   ```

2. Open http://127.0.0.1:8000 at 1920×1080 or 1440×900. Use the default dark theme,
   browser zoom 100%, and 1× vertical exaggeration for uploaded scenes. Close unrelated windows.
3. Run one upload before recording to warm the models. Process the intended GeoTIFFs once to
   populate DEM/OSM caches. Do not run GPU benchmarks or training during the recording.
4. Prepare the following local files. `demo/` and model weights are not part of git:

   | Segment | RGB image | Reference/settings |
   |---|---|---|
   | PNG + validation | `demo/gamus_DC_04_27.png` | Pixel size **0.33**; `demo/reference_heights/gamus_DC_04_27_reference_AGL.tif`; reference type **Height above ground (nDSM / AGL)**; **Fast** |
   | Absolute DSM | `demo/naip_philadelphia_cityhall.tif` | Leave pixel size blank; `demo/reference_heights/naip_philadelphia_cityhall_lidar_dsm.tif`; reference type **Absolute elevation (DSM)**; **Fast**, **SRTM** |
   | Disaster context | `demo/naip_pittsburgh_bridges.tif` | Remove the Philadelphia reference before uploading; leave pixel size blank; **Fast**, **SRTM** |
   | Optional flood-defence insert | `demo/naip_baton_rouge_levee.tif` | No reference; **Fast**; use the base selected in your recorded result |

5. Check that the Context tab actually contains the features you intend to show. If OSM is
   unavailable, use an earlier successful recording of the real app or skip that insert.
   Do not replace an empty result with invented names or icons.
6. Keep [the current landscape report](evaluation/landscapes-2026-10-04.md) ready for the
   evaluation segment. Plan cursor movements before recording; use slow orbiting and short routes.

## 0:00–0:25 — Problem and opening view

**Show:** title “AltiMap — Single-view elevation and 3D flythrough”, then the app with an uploaded
scene. Slowly orbit the city. Briefly show the original RGB image beside it in the video edit.

**Say:**

> AltiMap turns a single aerial or satellite RGB image into elevation maps and an interactive
> three-dimensional scene. We built it for the DepthWizard problem: supporting images both with
> and without geographic coordinates, and making the estimated heights available for inspection,
> validation and export. This demonstration shows the complete path from image upload to a
> navigable reconstruction.

## 0:25–1:00 — PNG input and the estimation pipeline

**Show:** Image tab → Your image. Enter pixel size **0.33**, add the GAMUS AGL reference,
select **Height above ground**, choose **Fast**, then upload `gamus_DC_04_27.png`.
Show progress and the resulting city. Use the upload control inside the Image tab so the
reference settings are visible. Shorten the processing wait if needed.

**Say:**

> First, we upload an ordinary PNG without geographic coordinates. We supply its known pixel
> size and attach a reference height map. A pretrained remote-sensing height backbone,
> fine-tuned on GAMUS, estimates height above ground and land cover. Building heights use our
> evaluated model blend, with a separate canopy specialist in extensive forest when available.
> The output is a relative surface product: heights above ground, without a known absolute
> elevation datum. The image becomes the texture for the reconstructed scene.

**On-screen caption:** “Uploaded model prediction · GSD 0.33 m/pixel”.

## 1:00–1:30 — Validate the PNG prediction

**Show:** Accuracy tab. Hold the metrics and scatter plot on screen. Then View → Height,
followed by Error. Keep the error layer visible long enough to show its legend.

**Say:**

> We validate the prediction against the attached height-above-ground reference. On this
> particular GAMUS tile, Fast quality reports about 2.83 metres RMSE, 1.68 metres MAE and
> 0.895 correlation. These are scene-specific results, not a promise for every image. The
> scatter plot and class-wise errors show where the estimate differs from the reference.
> The Error layer makes those differences visible across the scene.

**Use the numbers actually displayed.** The values above were verified locally on 2026-10-04.

## 1:30–2:15 — GeoTIFF input and absolute elevation

**Show:** Image tab. Replace the reference with the Philadelphia LiDAR DSM. Select
**Absolute elevation (DSM)**, clear the pixel-size field, retain **Fast** and **SRTM**, then
upload `naip_philadelphia_cityhall.tif`. Show Ground (SRTM), DSM range and downloads. In View,
switch City model → Exact DSM, then back. In Accuracy, show “Compared with the absolute DSM”.

If coarse-DEM street ramps distract from the building view, use **View → Ground display →
Flat ground** and keep its visualization label visible. Say: “This view removes terrain relief
to inspect the estimated buildings; it does not show measured street elevations.” Switch back
to **Estimated terrain** before showing flood-defence crest lines.

**Say:**

> A GeoTIFF provides coordinates, a coordinate system and pixel size. AltiMap uses those
> metadata to place the image and obtain a coarse elevation base. SRTM is the default;
> Copernicus is also available. The model contributes local surface detail, producing an
> absolute DSM on the source image’s grid. SRTM heights use the EGM96 datum, while Copernicus
> uses EGM2008. We explicitly select an absolute DSM reference so validation compares the
> correct elevation product. The Exact DSM view shows the per-pixel surface alongside the
> simplified city model used for exploration.

## 2:15–3:00 — Height analysis and navigation

**Show:** click a building for its height/area card. In View, show Slope and enable Contour
lines briefly. In Measure, enable Measure and double-click two visible locations for an A→B
profile. Turn Measure off. Enable Set points, place two or three points, then Play. Show a
short flythrough. Pause if still playing, reset the view, then demonstrate Walk briefly.

**Say:**

> The reconstruction supports inspection, not just viewing. Selecting a building reveals its
> estimated height and footprint. Slope and contour layers help explore the surface. A
> two-point measurement draws a height profile and reports the distance between the points.
> We can orbit the scene, fly manually, or create a camera route by placing waypoints.
> First-person Walk follows the estimated ground and blocks movement through building cells.
> All these tools operate inside the same browser interface.

## 3:00–3:40 — Disaster-management context

**Show:** remove the reference, upload `naip_pittsburgh_bridges.tif`, then open Context.
Show actual bridge/facility results and their scene markers. Optionally insert a short clip of
Baton Rouge’s mapped levee and its crest line; shorten this segment elsewhere to keep five minutes.

**Say:**

> Geographic coordinates also let us retrieve OpenStreetMap context for the image’s area.
> Here we can inspect mapped bridges and critical facilities such as hospitals, schools or
> emergency services, where those features are available. Mapped levees and embankments can
> also be preserved in the terrain and shown with estimated crest elevations. These features
> come from map data, rather than being recognised as hospitals or flood defences by the
> height model. They provide useful planning context, while their accuracy still depends on
> the map coverage and elevation estimates.

## 3:40–4:20 — Evaluation across landscapes

**Show:** the current report’s absolute DSM table. Highlight city, sparse suburb, hills and
forest; highlight the SRTM rows. Label this screen “External NAIP / USGS 3DEP evaluation,
Fast quality, all valid pixels”. Optionally show a brief montage of the four uploaded scenes.

**Say:**

> We also evaluate across four external landscapes using USGS reference elevation data.
> With SRTM, absolute DSM RMSE is approximately 38.87 metres in the dense city, 4.28 in the
> suburb, 6.24 in the hills and 7.44 in forest. The model improves on the base DEM in the first
> three scenes, but forest remains slightly worse. Tall urban structures and canopy heights
> are continuing limitations. Agreement with a public DEM measures consistency; independent
> LiDAR comparison is the stronger accuracy check. Indian satellite scenes still need local
> reference data before we can claim their accuracy.

## 4:20–5:00 — Export and close

**Show:** return to a processed scene. In Image, download nDSM and absolute DSM GeoTIFFs
and the buildings GeoJSON where available. Click Export in the top bar to save the GLB.
If QGIS/Blender is already installed and prepared, insert a short clip opening those files;
otherwise show the downloaded files without claiming an external-tool demonstration.
Finish with a slow orbit and a closing title.

**Say:**

> The results are available beyond the viewer. We export float32 elevation GeoTIFFs with
> metadata sidecars, building footprints as GeoJSON, and the current three-dimensional scene
> as a GLB in metres. The unified application serves the interface and inference API from one
> local process, with source code, setup instructions and measured evaluation results in the
> repository. AltiMap brings single-image elevation estimation, reference validation and
> interactive exploration into one workflow. Our next priority is stronger validation on
> Indian satellite imagery and better recovery of tall structures and forest canopy.

**Closing title:** “AltiMap · Image → Elevation → Validation → 3D exploration”.

## Keep the recording accurate

- Use **uploaded predictions** for the main demonstration. Sample scenes are LiDAR reference
  previews; if shown, retain their reference label.
- Keep uploaded scenes at **1× vertical exaggeration** while discussing physical heights.
- If PNG GSD is unknown, show the **experimental 0.33 m/pixel assumption**. A known-length
  scale measurement can set GSD and reprocess; it does not establish an absolute datum.
- Match **reference type and datum** to the file. Do not rely on Auto-detect for the main demo.
- Only present metrics with their dataset, scene, GSD and quality. Do not call public-DEM
  agreement independent ground-truth validation or claim survey-grade accuracy.
- Leave a caption on edited processing waits. Scene-sized GeoTIFFs can take minutes on a
  laptop GPU even though a warmed small GAMUS tile takes seconds.
- If the recording must be three minutes, omit the disaster-context insert, compress the
  pipeline explanation and show only one profile and one short route. Retain both input types,
  reference validation, export and the accuracy limitation.
