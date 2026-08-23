# AltiMap — Single-View Elevation Estimation and 3D Visualization

**Date:** 2026-08-23
**Status:** Design approved, pending spec review
**Context:** Hackathon submission, timeline of weeks. Graded 50% on DSM accuracy (RMSE / MAE / correlation against LiDAR reference, stratified across urban, sparse, hilly, and forested landscapes) and 50% on rendering quality, navigability, and standalone deployment.

---

## 1. Problem

Turn a single overhead optical RGB image into an elevation map, and let a user fly through the result in 3D.

Two input paths, two outputs:

| Input | Output | Datum |
|---|---|---|
| PNG / JPG, no spatial metadata | Relative DSM (rDSM) | `relative` |
| GeoTIFF with CRS and transform | Absolute DSM, metric heights | `ellipsoidal` |

The stated difficulty is that monocular depth foundation models are trained on natural egocentric imagery and emit *relative, affine-invariant* depth, so they suffer a domain gap on nadir imagery and carry no absolute scale.

## 2. Core design decision

**Predict nDSM — height above local ground, in metres — not depth, and not absolute elevation.** Absolute elevation is then composed:

```
DSM = nDSM (learned, metric, high-frequency)
    + DTM (public bare-earth DEM, low-frequency)
```

This formulation is independently validated by the training data: USGS 3DEP publishes `3dep-lidar-hag` ("height above ground"), a **ready-made nDSM product**. The supervision target exists directly, with no DSM−DTM differencing step and no co-registration between two elevation rasters to get wrong.

This is the decision the whole project rests on. It splits the problem along the axis where data availability actually divides:

- **High-frequency structure** (buildings, trees, walls) is what a learned model can extract from a single RGB image, and is what LiDAR reference data supervises well.
- **Absolute datum** (is this hilltop at 40 m or 1400 m?) is unrecoverable from a single image in principle, but is freely available from public DEMs. At 10–30 m resolution such a DEM carries no building signal — but for terrain trend it is entirely sufficient.

**DTM source, and a trap to avoid.** The bare-earth component must be a genuine *terrain* model. Copernicus GLO-30 is a **surface** model (DSM) — using it here would double-count buildings and canopy, inflating every height by whatever the 30 m DEM already includes. Sources, in preference order:

| Source | Resolution | Bare earth? | Coverage |
|---|---|---|---|
| `3dep-lidar-dtm` | 2 m | Yes | Partial, workunit-based |
| `3dep-seamless` | 10 m | Yes | CONUS + AK/HI |
| Copernicus GLO-30 + morphological opening | 30 m | **Approximated** | Global |

CONUS uses `3dep-seamless`. Outside it, GLO-30 filtered by morphological opening is the fallback, and **must be reported as an approximation** rather than presented as bare earth.

Consequences:

- **Scale ambiguity never arises.** The network emits metres by construction. There is no relative-to-metric calibration step to get wrong.
- Fitting a scale factor against a 30 m DEM — the literal reading of the brief — is *avoided* rather than solved. That approach can only recover terrain trend, and forcing building-scale predictions to match a 30 m surface actively destroys the detail we want.
- The rDSM path is the same network with the same metric output, written without a CRS. Non-georeferenced input therefore gets **metric** heights, which is strictly stronger than the brief requires.

## 3. Architecture

Four subsystems joined by one narrow contract.

**One spec, several implementation plans.** This document covers the whole system because the contract in §3.1 is what makes the subsystems separable — that reasoning has to live in one place. Each subsystem then gets its own implementation plan and build cycle, and can be worked on independently once the contract is fixed.

### 3.1 The contract

Producers emit and the viewer consumes exactly:

1. A **Cloud-Optimized GeoTIFF**, float32 elevation in metres.
2. The **source RGB**, on the identical grid.
3. A **JSON sidecar**:

```json
{
  "gsd_m": 2.0,
  "source_gsd_m": 0.6,
  "datum": "ellipsoidal",          // or "relative"
  "vertical_unit": "m",
  "model_version": "da3base-ndsm-v1",
  "height_range_m": [0.0, 84.3],   // min/max of the emitted band
  "tile_overlap_px": 130,
  "dtm_source": "copernicus-glo30" // or null for the rDSM path
}
```

`height_range_m` exists so the viewer can set up its colour ramp and camera bounds without scanning the raster.

Nothing else crosses the boundary. The viewer never imports ML code; the ML code does not know the viewer exists. Either side can be rebuilt or replaced independently, and each is testable alone against a fixture triple.

### 3.2 Subsystems

**`elevation/` (Python)** — ingest → tile → infer → stitch → compose → write.

| Module | Responsibility |
|---|---|
| `io` | rasterio read/write, CRS and transform handling, format dispatch |
| `tiling` | 518 px windows, 25% overlap, cosine-feathered reassembly |
| `model` | Encoder (swappable) + DPT nDSM head, checkpoint loading |
| `terrain` | Bare-earth DEM fetch (`3dep-seamless`, or filtered GLO-30 outside CONUS), reproject, smooth to DTM |
| `compose` | nDSM + DTM → DSM, or rDSM passthrough |
| `writer` | COG + sidecar emission |

**`viz/` (React + Three.js)** — loads the contract triple, renders a chunked displaced grid mesh, provides navigation and inspection tools.

**`app/` (FastAPI + PyInstaller)** — thin orchestration only, no algorithms. Serves static frontend, exposes `POST /jobs` → `GET /jobs/{id}` → `GET /jobs/{id}/dsm.tif`, launches a pywebview window.

**`live/` (live-video 3D mode)** — showcase feature. **Explicitly last in build order and explicitly cuttable**; removing it must not touch subsystems 1–3.

**`eval/` (CLI, outside the app)** — accuracy measurement. See §6.

### 3.3 The shared renderer abstraction

```
HeightFieldSource (interface)
  ├── GeoTiffSource   → DSM GeoTIFF + RGB          (satellite mode)
  └── LiveDepthSource → per-frame depth + video    (live mode)
        ↓
   TerrainMesh — chunked grid, vertex displacement in shader
```

Live mode is a second *data source*, not a second renderer. This is what makes it cheap to add and cheap to delete.

## 4. Elevation model

### 4.1 Encoder — swappable, fully open weights only

**Hard requirement: permissively-licensed, ungated weights.** No non-commercial licences, no access requests, no custom vendor licences. This rules out several otherwise-attractive options (§4.2) and costs less than expected.

Verified 2026-08-23 against HuggingFace model pages.

| Option | Params | VRAM (train, 6 GB budget) | Licence | Domain match |
|---|---|---|---|---|
| **`depth-anything/DA3-BASE`** | 135 M | ~4.2 GB @ bs2 + ckpt | **Apache-2.0** | natural images, strong depth prior |
| **`torchgeo/swin_v2_b_naip_rgb_satlas`** | ~88 M | ~3.5 GB @ bs2 + ckpt | **Apache-2.0** | **NAIP aerial RGB, ~0.6–1 m** |
| `depth-anything/DA3-SMALL` | 34 M | ~2.2 GB @ bs4 | Apache-2.0 | natural images |
| `depth-anything/Depth-Anything-V2-Small` | 25 M | ~2.0 GB @ bs4 | Apache-2.0 | natural images |
| `MVRL/scalemae-vitlarge-800` frozen + head | ~30 M trainable | ~1.5 GB @ bs8 | Apache-2.0 | fMoW-RGB, **GSD-conditional** |

### 4.1.1 Published evidence against the obvious choice — read this first

**A plain U-Net beat Depth Anything V2 on this exact task.** On DFC2023, building-wise RMSE: DAv2-S **6.4048** vs U-Net **4.9312** (arXiv:2506.02534). Same result direction on GBH. The stated cause is architectural, not incidental:

> "Transformer-based architectures typically require large-scale datasets to reach their full capacity, emphasizing global context over local detail, which can lead to blurred building boundaries."

Depth foundation models won in that literature only when given specific extra machinery:

| Intervention | Effect | Source |
|---|---|---|
| Sparse-LiDAR affine anchor (HRF) | MAE 5.69 → **4.32** | arXiv:2505.06905 |
| **BitFit PEFT** (beat LoRA, Adapter, VPT, *and* full fine-tuning) | MAE **−30.9%**, F1-HE +44.2% | same |

**Consequences for this design:**

1. A **convolutional/hierarchical baseline is a co-primary, not a courtesy baseline.** It may simply win.
2. The transformer path uses **BitFit-style parameter-efficient fine-tuning, not full fine-tuning** — published as the best of five PEFT strategies on this task, and it also slashes VRAM (bias-only training means almost no optimizer state), which suits 6 GB.
3. This strengthens the Satlas Swin option independently: Swin is *hierarchical*, so it preserves local detail far better than a plain ViT — directly addressing the "blurred building boundaries" failure mode.

### 4.1.2 Decision: three encoders, run as a genuine ablation

| # | Encoder + head | Hypothesis under test |
|---|---|---|
| 1 | **U-Net / Swin-V2-B Satlas NAIP + UPerNet** | Local detail and domain-matched pretraining beat a general depth prior |
| 2 | **`DA3-BASE` + DPT, BitFit PEFT** | A strong depth prior wins *if* fine-tuned parameter-efficiently |
| 3 | `DA3-SMALL` + DPT, BitFit | Cheap control for the capacity question |

Start with **#1**, because the literature says it is the one most likely to work and it is also the cheapest to train. #2 is the interesting result either way — confirming or refuting the premise the brief suggests.

This is why §3.2's encoder abstraction is load-bearing rather than speculative generality: the evidence genuinely does not settle which backbone wins.

**Stretch contender: Scale-MAE.** Its GSD-conditional positional encoding directly targets the GSD-sensitivity risk in §11, making it the natural response if that risk materialises. Cost: it is a raw MAE checkpoint, not `transformers`-loadable, so it needs custom loading code. Deferred unless needed.

**Frozen-encoder feature caching** remains available for any of these: freeze the encoder, precompute features once, then train the head on cached features in minutes rather than hours — turning a scarce experiment budget on one laptop GPU into an abundant one. Cost is disk, ~2 MB/tile for ViT-L-scale tokens, so 10 k tiles is ~20 GB against 32 GB free. Mitigated by fp16 caching on a subset. A real trade-off, not a free win.

**Licence note for the record:** Depth Anything **V2 Base and Large are both `cc-by-nc-4.0`**, not just Large. Only V2-Small is Apache-2.0. Depth Anything 3 fixes this — `DA3-SMALL`, `DA3-BASE`, and `DA3METRIC-LARGE` are all Apache-2.0 (`DA3-LARGE` is not). This is why the project targets DA3 rather than V2.

### 4.2 Rejected on licence or access grounds

Recorded so the decision is not silently revisited:

| Model | Why rejected |
|---|---|
| `facebook/dinov3-*` (incl. `-pretrain-sat493m`) | Gated (manual approval) + Meta custom licence, not open source |
| `facebook/dinov3-vitl16-chmv2-dpt-head` | Same. **Near-exact prior art** — a DPT head doing dense canopy-height regression on satellite imagery — so its published architecture remains a useful reference even though the weights are unusable here |
| `Depth-Anything-V2-Base` / `-Large` | `cc-by-nc-4.0` |
| `depth-anything/DA3-LARGE` | `cc-by-nc-4.0` (but `DA3METRIC-LARGE` is Apache-2.0) |
| `apple/DepthPro` | `apple-amlr` research licence; also a hard 1536×1536 input |
| `lpiccinelli/unidepth-v2-*` | CC BY-NC 4.0; also conditions on camera intrinsics |
| `JUGGHM/Metric3D` | No licence tag at all — unusable without clarification |

Losing DINOv3 costs less than it appears: Satlas NAIP provides comparable aerial-domain pretraining at a comparable GSD under Apache-2.0, and dropping the access request removes both the longest-lead item and a licence-eligibility risk from the plan.

### 4.3 Head and loss

**Head is encoder-dependent:** DPT decoder for plain-ViT encoders (DA3), UPerNet/FPN for hierarchical ones (Swin, U-Net). Single-channel output, ReLU final activation — heights above ground are non-negative by definition.

#### The long tail is the central modelling problem

Height distributions are severely long-tailed: near-zero background is the head, tall buildings the tail. Networks trained by plain regression are consequently **biased toward underestimating tall buildings**. The magnitude is startling — same model on GBH, all-pixel RMSE **4.89 m** against a height-balanced building RMSE of **37.47 m**. This, not the choice of backbone, is what separates SOTA from mediocre.

#### Loss design

**Plain L1 regression is the field's default but is measurably not its best.** The cleanest published ablation — identical architecture, Vaihingen (arXiv:2006.02801, Table II):

| Objective | RMSE (m) | Rel |
|---|---|---|
| MSE regression | 3.217 | 0.814 |
| Multi-class CE + SID | 2.192 | – |
| **Ordinal + SID discretisation** | **1.698** | **0.314** |

A **~47% RMSE reduction from the loss function alone.** Ignoring this would be the single largest avoidable error in the project.

**Design — staged, so each addition is measurable:**

1. **Stage A (baseline):** L1 in metres + gradient-matching on first differences. Robust to LiDAR outliers, directly aligned with graded MAE, gradient term sharpens the edges L1 blurs.
2. **Stage B (expected primary):** add an **ordinal-classification head over spacing-increasing-discretised (SID) height bins**, with the final height as the bin-distribution expectation. This is the intervention with the strongest published evidence.
3. **Stage C (stretch, if time):** HTC-DC Net's **head-tail-cut loss** — separating foreground and background so the long tail stops being swamped — plus a KL term against a reference distribution (Gaussian foreground, uniform background). Its published total is `Lh + 0.01·Lb + Lhtc + Ldist`.

**SILog is deliberately excluded**, and this is now evidence-backed rather than merely argued: no paper in this literature uses SILog as a primary objective. It is scale-invariant, and absolute scale is exactly what we are predicting — the recent foundation-model work treats scale-ambiguous output as *the defect to be fixed*, not a property to adopt. Log-space quantities (Rel, δ1, RMSE-log) belong in *reporting*, inherited from indoor MDE, not in the objective.

### 4.4 Resolution — input 0.6 m, output 2 m

**Output resolution is capped by reference-data resolution.** NAIP imagery is 0.6 m/px, but every `3dep-lidar-*` raster is **2 m** and no finer variant is published. Therefore:

- **Input:** 0.6 m/px RGB tiles.
- **Supervision and output:** **2 m/px** nDSM.
- The model consumes fine imagery and emits a coarser height field — a 1:3.33 ratio, which is unremarkable for dense regression (DPT natively decodes at a fraction of input resolution).

This is a deliberate acceptance, not an oversight. Predicting at 0.6 m would be unverifiable: there is no 0.6 m reference data to validate it against, so the extra detail would be unfalsifiable decoration. Reported RMSE is therefore a **2 m-grid** figure, and the spec says so wherever numbers appear.

*Escape hatch, deliberately not taken:* rasterizing `3dep-lidar-copc` / `s3://usgs-lidar-public` point clouds with PDAL would yield 0.5 m targets. It costs days and is out of scope (§13).

- Georeferenced input is resampled to 0.6 m from its GeoTIFF transform; PNG/JPG takes a user-supplied GSD, defaulting to 0.6.
- Tiled inference at 518 px with 25% overlap and cosine-feathered blending, so arbitrarily large scenes work.

## 5. Visualization

### 5.1 Rendering

**Displace in the vertex shader, never on the CPU.** The DSM uploads as a float32 `DataTexture`; the mesh is a static grid whose vertices are lifted in the vertex shader.

Everything else follows from this: swapping DSMs costs one texture upload, a continuous vertical-exaggeration slider is free, and geometry is never rebuilt. Normals come from finite differences on the height texture in the fragment shader — needed anyway for the slope readout, so slope shading is free.

**Projection accuracy is structural, not computed.** DSM and RGB share one grid, so UVs are identity. There is no reprojection step and therefore no reprojection error.

**LOD:** quadtree over 128×128-quad chunks, two or three levels, vertical skirts on chunk edges to hide seam cracks. Nothing more elaborate is warranted at single-scene scale.

### 5.2 Navigation

Pointer-locked first-person flight (WASD + mouse look), orbit mode for overview, and a spline **drone-path mode** for hands-off cinematic flythrough. The last matters because judges watch demos: a smooth automated pass reads as far more finished than a fumbled mouse. Camera clamps above terrain by sampling the height texture.

### 5.3 Inspection and validation tools

This is where the brief's "validate estimated height values against reference datasets" is earned, so it gets real UI:

- **Height probe** — click for metres. Samples the *DSM data* at that texel, never the displaced geometry, so exaggeration cannot corrupt the reading.
- **Slope readout** — degrees, from the DSM gradient.
- **Profile line** — two clicks produce a 2D elevation transect overlay.
- **Reference diff** — load a reference DSM; get a signed-error colormap on the terrain, an error histogram, and live RMSE / MAE / correlation. **Computed by the same code path as `eval/`**, so the demo and the report can never disagree.

### 5.4 Live-video mode (Phase 6, cuttable)

Phase 1 — per-frame relief: `getUserMedia` → video texture → Depth Anything in-browser via WebGPU at reduced resolution → depth texture → the same displaced mesh, unprojected with an assumed pinhole FOV instead of orthographically. An EMA on the depth texture damps the frame-to-frame flicker monocular depth is prone to.

Phase 2 — pose-tracked fusion behind a toggle: accumulate unprojected depth into a persistent point cloud or TSDF.

**Honest framing:** phase 1 is a live 2.5D relief of the current view, *not* a persistent model. Geometry visibly reshapes as the camera moves, because there is no memory between frames. Only phase 2 produces an accumulating 3D model, and it requires pose tracking plus cross-frame scale alignment.

Fallback if browser WebGPU inference proves too slow: run depth on the Python backend and stream over a WebSocket.

## 6. Evaluation

`eval/` is a standalone CLI — `eval pred.tif ref.tif --landcover wc.tif --out report.json` — built **first**, because half the score is a number it produces.

Three details matter more than the metric formulas:

1. **Co-registration before metrics.** A one-pixel horizontal misalignment between predicted and reference DSM will dominate RMSE and make a good model look bad. The harness estimates planar shift by phase correlation and vertical bias by median difference, then **reports raw and corrected numbers side by side** — never silently corrected. A hidden correction reads as a thumb on the scale.
2. **Masking.** Nodata and water. Heights over water are meaningless and only add noise.
3. **Percentiles alongside RMSE.** P50 and P90 absolute error reveal whether errors are broad or concentrated; RMSE alone hides outlier behaviour.

### 6.1 Metrics — all-pixel RMSE alone is misleading

**A single all-pixel RMSE will flatter the model badly and a sharp judge will catch it.** Published example, same model on GBH: all-pixel RMSE **4.89 m** alongside height-balanced building RMSE **37.47 m**. All-pixel error is dominated by vast near-zero-height background, so it mostly measures how well the model predicts "the ground is at ground level."

The harness therefore reports a **metric matrix**, not a number:

| Metric | Scope | Why |
|---|---|---|
| RMSE, MAE | all pixels | Comparability with the brief and prior work |
| RMSE, MAE | **building pixels only** | The honest difficulty measure |
| RMSE | **building-wise** (one median height per instance) | How the DFC/GBH literature reports; instance-level utility |
| RMSE | **height-balanced** (equal weight per height stratum) | Exposes tall-building underestimation (§4.3) |
| **δ1** | all pixels | `max(y/ŷ, ŷ/y) < 1.25`; the *official DFC2023 Track 2 metric* |
| Pearson r | all pixels | Required by the brief |
| Median error, P50/P90 absolute | all pixels | Distinguishes broad from concentrated error |

Building masks come from the reference nDSM thresholded and cleaned, or from OSM footprints where available — the method is recorded in the report because it affects the numbers.

### 6.2 Target numbers

**DFC2023 is nearly an exact match for our configuration** — 0.5 m optical input with 2 m nDSM labels, against our 0.6 m NAIP with 2 m HAG. Its published results are therefore the directly relevant targets:

| Level | Building-wise RMSE | δ1 |
|---|---|---|
| Weak | > 6.0 m | < 0.5 |
| Plain U-Net reference | ~4.93 m | – |
| Depth Anything V2 S reference | ~6.40 m | – |
| **SOTA (best published pipeline)** | **~4.17 m** | – |
| **DFC2023 Track 2 winner (HGDNet)** | – | **0.8012** |

**RMSE is uninterpretable without dataset, GSD, and pixel-versus-building scope.** The literature's own trend, one architecture family across GSDs: 0.09 m → 1.30 m RMSE, 1.3 m → 2.12 m, 3 m → 4.49 m. Every number we report carries all three qualifiers.

*Unverified:* the DFC2023 outcome paper's full leaderboard (IEEE paywall, OpenReview CAPTCHA); one HTC-DC Net table entry (B7 building-pixel RMSE 1.5173) is internally inconsistent and probably a typo — cite the B5 figure instead if a building-pixel number from that paper is needed.

**Stratification** into urban / sparse / hilly / forested from ESA WorldCover 10 m land cover, plus DTM-derived terrain roughness for "hilly". Automatic and defensible, no hand-labelling. Reported per class, because the rubric grades stability *across* classes rather than in aggregate.

**Baselines:**

| Baseline | What it establishes |
|---|---|
| Constant predictor (mean nDSM) | **The honest floor** — fail to beat it and nothing was learned |
| DTM only (nDSM = 0) | What the free 30 m DEM alone achieves |
| Zero-shot depth + affine fit to DEM | The literal-brief reading |
| Ours | — |

The first row is the one most published work quietly omits. Including it is what makes the rest of the table credible.

**Validation split: hold out an entire city, not random tiles.** Random tile splits leak, because adjacent tiles share buildings and lighting; a random-split number is a fantasy that will not survive a judge asking about generalisation.

## 7. Data

**Source: Microsoft Planetary Computer STAC.** Verified live 2026-08-23 by real HTTP requests — no API key, anonymous SAS token issuance working.

| Collection | Product | Role |
|---|---|---|
| `naip` | 0.6 m RGB+NIR orthophoto | Model input |
| `3dep-lidar-hag` | **2 m height-above-ground** | **Supervision target (ready-made nDSM)** |
| `3dep-seamless` | 10 m / 30 m bare-earth DEM | DTM component |
| `esa-worldcover` | 10 m land cover | Evaluation stratification |
| `cop-dem-glo-30` | 30 m surface model | Global DTM fallback (needs filtering, §2) |

Access pattern:

```
POST https://planetarycomputer.microsoft.com/api/stac/v1/search
GET  https://planetarycomputer.microsoft.com/api/sas/v1/token/{account}/{container}
```

via `pystac-client` + `planetary-computer.sign()`, reading windows through GDAL `/vsicurl/`.

### 7.1 Disk is a non-issue — do not download rasters

All assets are COGs serving `Accept-Ranges: bytes` (verified: range GETs return HTTP 206 with valid TIFF magic). Windowed reads through `/vsicurl/` fetch only the internal tiles touched, so **peak disk is the extracted patch cache, not the source rasters** — which matters, since one NAIP tile is 467 MB against 32 GB free.

**The incremental download-extract-delete pipeline is therefore cut from the design.** Patches are extracted directly from remote COGs. If a local mirror is ever wanted, one NAIP + HAG pair peaks at ~490 MB, comfortably inside budget.

Measured reference pair (Wasatch, Utah): NAIP 467.20 MB at 12320×9610 / 0.6 m / EPSG:26912; HAG 16.45 MB at 4097×4097 / 2 m.

### 7.2 Coverage is patchy — AOI selection is a phase-0 gate

`3dep-lidar-hag` coverage is workunit-based, not seamless. Verified empirically: **0 hits over Raleigh NC, hits over Utah.** Candidate AOIs must be STAC-searched for *both* `naip` and `3dep-lidar-hag` coverage before any training commitment. Selecting AOIs spanning urban, sparse, hilly, and forested land cover is a phase-0 deliverable, because the rubric grades stability across exactly those classes.

**Sample broadly, not deeply — label diversity beats label quality.** The strongest published finding on generalisation (arXiv:2506.02534): a U-Net trained on high-quality labels alone scored 4.33 m in-domain but degraded to **9.41 m out-of-domain** (Rio de Janeiro 13.43 m). Training instead on *diverse but imperfect* labels cut out-of-domain error to **3.83–4.83 m** while barely harming in-domain performance.

So AOI selection maximises geographic and morphological spread — many workunits, fewer tiles each — rather than exhaustively mining one well-covered region. This is a data-strategy decision with a larger expected effect on the score than any architecture choice in §4.1.

### 7.3 Rejected sources

| Source | Why |
|---|---|
| `s3://naip-analytic`, `naip-source`, `naip-visualization` | Requester-pays — verified 403 anonymous. Planetary Computer serves the same imagery free |
| `s3://prd-tnm/` 1 m elevation | Anonymous access works, but **DTM only** — no CONUS DSM, so no nDSM derivable |
| `StagedProducts/Elevation/DSM/` | Alaska IFSAR only, ~5 m radar-derived |
| GeoNRW, DFC2019, DFC2023 | Monolithic multi-GB archives, strictly worse than per-tile COG access. Not evaluated further |

### 7.4 Unverified

- Vertical accuracy figures (3DEP QL1/QL2 ~10 cm RMSEz, Copernicus <4 m LE90) are spec values, not confirmed from documents.
- STAC `license: proprietary` on `naip`/`3dep-*` is a Planetary Computer metadata artifact; the underlying USGS products are US public domain, but no licence document was fetched to confirm.
- Long-term availability of the Planetary Computer hosted API. It responded fully on 2026-08-23; worth re-checking before building a hard dependency on the SAS endpoint.
- Target RMSE figures — research in flight. Deliberately left unstated rather than guessed; report numbers must be citation-traceable.

## 8. Environment

- **System Python is 3.14.6, which has no PyTorch wheels.** The project pins its own **Python 3.12** via `uv`.
- GPU: RTX 3050 Laptop, **6 GB VRAM** — the binding constraint on model choice (§4.1).
- Disk: **32 GB free** on `/home`. Originally expected to be the binding constraint on data strategy, but COG range reads dissolved it (§7.1). It now only bounds the optional feature cache (§4.1).
- RAM: 15 GB. Tile cache lives on disk, not in memory.

## 9. Packaging

`uv` for environment management. PyInstaller bundles FastAPI, torch, rasterio, and the built frontend into one executable that launches the server and opens a pywebview window. The same build also runs as a plain web app for browser demos.

Two deliberate choices:

- **rasterio's bundled-GDAL wheels, not system GDAL.** GDAL under PyInstaller is a well-known source of packaging pain.
- **Model weights download on first run rather than being bundled.** Keeps the installer small and avoids redistributing weights whose licence may prohibit redistribution.

## 10. Phasing

Each phase ends somewhere shippable.

| Phase | Deliverable |
|---|---|
| 0 | Spikes: **AOI coverage search** (§7.2), PyInstaller + rasterio, browser WebGPU depth, Satlas/torchgeo loading path |
| 1 | `eval/` + data pipeline, tested on fixtures |
| 2 | Zero-shot approach end-to-end — **contract is real, viz unblocked** |
| 3 | Training; the model that earns the accuracy score |
| 4 | Viewer with inspection and validation tools |
| 5 | Standalone executable |
| 6 | Live-video mode — **cuttable** |

**Phase 2 is load-bearing.** A weak-but-working pipeline producing a real GeoTIFF unblocks all visualization work to proceed in parallel with training.

Build order rationale: `eval/` and the data pipeline come first because you cannot improve what you cannot measure. The zero-shot approach then lands nearly free and becomes both the baseline row in the results table and the fixture generator for the viewer.

## 11. Risks

| Risk | Mitigation |
|---|---|
| **Transformer backbone underperforms a plain U-Net** — published, not hypothetical (§4.1.1) | U-Net/Swin is a co-primary; BitFit PEFT on the transformer path |
| **Tall-building underestimation from the long tail** — 4.89 m all-pixel vs 37.47 m height-balanced in published work | Ordinal/SID head (§4.3 stage B); report height-balanced RMSE so it cannot hide |
| **Label misalignment** between 0.6 m imagery and 2 m HAG labels | Documented in DFC2023 at the same ratio; accept an irreducible error floor and state it rather than chase it |
| **Cross-city generalisation gap of 2–3.3×** | Diverse-AOI sampling (§7.2); hold out whole regions |
| Shadow and illumination bias; systematic tall-building underestimation | Known and documented; report per-class, do not claim robustness we have not measured |
| Vegetation ~2× harder than impervious surfaces (2.14 m vs 1.01 m on Vaihingen) | Expected, not a bug; report forest class separately |
| Co-registration error inflating RMSE | Phase-correlation alignment; report raw and corrected |
| PyInstaller + GDAL packaging failure | Spike in phase 0; rasterio bundled wheels |
| DINOv3 access refused or licence-incompatible | No fallback needed — `DA3-BASE` is already the primary; the DINOv3 contender is simply dropped and §4.1's comparison loses a row |
| Cross-city generalisation gap | Hold out an entire city, not random tiles |
| Forest canopy height from RGB is genuinely hard | Report per-class; do not hide weak forest numbers |
| 6 GB VRAM limits model size | Frozen encoder + feature caching (§4.1) |
| Browser WebGPU depth too slow | Backend inference over WebSocket |

## 12. Testing

- **Tiling round-trip identity** — tile then stitch must reproduce the input exactly. Catches the class of bug most likely to silently corrupt output.
- **CRS and transform preservation** through the full pipeline.
- **Sidecar schema validation.**
- **Golden-file test** — a fixture GeoTIFF through the whole pipeline to a known result.
- **Metric regression test** — assert RMSE stays under threshold on a held-out fixture, so accuracy cannot silently regress.
- **Frontend** — contract fixture loads and renders.

## 13. Explicitly out of scope

- Multi-view or stereo reconstruction. Single-view is the brief.
- Semantic multi-task heads (building footprint segmentation). Considered and cut as YAGNI; marginal RMSE gain for meaningful training complexity. Revisit only if phase 3 finishes early.
- Depth Anything V2 metric variants (Hypersim indoor, VKITTI driving) — neither prior transfers to nadir imagery, and both would unhelpfully constrain the output range.
- UniDepth and Metric3D V2 — both condition on camera intrinsics, which have no meaningful analogue for a nadir orthophoto.
- Rasterizing LiDAR point clouds with PDAL for 0.5 m targets (§4.4). Days of work for detail we cannot validate.
- Multi-spectral input. NAIP carries NIR and it would likely help, but the deliverable accepts RGB PNG/JPG, so RGB-only keeps training and inference domains identical.

## 14. Key references

Accuracy targets and design evidence trace to these:

- **HTC-DC Net** — SOTA classification-regression hybrid, long-tail analysis, cross-city numbers. [arXiv:2309.16486](https://arxiv.org/abs/2309.16486), [code](https://github.com/zhu-xlab/HTC-DC-Net)
- **Deep ordinal regression for height** — the MSE-vs-ordinal loss ablation (§4.3). [arXiv:2006.02801](https://arxiv.org/pdf/2006.02801)
- **Weak supervision from imperfect labels** — DAv2-loses-to-U-Net result, and diversity-beats-quality (§4.1.1, §7.2). [arXiv:2506.02534](https://arxiv.org/pdf/2506.02534)
- **Sparse LiDAR-guided correction** — BitFit PEFT result, shadow-bias analysis. [arXiv:2505.06905](https://arxiv.org/html/2505.06905)
- **HGDNet** — DFC2023 Track 2 winner, δ1 = 0.8012. [arXiv:2308.05387](https://arxiv.org/abs/2308.05387)
- **IM2HEIGHT** [arXiv:1802.10249](https://arxiv.org/abs/1802.10249); **TSE-Net** (height-balanced metrics) [arXiv:2511.13552](https://arxiv.org/pdf/2511.13552)
- **Microsoft Planetary Computer STAC** — [planetarycomputer.microsoft.com](https://planetarycomputer.microsoft.com/api/stac/v1)
