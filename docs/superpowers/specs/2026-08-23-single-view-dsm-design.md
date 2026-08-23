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
    + DTM (smoothed public 30 m DEM, low-frequency)
```

This is the decision the whole project rests on. It splits the problem along the axis where data availability actually divides:

- **High-frequency structure** (buildings, trees, walls) is what a learned model can extract from a single RGB image, and is what LiDAR reference data supervises well.
- **Absolute datum** (is this hilltop at 40 m or 1400 m?) is unrecoverable from a single image in principle, but is freely and accurately available from Copernicus GLO-30. At 30 m resolution a public DEM carries no building signal — but for terrain trend it is entirely sufficient.

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
  "gsd_m": 0.5,
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
| `terrain` | Copernicus GLO-30 fetch, reproject, smooth to DTM |
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

### 4.1 Encoder — swappable, two candidates

Verified 2026-08-23 against HuggingFace model pages.

| Option | Trainable params | VRAM (train, 6 GB budget) | Licence | Domain match |
|---|---|---|---|---|
| `depth-anything/Depth-Anything-V2-Small` | 25 M | ~2.0 GB @ bs4 | Apache-2.0 | natural images |
| **`depth-anything/DA3-BASE`** | 135 M | ~4.2 GB @ bs2 + ckpt | **Apache-2.0** | natural images |
| **`facebook/dinov3-vitl16-pretrain-sat493m`** frozen + trained DPT head | ~30 M | **~1.5 GB @ bs8** | Meta custom, **gated** | **0.6 m Maxar ortho** |

**Licence constraint:** Depth Anything V2 **Base and Large are both `cc-by-nc-4.0`** — non-commercial. Only V2-Small is Apache-2.0. Depth Anything 3 resolves this: `DA3-SMALL` (34 M) and `DA3-BASE` (135 M) are Apache-2.0, as is `DA3METRIC-LARGE` (334 M).

**Counterintuitive memory result:** freezing a 303 M encoder is *cheaper* than fine-tuning a 25 M one, because AdamW's two fp32 moment buffers dominate the footprint and a frozen encoder has none. This buys a domain-matched ViT-L for a third of the memory of the small fine-tune.

**Frozen-encoder feature caching.** A frozen encoder means features can be precomputed once and the head trained on cached features in minutes rather than hours — converting a scarce experiment budget on a single laptop GPU into an abundant one. Cost: ~2 MB/tile for ViT-L tokens, so 10 k tiles is ~20 GB against 32 GB free disk. Mitigated by fp16 caching on a subset. A real trade-off, not a free win.

**Decision:** `DA3-BASE` fine-tune as the licence-safe primary. DINOv3-SAT frozen + DPT head as a parallel contender, gated on the access request and a licence review for competition eligibility. **File the DINOv3 access request immediately** — it is the longest-lead item in the project.

### 4.2 Prior art to mirror

`facebook/dinov3-vitl16-chmv2-dpt-head` (336 M, `CHMv2ForDepthEstimation`) is a DPT decoder performing dense canopy-height regression on satellite imagery over a frozen DINOv3-SAT backbone. This is structurally the architecture proposed here, already trained. Its head-only checkpoint (~135 MB) is the reference implementation for §4.3. Gated, Meta custom licence; too large to train on 6 GB, but valuable as an architecture template and inference-only baseline.

*Unverified:* its input resolution and training GSD are not stated on the model card.

### 4.3 Head and loss

**Head:** DPT decoder over four encoder stages, single-channel output, ReLU final activation — heights above ground are non-negative by definition.

**Loss:** L1 in metres (primary) + a gradient-matching term on first differences.

- L1 is robust to LiDAR outliers and directly aligned with the graded MAE.
- The gradient term sharpens building edges, which plain L1 blurs.
- **SILog is deliberately excluded.** It is scale-invariant, and absolute scale is precisely what we are trying to predict; a scale-invariant loss discards the signal we need.
- Relative term weighting is a hyperparameter for the ablation table.

### 4.4 Resolution and inference

- Canonical working resolution **0.5 m/px**.
- Georeferenced input is resampled from its GeoTIFF transform; PNG/JPG takes a user-supplied GSD, defaulting to 0.5.
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

**Metrics:** RMSE, MAE, Pearson correlation, median error, P50/P90 absolute error.

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

**Status: unverified at time of writing.** Research on dataset availability and published accuracy targets was interrupted by a rate limit and is being re-run. Candidates under evaluation:

- USGS 3DEP + NAIP via Microsoft Planetary Computer STAC — attractive because tile-level programmatic access suits a disk-constrained incremental loop
- IEEE GRSS DFC2019 (US3D), DFC2023 Track 2
- GeoNRW
- Copernicus DEM GLO-30 (for the DTM component, needed regardless)
- ESA WorldCover 10 m (for evaluation stratification)

**Hard constraint: 32 GB free disk.** The data pipeline must download a tile, extract patches, and delete the source incrementally — never materialising a full dataset. Tile cache stored as lossless-compressed RGB PNG + 16-bit height PNG pairs to keep the working set small.

Target RMSE figures are deliberately left unstated pending verification, rather than guessed. Numbers going into a report must be traceable to a citation.

## 8. Environment

- **System Python is 3.14.6, which has no PyTorch wheels.** The project pins its own **Python 3.12** via `uv`.
- GPU: RTX 3050 Laptop, **6 GB VRAM** — the binding constraint on model choice (§4.1).
- Disk: **32 GB free** on `/home` — the binding constraint on data strategy (§7).
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
| 0 | Spikes: data fetch, PyInstaller + rasterio, browser WebGPU depth, DINOv3 access request filed |
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
| Co-registration error inflating RMSE | Phase-correlation alignment; report raw and corrected |
| Dataset downloads exceed 32 GB | Incremental tile-level fetch from Planetary Computer |
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
