# Design: depth diagnostics and the rDSM viewer

**Date:** 2026-08-25
**Status:** approved, not yet planned
**Depends on:** `docs/superpowers/spikes/2026-08-24-da3-nadir-domain-gap.md`
**Implements:** design doc §5.1–§5.3 (rendering core and inspection), the brief's Visualization Layer milestone

## 1. Why

Two things drive this.

The 2026-08-24 spike established that DA3-SMALL fits a tilted plane to nadir imagery instead of reading relief. That was measured on n=28 with `relief_ratio` as the headline statistic — and `relief_ratio` turned out to be **confounded with the artifact it was meant to detect** (the five highest-relief images all had dominant-axis |corr| ≥ 0.94). The diagnostics need a statistic that is not confounded, and a sample large enough to trust.

Separately, the brief weights Visualization at **50%** and asks for a standalone interactive platform. The viewer is not a demo accessory; it is half the deliverable. Building its rendering core now, against data we already have, de-risks the half of the score that cannot be recovered late.

The brief also settles the units question for this dataset:

> "For non-georeferenced imagery, relative height may be used directly in the visualization stage."

These Roboflow JPGs carry no spatial metadata, so they exercise the **rDSM path only**. Nothing in this document produces or displays a metre. The metric path needs the georeferenced NAIP + 3DEP HAG AOIs secured in Task 1 and is out of scope here.

## 2. The correction this design is built on

A plane fit to DA3's depth output explains, over the 28-image sample:

| plane fit R² | mean | median | min | max |
|---|---|---|---|---|
| plane | 0.744 | 0.889 | 0.155 | 0.982 |
| quadratic | 0.837 | 0.946 | 0.467 | 0.991 |

A quadratic adds only ~9 points on average, so **a plane is the right thing to subtract** — the extra degrees of freedom buy little and would start absorbing real relief.

`plane_r2` becomes the headline metric, replacing `relief_ratio`. It is the fraction of the scene DA3 explained with a ramp, so `1 - plane_r2` is the fraction that is something else. This ordering is also useful: `Lake` 0.155, `Airport` 0.206 and both `City` images (0.60, 0.63) carry real non-ramp structure, while `Desert`/`Highway`/`River` at ≥0.98 are essentially pure artifact.

## 3. Components

Three units with clean boundaries.

**`spikes/04_da3_nadir_check.py` is left alone.** It is the artifact that produced the 08-24 findings doc, and retrofitting it to import `viewer/metrics.py` would make those findings non-reproducible — the numbers in that doc would no longer correspond to the code that generated them. The enriched analysis lives in the new components instead, and supersedes the spike rather than editing it. The spike's `relief_ratio`-led framing stays on the record as what was believed on 08-24.

### 3.1 `viewer/metrics.py` — pure numpy, torch-free

Deliberately free of any torch or DA3 import, so the existing `pytest` suite can cover it in the main `.venv` without the 4.5 GB `.venv-da3`. This constraint is load-bearing, not incidental — do not import torch here.

```
fit_plane(depth)            -> (a, b, c), r2
detrend(depth, plane)       -> residual
scene_metrics(depth, rgb, conf) -> dict
```

The plane is fitted in normalized coordinates `u = x/W`, `v = y/H`, model `z = a + b·u + c·v`, by least squares. `r2 = 1 - SS_res/SS_tot`.

Metrics produced:

| name | definition | reads as |
|---|---|---|
| `plane_r2` | variance explained by the fitted ramp | how much of the scene is artifact |
| `residual_relief` | `std(residual) / median(depth)` | relief surviving detrend |
| `structure_alignment` | Pearson r between \|∇residual\| and \|∇luminance\| (Sobel) | does residual relief land on image edges |
| `conf_texture_corr` | Pearson r between confidence and \|∇luminance\| | is confidence a texture detector |
| `row_corr`, `col_corr` | retained from the 08-24 spike | ramp orientation |

`structure_alignment` is the one that answers "does it see buildings, or just tilt?" — if DA3 reads real structure, residual gradients should co-locate with RGB edges. `conf_texture_corr` converts an eyeball observation from 08-24 into a number.

RGB arrives at 640² and depth at 504², so RGB is resized to the depth grid (area interpolation) before any gradient comparison.

**Degenerate inputs:** constant depth gives `SS_tot = 0`; `median = 0` breaks `residual_relief`; a constant channel breaks Pearson. Each returns `nan` for that metric rather than raising, and `nan` propagates to the report as `n/a` instead of silently becoming 0.

### 3.2 `viewer/export_scenes.py` — depth → static assets

Imports torch lazily, runs under `.venv-da3`.

```
viewer/data/
  index.json
  scenes/<scene_id>/
    depth.bin     float32 little-endian, H*W, row-major
    rgb.jpg       resized to the depth grid
    meta.json
    terrain.glb   optional
```

`meta.json` carries `id`, `class`, `source_image`, `width`, `height`, `depth_min/max/median`, `residual_min/max`, the plane coefficients `{a, b, c}`, and the full metrics dict.

`scene_id` is `<class>__<source-stem-first-16-chars>`, matching the naming the 08-24 spike already writes for its previews, so assets and previews line up by eye. Source stems are Roboflow hashes and unique within a split; the split name prefixes the id when more than one split is exported.

**Two outputs, two scopes.** Metrics over all 1000 images land in `viewer/data/metrics_full.json` (one record per image, no assets) plus a printed per-class table with n, median and IQR. Assets land under `scenes/` for the curated 42 only. `index.json` lists the curated scenes and additionally carries the full-run per-class aggregates, so the dashboard can show how a displayed scene sits against its class distribution without loading 1000 records.

**Raw depth ships once.** Detrending happens in the vertex shader from three uniforms, so the raw ⟷ detrended toggle costs one texture, not two, and switches instantly.

**Curation.** Metrics run over all 1000 images (~2 min at 0.12 s/image); *assets* export for a curated subset. Default: **3 per class = 42 scenes** — the 2 lowest `plane_r2` and the single highest, per class. That guarantees both the informative cases and a deliberate contrast case in every class. At ~1 MB depth + ~2 MB GLB per scene this is ~130 MB; `--no-glb` drops it to ~45 MB. `/home` is at 92%, so this is a real constraint and the default is chosen against it.

**`terrain.glb`** is an orthographic displaced grid, RGB projected on as texture, UVs identity per §5.1. Default `--glb-res 256` (65k verts) rather than the full 504² (254k), for file size. This is the literal "3D model" deliverable — openable in Blender or any glTF viewer, independent of the dashboard — and it serves the brief's *"project the original optical image onto a generated 3D terrain mesh."*

**Height sign.** Depth is distance from sensor, so for a nadir view **larger depth means lower ground**. Height is `depth_max - depth`, normalized to [0,1]. Getting this backwards renders every city as a pit; it is called out here because it is silent when wrong.

### 3.3 `viewer/web/` — the dashboard

Static HTML/JS/three.js, served by `python -m http.server`. three.js is **vendored and pinned locally**, never CDN — the brief requires standalone deployment and this must run with no network.

Layout: scene list with thumbnails and per-scene metrics on the left, 3D canvas centre, controls and readouts right.

**Rendering core, per §5.1 exactly.** Depth uploads as a float32 `DataTexture`; the mesh is a static grid lifted in the vertex shader; normals come from finite differences in the fragment shader. Consequences: the exaggeration slider is free, a scene swap is one texture upload, geometry is never rebuilt, and slope shading comes along for nothing.

Vertex displacement, both modes normalized to [0,1] so the exaggeration slider behaves identically across scenes and modes:

```
raw:        h = (depth_max - d) / (depth_max - depth_min)
detrended:  r = d - (a + b·u + c·v)
            h = (residual_max - r) / (residual_max - residual_min)
```

**Scene list sorts by ascending `plane_r2` by default.** Alphabetical ordering opens on `Agriculture` (R² 0.96, near-pure ramp); ascending `plane_r2` opens on `Lake` (0.155), where there is something to see. Class-alphabetical remains available as a sort option.

Controls: exaggeration slider, raw ⟷ detrended toggle, colormap select, orbit / pointer-locked fly.

**Height probe** samples the `Float32Array` held in JS — the *data*, never the displaced geometry — so exaggeration cannot corrupt the reading. Labelled "relative", no units.

**Slope readout carries an honesty constraint.** Slope in degrees requires horizontal and vertical axes to share units, and here the vertical axis is unitless. So the readout is labelled **"display slope — depends on exaggeration"**, not presented as a physical ground slope. It becomes a real measurement only on the georeferenced path. Presenting an unqualified degree figure here would contradict the eval harness's own standard that a number is meaningless without dataset, GSD and scope.

## 4. Data flow

```
images ──DA3──> depth (504²) ──metrics.py──> plane, metrics
                    │                              │
                    └──────> export_scenes.py <────┘
                                   │
                    depth.bin + rgb.jpg + meta.json + terrain.glb
                                   │
                              fetch (lazy, per scene)
                                   │
                    DataTexture ──> vertex displacement ──> canvas
```

Metrics are computed once at export and travel in `meta.json`. The browser never recomputes them, so the dashboard and the findings doc cannot disagree — the same property §5.3 asks of the reference-diff path.

## 5. Error handling

- **No WebGL2** → explicit message, not a blank canvas.
- **Scene fetch failure** → that row shows an error state; the rest of the dashboard stays live.
- **Non-finite depth** → validated at export; fails loudly naming the scene, rather than shipping a `.bin` that renders as a spike.
- **Degenerate plane fit** (constant depth) → guarded; metric returns `nan`, export continues.
- **`Prediction.is_metric`** is an empty addict `Dict`, not an int (see 08-24 findings §4). Test truthiness.

## 6. Testing

`viewer/metrics.py` is pure numpy specifically so it can be tested in the main `.venv`:

- `fit_plane` recovers known coefficients from a synthetic ramp; `r2 == 1.0` on an exact plane.
- `detrend` of an exact plane returns ≈ 0.
- Constant-input cases return `nan`, not `ZeroDivisionError` and not `0.0`.
- Export round-trip: written `depth.bin` reads back at the declared shape and dtype, values finite.

The viewer is verified by running it and looking at it. No browser-automation harness — disproportionate at this scope.

## 7. Out of scope

Deferred deliberately, recorded so it is not silently revisited:

- **LOD quadtree** (§5.1) — one 504² scene does not need it.
- **Drone-path mode** (§5.2) — demo polish, not structural; add once there is terrain worth flying over.
- **Reference diff** (§5.3) — needs ground truth this dataset does not have.
- **Upload backend** — a brief deliverable, but it needs a live GPU process; the static export format is designed so it bolts on without changing the viewer.
- **Any metre value.** rDSM path only.
- **Oblique-vs-nadir detection.** The 08-24 confound stands: some of that sample is genuinely oblique, where a depth ramp is *correct* behaviour. I do not have a reliable detector, and an unreliable one would launder a guess into a number. The NAIP orthophoto run resolves this properly.

## 8. Known limits of what this produces

- `structure_alignment` is a **proxy**. RGB edges and elevation edges co-locate for buildings but not for painted road markings or field boundaries, which are flat. A high score is suggestive, not proof of structure recovery.
- The viewer displays relative height. It cannot validate anything against reference data yet, so it does not address the brief's "validate estimated height values" requirement — that arrives with the reference-diff work on georeferenced data.
- Detrending is a correction applied *after* the model, not a fix to the model. It makes residual relief visible; it does not make DA3 better at nadir imagery. The frozen-encoder-plus-trained-head work in design doc §4.1 remains the actual answer.
