# 24-hour metric height model — Implementation Plan

> **For agentic workers:** executed inline in the session that wrote it. Steps use checkbox syntax.
> Deviation from the writing-plans template, on purpose: under a 24 h deadline the code lives only
> in the files each task names (not duplicated here), and there are **no commits** (the user's call).

**Goal:** A fine-tuned RS3DAda height model that turns PNG/JPG/GeoTIFF into metric nDSM / absolute
DSM, scored on GAMUS, served in the existing terrain-studio frontend.

**Architecture:** VM-side scripts (model wrapper, eval, train) depend only on torch, h5py, the
cloned SynRS3D repo and `viewer/gamus_dataset.py`. They reach the VM through a private Hugging Face
repo, because the VM is only reachable through a pasted sshx terminal. The trained checkpoint comes
back the same way. The laptop runs the demo: `viewer/server.py` + `frontend/`.

**Tech stack:** PyTorch (cu128), SynRS3D `models.dpt.DPT_DINOv2` (DINOv2 ViT-L + DPT), h5py,
rasterio, FastAPI, React/three.js.

**Spec:** `docs/superpowers/specs/2026-09-29-24h-metric-height-design.md`

## Global constraints

- Heights are metres, float32, NaN nodata. Output clamped ≥ 0 for nDSM.
- GAMUS GSD is **0.33 m** (GAMUS paper §1). Inputs with a known GSD are resampled to 0.33 m.
- ViT patch 14: every model input side must be a multiple of 14 (train crops 518, infer windows 518).
- RS3DAda normalisation: mean (123.675, 116.28, 103.53), std (58.395, 57.12, 57.375) on 0–255 RGB.
- RS3DAda heads: `[{'name': 'regression', 'nclass': 1}, {'name': 'segmentation', 'nclass': 8}]`;
  forward returns a dict keyed by head name.
- Segmentation classes are OpenEarthMap order: 0 bareland, 1 rangeland, 2 developed, 3 road, 4 tree,
  5 water, 6 agriculture, 7 building. GAMUS → OEM map: ground→0, low_veg→1, buildings→7,
  water→5, roads→3, trees→4, background→ignore.
- `viewer/metrics.py` must stay torch-free. New torch code imports torch lazily inside functions.
- No git commits.

---

### Task 1: Contract accepts orthometric datum

**Files:** Modify `src/altimap/contract.py` (`VALID_DATUMS`); test in `tests/test_contract.py`.
- [ ] Failing test: `Sidecar(..., datum="orthometric")` constructs. Run, see it fail.
- [ ] Add `"orthometric"` to `VALID_DATUMS`. Run `pytest tests/test_contract.py`, all pass.

### Task 2: Height metrics (pure numpy)

**Files:** Create `viewer/height_metrics.py`, `tests/test_height_metrics.py`.
**Produces:** `height_scores(pred, ref, classes=None, building_class=3) -> dict` with keys `rmse`,
`mae`, `pearson`, `building_rmse`, `n`. NaN in either input is excluded; degenerate input → `nan`.
- [ ] Tests: perfect prediction → 0 errors, r = 1; constant pred → pearson nan; NaN pixels ignored;
  building_rmse uses only class == 3 pixels and is nan when there are none.
- [ ] Implement; `pytest tests/test_height_metrics.py` passes.

### Task 3: Model wrapper with sliding-window inference

**Files:** Create `viewer/height_model.py`, `tests/test_height_model.py` (pure parts only).
**Produces:** `window_starts(length, patch, stride) -> list[int]`,
`feather(patch) -> np.ndarray (patch, patch) float32`, `load_model(ckpt, synrs3d_dir, device)`,
`predict(model, rgb_uint8, device, patch=518, tta=False) -> (height float32 HxW, logits_argmax uint8 HxW)`.
- [ ] Tests: `window_starts` covers the full length with the last window flush to the end, handles
  length < patch (single start 0 after padding); `feather` is positive everywhere and peaks centrally.
- [ ] Implement; pure tests pass locally. Torch path verified on the VM in Task 5.

### Task 4: Evaluation CLI

**Files:** Create `viewer/height_eval.py`.
**Consumes:** `height_scores`, `load_model`, `predict`, `gamus_dataset.find_tiles/load_tile`.
**Produces:** `python -m viewer.height_eval --gamus DIR --split test --ckpt PATH --out results.json`
prints per-city and overall scores for the model and the `zero` and `train-mean` baselines, and a
segmentation confusion check (share of GAMUS building pixels predicted as OEM building).

### Task 5 (VM): zero-shot baseline — **first real numbers**

- [ ] Upload code to the HF repo from the laptop; download it on the VM.
- [ ] `python -m viewer.height_eval --split test --limit 300` with the stock RS3DAda checkpoint.
- [ ] Sanity: model RMSE < train-mean baseline RMSE; building-as-building share > 0.5 (else the
  class map is wrong: fix the map in `height_train.py` before training).

### Task 6: Training script

**Files:** Create `viewer/height_train.py`.
**Produces:** `python -m viewer.height_train --gamus DIR --ckpt RS3DAda.pth --out runs/ft --hours 8`.
BitFit on the encoder (`pretrained.*` params except biases frozen), decoder + heads trained,
loss `L1 + 0.5·gradL1 + 0.2·CE(ignore=255)`, bf16 autocast, batch 4, random 518 crops +
flips/rot90 + colour jitter, AdamW lr 1e-4, cosine + 500 warmup, val every 1500 steps on 300 val
tiles, keeps `best.pth` by val RMSE, stops at `--hours`.
- [ ] On VM: `--hours 0.05` smoke run finishes, writes `best.pth`, loss decreases.
- [ ] On VM: real run in tmux. Upload `best.pth` to HF after it finishes.

### Task 7: Estimation module (laptop + VM)

**Files:** Create `viewer/estimate.py`; test `tests/test_estimate.py` (pure parts).
**Produces:** `target_shape(shape, src_gsd, dst_gsd=0.33)`,
`estimate(path, model, device, gsd_m=None) -> dict` that writes `ndsm.tif` (always) and `dsm.tif`
(georeferenced input with DEM) via `write_elevation_cog` + `Sidecar`, and returns arrays for the UI.
DEM = Copernicus GLO-30 (`cop-dem-glo-30`) through `viewer.dem.DemSource`, morphological opening
(grey_opening, 150 m footprint) to approximate bare earth, `datum="orthometric"`.
- [ ] Tests: `target_shape` scaling; no-GSD input keeps its shape.

### Task 8: Server endpoint + frontend

**Files:** Modify `viewer/server.py` (new `POST /api/estimate`, `GET /api/estimate/{id}/{file}`),
`frontend/src/main.jsx` (upload tab calls `/api/estimate`, shows max height in metres, links the
DSM/nDSM GeoTIFF download).
- [ ] Response keeps the classify-static shape (`rgb`, `height`, `classes` data URIs) plus `max_m`,
  `georeferenced`, `downloads`.
- [ ] Manual: start server + `npm run dev`, upload a GAMUS test tile PNG, see terrain, max in metres,
  download works.

### Task 9: Final evaluation + README numbers

- [ ] On VM: `height_eval` on all 1000 downloaded test tiles with `best.pth` and with zero-shot.
- [ ] Put both tables (per city, overall, building RMSE, baselines) in `README.md`.
