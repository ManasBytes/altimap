# Spike: DA3-SMALL on nadir remote-sensing imagery

**Date:** 2026-08-24
**Question:** The design doc (§1) asserts that monocular depth foundation models suffer a domain gap on nadir imagery. Is that true of DA3-SMALL, and what does the failure actually look like?

## Verdict

**The domain gap is real, it is severe, and it has a specific shape: DA3-SMALL fits a tilted plane to nadir imagery instead of reading relief.**

Across a 28-image sample stratified over all 14 landscape classes, depth correlates with the **row index** at mean |r| = **0.770** (median 0.831), against mean |r| = **0.234** for the column index. 26 of 28 images have a *negative* row correlation — depth decreasing from top of frame to bottom — which is exactly the egocentric ground-plane prior: the model places the top of the image far away and the bottom near, as it would for a forward-facing photo of a road. 25 of 28 images have a dominant-axis correlation above 0.5; 13 are above 0.9.

This does not kill the DA3-encoder approach, but it does confirm that **a frozen DA3 encoder plus a trained head is doing real work, not cosmetic work.** The raw model output is not a weak elevation signal that needs rescaling — on these images it is largely not an elevation signal at all.

## What was run

```
uv venv --python 3.12 .venv-da3
uv pip install --torch-backend=auto torch torchvision
uv pip install einops addict omegaconf imageio imageio-ffmpeg tqdm \
    opencv-python-headless huggingface_hub safetensors matplotlib trimesh \
    evo "moviepy==1.0.3" pycolmap plyfile
uv pip install --no-deps "git+https://github.com/ByteDance-Seed/depth-anything-3"
```

`spikes/04_da3_nadir_check.py`, model `depth-anything/DA3-SMALL` (Apache-2.0), `process_res=504`, on an RTX 3050 6 GB (torch 2.13.0+cu132, bf16 autocast).

Data: `~/Downloads/Remote Sensing Data.v2i.yolov8`, train split, 700 images, all 700 labelled, 2 images per class taken in filename order (deterministic).

**Performance: ~0.11–0.12 s/image mean on the 3050** across two runs, including preprocessing and the forward pass. A full 1000-image pass is ~2 minutes. Inference cost is not a constraint on this hardware.

## Results (verbatim)

```
class         row_corr  col_corr   relief   n
---------------------------------------------
Agriculture     -0.964    -0.101    0.103   2
Airport         -0.495    -0.201    0.051   2
Beach           -0.787    +0.179    0.198   2
City            -0.671    +0.133    0.065   2
Desert          -0.607    -0.290    0.069   2
Forest          -0.930    +0.115    0.062   2
Grassland       -0.946    -0.114    0.086   2
Highway         -0.962    -0.167    0.320   2
Lake            -0.638    -0.115    0.080   2
Mountain        -0.299    -0.629    0.053   2
Parking         -0.762    -0.023    0.064   2
Port            -0.058    -0.467    0.048   2
Railway         -0.817    +0.091    0.089   2
River           -0.976    -0.034    0.104   2
---------------------------------------------
mean |row_corr| 0.770   mean |col_corr| 0.234
images with |row_corr| > 0.5: 24/28
images with dominant-axis |corr| > 0.5: 25/28  (> 0.9: 13)
row_corr negative (top of frame read as far): 26/28
is_metric: {False}  (False = relative depth, as expected)
mean 0.12s/image on cuda
```

`relief` is `std(depth) / median(depth)` — depth is affine-invariant, so an absolute standard deviation carries no meaning and this ratio is used instead.

## Findings

### 1. The prior fires on the row axis, but not only the row axis

`Mountain` (row −0.299, col −0.629) and `Port` (row −0.058, col −0.467) invert the pattern: the tilted plane runs left-to-right instead of top-to-bottom. So the artefact is "an arbitrarily-oriented depth ramp", not specifically "a vertical depth ramp". Any future check should use `max(|row_corr|, |col_corr|)` rather than `|row_corr|` alone — on that measure **25 of 28** images exceed 0.5 and **13** exceed 0.9, which is worse than the row-only headline.

The `Mountain__002` preview is the clearest single example: the RGB shows a dense dendritic drainage network with obvious ridges and valleys, and the predicted depth is a smooth left-to-right ramp that ignores every one of them.

### 2. High "relief" is a symptom of the prior, not evidence against it

The five highest-relief images all have a dominant-axis |correlation| ≥ 0.94. Correlation between |row_corr| and relief across the sample is **+0.375** — positive, i.e. the harder the model tilts the plane, the more depth variation it reports. The variation is the ramp's own range.

`Highway__069` is the extreme case: relief 0.420, the highest in the sample, row_corr −0.99. Its depth map is a horizon band across the top with a smooth ground plane below — a textbook driving-scene reconstruction imposed on an aerial photograph.

**Consequence: relief magnitude must not be used as a "does this model see structure" proxy anywhere in the eval harness.** It is confounded with the exact failure it would be trying to detect.

### 3. The confidence maps track image texture, not depth quality

In every preview inspected, the confidence map lights up on high-frequency texture — building edges, tree canopy, road markings — while the depth over those same regions is a featureless ramp. Confidence is high precisely where the depth is most wrong. It is a texture detector here, not a reliability estimate, and should not be used to weight or mask predictions on this domain without being re-validated against ground truth.

### 4. `Prediction.is_metric` is not an int

`Prediction.is_metric` is annotated `int` but arrives as an **empty addict `Dict`** for the non-metric models. `output_processor.py:71` calls `getattr(model_output, "is_metric", 0)` on an addict `Dict`, and addict's `__getattr__` manufactures an empty `Dict` for any missing key rather than raising — so the `0` default is unreachable. `int(prediction.is_metric)` raises `TypeError`. Truthiness works (empty `Dict` is falsy).

**Anything in AltiMap that branches on this flag must test truthiness, not assume an int.**

### 5. Installing the package needs `--no-deps`

`depth-anything-3` declares a kitchen-sink dependency list including `open3d`, `gsplat`, `gradio`, `fastapi`, `pre-commit` and `e3nn`, none of which the depth inference path touches. But `api.py` transitively imports the whole `utils/export` package at module load, which does drag in `moviepy` (1.x only — it imports `moviepy.editor`), `pycolmap`, `trimesh`, `plyfile` and `evo` even when you only want a depth array. The `--no-deps` list above is the working minimum. `gsplat` stays absent and only produces a startup warning.

Their `numpy<2` pin was **not** respected — numpy 2.5.2 is installed and nothing in the depth path complained.

## Limits of this spike — do not over-read it

- **n = 28.** Two images per class. The per-class rows are means of two numbers and should not be treated as class rankings; only the aggregate is meaningful.
- **No ground truth.** This dataset has no DSM, so nothing here is an accuracy measurement. The spike establishes that the output does not *look like* elevation; it cannot say how far off it is. The RMSE numbers in design doc §9 remain the only quantitative reference points and they come from the literature, not from here.
- **Not all of this imagery is nadir.** `Highway__069` is visibly oblique, and several others are aerial photographs rather than orthorectified imagery. The dataset is a captioning/classification set, not a photogrammetry set. A genuinely oblique image *should* produce a depth ramp — that is correct behaviour, not a failure — so some fraction of the 25/28 is legitimate. This weakens the headline number by an unmeasured amount. Re-running on NAIP orthophotos from Task 1's AOIs would remove the confound and is the obvious follow-up.
- **Class labels are approximate.** `City__006` is plainly an airport, not a city. Roboflow labels here are single whole-frame boxes and are not reliable enough to build stratification on for anything load-bearing.
- **DA3-SMALL only.** DA3-BASE (also Apache-2.0) was not tested. It is plausible the larger backbone is less prone to the prior; unverified.

## Reproducing

```
.venv-da3/bin/python spikes/04_da3_nadir_check.py --per-class 2 --save-npy
.venv-da3/bin/python spikes/04_da3_nadir_check.py --split test --per-class 8   # wider sweep
```

Outputs land in `spikes/out/da3_nadir/`: one RGB | depth | confidence PNG per image, `records.json` with per-image metrics, and `.npy` float32 depth arrays under `--save-npy`.

`.venv-da3` is ~4.5 GB and gitignored. Delete it when this line of work is done — `/home` was at 91% during this spike.
