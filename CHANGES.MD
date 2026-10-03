# 48-hour hardening sprint changes

Branch: `codex/48h-final-hardening` (local only; no GitHub changes made).

## Final building-fusion decision

Production building routing now uses **25% v1 + 75% v2**. The change passed three predeclared
gates: corrected full validation, corrected untouched test, and a 24-tile TTA safety check. The
focused TTA run improved pooled overall RMSE from **3.1690 m to 3.0916 m** and building RMSE from
**5.1284 m to 4.9140 m**; the ordinary-scene overall regression was only **0.04%**.

The fusion evaluator was corrected before this decision: GAMUS `-5 m` no-data sentinels are now
excluded with `clean_height()`, and caches record their TTA mode so incompatible predictions cannot
be mixed. See `BUILDING_FUSION_075_REPORT.md` for the complete evidence and impact analysis.

## Implemented

- Added pooled bias and building MAE to height metrics, plus reusable reference-height bins:
  `0–2`, `2–5`, `5–10`, `10–20`, `20–50`, and `>50 m`.
- Added the same height-bin reports to `viewer.height_eval`.
- Added `viewer.fusion_eval`, which caches v1/v2 predictions and compares fixed building weights
  `0.00/0.25/0.50/0.75/1.00` and five probability-based soft-routing threshold pairs. It does not
  alter routing while experiments run.
- Added an optional building probability return from the existing height model while preserving
  the old two-value inference contract.
- Changed absolute elevation outputs to GDAL's COG driver and added a real COG test.
- Added complete-patch DEM caching and preserved unresolved DEM pixels as nodata instead of
  converting failed inland coverage to 0 m.
- Added GeoTIFF warnings for rotated/sheared and materially non-square grids; absolute DSM export
  is skipped for those unsafe transforms instead of silently producing incorrect coordinates.
- Made PNG/JPG unknown-GSD behavior explicit in API metadata and the viewer: `0.33 m/pixel` is
  an experimental assumption, not silently known metric scale.
- Restored the conditional `Built with DINOv3` attribution when CHMv2 is active.
- Pinned frontend dependency versions and verified the existing lockfile with `npm ci --dry-run`.
- Added `THIRD_PARTY_NOTICES.md` with the model/data attribution checklist.

## Verification

- `.venv\Scripts\python.exe -m pytest -q` → **122 passed, 1 skipped**.
- `.venv\Scripts\python.exe -m compileall -q viewer src` → passed.
- `frontend\npm ci --dry-run` → passed.
- `frontend\npm run build` → passed.

The repository does not track model/data artifacts. For local validation, the v2 checkpoint is
in `viewer/cache/best_v2.pth` (ignored) and the six-scene GAMUS slice is under `E:\TEST\GAMUS`.

## Local validation evidence

For a six-scene GAMUS validation slice (`DC_02_26`, `DC_04_23`, `DC_04_27`, `PHL_6150`–`PHL_6152`),
the cached comparison produced:

| Routing | RMSE (m) | MAE (m) | Building RMSE (m) | Bias (m) |
|---|---:|---:|---:|---:|
| v1 only / hard weight 0.00 | 3.606 | 1.774 | 2.050 | -0.236 |
| then-current hard weight 0.50 | 3.597 | 1.764 | 1.989 | -0.230 |
| hard weight 0.75 | 3.596 | 1.764 | 1.987 | -0.227 |
| best tested soft 0.1–0.5 | 3.595 | 1.763 | 1.987 | -0.227 |

This historical six-tile slice was not enough evidence to replace the then-current 0.50 route. The
later full-split experiment below supplied the required evidence. Raw slice results remain in
`E:\TEST\fusion-six.json` and `E:\TEST\height-six.json`.

## Full GAMUS routing experiment

The larger experiment used every downloaded validation and test tile. The full GAMUS train split is
also now present; the verifier found no incomplete train, validation, or test triplets.

- Training: **5,004** tiles, DC, NYC, and PHL.
- Validation: **859** tiles, DC and PHL, all cached v1/v2 predictions.
- Final test: **2,861** tiles, DC, NYC, and PHL, all cached v1/v2 predictions.
- Inference: no TTA, BF16 autocast, RTX 3050, batch size 8 for the test pass.
- Routing candidates: fixed weights `0.00/0.25/0.50/0.75/1.00` and soft probability pairs
  `0.1-0.5/0.2-0.6/0.3-0.7/0.4-0.8/0.5-0.9`.

After excluding invalid reference sentinels, validation selected `hard_w0.75`: overall RMSE
**2.740 m**, MAE **1.287 m**, and building RMSE **3.276 m**. The former `hard_w0.50` route measured
**2.751 m** overall and **3.335 m** on buildings.

On the untouched test split, `hard_w0.50` measured **3.757 m** overall / **5.231 m** building RMSE;
`hard_w0.75` measured **3.697 m** overall / **4.993 m** building RMSE. The later focused TTA check
also passed, so production routing was changed to `hard_w0.75`.

Full artifacts:

- `E:\TEST\fusion-validation-corrected.json` and `.csv`
- `E:\TEST\fusion-test-corrected.json` and `.csv`
- `E:\TEST\fusion-tta24-{all,tall,ordinary}.json` and `.csv`
- `E:\TEST\height-validation.json`
- `E:\TEST\height-test.json`
- `E:\TEST\fusion-evaluation-summary.md`

The refreshed height reports use the real training mean (**8.5037 m**) for their train-mean
baseline; it is no longer a zero placeholder.
