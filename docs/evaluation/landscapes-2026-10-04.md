# Current landscape evaluation (2026-10-04)

The pipeline at `de3b9e5` uses v1 heights, 25% v1 / 75% v2 on buildings, CHMv2
in extensive forest, and the DEM-consistent absolute DSM. These are single-pass
(Fast quality, no flip averaging), all-valid-pixel scores against USGS 3DEP on
four NAIP scenes. They are external scene checks, not a dataset-wide accuracy guarantee.
The imagery and reference acquisition dates can differ; optical roof displacement
and the 30 m base DEM also contribute to error.

Raw metrics, including bias and the zero-height baseline, are in
[landscapes-2026-10-04.json](landscapes-2026-10-04.json). Undefined correlation is
stored as `null`. No imagery, weights, or raster outputs are bundled in this report.

## Absolute DSM against reference DSM

All errors are metres. The DEM-alone baseline uses the same base and grid.

| Landscape | GSD (m/px) | Base | DSM RMSE | DSM MAE | DSM r | DEM-alone RMSE |
|---|---|---|---|---|---|---|
| dense city | 0.3 | srtm | 38.87 | 30.41 | 0.243 | 40.56 |
| suburb (sparse) | 0.6 | srtm | 4.28 | 3.35 | 0.774 | 4.71 |
| hilly town | 0.6 | srtm | 6.24 | 4.62 | 0.917 | 7.26 |
| forest | 0.6 | srtm | 7.44 | 5.87 | 0.993 | 7.13 |
| dense city | 0.3 | glo30 | 34.36 | 21.59 | 0.391 | 36.26 |
| suburb (sparse) | 0.6 | glo30 | 3.62 | 2.79 | 0.835 | 4.02 |
| hilly town | 0.6 | glo30 | 4.75 | 2.95 | 0.941 | 5.53 |
| forest | 0.6 | glo30 | 8.75 | 6.84 | 0.992 | 8.53 |

## Height above ground against reference HAG

These are the SRTM-route nDSM scores; changing the base does not materially change
nDSM here (hills RMSE differs by less than 0.01 m due to bridge heights).

| Landscape | GSD (m/px) | nDSM RMSE | nDSM MAE | Bias | r | Predict-zero RMSE |
|---|---|---|---|---|---|---|
| dense city | 0.3 | 31.58 | 17.62 | +1.59 | 0.554 | 42.35 |
| suburb (sparse) | 0.6 | 4.47 | 3.21 | +1.42 | 0.612 | 6.48 |
| hilly town | 0.6 | 3.55 | 1.66 | -0.14 | 0.782 | 6.48 |
| forest | 0.6 | 15.67 | 12.94 | -10.75 | 0.362 | 24.66 |

The model beats predict-zero on all four scenes. Absolute DSM RMSE improves over
both base DEMs in the city, suburb, and hills. In forest it is worse by 0.31 m on
SRTM and 0.22 m on GLO-30. High forest DSM correlation mostly reflects terrain
relief: canopy HAG correlation is only 0.362, with a −10.75 m bias. Dense-city
DSM error remains large (38.87 m on SRTM, 34.36 m on GLO-30).

Eight synthetic ground control points drawn from LiDAR reduce the SRTM city
DSM RMSE from 38.87 to 37.18 m on the remaining pixels. The correction is declined
on the suburb/hills and all three GLO-30 sites; forest has no eligible bare-ground
pixels. This checks the calibration rule; it is not a claim about surveyed field GCPs.

References use NAVD88. Exports use EGM96 on SRTM and EGM2008 on GLO-30;
residual datum differences are retained. `dtm_source` in the JSON records where
3DEP seamless terrain replaced unavailable LiDAR DTM/HAG.

## Reproduce

Follow SETUP.md for checkpoints and dependencies. Place the four named NAIP
images in `demo/`; reference rasters are downloaded and cached by the evaluator.
Run each command with `--base-dem srtm`, then repeat with `--base-dem glo30`:

```bash
.venv-da3/bin/python -m viewer.dsm_eval --base-dem srtm --gsd 0.3 \
  --scenes naip_philadelphia_cityhall.tif --out city-srtm.json
.venv-da3/bin/python -m viewer.dsm_eval --base-dem srtm --gsd 0.6 \
  --scenes naip_dc_suburbs_chevychase.tif naip_pittsburgh_hills.tif \
  naip_smoky_mountains_forest.tif --out rest-srtm.json
```
