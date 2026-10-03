# AltiMap GAMUS building-routing evaluation

Date: 2026-10-03<br>
Decision: `hard_w0.75` adopted locally after corrected validation, untouched-test confirmation, and
a focused TTA safety check.

## Dataset

| Split | Tiles | Cities | Triplets |
|---|---:|---|---|
| train | 5,004 | DC, NYC, PHL | complete |
| validation | 859 | DC, PHL | complete |
| test | 2,861 | DC, NYC, PHL | complete |

Each tile has matching RGB, reference height, and class data. v1/v2 predictions were cached once;
routing variants were then evaluated without rerunning the models.

## Metric correction

The first reports included GAMUS `-5 m` no-data sentinels. The evaluator now applies
`clean_height()` before every metric. This removed 9,272,575 invalid validation pixels and
9,881,231 invalid test pixels. The routing conclusion did not change. Corrected artifacts have
`-corrected` in their filenames; earlier reports are audit history only.

## Corrected validation results

| Routing | Overall RMSE | Overall MAE | Building RMSE | Building MAE | Bias |
|---|---:|---:|---:|---:|---:|
| hard 0.00 | 2.8160 | 1.3241 | 3.6272 | 2.1081 | -0.1099 |
| hard 0.25 | 2.7765 | 1.3072 | 3.4541 | 2.0123 | -0.1077 |
| hard 0.50 (old production) | 2.7511 | 1.2948 | 3.3352 | 1.9405 | -0.1054 |
| hard 0.75 (selected) | **2.7402** | 1.2873 | **3.2763** | 1.8958 | -0.1032 |
| hard 1.00 | 2.7440 | **1.2852** | 3.2807 | **1.8813** | -0.1010 |
| soft 0.1-0.5 | 2.7494 | 1.2940 | 3.3318 | 1.9387 | -0.1040 |
| soft 0.2-0.6 | 2.7507 | 1.2944 | 3.3352 | 1.9401 | -0.1058 |
| soft 0.3-0.7 | 2.7520 | 1.2948 | 3.3394 | 1.9418 | -0.1076 |
| soft 0.4-0.8 | 2.7535 | 1.2952 | 3.3449 | 1.9440 | -0.1095 |
| soft 0.5-0.9 | 2.7553 | 1.2958 | 3.3521 | 1.9470 | -0.1116 |

## Corrected untouched-test confirmation

| Routing | Overall RMSE | Overall MAE | Building RMSE | Building MAE | Bias |
|---|---:|---:|---:|---:|---:|
| hard 0.00 / v1 only | 3.9623 | 1.5963 | 5.9289 | 2.3608 | -0.1514 |
| hard 0.50 (old production) | 3.7573 | 1.5513 | 5.2308 | 2.1389 | -0.1159 |
| hard 0.75 (selected) | **3.6972** | **1.5402** | **4.9934** | **2.0792** | -0.0981 |

The test split was read only after validation selected 0.75. Relative to 0.50, the selected route
improves overall RMSE by 1.60% and building RMSE by 4.54%.

## Focused 24-tile TTA check

| Subset | 0.50 overall | 0.75 overall | 0.50 building | 0.75 building |
|---|---:|---:|---:|---:|
| all 24 | 3.1690 | **3.0916** | 5.1284 | **4.9140** |
| tall 8 | 4.4430 | **4.2755** | 7.4559 | **7.1108** |
| ordinary 16 | **2.2774** | 2.2783 | **2.0104** | 2.0181 |

The pooled TTA result favors 0.75. The ordinary-subset overall regression is only 0.04%, well below
the predeclared 2% stop threshold. Production therefore changed to `25% v1 + 75% v2` on building
pixels. Non-building routing is unchanged.

## Final artifacts

- `E:\TEST\fusion-validation-corrected.json` and `.csv`
- `E:\TEST\fusion-test-corrected.json` and `.csv`
- `E:\TEST\fusion-notta24-{all,tall,ordinary}.json` and `.csv`
- `E:\TEST\fusion-tta24-{all,tall,ordinary}.json` and `.csv`
- `E:\TEST\height-validation.json`
- `E:\TEST\height-test.json`
- `E:\TEST\gamus-verification.json`

See `BUILDING_FUSION_075_REPORT.md` for causes, production impact, limitations, and code touchpoints.
No GitHub or GitLab operation was performed.
