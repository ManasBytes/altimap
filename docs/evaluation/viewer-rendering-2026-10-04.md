# Viewer rendering checks (2026-10-04)

The Philadelphia City Hall image shows oblique facades as well as roofs. LoD1 blocks follow
predicted image-space building regions, not surveyed ground footprints or detailed roof meshes.
The height model and coarse terrain remain sources of substantial error in this scene.

With SRTM and the existing 300 m bare-earth opening, the estimated ground spans 5.54–32.28 m
across the 307.2 m image. The 99th-percentile gradient magnitude is 0.50 m/m (about 27°).
These are diagnostics of the estimate, not measured street slopes. The coarse DEM retains
urban surface contamination, which becomes unrealistic street ramps in the viewer.

The new optional **Flat ground** view removes terrain relief from the display. It preserves
building nDSM heights and horizontal coordinates, puts trees on the plane and facilities on
the appropriate flattened roofs or ground, and hides absolute flood-defence crest lines.
GeoTIFFs and original elevation readouts are unaffected. This is a visualization option,
not an elevation accuracy improvement.

Roof seeds now require a 1 m interior radius in addition to their existing 4 m² area threshold.
This prevents long, thin ridges from seeding wall-like blocks. Re-scoring the 40 cached GAMUS
validation tiles with `viewer.city_eval` gave:

| Display geometry metric | Before | After |
|---|---:|---:|
| Footprint IoU | 0.84781 | 0.84784 |
| Edge F1 within 1 m | 0.67386 | 0.67325 |
| Building height RMSE (m) | 3.0486 | 3.0700 |
| Building parts | 2036 | 1793 |

The change reduces fragmentation with essentially unchanged footprint scores and a small
increase in display height RMSE. It does not improve the dense-city elevation benchmark.
The [landscape report](landscapes-2026-10-04.md) remains applicable to the raw raster pipeline.

Validation: 138 pytest tests passed, one skipped; two Node display tests passed; frontend build
passed. A real Fast/SRTM upload of `demo/naip_philadelphia_cityhall.tif` was checked in the
browser in estimated, flat, Walk and Exact DSM modes, with no runtime errors. The flat GLB
has a constant-height terrain mesh and `flat visualization` metadata. Its nDSM and DSM rasters
were identical in pixel values, CRS and transform to a pre-change Fast/SRTM upload.
