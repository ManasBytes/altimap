# Spike: Planetary Computer STAC coverage and windowed reads

**Date:** 2026-08-23
**Question:** Can we get co-located NAIP + 3DEP HAG across urban, sparse, hilly, and forested AOIs, reading windows without downloading whole rasters?

## Verdict

**GO.**

12 of 23 probed AOIs have both NAIP and `3dep-lidar-hag` coverage (`NAIP > 0 AND HAG > 0`). All four landscape classes (urban, sparse, hilly, forested) have at least one AOI with real, verified land-cover backing — not just the requester's guessed label. Windowed COG reads work over the network for both collections and pull correctly-shaped, real (non-nodata) pixel data without downloading the full raster. Resolutions match the plan's assumptions almost exactly (NAIP 0.6 m confirmed; HAG 2.0 m confirmed).

One correction to the plan's assumptions: the CRS returned for `3dep-lidar-hag` is a **compound CRS** (horizontal UTM + a NAVD88 vertical datum component), not a plain projected CRS. See "Deviations" below — this affects how Task 9 should read/reproject these rasters.

## Environment setup (Step 2)

```
uv venv --python 3.12 .venv-spike
uv pip install --python .venv-spike/bin/python pystac-client planetary-computer rasterio
```

Result: installed cleanly, no resolution errors. Resolved 51 packages (pystac-client 0.9.0, planetary-computer 1.0.0, pystac 1.15.2, rasterio 1.5.1) in ~3s, installed in <1s. No finding here — dependency resolution was uneventful.

## AOI coverage results

The brief's original 8 candidates were run first. Because 3DEP lidar coverage is workunit-based (not seamless — confirmed: 4 of the original 8 candidates came back with `3dep-lidar-hag = 0`), the candidate list was expanded per the task's ambiguity resolution #1: more candidates were added in regions with known USGS 3DEP lidar programs (Utah AGRC statewide lidar, Colorado Front Range, Vermont statewide lidar, Ohio statewide lidar, NC QL2, WA Cascades) until all four landscape classes had at least one AOI with real HAG coverage. A second round of probing (5 more candidates) was needed specifically to find a genuine "hilly" AOI after the ESA WorldCover check (see below) revealed the brief's `wasatch_hilly` candidate is actually mostly built-up/urban, not hilly terrain. 23 AOIs were probed in total; every one is recorded below, including the 11 failures.

Full output of `spikes/01_stac_coverage.py` (verbatim):

```
AOI                                    naip   3dep-lidar-hag    3dep-seamless   esa-worldcover
----------------------------------------------------------------------------------------------------
salt_lake_city_urban                     30                4                2                2
wasatch_hilly                            45                6                2                2
denver_urban                             28                4                4                4
portland_forested                        56                0                2                2
phoenix_sparse                           63                0                4                2
seattle_urban                            24                0                2                2
asheville_forested                       54                0                2                2
iowa_cropland                            72                0                2                2
provo_sparse                             30               13                2                2
moab_sparse                              30                5                2                2
wasatch_forested                         30                6                2                2
columbus_urban                           40                0                8                2
hocking_hills_forested                   42                0                2                2
asheville_urban                          36                0                2                2
blue_ridge_forested                      36                0                2                2
burlington_vt_urban                      79                2                2                2
green_mountains_forested                 42               15                2                2
king_county_forested                     36                0                2                2
wasatch_bench_hilly                      30                6                2                2
boulder_foothills_hilly                  42                6                4                2
colorado_springs_foothills_hilly               63                0                2                2
golden_co_foothills_hilly                28                6                2                2
provo_bench_hilly                        20               15                2                2

Viable AOIs (NAIP > 0 AND HAG > 0): ['salt_lake_city_urban', 'wasatch_hilly', 'denver_urban', 'provo_sparse', 'moab_sparse', 'wasatch_forested', 'burlington_vt_urban', 'green_mountains_forested', 'wasatch_bench_hilly', 'boulder_foothills_hilly', 'golden_co_foothills_hilly', 'provo_bench_hilly']
Count: 12 / 23
```

**Zero-HAG-coverage AOIs (confirms §7.2's workunit-based-coverage risk is real):** `portland_forested`, `phoenix_sparse`, `seattle_urban`, `asheville_forested`, `iowa_cropland`, `columbus_urban`, `hocking_hills_forested`, `asheville_urban`, `blue_ridge_forested`, `king_county_forested`, `colorado_springs_foothills_hilly` — 11 of 23 candidates, all otherwise-reasonable choices, have zero `3dep-lidar-hag` items despite dense NAIP coverage (20-79 items each). Notably this includes both of the brief's "forested" guesses (`portland_forested`, `asheville_forested`) and both "sparse"/cropland guesses (`phoenix_sparse`, `iowa_cropland`) — none of the brief's original class-specific guesses outside Utah survived. This corroborates the spec's Raleigh-NC-vs-Utah finding: 3DEP HAG coverage clusters heavily around a handful of state lidar programs (Utah AGRC, Colorado Front Range, Vermont) rather than being predictable from "this region probably has lidar."

### Landscape-class relabeling (ESA WorldCover majority land cover, `spikes/03_landcover_check.py`)

Per the task's ambiguity resolution #2, the requester's landscape-class labels were treated as guesses and checked against the actual dominant ESA WorldCover class inside each bbox (10 m pixels, full bbox extent):

```
salt_lake_city_urban       n= 1152000  built-up=62%, tree cover=19%, grassland=13%
wasatch_hilly              n= 1440000  built-up=59%, tree cover=23%, grassland=12%
denver_urban               n=  576000  built-up=75%, tree cover=16%, grassland=5%
provo_sparse               n= 1152000  built-up=36%, tree cover=31%, grassland=20%
moab_sparse                n= 1152000  bare/sparse vegetation=47%, shrubland=30%, grassland=10%
wasatch_forested           n= 1152000  tree cover=78%, grassland=20%, bare/sparse vegetation=1%
burlington_vt_urban        n= 1152000  tree cover=45%, built-up=19%, water=17%
green_mountains_forested   n= 1152000  tree cover=85%, grassland=9%, cropland=5%
wasatch_bench_hilly        n= 1008000  tree cover=63%, built-up=17%, grassland=13%
boulder_foothills_hilly    n= 1152000  tree cover=67%, grassland=20%, built-up=12%
golden_co_foothills_hilly  n= 1152000  grassland=50%, tree cover=38%, built-up=11%
provo_bench_hilly          n= 1152000  tree cover=49%, grassland=34%, built-up=9%
```

**Relabeling decisions:**

- **`wasatch_hilly` is mislabeled.** 59% built-up — it is a suburban/urban AOI (Draper/Sandy, UT valley floor), not a hilly-terrain AOI. It is redundant with `salt_lake_city_urban` (62% built-up) and should **not** be used to represent the "hilly" class.
- **`provo_sparse` is mislabeled.** 36% built-up / 31% tree cover / 20% grassland is a mixed suburban-foothill signature, not sparse/desert. It should **not** be used to represent the "sparse" class.
- **`burlington_vt_urban` is a mixed/marginal urban AOI.** 45% tree cover / 19% built-up / 17% water — it's forest+lake+small-city, not a clean urban core. Kept out of the final selection since `salt_lake_city_urban` and `denver_urban` are both clean, strongly built-up urban AOIs.
- **A genuine "hilly" AOI was found by round-2 probing:** `golden_co_foothills_hilly` (grassland 50%, tree cover 38%, built-up 11%) is a real mixed grass/shrub/light-tree rolling-foothills signature, distinct from both the urban AOIs (dominated by built-up) and the forested AOIs (78-85% tree cover). This replaces `wasatch_hilly` as the "hilly" representative.
- `wasatch_forested` (78% tree cover) and `green_mountains_forested` (85% tree cover) both genuinely earn the "forested" label.
- `moab_sparse` (47% bare/sparse vegetation + 30% shrubland = 77% sparse-vegetation cover) genuinely earns the "sparse" label.

**Selected AOIs for training:**

| AOI | bbox (west, south, east, north — EPSG:4326) | Landscape class | NAIP items | HAG items | WorldCover majority |
|---|---|---|---|---|---|
| `salt_lake_city_urban` | (-111.95, 40.72, -111.85, 40.80) | urban | 30 | 4 | built-up 62% |
| `denver_urban` | (-105.05, 39.70, -104.95, 39.78) | urban | 28 | 4 | built-up 75% |
| `golden_co_foothills_hilly` | (-105.30, 39.70, -105.20, 39.78) | hilly | 28 | 6 | grassland 50%, tree 38%, built-up 11% |
| `moab_sparse` | (-109.60, 38.53, -109.50, 38.61) | sparse | 30 | 5 | bare/sparse veg 47%, shrubland 30% |
| `wasatch_forested` | (-111.65, 40.60, -111.55, 40.68) | forested | 30 | 6 | tree cover 78% |
| `green_mountains_forested` | (-72.90, 44.10, -72.80, 44.18) | forested | 42 | 15 | tree cover 85% |

All four landscape classes are covered by at least one AOI with genuine land-cover backing, and urban + forested each have two independent candidates for redundancy. Backup/rejected candidates (`wasatch_hilly`, `provo_sparse`, `burlington_vt_urban`, `wasatch_bench_hilly`, `boulder_foothills_hilly`, `provo_bench_hilly`) are kept in `spikes/01_stac_coverage.py` and `spikes/03_landcover_check.py` for reference — several are usable as additional urban/hilly training data even though they were not chosen as the primary per-class exemplar.

## Asset keys and raster properties

Full output of `spikes/02_windowed_read.py` (verbatim):

```
naip: ut_m_4011126_sw_12_060_20211113
  shape=(12320, 9600) crs=EPSG:26912 res=(0.6, 0.6) dtype=uint8
  nodata=None blocksize=(512, 512)
  windowed read OK: (512, 512) min=18.00 max=231.00

3dep-lidar-hag: USGS_LPC_UT_Wasatch_L4_2013_LAS_2016-hag-2m-2-5
  shape=(4097, 4097) crs=COMPD_CS["NAD83 / UTM zone 12N + NAVD88 height",PROJCS["NAD83 / UTM zone 12N",GEOGCS["NAD83",DATUM["North_American_Datum_1983",SPHEROID["GRS 1980",6378137,298.257222101,AUTHORITY["EPSG","7019"]],AUTHORITY["EPSG","6269"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4269"]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",-111],PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH],AUTHORITY["EPSG","26912"]],VERT_CS["NAVD88 height",VERT_DATUM["North American Vertical Datum 1988",2005,AUTHORITY["EPSG","5103"]],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Gravity-related height",UP],AUTHORITY["EPSG","5703"]]] res=(2.0, 2.0) dtype=float32
  nodata=-9999.0 blocksize=(512, 512)
  windowed read OK: (512, 512) min=-9999.00 max=-9999.00
```

The HAG windowed read above landed on an all-nodata corner of that particular tile (top-left 512x512 window happened to sit outside the tile's valid-data footprint). To confirm windowed reads genuinely pull real elevation data (not just prove the API call succeeds), a follow-up center-window read was run:

```
center window shape=(512, 512) valid_px=262144/262144
valid min=-0.22 max=33.00 mean=4.73
```

512x512 pixels, all valid (non-nodata), height-above-ground values ranging -0.22 m to 33.0 m with mean 4.73 m — physically reasonable for a mixed terrain/vegetation/building HAG surface (small negative values are lidar/DTM noise, not a bug). This confirms real windowed COG reads over the network work correctly for both collections; only the requested window is read, not the full raster (both rasters are far larger than 512x512: NAIP is 12320x9600, HAG is 4097x4097).

**Full asset-key inventory** (`list(item.assets.keys())`) for both collections:

- `naip` -> `['image', 'thumbnail', 'tilejson', 'rendered_preview']`
- `3dep-lidar-hag` -> `['data', 'thumbnail', 'tilejson', 'rendered_preview']`

- **NAIP asset key: `image`**, resolution **0.6 m** (confirmed, matches plan assumption), dtype `uint8`, nodata `None`, CRS `EPSG:26912` (plain projected CRS), blocksize 512x512.
- **HAG asset key: `data`**, resolution **2.0 m** (confirmed, matches plan assumption), dtype `float32`, nodata `-9999.0`, CRS is a **compound CRS** (`EPSG:26912` horizontal + `EPSG:5703`/NAVD88 vertical component, WKT `COMPD_CS[...]`) — not a plain single EPSG code, blocksize 512x512.

Task 9 should hard-code asset keys `image` (NAIP) and `data` (3dep-lidar-hag).

> **Caveat — these properties are a single-tile spot check.** dtype, nodata,
> CRS form, and blocksize above were read from **one** tile pair, in the
> Wasatch AOI only. They were *not* cross-checked against the other selected
> AOIs. 3DEP is delivered per workunit and different acquisitions can come
> from different vendors, so a different nodata convention or CRS form in
> another AOI is plausible. **Tasks 9 and 10 must not treat these as blanket
> facts about the collections** — read `src.nodata` and `src.crs` from each
> dataset at runtime rather than hard-coding these values, and spot-check at
> least one tile per selected AOI.

## Deviations from spec assumptions

1. **HAG CRS is a compound CRS, not a plain projected CRS.** `src.crs` for `3dep-lidar-hag` returns `COMPD_CS[...]` combining the horizontal UTM zone 12N (EPSG:26912) with a NAVD88 vertical datum component (EPSG:5703), rather than a single flat EPSG code like NAIP's `EPSG:26912`. Code that does `src.crs.to_epsg()` or similar simple EPSG extraction on the HAG source will likely get `None` or fail, and needs to either use `src.crs.to_wkt()` / handle the compound case, or extract just the horizontal sub-CRS before reprojecting/comparing against NAIP's CRS. This should be called out explicitly wherever Task 9 aligns NAIP and HAG rasters to a common grid.
2. **NAIP nodata is `None`**, not a sentinel value — code should not assume a nodata mask exists for NAIP tiles; edge/no-data areas (if any) are not flagged via the nodata mechanism.
3. Resolutions themselves match the plan exactly: NAIP 0.6 m, HAG 2.0 m — **no deviation** on the core numeric assumption in Global Constraints.
4. **3DEP HAG coverage is much sparser and more geographically clustered than assumed.** 11 of 23 probed AOIs (48%) had zero HAG coverage despite having 20-79 NAIP items each, including every "forested" and "sparse"/cropland guess outside Utah. Coverage clustered around Utah (AGRC statewide lidar), Colorado Front Range, and Vermont. Any production AOI registry (populated from this findings doc) must probe HAG coverage per-AOI before committing to a region — it is not safe to assume a region has HAG coverage just because it has NAIP coverage or a plausible-sounding state lidar program.
5. **The requester's landscape-class labels for 2 of 8 original candidates were wrong** (`wasatch_hilly` is actually urban/suburban; `provo_sparse` is actually mixed suburban-foothill) — see relabeling section above. Class labels must be verified against actual land cover, not assigned by AOI name/region alone.

## If NO-GO

Not applicable — verdict is GO. Fallback list retained from the brief for reference in case future AOI expansion again fails to find HAG coverage:

1. Widen the AOI candidate list — coverage is workunit-based, so more probes may find viable regions
2. Substitute `3dep-lidar-dsm` minus `3dep-lidar-dtm` for HAG where HAG is absent
3. Fall back to GeoNRW (spec §7.3 rejected it on convenience, not availability)
