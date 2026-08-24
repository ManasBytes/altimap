"""Spike: sanity-check the landscape-class labels against ESA WorldCover.

Not in the brief verbatim -- added because the brief's landscape-class labels
(urban/sparse/hilly/forested) are the requester's guesses, not ground truth.
This reads the ESA WorldCover 10m raster windowed to each viable AOI's bbox
and reports the majority land-cover class, so the findings doc can correct
mislabeled AOIs instead of preserving wrong guesses.

Throwaway.
"""

import numpy as np
import planetary_computer
import pystac_client
import rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

WORLDCOVER_CLASSES = {
    10: "tree cover",
    20: "shrubland",
    30: "grassland",
    40: "cropland",
    50: "built-up",
    60: "bare/sparse vegetation",
    70: "snow/ice",
    80: "water",
    90: "herbaceous wetland",
    95: "mangroves",
    100: "moss/lichen",
}

# All 12 AOIs with NAIP > 0 AND HAG > 0 from spikes/01_stac_coverage.py.
# bbox = (west, south, east, north) EPSG:4326
VIABLE = {
    "salt_lake_city_urban": (-111.95, 40.72, -111.85, 40.80),
    "wasatch_hilly": (-111.95, 40.55, -111.85, 40.65),
    "denver_urban": (-105.05, 39.70, -104.95, 39.78),
    "provo_sparse": (-111.70, 40.20, -111.60, 40.28),
    "moab_sparse": (-109.60, 38.53, -109.50, 38.61),
    "wasatch_forested": (-111.65, 40.60, -111.55, 40.68),
    "burlington_vt_urban": (-73.25, 44.44, -73.15, 44.52),
    "green_mountains_forested": (-72.90, 44.10, -72.80, 44.18),
    "wasatch_bench_hilly": (-111.85, 40.53, -111.75, 40.60),
    "boulder_foothills_hilly": (-105.35, 39.95, -105.25, 40.03),
    "golden_co_foothills_hilly": (-105.30, 39.70, -105.20, 39.78),
    "provo_bench_hilly": (-111.68, 40.28, -111.58, 40.36),
}


def main() -> None:
    client = pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)

    for name, bbox in VIABLE.items():
        search = client.search(collections=["esa-worldcover"], bbox=bbox, limit=1)
        item = next(search.items())
        href = item.assets["map"].href
        with rasterio.open(href) as src:
            win_bounds = transform_bounds("EPSG:4326", src.crs, *bbox)
            win = from_bounds(*win_bounds, transform=src.transform)
            arr = src.read(1, window=win)
        vals, counts = np.unique(arr, return_counts=True)
        order = np.argsort(-counts)
        total = counts.sum()
        top = [(WORLDCOVER_CLASSES.get(int(vals[i]), f"class {vals[i]}"), counts[i] / total) for i in order[:3]]
        top_str = ", ".join(f"{cls}={frac:.0%}" for cls, frac in top)
        print(f"{name:<26} n={total:>8}  {top_str}")


if __name__ == "__main__":
    main()
