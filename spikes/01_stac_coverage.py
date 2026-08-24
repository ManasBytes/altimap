"""Spike: does Planetary Computer serve co-located NAIP + 3DEP HAG for candidate AOIs?

Throwaway. Findings go to docs/superpowers/spikes/.
"""

import planetary_computer
import pystac_client

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

# Candidate AOIs spanning the four landscape classes the rubric grades.
# bbox = (west, south, east, north) in EPSG:4326
#
# NOTE: this list was expanded beyond the brief's original 8 candidates.
# 3DEP lidar coverage is workunit-based (not seamless), so the original
# guesses (e.g. portland_forested, asheville_forested) were probed and,
# where they came back with zero HAG hits, additional candidates in
# regions with well-known USGS 3DEP lidar programs (Utah AGRC statewide
# lidar, Ohio statewide lidar, North Carolina QL2 program, Vermont lidar,
# Puget Sound lidar consortium) were added until all four landscape
# classes had at least one viable AOI.
CANDIDATES = {
    "salt_lake_city_urban": (-111.95, 40.72, -111.85, 40.80),
    "wasatch_hilly": (-111.95, 40.55, -111.85, 40.65),
    "denver_urban": (-105.05, 39.70, -104.95, 39.78),
    "portland_forested": (-122.75, 45.48, -122.65, 45.56),
    "phoenix_sparse": (-112.10, 33.42, -112.00, 33.50),
    "seattle_urban": (-122.36, 47.58, -122.26, 47.66),
    "asheville_forested": (-82.60, 35.55, -82.50, 35.63),
    "iowa_cropland": (-93.70, 41.55, -93.60, 41.63),
    # --- expanded candidates (added mid-spike to fill gaps / find forest+sparse) ---
    "provo_sparse": (-111.70, 40.20, -111.60, 40.28),  # Utah AGRC statewide lidar
    "moab_sparse": (-109.60, 38.53, -109.50, 38.61),  # UT desert, sparse veg
    "wasatch_forested": (-111.65, 40.60, -111.55, 40.68),  # UT mountain forest, Uinta-Wasatch-Cache NF
    "columbus_urban": (-83.05, 39.94, -82.95, 40.02),  # OH statewide lidar
    "hocking_hills_forested": (-82.55, 39.42, -82.45, 39.50),  # OH forested hill country
    "asheville_urban": (-82.58, 35.57, -82.50, 35.63),  # NC QL2 lidar, urban core
    "blue_ridge_forested": (-82.85, 35.60, -82.75, 35.68),  # NC QL2, Pisgah NF
    "burlington_vt_urban": (-73.25, 44.44, -73.15, 44.52),  # VT statewide lidar
    "green_mountains_forested": (-72.90, 44.10, -72.80, 44.18),  # VT forested mountains
    "king_county_forested": (-121.90, 47.40, -121.80, 47.48),  # WA Cascade foothills forest
    # --- round 2: the ESA WorldCover check (spikes/03_landcover_check.py) showed
    # wasatch_hilly is 59% built-up (mislabeled -- it's urban/suburban, not
    # "hilly") and provo_sparse is a mixed suburban/foothill area, not sparse.
    # These candidates were added to find a genuine rolling-hills / mixed
    # grass-shrub-tree AOI to replace wasatch_hilly.
    "wasatch_bench_hilly": (-111.85, 40.53, -111.75, 40.60),  # UT east-bench foothills
    "boulder_foothills_hilly": (-105.35, 39.95, -105.25, 40.03),  # CO Boulder foothills
    "colorado_springs_foothills_hilly": (-104.90, 38.80, -104.80, 38.88),  # CO Front Range
    "golden_co_foothills_hilly": (-105.30, 39.70, -105.20, 39.78),  # CO Golden foothills
    "provo_bench_hilly": (-111.68, 40.28, -111.58, 40.36),  # UT Provo bench foothills
}

COLLECTIONS = ["naip", "3dep-lidar-hag", "3dep-seamless", "esa-worldcover"]


def main() -> None:
    client = pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)

    print(f"{'AOI':<26} " + " ".join(f"{c:>16}" for c in COLLECTIONS))
    print("-" * 100)

    viable = []
    for name, bbox in CANDIDATES.items():
        counts = []
        for coll in COLLECTIONS:
            try:
                search = client.search(collections=[coll], bbox=bbox, limit=50)
                n = len(list(search.items()))
            except Exception as exc:  # spike: surface the error, do not swallow
                n = f"ERR:{type(exc).__name__}"
            counts.append(n)
        print(f"{name:<26} " + " ".join(f"{str(c):>16}" for c in counts))

        naip_n, hag_n = counts[0], counts[1]
        if isinstance(naip_n, int) and isinstance(hag_n, int) and naip_n > 0 and hag_n > 0:
            viable.append(name)

    print(f"\nViable AOIs (NAIP > 0 AND HAG > 0): {viable}")
    print(f"Count: {len(viable)} / {len(CANDIDATES)}")


if __name__ == "__main__":
    main()
