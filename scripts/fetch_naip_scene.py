"""Cut a NAIP test scene (USDA aerial RGB, public) around a point, like the other demo/ scenes.

    .venv-da3/bin/python scripts/fetch_naip_scene.py --lat 40.4400 --lon -80.0060 \
        --name naip_pittsburgh_bridges          # downtown Pittsburgh, three rivers and bridges

Writes demo/<name>.tif: RGB uint8 GeoTIFF at NAIP's native resolution in its UTM CRS, from the
newest NAIP image covering the point (Microsoft Planetary Computer). viewer/dsm_eval.lidar()
then fetches the matching USGS 3DEP LiDAR for it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import viewer.dem  # noqa: F401  (GDAL / requests timeouts before any remote read)


def main() -> None:
    import planetary_computer
    import pystac_client
    import rasterio
    from rasterio.warp import transform
    from rasterio.windows import from_bounds

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--size-m", type=float, default=1200.0, help="side of the square scene")
    ap.add_argument("--name", required=True)
    args = ap.parse_args()

    cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                                    modifier=planetary_computer.sign_inplace, timeout=60)
    items = sorted(cat.search(collections=["naip"], intersects={"type": "Point", "coordinates": [args.lon, args.lat]},
                              max_items=20).items(), key=lambda i: i.datetime, reverse=True)
    if not items:
        raise SystemExit("no NAIP image covers this point (NAIP is US-only)")
    item = items[0]
    with rasterio.open(item.assets["image"].href) as src:
        (x,), (y,) = transform("EPSG:4326", src.crs, [args.lon], [args.lat])
        half = args.size_m / 2
        window = from_bounds(x - half, y - half, x + half, y + half, src.transform).round_offsets().round_lengths()
        rgb = src.read([1, 2, 3], window=window)
        profile = {"driver": "GTiff", "height": rgb.shape[1], "width": rgb.shape[2], "count": 3,
                   "dtype": "uint8", "crs": src.crs, "transform": src.window_transform(window),
                   "compress": "deflate", "photometric": "RGB"}
    out = Path("demo") / f"{args.name}.tif"
    out.parent.mkdir(exist_ok=True)
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(rgb.astype(np.uint8))
    print(f"{out}: {rgb.shape[2]} x {rgb.shape[1]} px, {src.res[0]:.2f} m, {item.datetime:%Y-%m-%d} ({item.id})")


if __name__ == "__main__":
    main()
