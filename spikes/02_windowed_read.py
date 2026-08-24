"""Spike: confirm windowed COG reads pull only the requested window."""

import planetary_computer
import pystac_client
import rasterio
from rasterio.windows import Window

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
BBOX = (-111.95, 40.55, -111.85, 40.65)  # Wasatch — known-good from research

client = pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)

for coll, asset_key in [("naip", "image"), ("3dep-lidar-hag", "data")]:
    item = next(client.search(collections=[coll], bbox=BBOX, limit=1).items())
    href = item.assets[asset_key].href
    print(f"\n{coll}: {item.id}")
    with rasterio.open(href) as src:
        print(f"  shape={src.shape} crs={src.crs} res={src.res} dtype={src.dtypes[0]}")
        print(f"  nodata={src.nodata} blocksize={src.block_shapes[0]}")
        win = Window(0, 0, 512, 512)
        arr = src.read(1, window=win)
        print(f"  windowed read OK: {arr.shape} min={arr.min():.2f} max={arr.max():.2f}")
