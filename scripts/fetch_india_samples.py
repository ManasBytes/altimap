"""Download Indian test scenes: 1.2 km crops of Maxar Open Data satellite imagery (Sikkim).

    .venv-da3/bin/python scripts/fetch_india_samples.py        # -> demo/india/*.tif

WorldView-2/3 pansharpened RGB from the Maxar Open Data Program's "North India Floods" event
(licence CC BY-NC 4.0, credit "Maxar Open Data Program"): pre-flood 2022 acquisitions of
Himalayan terrain, a hard and relevant case for Cartosat-style imagery over India. Each crop is
a windowed read of the public cloud-optimized GeoTIFF, so only ~40 MB is transferred per scene.
Files already present are skipped.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform
from rasterio.windows import Window

BASE = "https://maxar-opendata.s3.amazonaws.com/events/India-Floods-Oct-2023/ard/45"
# name -> (tile path, centre lon, lat, what it is)
SCENES = {
    "sikkim_chungthang_town": ("120220122202/2022-03-07/10300100CE8D0400", 88.6450, 27.6036,
                               "WorldView-2, 0.49 m, 13.6 deg off-nadir: valley town on the Teesta"),
    "sikkim_chungthang_forest": ("120220122202/2022-03-07/10300100CE8D0400", 88.6300, 27.5750,
                                 "WorldView-2, 0.49 m, 13.6 deg off-nadir: forested Himalayan slope"),
    "sikkim_namchi_town": ("120220211231/2022-03-14/1040010073381800", 88.3700, 27.1680,
                           "WorldView-3, 0.37 m, 26 deg off-nadir: district town on a ridge"),
}
SIZE_M = 1200.0
OUT = Path("demo/india")


def fetch(name: str, tile: str, lon: float, lat: float) -> Path:
    out = OUT / f"{name}.tif"
    if out.exists():
        return out
    href = f"{BASE}/{tile.split('/')[0]}/{tile.split('/')[1]}/{tile.split('/')[2]}-visual.tif"
    with rasterio.open(href) as src:
        (x,), (y,) = transform("EPSG:4326", src.crs, [lon], [lat])
        col, row = ~src.transform * (x, y)
        n = int(round(SIZE_M / src.res[0]))
        c0 = int(np.clip(col - n / 2, 0, src.width - n))
        r0 = int(np.clip(row - n / 2, 0, src.height - n))
        win = Window(c0, r0, n, n)
        rgb = src.read([1, 2, 3], window=win)
        profile = src.profile | {"driver": "GTiff", "width": n, "height": n, "count": 3,
                                 "transform": src.window_transform(win), "compress": "deflate",
                                 "tiled": True, "blockxsize": 256, "blockysize": 256}
        profile.pop("photometric", None)
    OUT.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(rgb)
        dst.update_tags(source=href, licence="CC BY-NC 4.0, Maxar Open Data Program")
    return out


if __name__ == "__main__":
    for name, (tile, lon, lat, what) in SCENES.items():
        path = fetch(name, tile, lon, lat)
        with rasterio.open(path) as src:
            valid = (src.read(1) > 0).mean()
        print(f"{path}  {src.width}x{src.height} px at {src.res[0]:.2f} m, {valid:.0%} image  ({what})")
