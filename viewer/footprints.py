"""Rasterise Overture building footprints onto a tile's depth grid.

Needs rasterio (CRS transform + polygon rasterisation), so it is kept out of
viewer/refine.py, which stays pure numpy and testable in the light venv.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DEFAULT_CACHE = Path("viewer/cache/buildings_atlanta.parquet")


class BuildingIndex:
    """In-memory footprints with a bbox filter.

    The Overture extract for the whole Atlanta AOI is ~166 k rows, which is
    small enough to hold and filter directly; the alternative is a remote query
    per tile, and there are 620 tiles.
    """

    def __init__(self, parquet_path: Path = DEFAULT_CACHE):
        import pyarrow.parquet as pq

        table = pq.read_table(parquet_path)
        cols = table.column_names
        self.geojson = table.column("geojson").to_pylist()
        self.height = (table.column("height").to_pylist()
                       if "height" in cols else [None] * len(self.geojson))
        self.floors = (table.column("num_floors").to_pylist()
                       if "num_floors" in cols else [None] * len(self.geojson))
        self.xmin = np.asarray(table.column("xmin").to_pylist(), dtype=np.float64)
        self.ymin = np.asarray(table.column("ymin").to_pylist(), dtype=np.float64)
        self.xmax = np.asarray(table.column("xmax").to_pylist(), dtype=np.float64)
        self.ymax = np.asarray(table.column("ymax").to_pylist(), dtype=np.float64)
        meta_path = Path(parquet_path).with_suffix(".json")
        self.meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    def __len__(self) -> int:
        return len(self.geojson)

    def select(self, bbox_lonlat: tuple[float, float, float, float]) -> list[int]:
        """Indices of footprints whose bbox intersects the tile."""
        w, s, e, n = bbox_lonlat
        hit = (self.xmax >= w) & (self.xmin <= e) & (self.ymax >= s) & (self.ymin <= n)
        return np.nonzero(hit)[0].tolist()

    def rasterize(self, bounds, crs: str, shape: tuple[int, int]):
        """-> (mask int32, known_m dict, stats dict).

        mask is 0 for ground and 1..N per building; known_m maps those ids to a
        metre height where Overture supplies one. Ids are assigned per tile, so
        they are only meaningful alongside their own mask.
        """
        from rasterio.features import rasterize
        from rasterio.transform import from_bounds
        from rasterio.warp import transform_bounds, transform_geom

        h, w = shape
        bbox_ll = transform_bounds(crs, "EPSG:4326", *bounds)
        idx = self.select(bbox_ll)

        transform = from_bounds(*bounds, w, h)
        shapes = []
        known: dict[int, float] = {}
        used = 0
        for i in idx:
            try:
                geom = transform_geom("EPSG:4326", crs, json.loads(self.geojson[i]))
            except Exception:
                continue
            used += 1
            bid = used
            shapes.append((geom, bid))
            metres = self.height[i]
            if metres is None and self.floors[i]:
                # Overture floor counts are far rarer than heights; 3 m/floor is
                # a convention, not a measurement, so it is only a fallback.
                metres = float(self.floors[i]) * 3.0
            if metres is not None and np.isfinite(metres) and metres > 0:
                known[bid] = float(metres)

        if not shapes:
            return np.zeros(shape, dtype=np.int32), {}, {
                "n_footprints": 0, "n_with_height": 0, "coverage": 0.0}

        mask = rasterize(shapes, out_shape=shape, transform=transform,
                         fill=0, dtype=np.int32, all_touched=False)
        return mask, known, {
            "n_footprints": int(len(shapes)),
            "n_with_height": int(len(known)),
            "coverage": float((mask > 0).mean()),
        }
