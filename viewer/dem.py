"""Windowed reads of a low-resolution reference DEM, for scale calibration.

3DEP seamless (1/3 arcsec, ~10 m) over Atlanta. All 620 georeferenced tiles in
the Off-nadir Scene10 dataset fall inside a single COG, so the STAC item is
resolved once and every tile is a windowed read against the same open handle --
620 separate STAC queries would dominate the runtime otherwise.

Signed hrefs from Planetary Computer expire, so the handle is reopened on
failure rather than cached for the life of the process.

Note 3dep-lidar-hag has NO coverage here (verified against a Provo control that
returns 13 items), which is why this falls back to a 10 m bare-earth DEM. That
constrains terrain level, not building height.
"""

from __future__ import annotations

import numpy as np

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "3dep-seamless"
# 1/3 arcsec tiles are suffixed -13, 1 arcsec are -1. Prefer the finer posting.
FINE_SUFFIX = "-13"


def bboxes_intersect(a: tuple[float, float, float, float],
                     b: tuple[float, float, float, float]) -> bool:
    """(w, s, e, n) overlap test. Pure so it is testable without rasterio/network."""
    aw, as_, ae, an = a
    bw, bs, be, bn = b
    return aw < be and bw < ae and as_ < bn and bs < an


class DemSource:
    """Lazily-resolved reference DEM with per-tile windowed reads.

    Handle reuse across calls with DIFFERENT bboxes is only valid when they
    fall inside the same source raster. This held for the Atlanta set (all 620
    tiles share one 3DEP COG) but not for the Inria set (10 cities, each its
    own Copernicus GLO-30 cell) -- naively reusing the first handle would
    silently read all-NaN windows for every city after the first and report
    "no DEM coverage" instead of erroring, which is how this bug was found:
    the calibration numbers for cities 2-10 were uniformly None.
    """

    def __init__(self, collection: str = COLLECTION):
        self.collection = collection
        self._item_id: str | None = None
        self._href: str | None = None
        self._handle = None
        self._handle_bounds_lonlat: tuple[float, float, float, float] | None = None

    def _resolve(self, bbox_lonlat: list[float]) -> str:
        import planetary_computer
        import pystac_client

        catalog = pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)
        items = list(catalog.search(collections=[self.collection], bbox=bbox_lonlat,
                                    max_items=10).items())
        if not items:
            raise LookupError(f"no {self.collection} coverage for {bbox_lonlat}")
        fine = [i for i in items if i.id.endswith(FINE_SUFFIX)]
        item = (fine or items)[0]
        self._item_id = item.id
        return item.assets["data"].href

    def _open(self, bbox_lonlat: list[float]):
        import rasterio
        from rasterio.warp import transform_bounds

        needs_reopen = (
            self._handle is None
            or self._handle_bounds_lonlat is None
            or not bboxes_intersect(tuple(bbox_lonlat), self._handle_bounds_lonlat)
        )
        if needs_reopen:
            if self._handle is not None:
                try:
                    self._handle.close()
                except Exception:
                    pass
            self._href = self._resolve(bbox_lonlat)
            self._handle = rasterio.open(self._href)
            self._handle_bounds_lonlat = transform_bounds(
                self._handle.crs, "EPSG:4326", *self._handle.bounds)
        return self._handle

    @property
    def item_id(self) -> str | None:
        return self._item_id

    def patch(self, bounds: list[float], crs: str, shape: tuple[int, int]) -> np.ndarray | None:
        """Elevation (metres) over `bounds` (in `crs`), resampled to `shape`.

        Returns None if the tile falls outside coverage or the read yields no
        valid pixels. nodata becomes NaN so the caller's fit can drop it.
        """
        import rasterio
        from rasterio.warp import transform_bounds
        from rasterio.windows import from_bounds

        from viewer.geo import resample_to

        bbox_lonlat = list(transform_bounds(crs, "EPSG:4326", *bounds))
        try:
            src = self._open(bbox_lonlat)
        except LookupError:
            return None

        for attempt in (0, 1):
            try:
                window = from_bounds(*transform_bounds(crs, src.crs, *bounds),
                                     transform=src.transform)
                arr = src.read(1, window=window, boundless=True,
                               fill_value=float("nan")).astype(np.float64)
                break
            except rasterio.RasterioIOError:
                # Most likely an expired signed href -- reopen once, then give up.
                if attempt == 1:
                    return None
                try:
                    self._handle.close()
                except Exception:
                    pass
                self._handle = None
                self._handle_bounds_lonlat = None
                src = self._open(bbox_lonlat)

        nodata = src.nodata
        if nodata is not None:
            arr = np.where(arr == nodata, np.nan, arr)
        # 3DEP uses a large negative sentinel; guard even when nodata is unset.
        arr = np.where(arr < -1e4, np.nan, arr)

        if not np.isfinite(arr).any():
            return None
        # Fill before resampling: ndimage.zoom would smear NaNs across the patch.
        if not np.isfinite(arr).all():
            arr = np.where(np.isfinite(arr), arr, np.nanmean(arr))
        return resample_to(arr, shape)

    def close(self) -> None:
        if self._handle is not None:
            try:
                self._handle.close()
            finally:
                self._handle = None
                self._handle_bounds_lonlat = None
