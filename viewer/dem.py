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

import os

import numpy as np

# GDAL's HTTP reads have no timeout by default: a stalled Planetary Computer transfer would hang
# an upload (or a benchmark) forever. Fail after 60 s instead, retrying transient errors; the
# caller then reports "no DSM" and still returns the nDSM. Env vars set by the user win.
for _k, _v in (("GDAL_HTTP_TIMEOUT", "60"), ("GDAL_HTTP_CONNECTTIMEOUT", "20"),
               ("GDAL_HTTP_MAX_RETRY", "4"), ("GDAL_HTTP_RETRY_DELAY", "2")):
    os.environ.setdefault(_k, _v)
HTTP_TIMEOUT_S = 60


def _default_request_timeout() -> None:
    """planetary_computer signs URLs with requests and no timeout, so a stalled token request
    hung uploads at "Writing absolute DSM". Give every requests call in this process that
    doesn't choose its own a (connect, read) timeout."""
    import requests

    if getattr(requests.Session.request, "_altimap_timeout", False):
        return
    original = requests.Session.request

    def request(self, method, url, **kwargs):
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = (20, HTTP_TIMEOUT_S)
        return original(self, method, url, **kwargs)

    request._altimap_timeout = True
    requests.Session.request = request


_default_request_timeout()

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

        catalog = pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace,
                                            timeout=HTTP_TIMEOUT_S)
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


GLO30_AWS = "https://copernicus-dem-30m.s3.amazonaws.com/{name}/{name}.tif"


def glo30_tile_urls(west: float, south: float, east: float, north: float) -> list[str]:
    """Copernicus GLO-30 1x1 degree tiles on AWS Open Data covering a lon/lat box. Tiles are
    named by their south-west corner: N40_00_W080_00 covers 40..41 N, 80..79 W."""
    import math

    urls = []
    for lat in range(math.floor(south), math.floor(north) + 1):
        for lon in range(math.floor(west), math.floor(east) + 1):
            name = (f"Copernicus_DSM_COG_10_{'N' if lat >= 0 else 'S'}{abs(lat):02d}_00_"
                    f"{'E' if lon >= 0 else 'W'}{abs(lon):03d}_00_DEM")
            urls.append(GLO30_AWS.format(name=name))
    return urls


def glo30(bounds, crs, shape: tuple[int, int]) -> np.ndarray | None:
    """Copernicus GLO-30 (m, EGM2008) reprojected onto the grid `bounds` x `shape` in `crs`.

    From the AWS Open Data copy first: plain HTTPS, no catalogue search or URL signing (the
    same tiles as Planetary Computer's cop-dem-glo-30, byte for byte, but Planetary
    Computer's token service stalled for 50-120 s at a time). Tiles missing on AWS are open
    ocean, filled with 0 m as GLO-30 does at sea. Falls back to Planetary Computer."""
    import rasterio
    from rasterio.warp import transform_bounds

    from viewer.geo import warp_to_grid

    urls = []
    for url in glo30_tile_urls(*transform_bounds(crs, "EPSG:4326", *bounds)):
        try:
            with rasterio.open(url):
                urls.append(url)
        except rasterio.RasterioIOError:
            pass  # no tile here: all sea (or AWS unreachable -> fallback below)
    if urls:
        dem = warp_to_grid(urls, bounds, crs, shape).astype(np.float64)
        return np.where(np.isfinite(dem), dem, 0.0)
    return DemSource("cop-dem-glo-30").patch(list(bounds), str(crs), shape)
