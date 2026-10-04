"""DSM accuracy by landscape and by pixel size, against USGS 3DEP airborne LiDAR.

    .venv-da3/bin/python -m viewer.dsm_eval                      # all scenes, 0.6 -> 10 m
    .venv-da3/bin/python -m viewer.dsm_eval --gsd 0.6 --tta      # the app's High quality only

The four demo NAIP scenes (0.6 m, RGB) cover the brief's landscape types: dense city,
leafy suburb, hilly town, forest. Each is scored as the app would produce it:

- nDSM (height above ground) vs LiDAR height-above-ground,
- the exported absolute DSM vs the LiDAR DSM, next to the base DEM alone (SRTM by default,
  --base-dem glo30 for Copernicus: what you get with no model at all),
- with 8 LiDAR ground points used as ground control points (scored on all other pixels).

For coarser pixel sizes the image is block-averaged (as a coarser sensor would see it)
and the LiDAR is averaged onto the same grid. LiDAR references are cached in
viewer/cache/dsm_eval/. LiDAR heights are NAVD88, GLO-30 EGM2008: they differ by well
under a metre over these sites, which is left in the numbers.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

import viewer.dem  # noqa: F401  (sets GDAL HTTP timeouts before the first LiDAR/DEM read)

SCENES = {  # demo file -> landscape
    "naip_philadelphia_cityhall.tif": "dense city",
    "naip_dc_suburbs_chevychase.tif": "suburb (sparse)",
    "naip_pittsburgh_hills.tif": "hilly town",
    "naip_smoky_mountains_forest.tif": "forest",
}
DEMO = Path("demo")
CACHE = Path("viewer/cache/dsm_eval")
LIDAR = ("dsm", "dtm", "hag")


def lidar(path: Path) -> dict[str, np.ndarray]:
    """3DEP LiDAR DSM / DTM / height-above-ground on the image's own grid (cached). Where
    Planetary Computer has no LiDAR DTM/HAG tile over the scene (Philadelphia), bare earth
    comes from 3DEP seamless (1/3 arcsec, LiDAR-derived, NAVD88) and HAG = DSM - DTM."""
    import rasterio

    from viewer.geo import warp_to_grid

    CACHE.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path) as src:
        bounds, crs, shape = tuple(src.bounds), src.crs, src.shape
    out = {}
    for kind in LIDAR:
        f = CACHE / f"{path.stem}_{kind}.npy"
        if not f.exists():
            import planetary_computer
            import pystac_client
            from rasterio.warp import transform_bounds

            cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                                            modifier=planetary_computer.sign_inplace)
            items = cat.search(collections=[f"3dep-lidar-{kind}"],
                               bbox=transform_bounds(crs, "EPSG:4326", *bounds), max_items=50).items()
            np.save(f, warp_to_grid([i.assets["data"].href for i in items], bounds, crs, shape))
        out[kind] = np.load(f)
    if np.isfinite(out["dtm"]).mean() < 0.99:
        from viewer.dem import DemSource

        out["dtm"] = DemSource("3dep-seamless").patch(bounds, str(crs), shape).astype(np.float32)
        out["dtm_source"] = "3dep-seamless"
    if np.isfinite(out["hag"]).mean() < 0.99:
        out["hag"] = np.maximum(out["dsm"] - out["dtm"], 0)
    return out


def block_mean(a: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """NaN-aware area average onto a coarser `shape`."""
    ok = np.isfinite(a)
    num = np.asarray(Image.fromarray(np.where(ok, a, 0).astype(np.float32), "F").resize(shape[::-1], Image.BOX))
    den = np.asarray(Image.fromarray(ok.astype(np.float32), "F").resize(shape[::-1], Image.BOX))
    return np.where(den > 0.5, num / np.maximum(den, 1e-6), np.nan)


def degrade(path: Path, gsd: float, tmp: Path) -> Path:
    """The scene as a coarser sensor would record it: block-averaged RGB, scaled transform."""
    import rasterio

    with rasterio.open(path) as src:
        scale = gsd / src.res[0]
        if abs(scale - 1) < 1e-6:
            return path
        shape = (round(src.height / scale), round(src.width / scale))
        rgb = np.transpose(src.read([1, 2, 3]), (1, 2, 0))
        small = np.asarray(Image.fromarray(rgb).resize(shape[::-1], Image.BOX))
        profile = src.profile | {"height": shape[0], "width": shape[1],
                                 "transform": src.transform * src.transform.scale(src.width / shape[1],
                                                                                  src.height / shape[0])}
    out = tmp / f"{path.stem}_{gsd:g}m.tif"
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(np.transpose(small, (2, 0, 1)))
    return out


def scores(pred: np.ndarray, ref: np.ndarray) -> dict:
    ok = np.isfinite(pred) & np.isfinite(ref)
    p, r = pred[ok].astype(np.float64), ref[ok].astype(np.float64)
    e = p - r
    return {"rmse": float(np.sqrt(np.mean(e ** 2))), "mae": float(np.mean(np.abs(e))),
            "bias": float(np.mean(e)), "r": float(np.corrcoef(p, r)[0, 1]) if p.std() > 0 else float("nan")}


def gcp_check(dsm: np.ndarray, ground: np.ndarray, ref: dict, transform, crs, n: int = 8,
              seed: int = 0) -> dict:
    """Use n LiDAR bare-ground points as GCPs, fitted against the bare-earth ground as the app
    does; score the corrected DSM on every other pixel."""
    from rasterio.warp import transform as warp

    from viewer.estimate import gcp_correction

    bare = np.isfinite(ref["dsm"]) & np.isfinite(ref["hag"]) & (ref["hag"] < 0.3) & np.isfinite(dsm)
    rows, cols = np.nonzero(bare)
    if len(rows) < n:
        return {"model": None, "note": f"only {len(rows)} bare-ground LiDAR pixels (closed canopy)"}
    pick = np.random.default_rng(seed).choice(len(rows), size=min(n, len(rows)), replace=False)
    rs, cs = rows[pick], cols[pick]
    xs, ys = zip(*[transform * (c + 0.5, r + 0.5) for r, c in zip(rs, cs)])
    lons, lats = warp(crs, "EPSG:4326", list(xs), list(ys))
    fix, info = gcp_correction(ground, transform, crs,
                               [(lo, la, float(ref["dsm"][r, c])) for lo, la, r, c in zip(lons, lats, rs, cs)])
    held = np.ones(dsm.shape, bool)
    held[rs, cs] = False
    ref_held = np.where(held, ref["dsm"], np.nan)
    return {"model": info.get("model"), "before": scores(dsm, ref_held),
            "after": scores(dsm + fix, ref_held) if fix is not None else None}


def main() -> None:
    import rasterio

    from viewer.dem import BASE_DEMS
    from viewer.estimate import estimate, load_pipeline

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gsd", type=float, nargs="+", default=None,
                    help="pixel sizes to test (default: native, then 0.6 1 2 5 10 m)")
    ap.add_argument("--tta", action="store_true", help="4-flip averaging (the app's High quality)")
    ap.add_argument("--ckpt", type=Path, default=Path("viewer/cache/best.pth"))
    ap.add_argument("--out", type=Path, default=Path("dsm_eval.json"))
    ap.add_argument("--scenes", nargs="+", default=list(SCENES), choices=list(SCENES))
    ap.add_argument("--base-dem", choices=tuple(BASE_DEMS), default="srtm",
                    help="DEM under the DSM, and the 'DEM alone' baseline (default SRTM, as the app)")
    args = ap.parse_args()

    models = load_pipeline(args.ckpt)  # app pipeline: v1 + 75%-weighted v2 buildings + CHMv2 forest
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for name in args.scenes:
            landscape = SCENES[name]
            path = DEMO / name
            ref = lidar(path)
            with rasterio.open(path) as src:
                native = round(float(src.res[0]), 2)
            gsds = args.gsd or [native] + [g for g in (0.6, 1.0, 2.0, 5.0, 10.0) if g > native + 0.01]
            for gsd in gsds:
                img = degrade(path, gsd, tmp)
                out = estimate(img, out_dir=tmp / "out", tta=args.tta, base_dem=args.base_dem, **models)
                shape = out["ndsm"].shape
                r = {k: (v if v.shape == shape else block_mean(v, shape))
                     for k, v in ref.items() if k in LIDAR}
                with rasterio.open(img) as src:
                    transform, crs, bounds = src.transform, src.crs, tuple(src.bounds)
                dem = BASE_DEMS[args.base_dem][0](bounds, crs, shape)  # the base DEM alone, no model
                row = {"scene": name, "landscape": landscape, "gsd_m": gsd, "base_dem": args.base_dem,
                       "dtm_source": ref.get("dtm_source", "3dep-lidar"),
                       "ndsm": scores(out["ndsm"], r["hag"]),
                       "ndsm_zero": scores(np.zeros(shape, np.float32), r["hag"]),
                       "dsm": scores(out["dsm"], r["dsm"]) if out["dsm"] is not None else None,
                       "dem_only": scores(dem, r["dsm"]) if dem is not None else None}
                if out["dsm"] is not None and gsd == gsds[0]:
                    row["gcp8"] = gcp_check(out["dsm"], out["ground"], r, transform, crs)
                rows.append(row)
                d, z = row["dsm"], row["dem_only"]
                print(f"{landscape:16s} {gsd:5.1f} m | nDSM RMSE {row['ndsm']['rmse']:5.2f} bias "
                      f"{row['ndsm']['bias']:+6.2f} r {row['ndsm']['r']:.2f} (zero: {row['ndsm_zero']['rmse']:5.2f})"
                      f" | DSM RMSE {d['rmse']:5.2f} r {d['r']:.3f} ({args.base_dem} alone {z['rmse']:5.2f} r {z['r']:.3f})"
                      + (f" | 8 GCPs {row['gcp8']['after']['rmse']:5.2f} ({row['gcp8']['model']})"
                         if row.get("gcp8", {}).get("after") else ""), flush=True)
                args.out.write_text(json.dumps(rows, indent=1))  # after every row: a crash keeps the rest


if __name__ == "__main__":
    main()
