"""Score the 3D city model's buildings on GAMUS val tiles (true building labels + LiDAR).

    .venv-da3/bin/python -m viewer.city_eval --fetch    # first run: download the 40 tiles (370 MB)
    .venv-da3/bin/python -m viewer.city_eval            # score the current viewer/city_model.py

Reports, pooled over tiles: footprint IoU against the reference building mask, edge F1
(share of outline pixels within 1 m of the true outline), height RMSE on pixels where
the model placed a building, and the number of buildings. Height predictions are cached
in viewer/cache/city_eval/, so re-scoring after a city_model change takes no GPU.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy import ndimage

from viewer.city_model import building_parts, footprints

GSD = 0.33
CACHE = Path("viewer/cache/city_eval")
# The 40 GAMUS val tiles behind the city-model numbers in CLAUDE.md / README.
EVAL_TILES = """DC_04_27 DC_12_17 DC_15_17 DC_15_40 DC_16_21 DC_16_33 DC_21_33 DC_26_47 DC_27_32 DC_29_07
DC_31_20 DC_41_60 DC_43_67 DC_48_26 DC_49_58 DC_58_46 PHL_6186 PHL_6188 PHL_6197 PHL_6249 PHL_6275
PHL_6279 PHL_6302 PHL_6334 PHL_6342 PHL_6417 PHL_6445 PHL_6483 PHL_6500 PHL_6556 PHL_6572 PHL_6630
PHL_6670 PHL_6740 PHL_6741 PHL_6743 PHL_6746 PHL_6782 PHL_6817 PHL_6922""".split()


def fetch(root: Path) -> None:
    """Download the eval tiles from the public GAMUS release into `root` (skips existing)."""
    from huggingface_hub import hf_hub_download

    for sid in EVAL_TILES:
        image = f"{sid}_RGB.h5" if sid.startswith("DC_") else f"{sid}_IMG.h5"  # DC vs other cities
        for sub, name in (("images", image), ("heights", f"{sid}_AGL.h5"), ("classes", f"{sid}_CLS.h5")):
            hf_hub_download("earthflow/GAMUS", f"{sub}/val/{name}", repo_type="dataset", local_dir=root)


def predictions(root: Path) -> list[Path]:
    from viewer.gamus_dataset import find_tiles, load_tile

    tiles = [t for t in find_tiles((root,)) if t.split == "val" and t.scene_id in EVAL_TILES]
    if len(tiles) < len(EVAL_TILES):
        raise SystemExit(f"{len(tiles)}/{len(EVAL_TILES)} eval tiles under {root}; run with --fetch")
    CACHE.mkdir(parents=True, exist_ok=True)
    todo = [t for t in tiles if not (CACHE / f"{t.scene_id}.npz").exists()]
    if todo:
        from viewer.height_model import OEM_TO_GAMUS, clean_height, load_model, predict

        model, dev = load_model(Path("viewer/cache/best.pth"), Path("viewer/cache/SynRS3D"))
        for t in todo:
            rgb, cls, ref = load_tile(t)
            h, oem = predict(model, rgb, dev)
            np.savez_compressed(CACHE / f"{t.scene_id}.npz", cls=cls, ref=clean_height(ref),
                                h=h.astype(np.float32), pcls=np.asarray(OEM_TO_GAMUS)[oem])
    v2 = Path("viewer/cache/best_v2.pth")  # the app fuses v2 into building heights
    todo = [t for t in tiles if v2.exists() and "h2" not in np.load(CACHE / f"{t.scene_id}.npz")]
    if todo:
        from viewer.height_model import load_model, predict

        model, dev = load_model(v2, Path("viewer/cache/SynRS3D"))
        for t in todo:
            d = dict(np.load(CACHE / f"{t.scene_id}.npz"))
            d["h2"] = predict(model, load_tile(t)[0], dev)[0].astype(np.float32)
            np.savez_compressed(CACHE / f"{t.scene_id}.npz", **d)
    return [CACHE / f"{t.scene_id}.npz" for t in tiles]


def footprint_raster(ndsm, classes, shape):
    """City-model buildings rasterized back onto the grid -> (mask, height raster, count)."""
    from rasterio.features import rasterize
    from shapely.geometry import Polygon

    rows, cols = shape
    shapes = []
    for b in footprints(*building_parts(ndsm, classes, GSD), shape=shape, gsd_m=GSD):
        p = Polygon([(u * cols, v * rows) for u, v in b["rings"][0]],
                    [[(u * cols, v * rows) for u, v in r] for r in b["rings"][1:]])
        if p.is_valid and not p.is_empty:
            shapes.append((p, b["h"]))
    if not shapes:
        return np.zeros(shape, bool), np.zeros(shape, np.float32), 0
    hr = rasterize(shapes, out_shape=shape, fill=0, dtype="float32")
    mask = rasterize([(p, 1) for p, _ in shapes], out_shape=shape, fill=0, dtype="uint8") > 0
    return mask, hr, len(shapes)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gamus", type=Path, default=Path("viewer/cache/gamus"))
    ap.add_argument("--fetch", action="store_true", help="download the eval tiles first")
    args = ap.parse_args()
    if args.fetch:
        fetch(args.gamus)
    tol = round(1.0 / GSD)  # 1 m, in pixels
    inter = union = hit_p = n_p = hit_r = n_r = count = 0
    se = n = 0.0
    for f in predictions(args.gamus):
        d = np.load(f)
        ref_b = d["cls"] == 3
        from viewer.estimate import fuse_heights

        h = fuse_heights(d["h"], d["h2"] if "h2" in d else None, d["pcls"])
        pm, hr, c = footprint_raster(h, d["pcls"], ref_b.shape)
        inter += (pm & ref_b).sum()
        union += (pm | ref_b).sum()
        pb, rb = pm & ~ndimage.binary_erosion(pm), ref_b & ~ndimage.binary_erosion(ref_b)
        hit_p += (ndimage.distance_transform_edt(~rb)[pb] <= tol).sum()
        n_p += pb.sum()
        hit_r += (ndimage.distance_transform_edt(~pb)[rb] <= tol).sum()
        n_r += rb.sum()
        ok = pm & ref_b & np.isfinite(d["ref"])
        se += float(((hr[ok] - d["ref"][ok]) ** 2).sum())
        n += ok.sum()
        count += c
    p, r = hit_p / max(n_p, 1), hit_r / max(n_r, 1)
    print(f"footprint IoU {inter / max(union, 1):.3f} | edge F1 @1 m {2 * p * r / max(p + r, 1e-9):.3f}"
          f" | height RMSE on buildings {np.sqrt(se / max(n, 1)):.2f} m | buildings {count}")


if __name__ == "__main__":
    main()
