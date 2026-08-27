"""GeoTIFF -> 3D terrain, with NO depth model in the loop.

Every other page in this app runs a monocular depth model and then either
displays its raw output (rDSM path) or tries to calibrate it against a
reference elevation source (absolute-DSM path) -- and that calibration mostly
fails, because DA3 does not encode real height (see validate.py: median
correlation +0.004 against 28,960 real building heights).

This page skips the model entirely. For a georeferenced GeoTIFF, the real
elevation is already one API call away -- the same DEM this project already
fetches for calibration (viewer/dem.py). Instead of asking a depth model to
guess height and then checking the guess, drape the real orthophoto directly
over the real DEM. No ML uncertainty, no gating, no "reject_reason". The
trade is resolution: Copernicus GLO-30 is 30 m posting against 0.5 m imagery,
so this is coarse-but-true rather than sharp-but-guessed -- the opposite
trade-off from the rest of the app.

Picks 5 tiles from the Inria set (100% georeferenced, unlike the 12% for the
off-nadir Atlanta set) spanning a deliberate range of real terrain: two alpine
(dramatic real relief), one coastal hillside (already verified interesting in
this project), two flat urban (a fair contrast case -- these SHOULD look
nearly flat, because at 30 m posting they genuinely are).

Height-encoding trick: the rg16 PNG format and the browser's DataTexture
pipeline both assume "depth" semantics (larger stored value = lower ground, so
the shader computes height = (max - stored) / range). Real elevation is the
opposite (larger = higher). Rather than add a second code path, this encodes
`fake_depth = elev_max - elevation` -- which round-trips through the exact
same shader math to the exact correct height, unmodified. See
tests/test_viewer_dem_direct.py for the proof.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from viewer.dem import DemSource
from viewer.footprints import BuildingIndex
from viewer.geo import encode_rg16, geo_meta_from_dataset
from viewer.refine import extrude_known_buildings
from viewer.terrain import build_terrain

ZIP_PATH = Path("/home/biplab-dev/Downloads/HR_0.5m.zip")
DEM_COLLECTION = "cop-dem-glo-30"
# Already covers exactly the 5 cities DEFAULT_TILES draws from -- see
# viewer/fetch_buildings.py's Inria multi-region extract earlier this session.
BUILDINGS_CACHE = Path("viewer/cache/buildings_inria.parquet")

# entry name, human label, one-line rationale -- chosen for genuine contrast,
# not at random, so the 5 tiles actually demonstrate something.
DEFAULT_TILES = [
    ("HR_0.5m/tyrol-w7.tif", "Tyrol West (Alps)", "alpine — real dramatic relief expected"),
    ("HR_0.5m/innsbruck5.tif", "Innsbruck (Alps)", "alpine valley — real dramatic relief expected"),
    ("HR_0.5m/sfo4.tif", "San Francisco Bay Area", "coastal hillside — verified interesting earlier in this project"),
    ("HR_0.5m/austin1.tif", "Austin", "flat urban — honest null case: should render nearly flat"),
    ("HR_0.5m/vienna3.tif", "Vienna", "flat urban — honest null case: should render nearly flat"),
]


def elevation_to_fake_depth(elev: np.ndarray) -> tuple[np.ndarray, float, float]:
    """elev (real metres) -> array that decodes to correct height through the
    existing depth-shaped pipeline. Returns (fake_depth, elev_min, elev_max)."""
    elev_min, elev_max = float(np.nanmin(elev)), float(np.nanmax(elev))
    fake_depth = elev_max - elev
    return fake_depth, elev_min, elev_max


def export_one(entry: str, label: str, note: str, zip_path: Path, dem: DemSource,
               out_dir: Path, glb_res: int, buildings: BuildingIndex | None) -> dict:
    import rasterio

    vsi = f"/vsizip/{zip_path}/{entry}"
    with rasterio.open(vsi) as src:
        rgb = np.transpose(src.read([1, 2, 3]), (1, 2, 0)).astype(np.uint8)
        geo = geo_meta_from_dataset(src)
    if not geo["georeferenced"]:
        raise ValueError(f"{entry} carries no CRS -- this page requires georeferenced input")

    h, w = geo["height"], geo["width"]
    patch = dem.patch(geo["bounds"], geo["crs"], (h, w))
    if patch is None:
        raise ValueError(f"no {DEM_COLLECTION} coverage for {entry}")
    if not np.isfinite(patch).all():
        # Fill any residual nodata with the local mean rather than fail --
        # a few missing DEM cells at a tile edge shouldn't sink the whole tile.
        patch = np.where(np.isfinite(patch), patch, np.nanmean(patch))

    scene_id = Path(entry).stem
    scene_dir = out_dir / "scenes" / scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)

    fake_depth, elev_min, elev_max = elevation_to_fake_depth(patch)
    encoded, lo, hi = encode_rg16(fake_depth.astype(np.float32))
    Image.fromarray(encoded).save(scene_dir / "elevation.png", compress_level=6)
    Image.fromarray(rgb).resize((w, h), Image.BILINEAR).save(scene_dir / "rgb.jpg", quality=90)

    footprint_m = geo["ground_m"][0]
    elev_range_m = elev_max - elev_min
    true_scale_exag = elev_range_m / footprint_m if footprint_m else 0.0

    height01 = (patch - elev_min) / elev_range_m if elev_range_m > 0 else np.zeros_like(patch)
    build_terrain(height01, scene_dir / "rgb.jpg", scene_dir / "terrain.glb",
                  res=glb_res, exaggeration=true_scale_exag)

    # --- optional second variant: real DEM + real Overture building heights,
    # still zero depth model. Reuses the shared viewer's "refined" toggle --
    # see terrain.js's loadScene(), which already knows how to load a
    # refined.png against meta.refined without any page-specific code.
    refined_meta = None
    if buildings is not None:
        mask, known_m, fp_stats = buildings.rasterize(geo["bounds"], geo["crs"], (h, w))
        if fp_stats["n_footprints"] > 0:
            fused, n_extruded = extrude_known_buildings(patch, mask, known_m)
            f_fake_depth, f_min, f_max = elevation_to_fake_depth(fused)
            f_encoded, f_lo, f_hi = encode_rg16(f_fake_depth.astype(np.float32))
            Image.fromarray(f_encoded).save(scene_dir / "refined.png", compress_level=6)

            f_range = f_max - f_min
            f_height01 = (fused - f_min) / f_range if f_range > 0 else np.zeros_like(fused)
            f_exag = f_range / footprint_m if footprint_m else 0.0
            build_terrain(f_height01, scene_dir / "rgb.jpg", scene_dir / "terrain_buildings.glb",
                          res=glb_res, exaggeration=f_exag)

            refined_meta = {
                "depth_lo": f_lo, "depth_hi": f_hi, "depth_min": f_lo, "depth_max": f_hi,
                "n_footprints": fp_stats["n_footprints"],
                "n_with_height": fp_stats["n_with_height"],
                "n_extruded": n_extruded,
                "true_scale_exaggeration": f_exag,
                "source": "overture-building-heights",
            }

    record = {
        "id": scene_id, "label": label, "note": note, "source_entry": entry,
        "width": w, "height": h,
        "encoding": "rg16-png-elevation",  # NOT "rg16-png" -- no depth model produced this
        "depth_lo": lo, "depth_hi": hi,
        # Matches the field names the shared viewer already reads for the raw-mode
        # range, so terrain.js needs zero changes to render this correctly.
        "depth_min": lo, "depth_max": hi,
        "geo": geo,
        "elevation_m": {"min": elev_min, "max": elev_max, "range": elev_range_m},
        "dem_source": DEM_COLLECTION, "dem_item": dem.item_id,
        "true_scale_exaggeration": true_scale_exag,
        "has_glb": True,
        "refined": refined_meta,
        "has_buildings_glb": refined_meta is not None,
        # Shaped exactly like the model-calibration "absolute" field elsewhere
        # in this app (see viewer/export_offnadir.py) so probe.js's existing
        # metricParams()/pickTerrain() pick this up with zero code changes.
        # "usable" is unconditionally true: this IS the reference elevation,
        # not a fit to it, so there is no quality gate to apply.
        "absolute": {
            "usable": True, "scale_m": elev_range_m, "offset_m": elev_min,
            "source": DEM_COLLECTION, "reference_is_bare_earth": False,
            "reference_posting_m": 30.0,
        },
    }
    (scene_dir / "meta.json").write_text(json.dumps(record))
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=ZIP_PATH)
    # Matches the live viewer's mesh density (see demdirect.js) so the
    # downloadable GLB shows the same building edge detail as the browser --
    # buildings carry real sub-metre footprint precision, and a coarser mesh
    # would blur that away regardless of how sharp the source data is.
    parser.add_argument("--glb-res", type=int, default=768)
    parser.add_argument("--out", type=Path, default=Path("viewer/web/data-demdirect"))
    parser.add_argument("--buildings", type=Path, default=BUILDINGS_CACHE)
    parser.add_argument("--no-buildings", action="store_true",
                        help="skip the real-building-height fusion variant")
    args = parser.parse_args()

    buildings = None
    if not args.no_buildings and args.buildings.exists():
        buildings = BuildingIndex(args.buildings)
        print(f"loaded {len(buildings)} building footprints from {args.buildings}")
    elif not args.no_buildings:
        print(f"no buildings cache at {args.buildings} -- run fetch_buildings.py first; "
              f"continuing with DEM-only output")

    dem = DemSource(collection=DEM_COLLECTION)
    records = []
    for entry, label, note in DEFAULT_TILES:
        print(f"{label} ({entry}) -- {note}")
        record = export_one(entry, label, note, args.zip, dem, args.out, args.glb_res, buildings)
        r = record["elevation_m"]
        print(f"  elevation {r['min']:.1f}-{r['max']:.1f} m (range {r['range']:.1f} m), "
              f"true-scale exaggeration {record['true_scale_exaggeration']:.4f}")
        if record["refined"]:
            f = record["refined"]
            print(f"  + buildings: {f['n_footprints']} footprints, {f['n_with_height']} with "
                  f"known height, {f['n_extruded']} extruded")
        records.append(record)
    dem.close()

    index = {
        "dataset": "dem-direct", "dem_source": DEM_COLLECTION,
        "note": "No depth model in this pipeline. Terrain is the real reference DEM, "
                "optionally fused with real Overture building heights.",
        "scenes": [{
            "id": r["id"], "label": r["label"], "note": r["note"],
            "elevation_range_m": round(r["elevation_m"]["range"], 1),
            "crs": r["geo"]["crs"], "ground_m": r["geo"]["ground_m"],
            "has_buildings": r["has_buildings_glb"],
            "n_extruded": (r["refined"] or {}).get("n_extruded", 0),
        } for r in records],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "index.json").write_text(json.dumps(index, indent=2))
    print(f"\nwrote {args.out / 'index.json'} ({len(records)} scenes)")


if __name__ == "__main__":
    main()
