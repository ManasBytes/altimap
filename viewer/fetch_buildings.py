"""Extract Overture building footprints + heights for an AOI, once, to disk.

Why Overture and not the alternatives, measured over the Atlanta extent:

    ms-buildings (Planetary Computer)  129 M US rows, geometry ONLY -- no height
    OpenStreetMap (Overpass)           157,879 buildings,  1.5% with any height
    Overture Maps                      166,006 buildings, 78.2% with height (m)

Heights are what make footprints useful here: they turn the relative-depth
residual inside a footprint into a metric anchor, which is the brief's
"minimal Ground Control Points" path. Footprints alone only fix geometry.

The remote query scans a large partitioned parquet and takes minutes, so it runs
once and caches locally; per-tile work is then an in-memory bbox filter.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

# Overture publishes dated releases; this is resolved from the bucket listing
# rather than hard-coded, because guessed release tags simply 404.
BUCKET = "s3://overturemaps-us-west-2/release"
LISTING = "https://overturemaps-us-west-2.s3.amazonaws.com/?list-type=2&prefix=release/&delimiter=/"
DEFAULT_OUT = Path("viewer/cache/buildings_atlanta.parquet")

# Union of the 620 georeferenced tile footprints in the Off-nadir Scene10 set.
ATLANTA = (-84.4954, 33.5934, -84.3049, 33.8088)


def latest_release() -> str:
    import re
    import urllib.request

    with urllib.request.urlopen(LISTING, timeout=60) as resp:
        body = resp.read().decode()
    releases = [r for r in re.findall(r"<Prefix>release/([^<]+)/</Prefix>", body) if r.strip()]
    if not releases:
        raise RuntimeError("could not list Overture releases")
    return sorted(releases)[-1]


def fetch(bbox: tuple[float, float, float, float], out: Path, release: str | None = None) -> dict:
    import duckdb

    release = release or latest_release()
    src = f"{BUCKET}/{release}/theme=buildings/type=building/*.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    w, s, e, n = bbox

    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial; INSTALL httpfs; LOAD httpfs; SET s3_region='us-west-2';")

    started = time.perf_counter()
    # bbox is a struct column in Overture and supports predicate pushdown, so
    # filtering on it avoids decoding geometry for the whole planet.
    con.execute(f"""
        COPY (
          SELECT id,
                 height,
                 num_floors,
                 bbox.xmin AS xmin, bbox.ymin AS ymin,
                 bbox.xmax AS xmax, bbox.ymax AS ymax,
                 ST_AsGeoJSON(geometry) AS geojson
          FROM read_parquet('{src}', hive_partitioning=1)
          WHERE bbox.xmin > {w} AND bbox.xmax < {e}
            AND bbox.ymin > {s} AND bbox.ymax < {n}
        ) TO '{out}' (FORMAT PARQUET)
    """)
    elapsed = time.perf_counter() - started

    stats = con.execute(f"""
        SELECT count(*), count(height), count(num_floors), median(height)
        FROM read_parquet('{out}')
    """).fetchone()
    meta = {
        "release": release,
        "bbox": list(bbox),
        "count": stats[0],
        "with_height": stats[1],
        "with_floors": stats[2],
        "median_height_m": stats[3],
        "seconds": round(elapsed, 1),
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bbox", nargs=4, type=float, default=list(ATLANTA),
                        metavar=("W", "S", "E", "N"))
    parser.add_argument("--release", default=None, help="Overture release, else latest")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    print(f"extracting Overture buildings for {args.bbox}")
    meta = fetch(tuple(args.bbox), args.out, args.release)
    pct = 100 * meta["with_height"] / max(meta["count"], 1)
    print(f"release {meta['release']}: {meta['count']} buildings in {meta['seconds']}s")
    print(f"  with height: {meta['with_height']} ({pct:.1f}%), median {meta['median_height_m']:.1f} m")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
