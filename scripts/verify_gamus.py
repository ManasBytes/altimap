"""Verify GAMUS image/height/class triplets before model evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from viewer.gamus_dataset import find_tiles, load_tile


def verify(root: Path) -> dict:
    summary = {"root": str(root), "splits": {}, "total": 0}
    for split in ("train", "val", "test"):
        groups = {}
        for kind, suffixes in (("images", ("RGB", "IMG")), ("classes", ("CLS",)),
                               ("heights", ("AGL",))):
            for suffix in suffixes:
                for path in (root / kind / split).glob(f"*_{suffix}.h5"):
                    groups.setdefault(path.name.rsplit("_", 1)[0], {})[kind] = path
        complete = [scene for scene, files in groups.items() if len(files) == 3]
        incomplete = sorted(scene for scene, files in groups.items() if len(files) != 3)
        split_tiles = [tile for tile in find_tiles((root,)) if tile.split == split]
        cities = sorted(Counter(tile.scene_id.split("_")[0] for tile in split_tiles))
        samples = []
        for tile in split_tiles[: min(3, len(split_tiles))]:
            rgb, classes, heights = load_tile(tile)
            samples.append({"scene": tile.scene_id, "rgb": list(rgb.shape),
                            "classes": list(classes.shape), "heights": list(heights.shape)})
        summary["splits"][split] = {"tiles": len(complete), "loader_tiles": len(split_tiles),
                                    "cities": cities, "incomplete": incomplete, "samples": samples}
        summary["total"] += len(complete)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    result = verify(args.root)
    print(json.dumps(result, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
