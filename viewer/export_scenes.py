"""Run DA3 over the dataset, curate scenes, write static viewer assets.

Two passes. Pass 1 computes metrics for every image and keeps no depth. Pass 2
re-runs DA3 on only the curated subset to write assets. Holding 1000 depth maps
would cost ~1 GB of RAM; re-running ~42 images costs ~5 s. The model is
deterministic, so pass 2 reproduces pass 1 exactly.

Run under .venv-da3 -- this imports torch. viewer/metrics.py deliberately
does not.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from viewer.metrics import luminance, scene_metrics
from viewer.terrain import build_terrain, height_field

DEFAULT_DATASET = Path("/home/biplab-dev/Downloads/Remote Sensing Data.v2i.yolov8")
DEFAULT_MODEL = "depth-anything/DA3-SMALL"

# Order matches data.yaml's `names`; the YOLO class index is a position here.
CLASS_NAMES = [
    "Agriculture", "Airport", "Beach", "City", "Desert", "Forest", "Grassland",
    "Highway", "Lake", "Mountain", "Parking", "Port", "Railway", "River",
]


def scene_id(class_name: str, image_path: Path, split: str, multi_split: bool) -> str:
    stem = image_path.stem[:16]
    return f"{split}__{class_name}__{stem}" if multi_split else f"{class_name}__{stem}"


def index_split(split_dir: Path) -> list[tuple[str, Path]]:
    """(class_name, image_path) for every labelled image in a split.

    Label files carry no trailing newline, so they are read individually --
    `cat labels/*.txt` welds the last row of one file onto the first row of the
    next and corrupts the class histogram.
    """
    out = []
    for image_path in sorted((split_dir / "images").glob("*.jpg")):
        label_path = split_dir / "labels" / f"{image_path.stem}.txt"
        rows = []
        if label_path.exists():
            rows = [r for r in label_path.read_text().splitlines() if r.strip()]
        if not rows:
            continue
        best = max(rows, key=lambda r: float(r.split()[3]) * float(r.split()[4]))
        out.append((CLASS_NAMES[int(best.split()[0])], image_path))
    return out


def curate(records: list[dict], per_class_low: int = 2, per_class_high: int = 1) -> list[dict]:
    """Lowest-plane_r2 scenes per class, plus one highest as deliberate contrast.

    Sorting by plane_r2 ascending puts the scenes with non-ramp structure first
    -- alphabetical ordering opens the dashboard on Agriculture (R2 0.96, near
    pure ramp), which is the least informative thing we have.
    """
    by_class: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        if np.isfinite(record["plane_r2"]):
            by_class[record["class"]].append(record)

    picked: list[dict] = []
    for name in CLASS_NAMES:
        ranked = sorted(by_class.get(name, []), key=lambda r: r["plane_r2"])
        chosen = ranked[:per_class_low]
        if per_class_high and len(ranked) > per_class_low:
            chosen = chosen + ranked[-per_class_high:]
        picked.extend(chosen)
    return picked


def class_aggregates(records: list[dict]) -> dict:
    """Per-class n / median / IQR for the metrics the dashboard shows."""
    out = {}
    for name in CLASS_NAMES:
        rows = [r for r in records if r["class"] == name]
        if not rows:
            continue
        entry = {"n": len(rows)}
        for key in ("plane_r2", "residual_relief", "structure_alignment", "conf_texture_corr"):
            values = np.array([r[key] for r in rows], dtype=float)
            values = values[np.isfinite(values)]
            if values.size == 0:
                entry[key] = {"median": None, "q1": None, "q3": None}
                continue
            entry[key] = {
                "median": float(np.median(values)),
                "q1": float(np.percentile(values, 25)),
                "q3": float(np.percentile(values, 75)),
            }
        out[name] = entry
    return out


def _load_model(model_name: str):
    import torch
    from depth_anything_3.api import DepthAnything3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"loading {model_name} onto {device}...")
    return DepthAnything3.from_pretrained(model_name).to(device=device), device


def _predict(model, image_path: Path, process_res: int):
    """-> (depth HxW float32, conf HxW float32 or None, is_metric bool)"""
    prediction = model.inference([str(image_path)], process_res=process_res)
    depth = np.asarray(prediction.depth[0], dtype=np.float32)
    conf = None
    if prediction.conf is not None:
        conf = np.asarray(prediction.conf[0], dtype=np.float32)
    if not np.isfinite(depth).all():
        raise ValueError(f"non-finite depth for {image_path.name}")
    # is_metric is an empty addict Dict for non-metric models, not an int --
    # output_processor.py:71 calls getattr(..., 0) on an addict Dict, whose
    # __getattr__ manufactures an empty Dict so the default is unreachable.
    # int() raises TypeError; truthiness reads correctly.
    return depth, conf, bool(prediction.is_metric)


def _luminance_on_depth_grid(image_path: Path, shape: tuple[int, int]) -> np.ndarray:
    """RGB arrives at 640, depth at 504 -- resize before any gradient compare."""
    rgb = Image.open(image_path).convert("RGB").resize((shape[1], shape[0]), Image.BILINEAR)
    return luminance(np.asarray(rgb, dtype=np.float64) / 255.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--splits", nargs="+", default=["train"], choices=["train", "valid", "test"])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument("--limit", type=int, default=0, help="cap images for a quick run")
    parser.add_argument("--metrics-only", action="store_true")
    parser.add_argument("--per-class-low", type=int, default=2)
    parser.add_argument("--per-class-high", type=int, default=1)
    parser.add_argument("--no-glb", action="store_true")
    parser.add_argument("--glb-res", type=int, default=256)
    parser.add_argument("--glb-exaggeration", type=float, default=0.15)
    parser.add_argument("--out", type=Path, default=Path("viewer/web/data"))
    args = parser.parse_args()

    multi_split = len(args.splits) > 1
    work: list[tuple[str, str, Path]] = []
    for split in args.splits:
        split_dir = args.dataset / split
        if not split_dir.is_dir():
            raise SystemExit(f"no such split: {split_dir}")
        work.extend((split, cls, path) for cls, path in index_split(split_dir))
    if args.limit:
        work = work[: args.limit]
    if not work:
        raise SystemExit("no labelled images found")
    print(f"{len(work)} labelled images across {len(args.splits)} split(s)")

    model, _device = _load_model(args.model)

    # --- Pass 1: metrics over everything, no depth retained -----------------
    records: list[dict] = []
    for i, (split, class_name, image_path) in enumerate(work, 1):
        depth, conf, is_metric = _predict(model, image_path, args.process_res)
        lum = _luminance_on_depth_grid(image_path, depth.shape)
        records.append({
            "id": scene_id(class_name, image_path, split, multi_split),
            "class": class_name,
            "split": split,
            "source_image": image_path.name,
            "width": int(depth.shape[1]),
            "height": int(depth.shape[0]),
            "is_metric": is_metric,
            **scene_metrics(depth, lum, conf),
        })
        if i % 50 == 0 or i == len(work):
            print(f"  [{i}/{len(work)}] metrics")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics_full.json").write_text(json.dumps(records, indent=2))

    aggregates = class_aggregates(records)
    print(f"\n{'class':<12} {'n':>4} {'plane_r2':>9} {'resid':>8} {'struct':>8} {'conf~tex':>9}")
    print("-" * 55)
    for name, entry in aggregates.items():
        def med(key: str) -> str:
            value = entry[key]["median"]
            return "  n/a" if value is None else f"{value:.3f}"
        print(f"{name:<12} {entry['n']:>4} {med('plane_r2'):>9} "
              f"{med('residual_relief'):>8} {med('structure_alignment'):>8} "
              f"{med('conf_texture_corr'):>9}")

    if args.metrics_only:
        print(f"\nmetrics-only: wrote {args.out / 'metrics_full.json'}")
        return

    # --- Pass 2: assets for the curated subset only -------------------------
    picked = curate(records, args.per_class_low, args.per_class_high)
    lookup = {scene_id(c, p, s, multi_split): (s, c, p) for s, c, p in work}
    print(f"\nexporting assets for {len(picked)} curated scenes")

    scenes_dir = args.out / "scenes"
    scenes_dir.mkdir(parents=True, exist_ok=True)
    for i, record in enumerate(picked, 1):
        _split, _class_name, image_path = lookup[record["id"]]
        depth, _conf, _ = _predict(model, image_path, args.process_res)

        scene_dir = scenes_dir / record["id"]
        scene_dir.mkdir(parents=True, exist_ok=True)
        depth.astype("<f4").tofile(scene_dir / "depth.bin")
        Image.open(image_path).convert("RGB").resize(
            (depth.shape[1], depth.shape[0]), Image.BILINEAR
        ).save(scene_dir / "rgb.jpg", quality=90)
        (scene_dir / "meta.json").write_text(json.dumps(record, indent=2))

        if not args.no_glb:
            # Detrended is the right height field for the GLB: a raw-depth mesh
            # is a tilted plane -- a faithful rendering of the artifact, but a
            # useless 3D asset.
            plane = (record["plane"]["a"], record["plane"]["b"], record["plane"]["c"])
            build_terrain(
                height_field(depth, plane),
                scene_dir / "rgb.jpg",
                scene_dir / "terrain.glb",
                res=args.glb_res,
                exaggeration=args.glb_exaggeration,
            )
        print(f"  [{i}/{len(picked)}] {record['id']}")

    (args.out / "index.json").write_text(json.dumps({
        "model": args.model,
        "process_res": args.process_res,
        "total_images_measured": len(records),
        "has_glb": not args.no_glb,
        "class_aggregates": aggregates,
        "scenes": [
            {k: r[k] for k in ("id", "class", "split", "source_image",
                               "plane_r2", "residual_relief",
                               "structure_alignment", "conf_texture_corr",
                               "row_corr", "col_corr")}
            for r in sorted(picked, key=lambda r: r["plane_r2"])
        ],
    }, indent=2))
    print(f"\nwrote {args.out / 'index.json'} ({len(picked)} scenes)")


if __name__ == "__main__":
    main()
