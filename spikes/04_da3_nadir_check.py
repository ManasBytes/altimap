"""Spike: does DA3-SMALL produce usable relief on nadir remote-sensing imagery?

The design doc (§1) states the core risk as an assertion rather than a measured
fact: monocular depth foundation models are trained on natural egocentric
imagery, so they emit relative affine-invariant depth and suffer a domain gap on
nadir views. Before any encoder-freezing or head-training work is scoped, it is
worth seeing what DA3-SMALL actually does to a nadir image.

The specific failure this looks for is the *egocentric ground-plane prior*: on a
forward-facing photo, depth almost always increases from the bottom of the frame
to the top, because that is where the ground recedes. A true nadir orthophoto has
no such structure -- every pixel is at roughly the same range from the sensor,
and what little depth variation exists is real relief (buildings, terrain). So a
strong depth-versus-row correlation on these images is the model hallucinating a
receding ground plane, not reading the scene. The column correlation is carried
alongside as a control: nadir imagery has no reason to favour either axis, so
|row| >> |col| is the fingerprint of the prior, while both being large just means
the image happens to contain a real gradient.

Reads the Roboflow "Remote Sensing Data v2" export, which carries a landscape
class per image (Agriculture, City, Mountain, ...). Sampling is stratified by
that class so the report says *where* the prior bites, not just whether it does.
DA3-SMALL is the model under test because it is Apache-2.0; DA3-LARGE is
cc-by-nc-4.0 and out of scope for this project (§4.1.1 licence note).

Relative depth only -- no metric claim is made or checked here. This spike
answers "is there signal", not "is the signal correct". Nothing in this dataset
has ground-truth elevation, so RMSE against a DSM is impossible by construction.

Throwaway.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

DEFAULT_DATASET = Path("/home/biplab-dev/Downloads/Remote Sensing Data.v2i.yolov8")
DEFAULT_MODEL = "depth-anything/DA3-SMALL"

# Order matches the `names` list in the dataset's data.yaml -- the YOLO class
# index is a position in that list, so this must not be re-sorted.
CLASS_NAMES = [
    "Agriculture",
    "Airport",
    "Beach",
    "City",
    "Desert",
    "Forest",
    "Grassland",
    "Highway",
    "Lake",
    "Mountain",
    "Parking",
    "Port",
    "Railway",
    "River",
]


def index_by_class(split_dir: Path) -> dict[str, list[Path]]:
    """Map landscape class -> image paths, from the YOLO label sidecars.

    Most label files in this export are empty (the annotator only labelled a
    subset), and the non-empty ones carry a single box covering nearly the whole
    frame -- this is a classification label wearing a detection label's clothes.
    Unlabelled images land under "_unlabelled" so the caller can still reach them
    but they never pollute a class bucket.

    Label files have no trailing newline, so they must be read individually.
    Concatenating them (`cat labels/*.txt`) silently welds the last row of one
    file onto the first row of the next and corrupts the class histogram.
    """
    by_class: dict[str, list[Path]] = defaultdict(list)
    for image_path in sorted((split_dir / "images").glob("*.jpg")):
        label_path = split_dir / "labels" / f"{image_path.stem}.txt"
        rows = []
        if label_path.exists():
            rows = [r for r in label_path.read_text().splitlines() if r.strip()]
        if not rows:
            by_class["_unlabelled"].append(image_path)
            continue
        # One box per image in practice; take the largest by area if that ever
        # stops being true, so the bucket reflects the dominant landscape.
        best = max(rows, key=lambda r: float(r.split()[3]) * float(r.split()[4]))
        index = int(best.split()[0])
        by_class[CLASS_NAMES[index]].append(image_path)
    return dict(by_class)


def stratified_sample(by_class: dict[str, list[Path]], per_class: int) -> list[tuple[str, Path]]:
    """Take the first `per_class` images of each labelled class, in name order.

    Deterministic on purpose: rerunning the spike after a code change must
    compare the same images, or the metrics move for the wrong reason.
    """
    sample = []
    for name in CLASS_NAMES:
        for image_path in by_class.get(name, [])[:per_class]:
            sample.append((name, image_path))
    return sample


def depth_stats(depth: np.ndarray) -> dict[str, float]:
    """Relief and axis-gradient diagnostics for one relative-depth map."""
    finite = np.isfinite(depth)
    values = depth[finite]
    if values.size == 0:
        return {"valid_frac": 0.0}

    median = float(np.median(values))
    rows, cols = np.mgrid[0 : depth.shape[0], 0 : depth.shape[1]]

    def corr(axis: np.ndarray) -> float:
        a = axis[finite].astype(np.float64)
        d = values.astype(np.float64)
        if a.std() == 0 or d.std() == 0:
            return 0.0
        return float(np.corrcoef(a, d)[0, 1])

    return {
        "valid_frac": float(finite.mean()),
        "min": float(values.min()),
        "max": float(values.max()),
        "median": median,
        # Relief relative to the scene's own scale. Affine-invariant depth has an
        # arbitrary unit, so an absolute std is meaningless -- this ratio is not.
        "relief_ratio": float(values.std() / median) if median != 0 else float("nan"),
        "row_corr": corr(rows),
        "col_corr": corr(cols),
    }


def save_preview(image_path: Path, depth: np.ndarray, conf: np.ndarray | None, out_path: Path, title: str) -> None:
    """Write an RGB | depth | confidence strip for eyeballing."""
    rgb = np.asarray(Image.open(image_path).convert("RGB"))
    panels = 2 if conf is None else 3
    fig, axes = plt.subplots(1, panels, figsize=(4.2 * panels, 4.6))

    axes[0].imshow(rgb)
    axes[0].set_title("RGB", fontsize=9)

    # turbo reversed: red = near, blue = far, matching how depth maps are
    # usually published. turbo also keeps small relief visible where the
    # perceptually-uniform maps flatten it -- which is the point here, since the
    # whole question is whether there is any relief to see.
    finite = np.isfinite(depth)
    shown = np.where(finite, depth, np.nan)
    axes[1].imshow(shown, cmap="turbo_r")
    axes[1].set_title("relative depth", fontsize=9)

    if conf is not None:
        axes[2].imshow(conf, cmap="magma")
        axes[2].set_title("confidence", fontsize=9)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--split", default="train", choices=["train", "valid", "test"])
    parser.add_argument("--per-class", type=int, default=2)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--process-res", type=int, default=504, help="DA3 internal working resolution")
    parser.add_argument("--out", type=Path, default=Path("spikes/out/da3_nadir"))
    parser.add_argument("--save-npy", action="store_true", help="also write raw float32 depth arrays")
    args = parser.parse_args()

    split_dir = args.dataset / args.split
    if not split_dir.is_dir():
        raise SystemExit(f"no such split: {split_dir}")

    by_class = index_by_class(split_dir)
    labelled = {k: v for k, v in by_class.items() if k != "_unlabelled"}
    print(f"{args.split}: {sum(len(v) for v in by_class.values())} images, "
          f"{sum(len(v) for v in labelled.values())} labelled across {len(labelled)} classes, "
          f"{len(by_class.get('_unlabelled', []))} unlabelled")
    missing = [c for c in CLASS_NAMES if c not in labelled]
    if missing:
        print(f"classes with no labelled image in this split: {', '.join(missing)}")

    sample = stratified_sample(by_class, args.per_class)
    if not sample:
        raise SystemExit("no labelled images to sample")
    print(f"sampling {len(sample)} images ({args.per_class}/class)\n")

    args.out.mkdir(parents=True, exist_ok=True)

    from depth_anything_3.api import DepthAnything3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"loading {args.model} onto {device}...")
    model = DepthAnything3.from_pretrained(args.model).to(device=device)

    records = []
    for i, (class_name, image_path) in enumerate(sample, 1):
        started = time.perf_counter()
        prediction = model.inference([str(image_path)], process_res=args.process_res)
        elapsed = time.perf_counter() - started

        depth = prediction.depth[0]
        conf = prediction.conf[0] if prediction.conf is not None else None
        stats = depth_stats(depth)

        # `Prediction.is_metric` is typed `int` but arrives as an empty addict
        # Dict for the non-metric models: output_processor.py:71 does
        # `getattr(model_output, "is_metric", 0)` on an addict Dict, whose
        # __getattr__ manufactures an empty Dict for any missing key instead of
        # raising, so the `0` default is unreachable. Empty Dict is falsy, so
        # truthiness reads correctly -- int() does not. Anything in AltiMap that
        # branches on this flag must not assume it is an int.
        is_metric = bool(prediction.is_metric)

        stem = f"{class_name}__{image_path.stem[:16]}"
        save_preview(
            image_path,
            depth,
            conf,
            args.out / f"{stem}.png",
            f"{class_name} | row_corr {stats['row_corr']:+.2f} | relief {stats['relief_ratio']:.3f}",
        )
        if args.save_npy:
            np.save(args.out / f"{stem}.npy", depth.astype(np.float32))

        records.append({
            "class": class_name,
            "image": image_path.name,
            "is_metric": is_metric,
            "seconds": round(elapsed, 3),
            **{k: round(v, 4) for k, v in stats.items()},
        })
        print(f"[{i:>2}/{len(sample)}] {class_name:<12} {elapsed:5.2f}s  "
              f"row_corr {stats['row_corr']:+.3f}  col_corr {stats['col_corr']:+.3f}  "
              f"relief {stats['relief_ratio']:.3f}")

    (args.out / "records.json").write_text(json.dumps(records, indent=2))

    print(f"\n{'class':<12} {'row_corr':>9} {'col_corr':>9} {'relief':>8} {'n':>3}")
    print("-" * 45)
    per_class = defaultdict(list)
    for r in records:
        per_class[r["class"]].append(r)
    for name in CLASS_NAMES:
        rows = per_class.get(name)
        if not rows:
            continue
        print(f"{name:<12} "
              f"{np.mean([r['row_corr'] for r in rows]):>+9.3f} "
              f"{np.mean([r['col_corr'] for r in rows]):>+9.3f} "
              f"{np.mean([r['relief_ratio'] for r in rows]):>8.3f} "
              f"{len(rows):>3}")

    abs_row = np.abs([r["row_corr"] for r in records])
    abs_col = np.abs([r["col_corr"] for r in records])
    # The ramp is not always vertical -- Mountain and Port tilt left-to-right --
    # so the dominant axis, not the row axis, is the honest summary statistic.
    dominant = np.maximum(abs_row, abs_col)
    print("-" * 45)
    print(f"mean |row_corr| {abs_row.mean():.3f}   mean |col_corr| {abs_col.mean():.3f}")
    print(f"images with |row_corr| > 0.5: {int((abs_row > 0.5).sum())}/{len(records)}")
    print(f"images with dominant-axis |corr| > 0.5: {int((dominant > 0.5).sum())}/{len(records)}"
          f"  (> 0.9: {int((dominant > 0.9).sum())})")
    print(f"row_corr negative (top of frame read as far): "
          f"{sum(1 for r in records if r['row_corr'] < 0)}/{len(records)}")
    print(f"is_metric: {set(r['is_metric'] for r in records)}  (False = relative depth, as expected)")
    print(f"mean {np.mean([r['seconds'] for r in records]):.2f}s/image on {device}")
    print(f"\nwrote previews + records.json to {args.out}")


if __name__ == "__main__":
    main()
