# Depth Diagnostics and rDSM Viewer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn DA3 relative-depth output into ramp-corrected diagnostics, exportable textured 3D terrain meshes, and an interactive three.js dashboard that renders the 28+ spike scenes as navigable relief.

**Architecture:** A pure-numpy metrics module (torch-free, so it tests in the main venv), a torch-side exporter that writes static per-scene assets, and a static three.js viewer that displaces a grid in the vertex shader from a float32 depth texture. Detrending happens in the shader from three plane coefficients, so raw ⟷ detrended is a uniform flip rather than a second dataset.

**Tech Stack:** Python 3.12, numpy, scipy, Pillow, trimesh, PyTorch + `depth_anything_3` (export only), three.js 0.185.1 (vendored), WebGL2.

## Global Constraints

- **`viewer/metrics.py` must not import torch, cv2, or `depth_anything_3`.** numpy and scipy only. This is what lets the existing pytest suite cover it in `.venv` without the 4.5 GB `.venv-da3`. Verified by a test that asserts the modules are absent from `sys.modules` after import.
- **three.js is pinned to 0.185.1 and vendored** under `viewer/web/vendor/three/`. No CDN reference at runtime — the brief requires standalone deployment.
- **No metre values anywhere in the UI or exports.** rDSM path only. Height axis is labelled "relative"; slope is labelled "display slope — depends on exaggeration".
- **Height is `depth_max - depth`.** Larger depth = further from sensor = lower ground. Reversing this renders cities as pits and looks plausible.
- **Python interpreters:** export runs under `.venv-da3/bin/python`; tests run under `.venv/bin/python`.
- **`Prediction.is_metric` is an empty addict `Dict`, not an int.** Test truthiness, never `int()`.
- **Do not modify `spikes/04_da3_nadir_check.py`** — it is the reproducible source of the 2026-08-24 findings doc.
- **Do not create or modify anything under `src/altimap/`** — Task 2 of the eval-harness plan is dispatched there.
- Dataset path: `/home/biplab-dev/Downloads/Remote Sensing Data.v2i.yolov8`.

## File Structure

| Path | Responsibility |
|---|---|
| `viewer/metrics.py` | Plane fit, detrend, per-scene metrics. Pure numpy/scipy. |
| `viewer/terrain.py` | Height-field → textured glTF mesh. Pillow + trimesh. |
| `viewer/export_scenes.py` | CLI: run DA3, curate, write static assets. |
| `viewer/web/index.html` | Dashboard shell, import map, panel layout. |
| `viewer/web/css/app.css` | Layout and theme. |
| `viewer/web/js/shaders.js` | GLSL source strings (shared `heightAt`). |
| `viewer/web/js/terrain.js` | Mesh, material, uniforms, scene loading. |
| `viewer/web/js/probe.js` | Ray-march picking + slope from the height field. |
| `viewer/web/js/app.js` | Wiring: scene list, controls, readouts, render loop. |
| `viewer/web/vendor/three/` | Pinned three.js files. |
| `tests/test_viewer_metrics.py` | Task 1 tests. |
| `tests/test_viewer_terrain.py` | Task 3 tests. |

---

### Task 1: `viewer/metrics.py` — plane fit and scene metrics

**Files:**
- Create: `viewer/metrics.py`
- Create: `viewer/__init__.py` (empty)
- Test: `tests/test_viewer_metrics.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `fit_plane(depth: np.ndarray) -> tuple[tuple[float, float, float], float]` returning `((a, b, c), r2)`
  - `plane_surface(plane: tuple[float, float, float], shape: tuple[int, int]) -> np.ndarray`
  - `detrend(depth: np.ndarray, plane: tuple[float, float, float]) -> np.ndarray`
  - `luminance(rgb: np.ndarray) -> np.ndarray`
  - `scene_metrics(depth, lum, conf=None) -> dict` with keys `plane`, `plane_r2`, `residual_relief`, `structure_alignment`, `conf_texture_corr`, `row_corr`, `col_corr`, `depth_min`, `depth_max`, `depth_median`, `residual_min`, `residual_max`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_viewer_metrics.py
import numpy as np
import pytest

from viewer.metrics import (
    detrend,
    fit_plane,
    luminance,
    plane_surface,
    scene_metrics,
)


def _ramp(h=32, w=48, a=2.0, b=3.0, c=5.0):
    """Exact plane z = a + b*u + c*v in normalized coords."""
    yy, xx = np.mgrid[0:h, 0:w]
    return a + b * (xx / w) + c * (yy / h)


def test_fit_plane_recovers_known_coefficients():
    (a, b, c), r2 = fit_plane(_ramp())
    assert a == pytest.approx(2.0, abs=1e-9)
    assert b == pytest.approx(3.0, abs=1e-9)
    assert c == pytest.approx(5.0, abs=1e-9)
    assert r2 == pytest.approx(1.0, abs=1e-12)


def test_detrend_of_exact_plane_is_zero():
    depth = _ramp()
    plane, _ = fit_plane(depth)
    assert np.abs(detrend(depth, plane)).max() < 1e-9


def test_plane_surface_reproduces_the_ramp():
    depth = _ramp()
    plane, _ = fit_plane(depth)
    assert np.allclose(plane_surface(plane, depth.shape), depth, atol=1e-9)


def test_constant_depth_yields_nan_r2_not_zero_and_not_raise():
    """SS_tot is 0 here. nan must propagate as 'unmeasurable', never as 0.0."""
    (_, b, c), r2 = fit_plane(np.full((16, 16), 3.0))
    assert np.isnan(r2)
    assert b == pytest.approx(0.0, abs=1e-9)
    assert c == pytest.approx(0.0, abs=1e-9)


def test_zero_median_depth_yields_nan_residual_relief():
    depth = np.zeros((16, 16))
    depth[:8] = -1.0
    depth[8:] = 1.0  # median is 0
    m = scene_metrics(depth, np.zeros((16, 16)))
    assert np.isnan(m["residual_relief"])


def test_scene_metrics_on_pure_ramp_reports_full_r2_and_no_residual():
    depth = _ramp()
    m = scene_metrics(depth, np.zeros_like(depth))
    assert m["plane_r2"] == pytest.approx(1.0, abs=1e-12)
    assert m["residual_relief"] == pytest.approx(0.0, abs=1e-9)
    assert m["col_corr"] > 0.5   # b > 0, depth rises with x
    assert m["row_corr"] > 0.5   # c > 0, depth rises with y


def test_structure_alignment_is_high_when_relief_sits_on_image_edges():
    """A bump in depth co-located with a bump in brightness should align."""
    depth = _ramp(64, 64)
    lum = np.zeros((64, 64))
    depth[20:30, 20:30] += 4.0
    lum[20:30, 20:30] = 1.0
    m = scene_metrics(depth, lum)
    assert m["structure_alignment"] > 0.5


def test_structure_alignment_is_low_when_relief_and_edges_are_unrelated():
    depth = _ramp(64, 64)
    lum = np.zeros((64, 64))
    depth[10:20, 10:20] += 4.0
    lum[40:50, 40:50] = 1.0
    m = scene_metrics(depth, lum)
    assert abs(m["structure_alignment"]) < 0.2


def test_conf_texture_corr_is_nan_without_confidence():
    m = scene_metrics(_ramp(), np.zeros((32, 48)))
    assert np.isnan(m["conf_texture_corr"])


def test_luminance_uses_rec601_weights():
    rgb = np.zeros((1, 1, 3))
    rgb[0, 0] = [1.0, 0.0, 0.0]
    assert luminance(rgb)[0, 0] == pytest.approx(0.299)


def test_metrics_module_is_torch_free():
    """Load-bearing: this module must test in .venv without .venv-da3."""
    import sys

    import viewer.metrics  # noqa: F401

    assert "torch" not in sys.modules
    assert "cv2" not in sys.modules
    assert "depth_anything_3" not in sys.modules
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_viewer_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'viewer'`

- [ ] **Step 3: Write the implementation**

```python
# viewer/metrics.py
"""Ramp-corrected diagnostics for monocular depth on nadir imagery.

DA3 fits a tilted plane to nadir scenes instead of reading relief (see
docs/superpowers/spikes/2026-08-24-da3-nadir-domain-gap.md). Everything here is
built around separating that plane from whatever survives underneath it.

Deliberately imports only numpy and scipy -- no torch, no cv2, no
depth_anything_3 -- so the main test venv can cover it without the 4.5 GB
inference venv. There is a test asserting this. Do not add heavy imports.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

NAN = float("nan")


def _normalized_grid(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """(u, v) in [0, 1), u along columns, v along rows."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    return xx / w, yy / h


def fit_plane(depth: np.ndarray) -> tuple[tuple[float, float, float], float]:
    """Least-squares plane `z = a + b*u + c*v` in normalized coordinates.

    Normalized rather than pixel coordinates so the coefficients are
    resolution-independent -- the shader applies them against UVs directly.

    Returns ((a, b, c), r2). r2 is nan for constant input, where the fit is
    exact but explains nothing, so a value of 1.0 would be actively misleading.
    """
    u, v = _normalized_grid(depth.shape)
    z = depth.ravel().astype(np.float64)
    design = np.column_stack([np.ones(z.size), u.ravel(), v.ravel()])
    coef, *_ = np.linalg.lstsq(design, z, rcond=None)

    residual = z - design @ coef
    ss_tot = float(((z - z.mean()) ** 2).sum())
    r2 = NAN if ss_tot == 0.0 else 1.0 - float((residual**2).sum()) / ss_tot
    return (float(coef[0]), float(coef[1]), float(coef[2])), r2


def plane_surface(plane: tuple[float, float, float], shape: tuple[int, int]) -> np.ndarray:
    a, b, c = plane
    u, v = _normalized_grid(shape)
    return a + b * u + c * v


def detrend(depth: np.ndarray, plane: tuple[float, float, float]) -> np.ndarray:
    """Depth with the fitted ramp removed -- the residual relief."""
    return depth - plane_surface(plane, depth.shape)


def luminance(rgb: np.ndarray) -> np.ndarray:
    """Rec. 601 luma. Accepts float [0,1] or uint8 [0,255]; returns input scale."""
    arr = rgb.astype(np.float64)
    return 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]


def gradient_magnitude(image: np.ndarray) -> np.ndarray:
    """Sobel gradient magnitude, edge-replicated."""
    arr = image.astype(np.float64)
    gx = ndimage.sobel(arr, axis=1, mode="nearest")
    gy = ndimage.sobel(arr, axis=0, mode="nearest")
    return np.hypot(gx, gy)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Correlation that returns nan instead of raising on degenerate input."""
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if x.size < 2 or x.std() == 0.0 or y.std() == 0.0:
        return NAN
    return float(np.corrcoef(x, y)[0, 1])


def scene_metrics(
    depth: np.ndarray,
    lum: np.ndarray,
    conf: np.ndarray | None = None,
) -> dict:
    """Full diagnostic record for one scene.

    `lum` must already be on the depth grid -- callers resize RGB before
    calling, since RGB arrives at 640 and depth at 504.
    """
    plane, r2 = fit_plane(depth)
    residual = detrend(depth, plane)
    median = float(np.median(depth))
    u, v = _normalized_grid(depth.shape)
    lum_edges = gradient_magnitude(lum)

    return {
        "plane": {"a": plane[0], "b": plane[1], "c": plane[2]},
        # Fraction of the scene DA3 explained with a ramp. 1 - plane_r2 is the
        # fraction that is something else. Replaces relief_ratio, which was
        # confounded with the artifact it was meant to detect.
        "plane_r2": r2,
        "residual_relief": float(residual.std() / median) if median != 0.0 else NAN,
        # Does residual relief land on image edges? Real structure should.
        "structure_alignment": _pearson(gradient_magnitude(residual), lum_edges),
        # Is confidence a reliability estimate or just a texture detector?
        "conf_texture_corr": _pearson(conf, lum_edges) if conf is not None else NAN,
        "row_corr": _pearson(v, depth),
        "col_corr": _pearson(u, depth),
        "depth_min": float(depth.min()),
        "depth_max": float(depth.max()),
        "depth_median": median,
        "residual_min": float(residual.min()),
        "residual_max": float(residual.max()),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_viewer_metrics.py -v`
Expected: PASS, 11 passed

- [ ] **Step 5: Confirm the existing suite still passes**

Run: `.venv/bin/python -m pytest -q`
Expected: all tests pass, no collection errors

- [ ] **Step 6: Commit**

```bash
git add viewer/__init__.py viewer/metrics.py tests/test_viewer_metrics.py
git commit -m "feat(viewer): add ramp-corrected depth metrics module"
```

---

### Task 2: `viewer/export_scenes.py` — DA3 run, curation, static assets

**Files:**
- Create: `viewer/export_scenes.py`

**Interfaces:**
- Consumes: `viewer.metrics.scene_metrics`, `viewer.metrics.luminance`
- Produces:
  - `viewer/data/metrics_full.json` — one record per image over the whole run
  - `viewer/data/index.json` — curated scene list + per-class aggregates
  - `viewer/data/scenes/<scene_id>/{depth.bin,rgb.jpg,meta.json}`
  - `curate(records, per_class_low=2, per_class_high=1) -> list[dict]`

Two passes on purpose: pass 1 computes metrics over every image, pass 2 re-runs DA3 on only the curated ~42 to write assets. Re-running 42 images costs ~5 s and avoids holding 1000 × 504² float32 (~1 GB) in memory.

- [ ] **Step 1: Write the module**

```python
# viewer/export_scenes.py
"""Run DA3 over the dataset, curate scenes, write static viewer assets.

Two passes. Pass 1 computes metrics for every image and keeps no depth. Pass 2
re-runs DA3 on only the curated subset to write assets. Holding 1000 depth maps
would cost ~1 GB of RAM; re-running 42 images costs ~5 s. The model is
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
    """(class_name, image_path) for every image in a split.

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
    parser.add_argument("--out", type=Path, default=Path("viewer/data"))
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

    model, device = _load_model(args.model)

    # --- Pass 1: metrics over everything, no depth retained -----------------
    records: list[dict] = []
    for i, (split, class_name, image_path) in enumerate(work, 1):
        depth, conf, is_metric = _predict(model, image_path, args.process_res)
        lum = _luminance_on_depth_grid(image_path, depth.shape)
        record = {
            "id": scene_id(class_name, image_path, split, multi_split),
            "class": class_name,
            "split": split,
            "source_image": image_path.name,
            "width": int(depth.shape[1]),
            "height": int(depth.shape[0]),
            "is_metric": is_metric,
            **scene_metrics(depth, lum, conf),
        }
        records.append(record)
        if i % 50 == 0 or i == len(work):
            print(f"  [{i}/{len(work)}] metrics")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics_full.json").write_text(json.dumps(records, indent=2))

    aggregates = class_aggregates(records)
    print(f"\n{'class':<12} {'n':>4} {'plane_r2':>9} {'resid':>8} {'struct':>8}")
    print("-" * 45)
    for name, entry in aggregates.items():
        print(f"{name:<12} {entry['n']:>4} "
              f"{entry['plane_r2']['median']:>9.3f} "
              f"{entry['residual_relief']['median']:>8.3f} "
              f"{entry['structure_alignment']['median']:>8.3f}")

    if args.metrics_only:
        print(f"\nmetrics-only: wrote {args.out / 'metrics_full.json'}")
        return

    # --- Pass 2: assets for the curated subset only -------------------------
    picked = curate(records, args.per_class_low, args.per_class_high)
    by_id = {(s, c, p): (s, c, p) for s, c, p in work}
    lookup = {scene_id(c, p, s, multi_split): (s, c, p) for s, c, p in by_id}
    print(f"\nexporting assets for {len(picked)} curated scenes")

    scenes_dir = args.out / "scenes"
    scenes_dir.mkdir(parents=True, exist_ok=True)
    for i, record in enumerate(picked, 1):
        split, class_name, image_path = lookup[record["id"]]
        depth, _conf, _ = _predict(model, image_path, args.process_res)

        scene_dir = scenes_dir / record["id"]
        scene_dir.mkdir(parents=True, exist_ok=True)
        depth.astype("<f4").tofile(scene_dir / "depth.bin")
        Image.open(image_path).convert("RGB").resize(
            (depth.shape[1], depth.shape[0]), Image.BILINEAR
        ).save(scene_dir / "rgb.jpg", quality=90)
        (scene_dir / "meta.json").write_text(json.dumps(record, indent=2))
        print(f"  [{i}/{len(picked)}] {record['id']}")

    (args.out / "index.json").write_text(json.dumps({
        "model": args.model,
        "process_res": args.process_res,
        "total_images_measured": len(records),
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
```

- [ ] **Step 2: Smoke-run on a small slice**

Run: `.venv-da3/bin/python -W ignore::SyntaxWarning -m viewer.export_scenes --limit 20 --out /tmp/claude-1000/-home-biplab-dev-Projects/46714303-f1a6-48cf-ac21-b616dbe00953/scratchpad/vdata`
Expected: prints a per-class table, then "exporting assets for N curated scenes", then an index.json line. No traceback.

- [ ] **Step 3: Verify the written assets parse**

```bash
.venv-da3/bin/python - <<'PY'
import json, numpy as np, pathlib
root = pathlib.Path("/tmp/claude-1000/-home-biplab-dev-Projects/46714303-f1a6-48cf-ac21-b616dbe00953/scratchpad/vdata")
index = json.loads((root / "index.json").read_text())
print("scenes:", len(index["scenes"]))
first = index["scenes"][0]["id"]
meta = json.loads((root / "scenes" / first / "meta.json").read_text())
depth = np.fromfile(root / "scenes" / first / "depth.bin", dtype="<f4")
assert depth.size == meta["width"] * meta["height"], (depth.size, meta["width"], meta["height"])
assert np.isfinite(depth).all()
print("ok:", first, depth.shape, depth.min(), depth.max())
PY
```
Expected: `ok: <scene id> ...` with finite min/max, no assertion error

- [ ] **Step 4: Full run over all three splits**

Run: `.venv-da3/bin/python -W ignore::SyntaxWarning -m viewer.export_scenes --splits train valid test --out viewer/data`
Expected: ~1000 images measured in ~2 min, per-class table with n ≈ 70 per class, ~42 scenes exported

- [ ] **Step 5: Commit**

```bash
git add viewer/export_scenes.py
git commit -m "feat(viewer): export DA3 scenes and full-dataset metrics"
```

---

### Task 3: `viewer/terrain.py` — textured glTF terrain meshes

**Files:**
- Create: `viewer/terrain.py`
- Modify: `viewer/export_scenes.py` (add `--glb-res`, `--no-glb`, `--glb-exaggeration`; call the builder in pass 2)
- Test: `tests/test_viewer_terrain.py`

**Interfaces:**
- Consumes: `viewer.metrics.detrend`
- Produces: `build_terrain(height01: np.ndarray, texture_path: Path, out_path: Path, res: int = 256, exaggeration: float = 0.15) -> None`, and `height_field(depth, plane=None) -> np.ndarray` returning [0,1] with 1 = highest ground.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_viewer_terrain.py
import numpy as np
import pytest

from viewer.metrics import fit_plane
from viewer.terrain import height_field


def test_height_field_inverts_depth_so_far_is_low():
    """Larger depth = further from sensor = LOWER ground. Reversing this
    renders every city as a pit, and looks plausible until you notice."""
    depth = np.array([[1.0, 2.0], [3.0, 4.0]])
    h = height_field(depth)
    assert h[0, 0] == pytest.approx(1.0)  # nearest -> highest
    assert h[1, 1] == pytest.approx(0.0)  # furthest -> lowest


def test_height_field_is_normalized_to_unit_range():
    h = height_field(np.random.default_rng(0).normal(size=(32, 32)))
    assert h.min() == pytest.approx(0.0)
    assert h.max() == pytest.approx(1.0)


def test_height_field_of_constant_depth_is_flat_not_nan():
    h = height_field(np.full((8, 8), 2.0))
    assert np.isfinite(h).all()
    assert np.allclose(h, 0.0)


def test_height_field_with_plane_removes_the_ramp():
    yy, xx = np.mgrid[0:32, 0:32]
    depth = 1.0 + 2.0 * (xx / 32)          # pure ramp
    depth[10:14, 10:14] -= 0.5             # a bump toward the sensor
    plane, _ = fit_plane(depth)
    h = height_field(depth, plane)
    # The bump should be the highest thing left once the ramp is gone.
    assert h[11, 11] > h[0, 0]
    assert h[11, 11] == pytest.approx(1.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_viewer_terrain.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'viewer.terrain'`

- [ ] **Step 3: Write the implementation**

```python
# viewer/terrain.py
"""Height field -> textured glTF terrain mesh.

The GLB is a standalone deliverable: openable in Blender or any glTF viewer
without the dashboard, and it is what satisfies the brief's "project the
original optical image onto a generated 3D terrain mesh".

trimesh and Pillow are imported lazily inside build_terrain so that
height_field stays testable in the main venv.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from viewer.metrics import detrend


def height_field(depth: np.ndarray, plane: tuple[float, float, float] | None = None) -> np.ndarray:
    """Depth -> height in [0, 1], where 1 is the highest ground.

    Depth is distance from the sensor, so for a nadir view larger depth means
    LOWER ground -- hence the inversion. Constant input returns all zeros
    rather than nan, so a flat scene renders flat instead of vanishing.
    """
    field = detrend(depth, plane) if plane is not None else np.asarray(depth, dtype=np.float64)
    lo, hi = float(field.min()), float(field.max())
    if hi == lo:
        return np.zeros_like(field, dtype=np.float64)
    return (hi - field) / (hi - lo)


def _resample(field: np.ndarray, res: int) -> np.ndarray:
    from PIL import Image

    if field.shape == (res, res):
        return field
    image = Image.fromarray(field.astype(np.float32), mode="F")
    return np.asarray(image.resize((res, res), Image.BILINEAR), dtype=np.float64)


def build_terrain(
    height01: np.ndarray,
    texture_path: Path,
    out_path: Path,
    res: int = 256,
    exaggeration: float = 0.15,
) -> None:
    """Write a textured, displaced grid as GLB.

    The grid spans [-0.5, 0.5] in X and Z, so every scene arrives at the same
    footprint regardless of source resolution and exaggeration means the same
    thing everywhere. UVs are identity per design doc 5.1 -- DSM and RGB share
    one grid, so there is no reprojection step and no reprojection error.
    """
    import trimesh
    from PIL import Image

    field = _resample(np.asarray(height01, dtype=np.float64), res)

    lin = np.linspace(-0.5, 0.5, res)
    xx, zz = np.meshgrid(lin, lin)
    vertices = np.column_stack([
        xx.ravel(),
        (field * exaggeration).ravel(),
        zz.ravel(),
    ]).astype(np.float32)

    # Two triangles per quad, wound so normals point up (+Y).
    idx = np.arange(res * res).reshape(res, res)
    tl, tr = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel()
    bl, br = idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    faces = np.concatenate([
        np.column_stack([tl, bl, tr]),
        np.column_stack([tr, bl, br]),
    ]).astype(np.int64)

    u = (xx.ravel() + 0.5)
    v = 1.0 - (zz.ravel() + 0.5)
    uv = np.column_stack([u, v]).astype(np.float32)

    material = trimesh.visual.material.SimpleMaterial(image=Image.open(texture_path))
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        visual=trimesh.visual.TextureVisuals(uv=uv, material=material),
        process=False,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(out_path)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_viewer_terrain.py -v`
Expected: PASS, 4 passed

- [ ] **Step 5: Wire GLB export into pass 2**

In `viewer/export_scenes.py`, add to `main()`'s argument block:

```python
    parser.add_argument("--no-glb", action="store_true")
    parser.add_argument("--glb-res", type=int, default=256)
    parser.add_argument("--glb-exaggeration", type=float, default=0.15)
```

Add the import at the top of the file:

```python
from viewer.terrain import build_terrain, height_field
```

And inside the pass-2 loop, immediately after the `meta.json` write:

```python
        if not args.no_glb:
            plane = (record["plane"]["a"], record["plane"]["b"], record["plane"]["c"])
            build_terrain(
                height_field(depth, plane),
                scene_dir / "rgb.jpg",
                scene_dir / "terrain.glb",
                res=args.glb_res,
                exaggeration=args.glb_exaggeration,
            )
```

Detrended is the right height field for the GLB: a raw-depth mesh is a tilted
plane, which is a faithful rendering of the artifact but a useless 3D asset.

- [ ] **Step 6: Verify a GLB is produced and loads**

```bash
.venv-da3/bin/python -W ignore::SyntaxWarning -m viewer.export_scenes \
  --limit 20 --glb-res 128 \
  --out /tmp/claude-1000/-home-biplab-dev-Projects/46714303-f1a6-48cf-ac21-b616dbe00953/scratchpad/vdata2
.venv-da3/bin/python - <<'PY'
import pathlib, trimesh
root = pathlib.Path("/tmp/claude-1000/-home-biplab-dev-Projects/46714303-f1a6-48cf-ac21-b616dbe00953/scratchpad/vdata2/scenes")
path = sorted(root.glob("*/terrain.glb"))[0]
scene = trimesh.load(path)
mesh = list(scene.geometry.values())[0] if hasattr(scene, "geometry") else scene
print(path.name, "verts", len(mesh.vertices), "faces", len(mesh.faces))
assert len(mesh.faces) == 2 * 127 * 127
print("bounds", mesh.bounds)
PY
```
Expected: `verts 16384 faces 32258`, bounds spanning about -0.5..0.5 in X and Z

- [ ] **Step 7: Commit**

```bash
git add viewer/terrain.py tests/test_viewer_terrain.py viewer/export_scenes.py
git commit -m "feat(viewer): export textured glTF terrain meshes"
```

---

### Task 4: Vendor three.js and build the dashboard shell

**Files:**
- Create: `viewer/web/index.html`, `viewer/web/css/app.css`, `viewer/web/js/app.js`
- Create: `viewer/web/vendor/three/` (4 downloaded files)

**Interfaces:**
- Consumes: `viewer/data/index.json`
- Produces: a served page listing scenes sorted by ascending `plane_r2`, with an empty canvas ready for Task 5.

- [ ] **Step 1: Vendor three.js 0.185.1**

```bash
mkdir -p viewer/web/vendor/three/controls
BASE=https://unpkg.com/three@0.185.1
curl -fsSL "$BASE/build/three.module.js"  -o viewer/web/vendor/three/three.module.js
curl -fsSL "$BASE/build/three.core.js"    -o viewer/web/vendor/three/three.core.js
curl -fsSL "$BASE/examples/jsm/controls/OrbitControls.js"      -o viewer/web/vendor/three/controls/OrbitControls.js
curl -fsSL "$BASE/examples/jsm/controls/PointerLockControls.js" -o viewer/web/vendor/three/controls/PointerLockControls.js
ls -la viewer/web/vendor/three viewer/web/vendor/three/controls
```
Expected: `three.module.js` ~650 KB, `three.core.js` ~1.4 MB, `OrbitControls.js` ~40 KB, `PointerLockControls.js` ~6 KB

- [ ] **Step 2: Write `viewer/web/index.html`**

```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AltiMap — rDSM viewer</title>
<link rel="stylesheet" href="./css/app.css">
<script type="importmap">
{
  "imports": {
    "three": "./vendor/three/three.module.js",
    "three/webgpu": "./vendor/three/three.module.js",
    "three/tsl": "./vendor/three/three.module.js",
    "three/addons/": "./vendor/three/"
  }
}
</script>
</head>
<body>
<div id="app">
  <aside id="scenes">
    <header>
      <h1>AltiMap</h1>
      <p class="sub">relative DSM — no metric scale</p>
      <label class="sortrow">sort
        <select id="sort">
          <option value="plane_r2">least ramp first</option>
          <option value="structure_alignment">most structure first</option>
          <option value="class">class A–Z</option>
        </select>
      </label>
    </header>
    <ul id="scene-list"></ul>
  </aside>

  <main id="stage">
    <canvas id="view"></canvas>
    <div id="overlay"><p id="status">loading…</p></div>
  </main>

  <aside id="controls">
    <section>
      <h2>Surface</h2>
      <div class="seg" id="mode">
        <button data-mode="raw" class="on">raw</button>
        <button data-mode="detrended">detrended</button>
      </div>
      <label>exaggeration <output id="exag-out">0.15</output>
        <input type="range" id="exag" min="0" max="1" step="0.01" value="0.15">
      </label>
      <label>colormap
        <select id="cmap">
          <option value="0">satellite RGB</option>
          <option value="1">height (turbo)</option>
          <option value="2">slope shade</option>
        </select>
      </label>
    </section>

    <section>
      <h2>Navigation</h2>
      <div class="seg" id="nav">
        <button data-nav="orbit" class="on">orbit</button>
        <button data-nav="fly">fly</button>
      </div>
      <p class="hint" id="nav-hint">drag to orbit, scroll to zoom</p>
    </section>

    <section>
      <h2>Probe</h2>
      <p class="hint">click the terrain</p>
      <dl id="probe">
        <dt>height</dt><dd id="p-height">—</dd>
        <dt>display slope</dt><dd id="p-slope">—</dd>
        <dt class="warn" id="slope-note">depends on exaggeration; not a ground slope</dt><dd></dd>
      </dl>
    </section>

    <section>
      <h2>Scene metrics</h2>
      <dl id="metrics">
        <dt>plane R²</dt><dd id="m-r2">—</dd>
        <dt>residual relief</dt><dd id="m-relief">—</dd>
        <dt>structure alignment</dt><dd id="m-struct">—</dd>
        <dt>conf ↔ texture</dt><dd id="m-conf">—</dd>
      </dl>
      <p class="hint" id="m-context">—</p>
      <a id="glb" class="dl" href="#" download>download terrain.glb</a>
    </section>
  </aside>
</div>
<script type="module" src="./js/app.js"></script>
</body>
</html>
```

- [ ] **Step 3: Write `viewer/web/css/app.css`**

```css
:root {
  --bg: #0e1116; --panel: #151a21; --line: #232b36;
  --fg: #e6edf3; --dim: #8b98a5; --accent: #4c9aff; --warn: #d9a441;
}
* { box-sizing: border-box; }
html, body { margin: 0; height: 100%; background: var(--bg); color: var(--fg);
  font: 13px/1.5 ui-sans-serif, system-ui, -apple-system, sans-serif; }
#app { display: grid; grid-template-columns: 260px 1fr 260px; height: 100%; }
aside { background: var(--panel); overflow-y: auto; }
#scenes { border-right: 1px solid var(--line); }
#controls { border-left: 1px solid var(--line); padding: 14px; }
header { padding: 14px; border-bottom: 1px solid var(--line); }
h1 { margin: 0; font-size: 15px; letter-spacing: .02em; }
h2 { font-size: 11px; text-transform: uppercase; letter-spacing: .08em;
  color: var(--dim); margin: 18px 0 8px; }
.sub { margin: 2px 0 10px; color: var(--dim); font-size: 11px; }
.sortrow { display: block; font-size: 11px; color: var(--dim); }
select, input[type=range] { width: 100%; margin-top: 4px; }
select { background: #0d1117; color: var(--fg); border: 1px solid var(--line);
  border-radius: 5px; padding: 4px; }
#scene-list { list-style: none; margin: 0; padding: 0; }
#scene-list li { display: flex; gap: 9px; padding: 8px 12px; cursor: pointer;
  border-bottom: 1px solid var(--line); align-items: center; }
#scene-list li:hover { background: #1b222c; }
#scene-list li.on { background: #1d2937; box-shadow: inset 2px 0 0 var(--accent); }
#scene-list img { width: 44px; height: 44px; object-fit: cover; border-radius: 3px;
  background: #000; }
#scene-list .name { font-size: 12px; }
#scene-list .meta { color: var(--dim); font-size: 10.5px; font-variant-numeric: tabular-nums; }
#scene-list li.err .name { color: var(--warn); }
#stage { position: relative; }
canvas { display: block; width: 100%; height: 100%; }
#overlay { position: absolute; inset: 0; display: grid; place-items: center;
  pointer-events: none; }
#overlay.hidden { display: none; }
#status { background: #0009; padding: 8px 14px; border-radius: 6px; color: var(--dim); }
.seg { display: flex; gap: 0; margin-bottom: 10px; }
.seg button { flex: 1; background: #0d1117; color: var(--dim); border: 1px solid var(--line);
  padding: 6px; cursor: pointer; font-size: 12px; }
.seg button:first-child { border-radius: 5px 0 0 5px; }
.seg button:last-child { border-radius: 0 5px 5px 0; }
.seg button.on { background: var(--accent); color: #06101f; border-color: var(--accent); }
label { display: block; margin: 10px 0; font-size: 11px; color: var(--dim); }
output { float: right; color: var(--fg); font-variant-numeric: tabular-nums; }
dl { display: grid; grid-template-columns: 1fr auto; gap: 3px 8px; margin: 0; }
dt { color: var(--dim); font-size: 11.5px; }
dd { margin: 0; text-align: right; font-variant-numeric: tabular-nums; }
dt.warn { grid-column: 1 / -1; color: var(--warn); font-size: 10.5px;
  line-height: 1.35; margin-top: 3px; }
.hint { color: var(--dim); font-size: 10.5px; margin: 6px 0; }
.dl { display: block; margin-top: 12px; padding: 7px; text-align: center;
  border: 1px solid var(--line); border-radius: 5px; color: var(--accent);
  text-decoration: none; font-size: 11.5px; }
.dl:hover { background: #1b222c; }
.dl.off { opacity: .35; pointer-events: none; }
```

- [ ] **Step 4: Write `viewer/web/js/app.js` (shell only — rendering lands in Task 5)**

```js
const DATA = '../data';

const els = {
  list: document.getElementById('scene-list'),
  sort: document.getElementById('sort'),
  status: document.getElementById('status'),
  overlay: document.getElementById('overlay'),
  mR2: document.getElementById('m-r2'),
  mRelief: document.getElementById('m-relief'),
  mStruct: document.getElementById('m-struct'),
  mConf: document.getElementById('m-conf'),
  mContext: document.getElementById('m-context'),
  glb: document.getElementById('glb'),
};

const state = { index: null, scenes: [], current: null };

const fmt = (v, d = 3) =>
  v === null || v === undefined || Number.isNaN(v) ? 'n/a' : Number(v).toFixed(d);

function sortScenes(key) {
  const s = [...state.index.scenes];
  if (key === 'class') return s.sort((a, b) => a.class.localeCompare(b.class));
  if (key === 'structure_alignment')
    return s.sort((a, b) => (b.structure_alignment ?? -Infinity) - (a.structure_alignment ?? -Infinity));
  return s.sort((a, b) => (a.plane_r2 ?? Infinity) - (b.plane_r2 ?? Infinity));
}

function renderList() {
  const scenes = sortScenes(els.sort.value);
  els.list.replaceChildren(...scenes.map((scene) => {
    const li = document.createElement('li');
    li.dataset.id = scene.id;
    li.innerHTML = `
      <img alt="" loading="lazy" src="${DATA}/scenes/${scene.id}/rgb.jpg">
      <div>
        <div class="name">${scene.class}</div>
        <div class="meta">R² ${fmt(scene.plane_r2, 2)} · str ${fmt(scene.structure_alignment, 2)}</div>
      </div>`;
    li.addEventListener('click', () => selectScene(scene.id));
    return li;
  }));
  if (!state.current && scenes.length) selectScene(scenes[0].id);
  else markActive();
}

function markActive() {
  for (const li of els.list.children) li.classList.toggle('on', li.dataset.id === state.current);
}

function showMetrics(scene) {
  els.mR2.textContent = fmt(scene.plane_r2);
  els.mRelief.textContent = fmt(scene.residual_relief);
  els.mStruct.textContent = fmt(scene.structure_alignment);
  els.mConf.textContent = fmt(scene.conf_texture_corr);

  // Situate this scene against its class distribution from the full 1000-image
  // run, so a single number is never read without its spread.
  const agg = state.index.class_aggregates?.[scene.class];
  els.mContext.textContent = agg
    ? `${scene.class}: n=${agg.n}, class median R² ${fmt(agg.plane_r2.median, 2)} (IQR ${fmt(agg.plane_r2.q1, 2)}–${fmt(agg.plane_r2.q3, 2)})`
    : '';

  els.glb.href = `${DATA}/scenes/${scene.id}/terrain.glb`;
  els.glb.classList.remove('off');
}

async function selectScene(id) {
  state.current = id;
  markActive();
  const scene = state.index.scenes.find((s) => s.id === id);
  showMetrics(scene);
  els.overlay.classList.remove('hidden');
  els.status.textContent = `loading ${scene.class}…`;
  try {
    await window.__loadScene?.(id, scene);
    els.overlay.classList.add('hidden');
  } catch (err) {
    els.status.textContent = `failed to load ${id}: ${err.message}`;
    els.list.querySelector(`[data-id="${id}"]`)?.classList.add('err');
  }
}

async function boot() {
  try {
    const res = await fetch(`${DATA}/index.json`);
    if (!res.ok) throw new Error(`index.json ${res.status}`);
    state.index = await res.json();
  } catch (err) {
    els.status.textContent =
      `could not load ${DATA}/index.json — run viewer/export_scenes.py first (${err.message})`;
    return;
  }
  els.sort.addEventListener('change', renderList);
  renderList();
}

boot();
export { state, els, fmt };
```

- [ ] **Step 5: Serve and confirm the shell renders**

```bash
cd viewer/web && python3 -m http.server 8777 &
sleep 1
curl -sf http://localhost:8777/index.html > /dev/null && echo "shell serves OK"
curl -sf http://localhost:8777/vendor/three/three.module.js | head -c 40
```
Expected: `shell serves OK` then the first bytes of the three.js module. Open http://localhost:8777/ — the left panel lists scenes sorted least-ramp-first, the centre is empty, the right panel shows metrics for the first scene.

- [ ] **Step 6: Commit**

```bash
git add viewer/web
git commit -m "feat(viewer): dashboard shell with vendored three.js 0.185.1"
```

---

### Task 5: Rendering core — float32 depth texture, vertex-shader displacement

**Files:**
- Create: `viewer/web/js/shaders.js`, `viewer/web/js/terrain.js`
- Modify: `viewer/web/js/app.js` (wire controls to uniforms)

**Interfaces:**
- Consumes: `viewer/data/scenes/<id>/{depth.bin,rgb.jpg,meta.json}`
- Produces: `initViewer(canvas) -> {loadScene(id, scene), setMode(m), setExaggeration(x), setColormap(c), heightAt(u,v), meta}` and installs `window.__loadScene`.

Per design doc §5.1: depth uploads as a float32 `DataTexture`, the mesh is a static grid lifted in the vertex shader, normals come from finite differences in the fragment shader. Swapping scenes is one texture upload; the exaggeration slider is free; slope shading costs nothing extra.

- [ ] **Step 1: Write `viewer/web/js/shaders.js`**

```js
// heightAt is needed in BOTH stages -- the vertex stage to displace, the
// fragment stage to derive normals by finite differences. GLSL has no shared
// includes here, so the source is defined once and injected into both.
export const HEIGHT_FN = /* glsl */`
uniform sampler2D uDepth;
uniform vec3  uPlane;      // a, b, c of z = a + b*u + c*v
uniform vec2  uRawRange;   // depth_min, depth_max
uniform vec2  uResRange;   // residual_min, residual_max
uniform float uExag;
uniform float uDetrend;    // 0 = raw, 1 = detrended

float heightAt(vec2 uv) {
  float d = texture2D(uDepth, uv).r;
  if (uDetrend > 0.5) {
    float r = d - (uPlane.x + uPlane.y * uv.x + uPlane.z * uv.y);
    return (uResRange.y - r) / max(uResRange.y - uResRange.x, 1e-8);
  }
  // Depth is distance from sensor: larger depth = lower ground. Hence max - d.
  return (uRawRange.y - d) / max(uRawRange.y - uRawRange.x, 1e-8);
}`;

export const VERT = /* glsl */`
${HEIGHT_FN}
varying vec2  vUv;
varying float vH;
void main() {
  vUv = uv;
  vH = heightAt(uv);
  // PlaneGeometry lies in XY; the mesh is rotated -90deg about X, so local +Z
  // becomes world up.
  vec3 p = vec3(position.xy, vH * uExag);
  gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0);
}`;

export const FRAG = /* glsl */`
${HEIGHT_FN}
uniform sampler2D uRgb;
uniform vec2  uTexel;    // 1/width, 1/height
uniform float uCmap;     // 0 rgb, 1 turbo, 2 slope shade
varying vec2  vUv;
varying float vH;

// Polynomial turbo approximation -- compact and good enough for relief.
vec3 turbo(float t) {
  t = clamp(t, 0.0, 1.0);
  return clamp(vec3(
    0.13572138 + t*(4.6153926 + t*(-42.66032258 + t*(132.13108234 + t*(-152.94239396 + t*59.28637943)))),
    0.09140261 + t*(2.19418839 + t*(  4.84296658 + t*(-14.18503333 + t*(  4.27729857 + t* 2.82956604)))),
    0.10667330 + t*(12.64194608 + t*(-60.58204836 + t*(110.36276771 + t*(-89.90310912 + t*27.34824973))))
  ), 0.0, 1.0);
}

vec3 surfaceNormal() {
  // Horizontal spacing is 1 world unit across the whole grid, so a one-texel
  // step is uTexel world units. Vertical scale is uExag.
  float hl = heightAt(vUv - vec2(uTexel.x, 0.0));
  float hr = heightAt(vUv + vec2(uTexel.x, 0.0));
  float hd = heightAt(vUv - vec2(0.0, uTexel.y));
  float hu = heightAt(vUv + vec2(0.0, uTexel.y));
  return normalize(vec3(
    (hl - hr) * uExag,
    2.0 * uTexel.x,
    (hd - hu) * uExag
  ));
}

void main() {
  vec3 n = surfaceNormal();
  vec3 lightDir = normalize(vec3(0.45, 0.8, 0.35));
  float lambert = 0.35 + 0.65 * max(dot(n, lightDir), 0.0);

  vec3 base;
  if (uCmap < 0.5)      base = texture2D(uRgb, vUv).rgb;
  else if (uCmap < 1.5) base = turbo(vH);
  else                  base = vec3(0.72);

  gl_FragColor = vec4(base * lambert, 1.0);
}`;
```

- [ ] **Step 2: Write `viewer/web/js/terrain.js`**

```js
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { VERT, FRAG } from './shaders.js';

const DATA = '../data';
const SEGMENTS = 384;   // mesh grid; independent of the 504^2 depth texture

export function initViewer(canvas) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  if (!renderer.capabilities.isWebGL2) {
    throw new Error('WebGL2 is required and is not available in this browser');
  }
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));

  // Float textures cannot always be linearly filtered. Probe rather than
  // assume: nearest sampling is exact at texel centres and merely slightly
  // blocky between them, which beats a black screen.
  const canFilterFloat = !!renderer.getContext().getExtension('OES_texture_float_linear');

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0e1116);
  const camera = new THREE.PerspectiveCamera(50, 1, 0.01, 100);
  camera.position.set(0, 0.85, 1.15);

  const orbit = new OrbitControls(camera, canvas);
  orbit.enableDamping = true;
  orbit.target.set(0, 0, 0);

  const uniforms = {
    uDepth:    { value: null },
    uRgb:      { value: null },
    uPlane:    { value: new THREE.Vector3() },
    uRawRange: { value: new THREE.Vector2(0, 1) },
    uResRange: { value: new THREE.Vector2(0, 1) },
    uTexel:    { value: new THREE.Vector2(1 / 504, 1 / 504) },
    uExag:     { value: 0.15 },
    uDetrend:  { value: 0 },
    uCmap:     { value: 0 },
  };

  const geometry = new THREE.PlaneGeometry(1, 1, SEGMENTS, SEGMENTS);
  const material = new THREE.ShaderMaterial({ vertexShader: VERT, fragmentShader: FRAG, uniforms });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.rotation.x = -Math.PI / 2;   // XY plane -> horizontal, local +Z is up
  scene.add(mesh);

  const state = { meta: null, depth: null, width: 0, height: 0 };

  function resize() {
    const { clientWidth: w, clientHeight: h } = canvas;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  new ResizeObserver(resize).observe(canvas);
  resize();

  renderer.setAnimationLoop(() => {
    orbit.update();
    renderer.render(scene, camera);
  });

  async function loadScene(id) {
    const base = `${DATA}/scenes/${id}`;
    const [metaRes, binRes] = await Promise.all([
      fetch(`${base}/meta.json`),
      fetch(`${base}/depth.bin`),
    ]);
    if (!metaRes.ok) throw new Error(`meta.json ${metaRes.status}`);
    if (!binRes.ok) throw new Error(`depth.bin ${binRes.status}`);

    const meta = await metaRes.json();
    const depth = new Float32Array(await binRes.arrayBuffer());
    const expected = meta.width * meta.height;
    if (depth.length !== expected) {
      throw new Error(`depth.bin has ${depth.length} values, meta declares ${expected}`);
    }

    const tex = new THREE.DataTexture(
      depth, meta.width, meta.height, THREE.RedFormat, THREE.FloatType,
    );
    tex.minFilter = tex.magFilter = canFilterFloat ? THREE.LinearFilter : THREE.NearestFilter;
    tex.wrapS = tex.wrapT = THREE.ClampToEdgeWrapping;
    tex.needsUpdate = true;
    uniforms.uDepth.value?.dispose();
    uniforms.uDepth.value = tex;

    const rgb = await new THREE.TextureLoader().loadAsync(`${base}/rgb.jpg`);
    rgb.colorSpace = THREE.SRGBColorSpace;
    uniforms.uRgb.value?.dispose();
    uniforms.uRgb.value = rgb;

    uniforms.uPlane.value.set(meta.plane.a, meta.plane.b, meta.plane.c);
    uniforms.uRawRange.value.set(meta.depth_min, meta.depth_max);
    uniforms.uResRange.value.set(meta.residual_min, meta.residual_max);
    uniforms.uTexel.value.set(1 / meta.width, 1 / meta.height);

    Object.assign(state, { meta, depth, width: meta.width, height: meta.height });
  }

  /** Height in [0,1] at UV, from the DATA -- never the displaced geometry, so
   *  exaggeration cannot corrupt a reading. Mirrors the GLSL heightAt exactly. */
  function heightAt(u, v) {
    if (!state.depth) return NaN;
    const x = Math.min(state.width - 1, Math.max(0, Math.round(u * state.width)));
    const y = Math.min(state.height - 1, Math.max(0, Math.round(v * state.height)));
    const d = state.depth[y * state.width + x];
    const m = state.meta;
    if (uniforms.uDetrend.value > 0.5) {
      const r = d - (m.plane.a + m.plane.b * u + m.plane.c * v);
      return (m.residual_max - r) / Math.max(m.residual_max - m.residual_min, 1e-8);
    }
    return (m.depth_max - d) / Math.max(m.depth_max - m.depth_min, 1e-8);
  }

  return {
    loadScene,
    heightAt,
    orbit,
    camera,
    mesh,
    get meta() { return state.meta; },
    get exaggeration() { return uniforms.uExag.value; },
    setMode: (m) => { uniforms.uDetrend.value = m === 'detrended' ? 1 : 0; },
    setExaggeration: (x) => { uniforms.uExag.value = x; },
    setColormap: (c) => { uniforms.uCmap.value = Number(c); },
  };
}
```

- [ ] **Step 3: Wire the viewer into `app.js`**

Add to the top of `viewer/web/js/app.js`, after the `DATA` constant:

```js
import { initViewer } from './terrain.js';

let viewer = null;
try {
  viewer = initViewer(document.getElementById('view'));
  window.__loadScene = (id) => viewer.loadScene(id);
} catch (err) {
  document.getElementById('status').textContent = err.message;
}
```

And append this control wiring just before the `boot();` call:

```js
document.getElementById('exag').addEventListener('input', (e) => {
  const v = Number(e.target.value);
  document.getElementById('exag-out').textContent = v.toFixed(2);
  viewer?.setExaggeration(v);
});
document.getElementById('cmap').addEventListener('change', (e) => viewer?.setColormap(e.target.value));
for (const btn of document.querySelectorAll('#mode button')) {
  btn.addEventListener('click', () => {
    document.querySelectorAll('#mode button').forEach((b) => b.classList.toggle('on', b === btn));
    viewer?.setMode(btn.dataset.mode);
  });
}
```

- [ ] **Step 4: Verify in the browser**

Run: `cd viewer/web && python3 -m http.server 8777`
Open http://localhost:8777/ and confirm:
- terrain renders textured with the satellite image
- the exaggeration slider changes relief smoothly with no stutter (geometry is not rebuilt)
- **raw** shows a tilted plane; **detrended** flattens the ramp and reveals residual structure
- switching scenes is near-instant
- the browser console is free of GL errors

- [ ] **Step 5: Commit**

```bash
git add viewer/web/js/shaders.js viewer/web/js/terrain.js viewer/web/js/app.js
git commit -m "feat(viewer): vertex-shader terrain displacement from float32 depth"
```

---

### Task 6: Navigation and inspection — fly mode, height probe, slope

**Files:**
- Create: `viewer/web/js/probe.js`
- Modify: `viewer/web/js/terrain.js` (expose fly controls), `viewer/web/js/app.js` (wire probe + nav)

**Interfaces:**
- Consumes: `initViewer(...).heightAt`, `.camera`, `.mesh`, `.exaggeration`
- Produces: `pickTerrain(event, canvas, viewer) -> {u, v, height, slopeDeg} | null`

- [ ] **Step 1: Write `viewer/web/js/probe.js`**

```js
import * as THREE from 'three';

const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();

/**
 * Pick a point on the displaced terrain.
 *
 * The mesh is displaced in the vertex shader, so a CPU raycast hits the FLAT
 * plane -- at high exaggeration that is visibly not the surface under the
 * cursor. So: raycast the flat plane to get an entry point, then march the ray
 * against the height field and bisect the crossing.
 */
export function pickTerrain(event, canvas, viewer) {
  if (!viewer?.meta) return null;
  const rect = canvas.getBoundingClientRect();
  pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(pointer, viewer.camera);

  const origin = raycaster.ray.origin.clone();
  const dir = raycaster.ray.direction.clone();
  const exag = viewer.exaggeration;

  const toUv = (p) => [p.x + 0.5, 0.5 - p.z];
  const inside = ([u, v]) => u >= 0 && u <= 1 && v >= 0 && v <= 1;
  const above = (t) => {
    const p = origin.clone().addScaledVector(dir, t);
    const uv = toUv(p);
    if (!inside(uv)) return null;
    return p.y - viewer.heightAt(uv[0], uv[1]) * exag;
  };

  const STEP = 0.01;
  const MAX_T = 12.0;
  let prevT = 0;
  let prevSign = above(0);
  let hitT = null;
  for (let t = STEP; t < MAX_T; t += STEP) {
    const s = above(t);
    if (s === null) { prevT = t; prevSign = null; continue; }
    if (prevSign !== null && prevSign > 0 && s <= 0) {
      let lo = prevT;
      let hi = t;
      for (let i = 0; i < 24; i++) {           // bisect to ~1e-9 world units
        const mid = (lo + hi) / 2;
        const sm = above(mid);
        if (sm === null) break;
        if (sm > 0) lo = mid; else hi = mid;
      }
      hitT = (lo + hi) / 2;
      break;
    }
    prevT = t;
    prevSign = s;
  }
  if (hitT === null) return null;

  const hit = origin.clone().addScaledVector(dir, hitT);
  const [u, v] = toUv(hit);
  if (!inside([u, v])) return null;

  const height = viewer.heightAt(u, v);

  // Slope from central differences on the height field. Horizontal spacing is
  // one world unit across the grid, so a one-texel step is 1/width. Vertical
  // scale is the current exaggeration -- which is exactly why this is reported
  // as a DISPLAY slope, not a ground slope. A real slope needs metric height.
  const du = 1 / viewer.meta.width;
  const dv = 1 / viewer.meta.height;
  const dhdu = (viewer.heightAt(u + du, v) - viewer.heightAt(u - du, v)) * exag / (2 * du);
  const dhdv = (viewer.heightAt(u, v + dv) - viewer.heightAt(u, v - dv)) * exag / (2 * dv);
  const slopeDeg = (Math.atan(Math.hypot(dhdu, dhdv)) * 180) / Math.PI;

  return { u, v, height, slopeDeg };
}
```

- [ ] **Step 2: Add fly controls to `terrain.js`**

Add the import at the top:

```js
import { PointerLockControls } from 'three/addons/controls/PointerLockControls.js';
```

Inside `initViewer`, after the `orbit` setup:

```js
  const fly = new PointerLockControls(camera, canvas);
  const keys = new Set();
  addEventListener('keydown', (e) => keys.add(e.code));
  addEventListener('keyup', (e) => keys.delete(e.code));
  let mode = 'orbit';
```

Replace the animation loop with:

```js
  const clock = new THREE.Clock();
  renderer.setAnimationLoop(() => {
    const dt = clock.getDelta();
    if (mode === 'orbit') {
      orbit.update();
    } else if (fly.isLocked) {
      const speed = (keys.has('ShiftLeft') ? 1.6 : 0.55) * dt;
      if (keys.has('KeyW')) fly.moveForward(speed);
      if (keys.has('KeyS')) fly.moveForward(-speed);
      if (keys.has('KeyA')) fly.moveRight(-speed);
      if (keys.has('KeyD')) fly.moveRight(speed);
      if (keys.has('KeyE')) camera.position.y += speed;
      if (keys.has('KeyQ')) camera.position.y -= speed;
      // Clamp above terrain by sampling the height field (design doc 5.2).
      const u = camera.position.x + 0.5;
      const v = 0.5 - camera.position.z;
      if (u >= 0 && u <= 1 && v >= 0 && v <= 1) {
        const floor = heightAt(u, v) * uniforms.uExag.value + 0.02;
        if (camera.position.y < floor) camera.position.y = floor;
      }
    }
    renderer.render(scene, camera);
  });
```

Add to the returned object:

```js
    setNavMode: (m) => {
      mode = m;
      orbit.enabled = m === 'orbit';
      if (m === 'fly') fly.lock();
      else fly.unlock();
    },
```

- [ ] **Step 3: Wire probe and nav into `app.js`**

Add the import beside the others:

```js
import { pickTerrain } from './probe.js';
```

Append before `boot();`:

```js
const canvasEl = document.getElementById('view');
canvasEl.addEventListener('click', (event) => {
  const hit = pickTerrain(event, canvasEl, viewer);
  document.getElementById('p-height').textContent =
    hit ? `${hit.height.toFixed(3)} rel` : '—';
  document.getElementById('p-slope').textContent =
    hit ? `${hit.slopeDeg.toFixed(1)}°` : '—';
});

for (const btn of document.querySelectorAll('#nav button')) {
  btn.addEventListener('click', () => {
    document.querySelectorAll('#nav button').forEach((b) => b.classList.toggle('on', b === btn));
    viewer?.setNavMode(btn.dataset.nav);
    document.getElementById('nav-hint').textContent =
      btn.dataset.nav === 'fly'
        ? 'click to lock pointer · WASD move · QE up/down · Shift fast · Esc release'
        : 'drag to orbit, scroll to zoom';
  });
}
```

- [ ] **Step 4: Verify in the browser**

Open http://localhost:8777/ and confirm:
- clicking the terrain reports a relative height and a display slope
- the probed height is stable when the exaggeration slider moves (it samples data, not geometry)
- the probe lands under the cursor even at exaggeration 1.0 (ray marching, not flat-plane picking)
- fly mode locks the pointer, WASD moves, and the camera does not sink through terrain
- Esc releases the pointer and orbit mode still works

- [ ] **Step 5: Run the full Python suite once more**

Run: `.venv/bin/python -m pytest -q`
Expected: all tests pass

- [ ] **Step 6: Commit**

```bash
git add viewer/web/js/probe.js viewer/web/js/terrain.js viewer/web/js/app.js
git commit -m "feat(viewer): fly navigation, ray-marched height probe, slope readout"
```

---

## Self-Review

**Spec coverage:**

| Spec section | Task |
|---|---|
| §3.1 `viewer/metrics.py`, all six metrics, nan on degenerate | Task 1 |
| §3.1 torch-free constraint | Task 1 (asserted by test) |
| §3.2 asset layout, `scene_id`, two-scope outputs | Task 2 |
| §3.2 curation (2 low + 1 high per class) | Task 2 |
| §3.2 `terrain.glb`, height sign | Task 3 |
| §3.3 vendored three.js, no CDN | Task 4 |
| §3.3 rendering core, shader detrend, exaggeration | Task 5 |
| §3.3 sort by ascending `plane_r2` | Task 4 |
| §3.3 height probe samples data | Task 5 (`heightAt`), Task 6 (picking) |
| §3.3 display-slope labelling | Task 4 (markup), Task 6 (value) |
| §4 metrics computed once, carried in meta | Task 2 |
| §5 error handling: WebGL2, fetch failure, non-finite, degenerate, `is_metric` | Tasks 1, 2, 4, 5 |
| §6 testing | Tasks 1, 3 |

**Gaps found and closed during review:**
- The spec's §5 "non-finite depth fails loudly naming the scene" had no implementation — added to `_predict` in Task 2.
- `meta.json` needed `residual_min`/`residual_max` for the shader's detrended normalization; added to `scene_metrics` in Task 1.
- Float-texture linear filtering is not guaranteed in WebGL2; Task 5 probes `OES_texture_float_linear` and falls back to `NearestFilter`.
- Naive raycasting would hit the undisplaced plane; Task 6 ray-marches the height field instead.

**Type consistency:** `heightAt(u, v)` has the same name and semantics in GLSL (`shaders.js`), JS (`terrain.js`), and its consumer (`probe.js`), and all three apply the same raw/detrended branch. `plane` is `{a, b, c}` in JSON and a 3-tuple in Python throughout.

**Known deviation from the plan template:** Tasks 4, 5 and 6 are browser code verified by running the page, not by automated test. A browser-automation harness is disproportionate at this scope; the Python layer beneath it is covered by pytest.
