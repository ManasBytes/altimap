# AltiMap Phase 0–1: Evaluation Harness and Data Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the measurement foundation — a `eval` CLI that scores a predicted elevation raster against reference LiDAR across the metric matrix the rubric demands, plus the data pipeline that extracts training patches from remote COGs.

**Architecture:** Pure-function metrics operating on numpy arrays and boolean masks, wrapped by thin I/O layers. Nothing in `eval/` knows about models; nothing in `data/` knows about metrics. Both depend only on `contract.py`, which defines the GeoTIFF-plus-sidecar interface from spec §3.1. Training patches are read as windows from remote Cloud-Optimized GeoTIFFs, never downloaded whole.

**Tech Stack:** Python 3.12 (pinned via `uv` — system Python 3.14 has no PyTorch wheels), rasterio, numpy, scipy, scikit-image, pystac-client, planetary-computer, pytest.

**Spec:** `docs/superpowers/specs/2026-08-23-single-view-dsm-design.md`

## Global Constraints

- **Python 3.12** exactly. System Python is 3.14.6 and has no PyTorch wheels. Pin via `.python-version`.
- **GDAL comes from rasterio's bundled wheels.** Never depend on system GDAL — it breaks PyInstaller packaging in Phase 5.
- **Never download whole rasters.** All remote raster access is windowed via `/vsicurl/`. One NAIP tile is 467 MB; disk has 32 GB free.
- **Input GSD 0.6 m, output/label GSD 2.0 m.** Output resolution is capped by reference-data resolution (spec §4.4).
- **Elevation rasters are float32, metres.** Nodata is `NaN`, never a sentinel like -9999.
- **Every reported metric carries dataset, GSD, and scope** (all-pixel / building-pixel / building-wise). An unqualified RMSE is uninterpretable (spec §6.2).
- **Alignment corrections are reported, never silently applied.** Raw and corrected numbers appear side by side (spec §6).
- Licence policy: permissively-licensed, ungated weights and data only (spec §4.1).

---

### Task 1: Spike — verify Planetary Computer access and select AOIs

**This is a spike, not TDD.** It answers go/no-go questions and produces a decision record. No production code, no tests. Everything here is throwaway except the written findings.

Spec §7.2 makes this a hard gate: `3dep-lidar-hag` coverage is workunit-based, verified as 0 hits over Raleigh NC but present over Utah. If we cannot find AOIs with both NAIP and HAG coverage across the four landscape classes, the entire data strategy in §7 changes.

**Files:**
- Create: `spikes/01_stac_coverage.py`
- Create: `docs/superpowers/spikes/2026-08-23-stac-coverage-findings.md`

- [ ] **Step 1: Create the project directory structure**

```bash
cd /home/biplab-dev/Projects/AltiMap
mkdir -p spikes docs/superpowers/spikes src/altimap tests
```

- [ ] **Step 2: Create a throwaway environment with STAC tooling**

```bash
cd /home/biplab-dev/Projects/AltiMap
uv venv --python 3.12 .venv-spike
uv pip install --python .venv-spike/bin/python pystac-client planetary-computer rasterio
```

Expected: installs without error. If `pystac-client` fails to resolve, that itself is a finding — record it.

- [ ] **Step 3: Write the coverage probe script**

Create `spikes/01_stac_coverage.py`:

```python
"""Spike: does Planetary Computer serve co-located NAIP + 3DEP HAG for candidate AOIs?

Throwaway. Findings go to docs/superpowers/spikes/.
"""

import planetary_computer
import pystac_client

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

# Candidate AOIs spanning the four landscape classes the rubric grades.
# bbox = (west, south, east, north) in EPSG:4326
CANDIDATES = {
    "salt_lake_city_urban": (-111.95, 40.72, -111.85, 40.80),
    "wasatch_hilly": (-111.95, 40.55, -111.85, 40.65),
    "denver_urban": (-105.05, 39.70, -104.95, 39.78),
    "portland_forested": (-122.75, 45.48, -122.65, 45.56),
    "phoenix_sparse": (-112.10, 33.42, -112.00, 33.50),
    "seattle_urban": (-122.36, 47.58, -122.26, 47.66),
    "asheville_forested": (-82.60, 35.55, -82.50, 35.63),
    "iowa_cropland": (-93.70, 41.55, -93.60, 41.63),
}

COLLECTIONS = ["naip", "3dep-lidar-hag", "3dep-seamless", "esa-worldcover"]


def main() -> None:
    client = pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)

    print(f"{'AOI':<26} " + " ".join(f"{c:>16}" for c in COLLECTIONS))
    print("-" * 100)

    viable = []
    for name, bbox in CANDIDATES.items():
        counts = []
        for coll in COLLECTIONS:
            try:
                search = client.search(collections=[coll], bbox=bbox, limit=50)
                n = len(list(search.items()))
            except Exception as exc:  # spike: surface the error, do not swallow
                n = f"ERR:{type(exc).__name__}"
            counts.append(n)
        print(f"{name:<26} " + " ".join(f"{str(c):>16}" for c in counts))

        naip_n, hag_n = counts[0], counts[1]
        if isinstance(naip_n, int) and isinstance(hag_n, int) and naip_n > 0 and hag_n > 0:
            viable.append(name)

    print(f"\nViable AOIs (NAIP > 0 AND HAG > 0): {viable}")
    print(f"Count: {len(viable)} / {len(CANDIDATES)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the probe**

```bash
cd /home/biplab-dev/Projects/AltiMap
.venv-spike/bin/python spikes/01_stac_coverage.py
```

Expected: a table of item counts per AOI per collection. **The go/no-go question: are there at least four viable AOIs, covering urban, sparse, hilly, and forested?**

- [ ] **Step 5: Verify a real windowed read works without downloading**

This confirms the central assumption of spec §7.1. Create `spikes/02_windowed_read.py`:

```python
"""Spike: confirm windowed COG reads pull only the requested window."""

import planetary_computer
import pystac_client
import rasterio
from rasterio.windows import Window

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
BBOX = (-111.95, 40.55, -111.85, 40.65)  # Wasatch — known-good from research

client = pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)

for coll, asset_key in [("naip", "image"), ("3dep-lidar-hag", "data")]:
    item = next(client.search(collections=[coll], bbox=BBOX, limit=1).items())
    href = item.assets[asset_key].href
    print(f"\n{coll}: {item.id}")
    with rasterio.open(href) as src:
        print(f"  shape={src.shape} crs={src.crs} res={src.res} dtype={src.dtypes[0]}")
        print(f"  nodata={src.nodata} blocksize={src.block_shapes[0]}")
        win = Window(0, 0, 512, 512)
        arr = src.read(1, window=win)
        print(f"  windowed read OK: {arr.shape} min={arr.min():.2f} max={arr.max():.2f}")
```

Run it:

```bash
.venv-spike/bin/python spikes/02_windowed_read.py
```

Expected: prints shape, CRS, resolution for both. **Confirm NAIP resolution is ~0.6 m and HAG is ~2.0 m** — the plan's Global Constraints depend on this. Note the exact asset keys printed; Task 9 needs them.

- [ ] **Step 6: Write the findings document**

Create `docs/superpowers/spikes/2026-08-23-stac-coverage-findings.md`:

```markdown
# Spike: Planetary Computer STAC coverage and windowed reads

**Date:** 2026-08-23
**Question:** Can we get co-located NAIP + 3DEP HAG across urban, sparse, hilly, and forested AOIs, reading windows without downloading whole rasters?

## Verdict

<GO or NO-GO — state plainly>

## AOI coverage results

<paste the table from step 4 verbatim>

**Selected AOIs for training:**

| AOI | bbox | Landscape class | NAIP items | HAG items |
|---|---|---|---|---|
| ... | ... | ... | ... | ... |

## Asset keys and raster properties

<paste step 5 output verbatim>

- NAIP asset key: `<key>`, resolution `<res>`, dtype `<dtype>`, nodata `<value>`
- HAG asset key: `<key>`, resolution `<res>`, dtype `<dtype>`, nodata `<value>`

## Deviations from spec assumptions

<Anything that contradicts spec §7. If NAIP is not 0.6 m or HAG is not 2 m,
say so — Global Constraints in the plan must be corrected.>

## If NO-GO

Fallbacks in priority order:
1. Widen the AOI candidate list — coverage is workunit-based, so more probes may find viable regions
2. Substitute `3dep-lidar-dsm` minus `3dep-lidar-dtm` for HAG where HAG is absent
3. Fall back to GeoNRW (spec §7.3 rejected it on convenience, not availability)
```

- [ ] **Step 7: Commit the spike and findings**

```bash
cd /home/biplab-dev/Projects/AltiMap
cat > .gitignore <<'EOF'
.venv*/
__pycache__/
*.pyc
.pytest_cache/
data/
*.tif
!tests/fixtures/*.tif
EOF
git add .gitignore spikes/ docs/superpowers/spikes/
git commit -m "spike: verify Planetary Computer STAC coverage and windowed COG reads"
```

- [ ] **Step 8: STOP and report**

Report the verdict before proceeding. **If NO-GO, do not continue to Task 2** — the data strategy needs redesign and that is a spec change, not a plan change.

---

### Task 2: Project scaffold and the contract module

Spec §3.1 defines the contract every subsystem communicates through. It is built first because both `eval/` and `data/` depend on it and nothing depends on them.

**Files:**
- Create: `pyproject.toml`
- Create: `.python-version`
- Create: `src/altimap/__init__.py`
- Create: `src/altimap/contract.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/test_contract.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `Sidecar` frozen dataclass with fields `gsd_m: float`, `source_gsd_m: float`, `datum: str`, `vertical_unit: str`, `model_version: str`, `height_range_m: tuple[float, float]`, `tile_overlap_px: int`, `dtm_source: str | None`
  - `Sidecar.to_json(path: Path) -> None`
  - `Sidecar.from_json(path: Path) -> Sidecar`
  - `write_elevation_cog(path: Path, array: np.ndarray, transform: Affine, crs: CRS, sidecar: Sidecar) -> None`
  - `read_elevation(path: Path) -> tuple[np.ndarray, Affine, CRS]` — returns float32 array with nodata as NaN
  - `make_synthetic_elevation(shape, transform, crs)` test fixture helper in `conftest.py`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "altimap"
version = "0.1.0"
description = "Single-view optical imagery to metric elevation models"
requires-python = "==3.12.*"
dependencies = [
    "numpy>=2.1",
    "rasterio>=1.4",
    "scipy>=1.14",
    "scikit-image>=0.24",
    "pystac-client>=0.8",
    "planetary-computer>=1.0",
]

[project.scripts]
altimap-eval = "altimap.eval.cli:main"

[project.optional-dependencies]
dev = ["pytest>=8.3", "pytest-cov>=5.0"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/altimap"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "network: requires network access to Planetary Computer",
]
```

- [ ] **Step 2: Pin Python and create the environment**

```bash
cd /home/biplab-dev/Projects/AltiMap
echo "3.12" > .python-version
uv venv --python 3.12
uv pip install -e ".[dev]"
```

Expected: resolves and installs. Verify:

```bash
.venv/bin/python -c "import rasterio, numpy, skimage; print(rasterio.__version__, numpy.__version__)"
```

Expected: prints versions without error.

- [ ] **Step 3: Write the failing test**

Create `tests/__init__.py` (empty file), then `tests/test_contract.py`:

```python
import json
from pathlib import Path

import numpy as np
import pytest
from rasterio.crs import CRS
from rasterio.transform import from_origin

from altimap.contract import Sidecar, read_elevation, write_elevation_cog


def test_sidecar_roundtrip(tmp_path: Path) -> None:
    sidecar = Sidecar(
        gsd_m=2.0,
        source_gsd_m=0.6,
        datum="ellipsoidal",
        vertical_unit="m",
        model_version="test-v1",
        height_range_m=(0.0, 84.3),
        tile_overlap_px=130,
        dtm_source="3dep-seamless",
    )
    path = tmp_path / "dsm.json"
    sidecar.to_json(path)
    assert Sidecar.from_json(path) == sidecar


def test_sidecar_rejects_bad_datum() -> None:
    with pytest.raises(ValueError, match="datum"):
        Sidecar(
            gsd_m=2.0,
            source_gsd_m=0.6,
            datum="nonsense",
            vertical_unit="m",
            model_version="test-v1",
            height_range_m=(0.0, 1.0),
            tile_overlap_px=0,
            dtm_source=None,
        )


def test_elevation_cog_roundtrip_preserves_georeferencing(tmp_path: Path) -> None:
    array = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    transform = from_origin(500000.0, 4500000.0, 2.0, 2.0)
    crs = CRS.from_epsg(32612)
    sidecar = Sidecar(
        gsd_m=2.0,
        source_gsd_m=0.6,
        datum="ellipsoidal",
        vertical_unit="m",
        model_version="test-v1",
        height_range_m=(1.0, 4.0),
        tile_overlap_px=0,
        dtm_source=None,
    )
    path = tmp_path / "dsm.tif"
    write_elevation_cog(path, array, transform, crs, sidecar)

    out, out_transform, out_crs = read_elevation(path)
    np.testing.assert_allclose(out, array)
    assert out_transform == transform
    assert out_crs == crs
    assert (tmp_path / "dsm.json").exists()


def test_nodata_becomes_nan(tmp_path: Path) -> None:
    array = np.array([[1.0, np.nan], [3.0, 4.0]], dtype=np.float32)
    transform = from_origin(0.0, 0.0, 2.0, 2.0)
    sidecar = Sidecar(
        gsd_m=2.0,
        source_gsd_m=0.6,
        datum="relative",
        vertical_unit="m",
        model_version="test-v1",
        height_range_m=(1.0, 4.0),
        tile_overlap_px=0,
        dtm_source=None,
    )
    path = tmp_path / "dsm.tif"
    write_elevation_cog(path, array, transform, CRS.from_epsg(32612), sidecar)

    out, _, _ = read_elevation(path)
    assert np.isnan(out[0, 1])
    assert out[0, 0] == 1.0
```

- [ ] **Step 4: Run the test to verify it fails**

```bash
cd /home/biplab-dev/Projects/AltiMap
.venv/bin/pytest tests/test_contract.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'altimap.contract'`

- [ ] **Step 5: Implement the contract module**

Create `src/altimap/__init__.py`:

```python
"""AltiMap — single-view optical imagery to metric elevation models."""

__version__ = "0.1.0"
```

Create `src/altimap/contract.py`:

```python
"""The interface every AltiMap subsystem communicates through.

A produced elevation result is always three things: a Cloud-Optimized GeoTIFF
holding float32 metres, the source RGB on the identical grid, and a JSON
sidecar describing both. Nothing else crosses a subsystem boundary.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine

VALID_DATUMS = ("ellipsoidal", "relative")


@dataclasses.dataclass(frozen=True)
class Sidecar:
    """Metadata accompanying an elevation raster.

    `height_range_m` exists so the viewer can configure its colour ramp and
    camera bounds without scanning the raster.
    """

    gsd_m: float
    source_gsd_m: float
    datum: str
    vertical_unit: str
    model_version: str
    height_range_m: tuple[float, float]
    tile_overlap_px: int
    dtm_source: str | None

    def __post_init__(self) -> None:
        if self.datum not in VALID_DATUMS:
            raise ValueError(
                f"datum must be one of {VALID_DATUMS}, got {self.datum!r}"
            )
        if self.gsd_m <= 0:
            raise ValueError(f"gsd_m must be positive, got {self.gsd_m}")
        if self.vertical_unit != "m":
            raise ValueError(
                f"vertical_unit must be 'm' — metres are the project-wide unit, "
                f"got {self.vertical_unit!r}"
            )

    def to_json(self, path: Path) -> None:
        payload = dataclasses.asdict(self)
        payload["height_range_m"] = list(self.height_range_m)
        path.write_text(json.dumps(payload, indent=2) + "\n")

    @classmethod
    def from_json(cls, path: Path) -> Sidecar:
        payload: dict[str, Any] = json.loads(path.read_text())
        payload["height_range_m"] = tuple(payload["height_range_m"])
        return cls(**payload)


def sidecar_path_for(raster_path: Path) -> Path:
    """The sidecar lives beside the raster with a .json suffix."""
    return raster_path.with_suffix(".json")


def write_elevation_cog(
    path: Path,
    array: np.ndarray,
    transform: Affine,
    crs: CRS,
    sidecar: Sidecar,
) -> None:
    """Write a float32 elevation COG plus its sidecar.

    NaN is the nodata value throughout the project — sentinel values like
    -9999 silently corrupt statistics when a mask is forgotten.
    """
    if array.ndim != 2:
        raise ValueError(f"expected a 2D array, got shape {array.shape}")

    data = array.astype(np.float32, copy=False)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=np.nan,
        tiled=True,
        blockxsize=256,
        blockysize=256,
        compress="deflate",
        predictor=3,
    ) as dst:
        dst.write(data, 1)

    sidecar.to_json(sidecar_path_for(path))


def read_elevation(path: Path) -> tuple[np.ndarray, Affine, CRS]:
    """Read an elevation raster as float32 with nodata represented as NaN."""
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        if src.nodata is not None and not np.isnan(src.nodata):
            data[data == src.nodata] = np.nan
        return data, src.transform, src.crs
```

- [ ] **Step 6: Run the test to verify it passes**

```bash
.venv/bin/pytest tests/test_contract.py -v
```

Expected: 4 passed.

- [ ] **Step 7: Add the shared test fixture helper**

Create `tests/conftest.py`:

```python
"""Shared fixtures. Synthetic rasters keep tests fast and network-free."""

from __future__ import annotations

import numpy as np
import pytest
from rasterio.crs import CRS
from rasterio.transform import from_origin

TEST_CRS = CRS.from_epsg(32612)  # UTM 12N — covers the Utah AOIs
TEST_TRANSFORM = from_origin(500000.0, 4500000.0, 2.0, 2.0)


@pytest.fixture
def test_crs() -> CRS:
    return TEST_CRS


@pytest.fixture
def test_transform():
    return TEST_TRANSFORM


def make_synthetic_ndsm(shape: tuple[int, int] = (64, 64)) -> np.ndarray:
    """A synthetic nDSM with the long-tailed structure real data has.

    Mostly near-zero ground, a few tall rectangular 'buildings'. This shape
    matters: metrics that look fine on uniform noise fail on long tails.
    """
    arr = np.zeros(shape, dtype=np.float32)
    rng = np.random.default_rng(1234)
    arr += rng.normal(0.0, 0.15, shape).astype(np.float32)  # ground roughness
    arr[10:20, 10:20] = 12.0   # low building
    arr[30:38, 40:52] = 35.0   # mid building
    arr[45:50, 15:22] = 80.0   # tall building — the distribution tail
    return arr


@pytest.fixture
def synthetic_ndsm() -> np.ndarray:
    return make_synthetic_ndsm()
```

- [ ] **Step 8: Verify test collection still works**

Adding a `conftest.py` can break collection if it has an import error, so check before committing:

```bash
.venv/bin/pytest tests/ -v
```

Expected: 4 passed, no collection errors.

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml .python-version src/ tests/
git commit -m "feat: add project scaffold and contract module

The contract (float32 COG + JSON sidecar) is the interface every
subsystem communicates through, so it is built first. NaN is the
project-wide nodata value — sentinels corrupt statistics silently."
```

---

### Task 3: Core metrics over a mask

Pure functions on numpy arrays. No I/O, no rasterio. This is the heart of the 50% accuracy score, so it gets the most careful tests.

**Files:**
- Create: `src/altimap/eval/__init__.py`
- Create: `src/altimap/eval/metrics.py`
- Create: `tests/test_metrics.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `rmse(pred: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> float`
  - `mae(pred, ref, mask) -> float`
  - `bias(pred, ref, mask) -> float` — mean signed error, `pred - ref`
  - `pearson_r(pred, ref, mask) -> float`
  - `abs_error_percentile(pred, ref, mask, q: float) -> float`
  - `delta1(pred, ref, mask, floor_m: float = 1.0, threshold: float = 1.25) -> tuple[float, int]` — returns (score, n_pixels_used)
  - `CoreMetrics` dataclass and `core_metrics(pred, ref, mask) -> CoreMetrics`

- [ ] **Step 1: Write the failing test**

Create `tests/test_metrics.py`:

```python
import numpy as np
import pytest

from altimap.eval.metrics import (
    abs_error_percentile,
    bias,
    core_metrics,
    delta1,
    mae,
    pearson_r,
    rmse,
)


@pytest.fixture
def simple() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ref = np.array([[0.0, 10.0], [20.0, 30.0]], dtype=np.float32)
    pred = np.array([[1.0, 12.0], [17.0, 30.0]], dtype=np.float32)
    mask = np.ones((2, 2), dtype=bool)
    return pred, ref, mask


def test_rmse_known_value(simple) -> None:
    pred, ref, mask = simple
    # errors: 1, 2, -3, 0 -> squares 1, 4, 9, 0 -> mean 3.5 -> sqrt 1.8708
    assert rmse(pred, ref, mask) == pytest.approx(1.8708286, rel=1e-6)


def test_mae_known_value(simple) -> None:
    pred, ref, mask = simple
    # |errors| 1, 2, 3, 0 -> mean 1.5
    assert mae(pred, ref, mask) == pytest.approx(1.5)


def test_bias_is_signed(simple) -> None:
    pred, ref, mask = simple
    # errors 1, 2, -3, 0 -> mean 0.0
    assert bias(pred, ref, mask) == pytest.approx(0.0)


def test_perfect_prediction_gives_zero_error(simple) -> None:
    _, ref, mask = simple
    assert rmse(ref, ref, mask) == pytest.approx(0.0)
    assert mae(ref, ref, mask) == pytest.approx(0.0)
    assert pearson_r(ref, ref, mask) == pytest.approx(1.0)


def test_mask_excludes_pixels() -> None:
    ref = np.array([[0.0, 100.0]], dtype=np.float32)
    pred = np.array([[0.0, 0.0]], dtype=np.float32)
    mask = np.array([[True, False]])
    # The huge error is masked out
    assert rmse(pred, ref, mask) == pytest.approx(0.0)


def test_nan_pixels_are_excluded_even_if_masked_in() -> None:
    """NaN must never leak into a statistic — this is the bug that silently
    turns every metric into nan."""
    ref = np.array([[1.0, np.nan]], dtype=np.float32)
    pred = np.array([[2.0, 5.0]], dtype=np.float32)
    mask = np.ones((1, 2), dtype=bool)
    assert rmse(pred, ref, mask) == pytest.approx(1.0)


def test_empty_mask_returns_nan() -> None:
    ref = np.zeros((2, 2), dtype=np.float32)
    pred = np.zeros((2, 2), dtype=np.float32)
    mask = np.zeros((2, 2), dtype=bool)
    assert np.isnan(rmse(pred, ref, mask))


def test_pearson_r_detects_anticorrelation() -> None:
    ref = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    pred = np.array([[4.0, 3.0, 2.0, 1.0]], dtype=np.float32)
    mask = np.ones((1, 4), dtype=bool)
    assert pearson_r(pred, ref, mask) == pytest.approx(-1.0)


def test_pearson_r_of_constant_is_nan() -> None:
    """Zero variance means correlation is undefined, not zero."""
    ref = np.array([[5.0, 5.0, 5.0]], dtype=np.float32)
    pred = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    mask = np.ones((1, 3), dtype=bool)
    assert np.isnan(pearson_r(pred, ref, mask))


def test_abs_error_percentile() -> None:
    ref = np.zeros((1, 5), dtype=np.float32)
    pred = np.array([[0.0, 1.0, 2.0, 3.0, 100.0]], dtype=np.float32)
    mask = np.ones((1, 5), dtype=bool)
    assert abs_error_percentile(pred, ref, mask, 50.0) == pytest.approx(2.0)


def test_delta1_excludes_pixels_below_floor() -> None:
    """nDSM background is exactly zero, where the y/y-hat ratio is undefined.
    delta1 must restrict to pixels above a height floor and report how many
    it used, so the number is interpretable."""
    ref = np.array([[0.0, 0.0, 10.0, 20.0]], dtype=np.float32)
    pred = np.array([[5.0, 5.0, 10.5, 40.0]], dtype=np.float32)
    mask = np.ones((1, 4), dtype=bool)
    score, n_used = delta1(pred, ref, mask, floor_m=1.0)
    assert n_used == 2  # only the 10.0 and 20.0 reference pixels
    # 10 vs 10.5 -> ratio 1.05 < 1.25 (pass); 20 vs 40 -> ratio 2.0 (fail)
    assert score == pytest.approx(0.5)


def test_delta1_all_below_floor_returns_nan() -> None:
    ref = np.zeros((1, 3), dtype=np.float32)
    pred = np.zeros((1, 3), dtype=np.float32)
    mask = np.ones((1, 3), dtype=bool)
    score, n_used = delta1(pred, ref, mask, floor_m=1.0)
    assert np.isnan(score)
    assert n_used == 0


def test_core_metrics_bundles_everything(simple) -> None:
    pred, ref, mask = simple
    m = core_metrics(pred, ref, mask)
    assert m.rmse == pytest.approx(1.8708286, rel=1e-6)
    assert m.mae == pytest.approx(1.5)
    assert m.n_pixels == 4
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
.venv/bin/pytest tests/test_metrics.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'altimap.eval'`

- [ ] **Step 3: Implement the metrics module**

Create `src/altimap/eval/__init__.py`:

```python
"""Accuracy measurement. Deliberately independent of the model code."""
```

Create `src/altimap/eval/metrics.py`:

```python
"""Metrics over predicted vs reference elevation.

Every function takes (pred, ref, mask) and reduces to a scalar. Pixels that
are NaN in either raster are always excluded, regardless of the mask — a NaN
leaking into a sum turns the whole statistic into NaN, which is the most
common silent failure in this kind of code.

An empty selection returns NaN rather than raising, so a stratified report can
contain classes absent from a given scene without special-casing.
"""

from __future__ import annotations

import dataclasses

import numpy as np


def _select(
    pred: np.ndarray, ref: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten to the valid, in-mask pixels of both rasters."""
    if pred.shape != ref.shape or pred.shape != mask.shape:
        raise ValueError(
            f"shape mismatch: pred {pred.shape}, ref {ref.shape}, mask {mask.shape}"
        )
    valid = mask & np.isfinite(pred) & np.isfinite(ref)
    return pred[valid], ref[valid]


def rmse(pred: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> float:
    p, r = _select(pred, ref, mask)
    if p.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((p - r) ** 2)))


def mae(pred: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> float:
    p, r = _select(pred, ref, mask)
    if p.size == 0:
        return float("nan")
    return float(np.mean(np.abs(p - r)))


def bias(pred: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> float:
    """Mean signed error, pred - ref. Positive means over-prediction."""
    p, r = _select(pred, ref, mask)
    if p.size == 0:
        return float("nan")
    return float(np.mean(p - r))


def pearson_r(pred: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> float:
    """Pearson correlation. NaN when either input has zero variance —
    correlation is undefined there, not zero."""
    p, r = _select(pred, ref, mask)
    if p.size < 2:
        return float("nan")
    p_sd, r_sd = p.std(), r.std()
    if p_sd == 0 or r_sd == 0:
        return float("nan")
    return float(np.mean((p - p.mean()) * (r - r.mean())) / (p_sd * r_sd))


def abs_error_percentile(
    pred: np.ndarray, ref: np.ndarray, mask: np.ndarray, q: float
) -> float:
    p, r = _select(pred, ref, mask)
    if p.size == 0:
        return float("nan")
    return float(np.percentile(np.abs(p - r), q))


def delta1(
    pred: np.ndarray,
    ref: np.ndarray,
    mask: np.ndarray,
    floor_m: float = 1.0,
    threshold: float = 1.25,
) -> tuple[float, int]:
    """Threshold accuracy: fraction of pixels with max(r/p, p/r) < threshold.

    This is the official DFC2023 Track 2 metric, inherited from indoor depth
    estimation where depth is strictly positive. nDSM background is exactly
    zero, where the ratio is undefined, so we restrict to pixels whose
    *reference* height is at least `floor_m` and return the pixel count
    alongside the score so the restriction is visible in the report.

    Returns (score, n_pixels_used).
    """
    p, r = _select(pred, ref, mask)
    above = r >= floor_m
    p, r = p[above], r[above]
    if p.size == 0:
        return float("nan"), 0
    # Clamp predictions away from zero so the ratio stays finite; a prediction
    # at or below zero against a real structure is a failure either way.
    p_safe = np.maximum(p, 1e-3)
    ratio = np.maximum(r / p_safe, p_safe / r)
    return float(np.mean(ratio < threshold)), int(p.size)


@dataclasses.dataclass(frozen=True)
class CoreMetrics:
    """The metric bundle reported for every scope and stratum."""

    rmse: float
    mae: float
    bias: float
    pearson_r: float
    p50_abs_error: float
    p90_abs_error: float
    delta1: float
    delta1_n_pixels: int
    n_pixels: int


def core_metrics(
    pred: np.ndarray, ref: np.ndarray, mask: np.ndarray
) -> CoreMetrics:
    d1, d1_n = delta1(pred, ref, mask)
    p, _ = _select(pred, ref, mask)
    return CoreMetrics(
        rmse=rmse(pred, ref, mask),
        mae=mae(pred, ref, mask),
        bias=bias(pred, ref, mask),
        pearson_r=pearson_r(pred, ref, mask),
        p50_abs_error=abs_error_percentile(pred, ref, mask, 50.0),
        p90_abs_error=abs_error_percentile(pred, ref, mask, 90.0),
        delta1=d1,
        delta1_n_pixels=d1_n,
        n_pixels=int(p.size),
    )
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
.venv/bin/pytest tests/test_metrics.py -v
```

Expected: 13 passed.

- [ ] **Step 5: Commit**

```bash
git add src/altimap/eval/ tests/test_metrics.py
git commit -m "feat: add core elevation metrics

NaN pixels are excluded unconditionally — a NaN leaking into a sum
turns every metric into NaN, the most common silent failure here.
delta1 restricts to pixels above a height floor because nDSM
background is exactly zero, where its ratio is undefined."
```

---

### Task 4: Scoped metrics — building-pixel, building-wise, height-balanced

Spec §6.1: a single all-pixel RMSE flatters the model badly. Published example — the same model scores 4.89 m all-pixel and 37.47 m height-balanced. These scopes are what make the report honest.

**Files:**
- Create: `src/altimap/eval/scopes.py`
- Create: `tests/test_scopes.py`

**Interfaces:**
- Consumes: `altimap.eval.metrics.core_metrics`, `CoreMetrics`
- Produces:
  - `building_mask(ref: np.ndarray, threshold_m: float = 2.0, min_area_px: int = 9) -> np.ndarray`
  - `building_wise_rmse(pred, ref, mask, threshold_m=2.0, min_area_px=9) -> tuple[float, int]` — returns (rmse, n_instances)
  - `HEIGHT_STRATA: tuple[tuple[float, float], ...]`
  - `height_balanced_rmse(pred, ref, mask) -> tuple[float, dict[str, float]]` — returns (balanced_rmse, per_stratum_rmse)

- [ ] **Step 1: Write the failing test**

Create `tests/test_scopes.py`:

```python
import numpy as np
import pytest

from altimap.eval.scopes import (
    building_mask,
    building_wise_rmse,
    height_balanced_rmse,
)


def test_building_mask_thresholds_height() -> None:
    ref = np.zeros((10, 10), dtype=np.float32)
    ref[2:6, 2:6] = 15.0  # 16 px, above min_area
    mask = building_mask(ref, threshold_m=2.0, min_area_px=9)
    assert mask[3, 3]
    assert not mask[0, 0]
    assert mask.sum() == 16


def test_building_mask_drops_specks() -> None:
    """A single tall pixel is LiDAR noise, not a building."""
    ref = np.zeros((10, 10), dtype=np.float32)
    ref[5, 5] = 30.0
    mask = building_mask(ref, threshold_m=2.0, min_area_px=9)
    assert mask.sum() == 0


def test_building_mask_ignores_nan() -> None:
    ref = np.full((10, 10), np.nan, dtype=np.float32)
    ref[2:6, 2:6] = 15.0
    mask = building_mask(ref)
    assert mask.sum() == 16


def test_building_wise_uses_median_per_instance() -> None:
    """Two buildings; prediction is off by a known amount on each.
    Building-wise RMSE compares one median height per instance."""
    ref = np.zeros((20, 20), dtype=np.float32)
    ref[2:6, 2:6] = 10.0
    ref[12:16, 12:16] = 30.0
    pred = np.zeros((20, 20), dtype=np.float32)
    pred[2:6, 2:6] = 12.0   # +2
    pred[12:16, 12:16] = 27.0  # -3
    mask = np.ones((20, 20), dtype=bool)

    value, n_instances = building_wise_rmse(pred, ref, mask)
    assert n_instances == 2
    # errors +2 and -3 -> sqrt((4+9)/2) = sqrt(6.5)
    assert value == pytest.approx(np.sqrt(6.5))


def test_building_wise_no_buildings_returns_nan() -> None:
    ref = np.zeros((10, 10), dtype=np.float32)
    pred = np.zeros((10, 10), dtype=np.float32)
    mask = np.ones((10, 10), dtype=bool)
    value, n = building_wise_rmse(pred, ref, mask)
    assert np.isnan(value)
    assert n == 0


def test_height_balanced_weights_strata_equally() -> None:
    """The point of this metric: a huge, perfectly-predicted ground area must
    not drown out a small, badly-predicted tall area."""
    ref = np.zeros((100, 100), dtype=np.float32)
    pred = np.zeros((100, 100), dtype=np.float32)
    # One tall patch, predicted 10 m too low
    ref[0:5, 0:5] = 60.0
    pred[0:5, 0:5] = 50.0
    mask = np.ones((100, 100), dtype=bool)

    all_pixel = np.sqrt(np.mean((pred - ref) ** 2))
    balanced, per_stratum = height_balanced_rmse(pred, ref, mask)

    # All-pixel error is diluted to near zero by the 9975 perfect ground pixels
    assert all_pixel < 1.0
    # The balanced metric surfaces the real 10 m failure
    assert balanced > 4.0
    assert per_stratum["50.0-inf"] == pytest.approx(10.0)


def test_height_balanced_ignores_empty_strata() -> None:
    ref = np.zeros((10, 10), dtype=np.float32)
    pred = np.ones((10, 10), dtype=np.float32)
    mask = np.ones((10, 10), dtype=bool)
    balanced, per_stratum = height_balanced_rmse(pred, ref, mask)
    assert balanced == pytest.approx(1.0)
    # Only the lowest stratum is populated
    assert len([v for v in per_stratum.values() if not np.isnan(v)]) == 1
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
.venv/bin/pytest tests/test_scopes.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'altimap.eval.scopes'`

- [ ] **Step 3: Implement the scopes module**

Create `src/altimap/eval/scopes.py`:

```python
"""Metric scopes beyond all-pixel.

All-pixel RMSE over an nDSM is dominated by vast near-zero background, so it
mostly measures how well the model predicts that the ground is at ground
level. Published illustration: one model scoring 4.89 m all-pixel scored
37.47 m height-balanced on the same data. These scopes are what make a report
honest rather than flattering.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from altimap.eval.metrics import rmse

# Strata for the height-balanced metric, in metres. Open-ended at the top so
# the tail — where underestimation lives — always has a home.
HEIGHT_STRATA: tuple[tuple[float, float], ...] = (
    (0.0, 2.0),
    (2.0, 5.0),
    (5.0, 10.0),
    (10.0, 20.0),
    (20.0, 50.0),
    (50.0, float("inf")),
)


def _stratum_label(low: float, high: float) -> str:
    return f"{low}-{'inf' if np.isinf(high) else high}"


def building_mask(
    ref: np.ndarray, threshold_m: float = 2.0, min_area_px: int = 9
) -> np.ndarray:
    """Structures in the reference nDSM: above `threshold_m` and large enough
    to be a building rather than LiDAR speckle.

    Derived from the reference raster, never the prediction — otherwise the
    scope moves with the model being evaluated and the numbers stop being
    comparable between models.
    """
    candidate = np.isfinite(ref) & (ref >= threshold_m)
    if min_area_px <= 1:
        return candidate

    labels, n = ndimage.label(candidate)
    if n == 0:
        return np.zeros_like(candidate, dtype=bool)

    sizes = np.bincount(labels.ravel())
    too_small = np.flatnonzero(sizes < min_area_px)
    keep = ~np.isin(labels, too_small)
    return keep & candidate


def building_wise_rmse(
    pred: np.ndarray,
    ref: np.ndarray,
    mask: np.ndarray,
    threshold_m: float = 2.0,
    min_area_px: int = 9,
) -> tuple[float, int]:
    """One median height per building instance, then RMSE across instances.

    This is how the DFC and GBH literature reports building height, and it
    reflects instance-level utility: a user asks "how tall is that building",
    not "what is the per-pixel error".

    Returns (rmse, n_instances).
    """
    buildings = building_mask(ref, threshold_m, min_area_px) & mask
    labels, n = ndimage.label(buildings)
    if n == 0:
        return float("nan"), 0

    errors = []
    for idx in range(1, n + 1):
        sel = labels == idx
        p_vals = pred[sel]
        r_vals = ref[sel]
        finite = np.isfinite(p_vals) & np.isfinite(r_vals)
        if not finite.any():
            continue
        errors.append(
            float(np.median(p_vals[finite]) - np.median(r_vals[finite]))
        )

    if not errors:
        return float("nan"), 0
    arr = np.asarray(errors, dtype=np.float64)
    return float(np.sqrt(np.mean(arr**2))), len(errors)


def height_balanced_rmse(
    pred: np.ndarray, ref: np.ndarray, mask: np.ndarray
) -> tuple[float, dict[str, float]]:
    """RMSE computed per reference-height stratum, then averaged with equal
    weight per stratum.

    This is the metric that exposes tall-building underestimation, the central
    failure mode of height regression (spec §4.3). Without it, a model that
    systematically halves 80 m towers still posts a respectable all-pixel
    number.

    Returns (balanced_rmse, per_stratum_rmse).
    """
    per_stratum: dict[str, float] = {}
    for low, high in HEIGHT_STRATA:
        stratum = mask & np.isfinite(ref) & (ref >= low) & (ref < high)
        per_stratum[_stratum_label(low, high)] = rmse(pred, ref, stratum)

    populated = [v for v in per_stratum.values() if not np.isnan(v)]
    balanced = float(np.mean(populated)) if populated else float("nan")
    return balanced, per_stratum
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
.venv/bin/pytest tests/test_scopes.py -v
```

Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/altimap/eval/scopes.py tests/test_scopes.py
git commit -m "feat: add building-pixel, building-wise and height-balanced scopes

All-pixel RMSE over an nDSM is dominated by near-zero background.
Height-balanced RMSE is what exposes tall-building underestimation,
the central failure mode of height regression."
```

---

### Task 5: Co-registration alignment

Spec §6: a one-pixel horizontal misalignment dominates RMSE and makes a good model look bad. The harness estimates and reports the correction; it never silently applies it.

**Files:**
- Create: `src/altimap/eval/align.py`
- Create: `tests/test_align.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `Alignment` frozen dataclass with `row_shift: float`, `col_shift: float`, `vertical_bias_m: float`
  - `estimate_alignment(pred: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> Alignment`
  - `apply_alignment(pred: np.ndarray, alignment: Alignment) -> np.ndarray`

- [ ] **Step 1: Write the failing test**

Create `tests/test_align.py`:

```python
import numpy as np
import pytest

from altimap.eval.align import Alignment, apply_alignment, estimate_alignment
from tests.conftest import make_synthetic_ndsm


def test_recovers_known_integer_shift() -> None:
    ref = make_synthetic_ndsm((128, 128))
    # Shift by (+3 rows, -2 cols); np.roll moves content, mimicking
    # a mis-georeferenced prediction
    pred = np.roll(np.roll(ref, 3, axis=0), -2, axis=1)
    mask = np.ones_like(ref, dtype=bool)

    alignment = estimate_alignment(pred, ref, mask)
    assert alignment.row_shift == pytest.approx(-3.0, abs=0.5)
    assert alignment.col_shift == pytest.approx(2.0, abs=0.5)


def test_recovers_vertical_bias() -> None:
    ref = make_synthetic_ndsm((128, 128))
    pred = ref + 4.5
    mask = np.ones_like(ref, dtype=bool)

    alignment = estimate_alignment(pred, ref, mask)
    assert alignment.vertical_bias_m == pytest.approx(4.5, abs=0.1)
    assert alignment.row_shift == pytest.approx(0.0, abs=0.5)


def test_aligned_input_needs_no_correction() -> None:
    ref = make_synthetic_ndsm((128, 128))
    mask = np.ones_like(ref, dtype=bool)
    alignment = estimate_alignment(ref, ref, mask)
    assert alignment.row_shift == pytest.approx(0.0, abs=0.5)
    assert alignment.col_shift == pytest.approx(0.0, abs=0.5)
    assert alignment.vertical_bias_m == pytest.approx(0.0, abs=0.01)


def test_applying_alignment_reduces_error() -> None:
    """The end-to-end property that actually matters, and which is immune to
    sign-convention confusion."""
    ref = make_synthetic_ndsm((128, 128))
    pred = np.roll(np.roll(ref, 3, axis=0), -2, axis=1) + 4.5
    mask = np.ones_like(ref, dtype=bool)

    before = np.sqrt(np.nanmean((pred - ref) ** 2))
    alignment = estimate_alignment(pred, ref, mask)
    corrected = apply_alignment(pred, alignment)
    after = np.sqrt(np.nanmean((corrected - ref) ** 2))

    assert after < before / 5.0


def test_apply_alignment_leaves_identity_unchanged() -> None:
    arr = make_synthetic_ndsm((32, 32))
    identity = Alignment(row_shift=0.0, col_shift=0.0, vertical_bias_m=0.0)
    np.testing.assert_allclose(apply_alignment(arr, identity), arr)


def test_nan_regions_do_not_break_estimation() -> None:
    ref = make_synthetic_ndsm((128, 128))
    pred = ref + 2.0
    pred[0:10, 0:10] = np.nan
    mask = np.ones_like(ref, dtype=bool)
    alignment = estimate_alignment(pred, ref, mask)
    assert np.isfinite(alignment.row_shift)
    assert alignment.vertical_bias_m == pytest.approx(2.0, abs=0.1)
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
.venv/bin/pytest tests/test_align.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'altimap.eval.align'`

- [ ] **Step 3: Implement the alignment module**

Create `src/altimap/eval/align.py`:

```python
"""Co-registration between a predicted and a reference elevation raster.

A one-pixel horizontal misalignment dominates RMSE and makes a good model look
bad. This module measures the offset so it can be *reported*, and optionally
corrected, but never silently corrected — a hidden correction is a thumb on
the scale, and a reviewer should be able to see both numbers (spec §6).
"""

from __future__ import annotations

import dataclasses

import numpy as np
from scipy import ndimage
from skimage.registration import phase_cross_correlation


@dataclasses.dataclass(frozen=True)
class Alignment:
    """Correction to apply to the prediction to align it with the reference.

    `row_shift` and `col_shift` are in pixels; `vertical_bias_m` is the
    constant to subtract from the prediction, in metres.
    """

    row_shift: float
    col_shift: float
    vertical_bias_m: float


def estimate_alignment(
    pred: np.ndarray, ref: np.ndarray, mask: np.ndarray
) -> Alignment:
    """Estimate planar shift by phase correlation and vertical bias by median.

    The median is used for the vertical component rather than the mean because
    elevation residuals have heavy tails — a few gross outliers over water or
    at building edges would drag a mean estimate noticeably.
    """
    valid = mask & np.isfinite(pred) & np.isfinite(ref)
    if not valid.any():
        return Alignment(0.0, 0.0, 0.0)

    # phase_cross_correlation cannot see NaN, so fill holes with the in-mask
    # mean, which is spectrally neutral.
    fill_p = float(np.mean(pred[valid]))
    fill_r = float(np.mean(ref[valid]))
    p_filled = np.where(np.isfinite(pred), pred, fill_p).astype(np.float64)
    r_filled = np.where(np.isfinite(ref), ref, fill_r).astype(np.float64)

    # Returns the shift that must be applied to `moving` to match `reference`.
    shift, _, _ = phase_cross_correlation(
        r_filled, p_filled, upsample_factor=10, normalization=None
    )
    row_shift, col_shift = float(shift[0]), float(shift[1])

    vertical_bias = float(np.median(pred[valid] - ref[valid]))
    return Alignment(row_shift, col_shift, vertical_bias)


def apply_alignment(pred: np.ndarray, alignment: Alignment) -> np.ndarray:
    """Shift and de-bias a prediction. NaN is preserved outside the source."""
    shifted = ndimage.shift(
        pred,
        shift=(alignment.row_shift, alignment.col_shift),
        order=1,
        mode="constant",
        cval=np.nan,
    )
    return shifted - alignment.vertical_bias_m
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
.venv/bin/pytest tests/test_align.py -v
```

Expected: 6 passed. If `test_recovers_known_integer_shift` fails on sign, flip the argument order in `phase_cross_correlation` — but note `test_applying_alignment_reduces_error` is the authoritative check, since it is convention-independent.

- [ ] **Step 5: Commit**

```bash
git add src/altimap/eval/align.py tests/test_align.py
git commit -m "feat: add co-registration estimation and correction

A one-pixel misalignment dominates RMSE. Vertical bias uses the median
rather than the mean because elevation residuals have heavy tails."
```

---

### Task 6: Landscape stratification

Spec §6: the rubric grades stability *across* urban, sparse, hilly and forested landscapes, so per-class numbers are mandatory, not a nicety.

**Files:**
- Create: `src/altimap/eval/stratify.py`
- Create: `tests/test_stratify.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `WORLDCOVER_TO_CLASS: dict[int, str]`
  - `LANDSCAPE_CLASSES: tuple[str, ...]` — `("urban", "sparse", "forested", "hilly")`
  - `landcover_masks(worldcover: np.ndarray) -> dict[str, np.ndarray]`
  - `hilly_mask(dtm: np.ndarray, gsd_m: float, slope_std_threshold_deg: float = 5.0, window_px: int = 16) -> np.ndarray`
  - `water_mask(worldcover: np.ndarray) -> np.ndarray`

- [ ] **Step 1: Write the failing test**

Create `tests/test_stratify.py`:

```python
import numpy as np
import pytest

from altimap.eval.stratify import (
    LANDSCAPE_CLASSES,
    hilly_mask,
    landcover_masks,
    water_mask,
)


def test_worldcover_maps_builtup_to_urban() -> None:
    wc = np.full((4, 4), 50, dtype=np.uint8)  # 50 = built-up
    masks = landcover_masks(wc)
    assert masks["urban"].all()
    assert not masks["forested"].any()


def test_worldcover_maps_treecover_to_forested() -> None:
    wc = np.full((4, 4), 10, dtype=np.uint8)  # 10 = tree cover
    masks = landcover_masks(wc)
    assert masks["forested"].all()
    assert not masks["urban"].any()


def test_worldcover_maps_grass_and_crop_to_sparse() -> None:
    wc = np.array([[30, 40], [20, 60]], dtype=np.uint8)  # grass/crop/shrub/bare
    masks = landcover_masks(wc)
    assert masks["sparse"].all()


def test_water_is_excluded_from_every_landscape_class() -> None:
    """Heights over water are meaningless; including them only adds noise."""
    wc = np.full((4, 4), 80, dtype=np.uint8)  # 80 = permanent water
    masks = landcover_masks(wc)
    for name in ("urban", "sparse", "forested"):
        assert not masks[name].any()
    assert water_mask(wc).all()


def test_unknown_class_belongs_to_no_landscape() -> None:
    wc = np.full((4, 4), 200, dtype=np.uint8)  # not a WorldCover code
    masks = landcover_masks(wc)
    for name in ("urban", "sparse", "forested"):
        assert not masks[name].any()


def test_flat_terrain_is_not_hilly() -> None:
    dtm = np.zeros((64, 64), dtype=np.float32)
    assert not hilly_mask(dtm, gsd_m=2.0).any()


def test_steep_varied_terrain_is_hilly() -> None:
    """A ridged surface with strongly varying slope."""
    yy, xx = np.mgrid[0:64, 0:64]
    dtm = (30.0 * np.sin(xx / 4.0) * np.cos(yy / 5.0)).astype(np.float32)
    assert hilly_mask(dtm, gsd_m=2.0).mean() > 0.5


def test_uniform_slope_is_not_hilly() -> None:
    """A constant ramp is steep but not hilly — slope *variation* is what
    makes terrain hard, so a planar hillside should not qualify."""
    yy, _ = np.mgrid[0:64, 0:64]
    dtm = (yy * 2.0).astype(np.float32)
    assert not hilly_mask(dtm, gsd_m=2.0).any()


def test_landscape_classes_constant_is_complete() -> None:
    assert set(LANDSCAPE_CLASSES) == {"urban", "sparse", "forested", "hilly"}
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
.venv/bin/pytest tests/test_stratify.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'altimap.eval.stratify'`

- [ ] **Step 3: Implement the stratification module**

Create `src/altimap/eval/stratify.py`:

```python
"""Partition a scene into the landscape classes the rubric grades.

Land cover comes from ESA WorldCover 10 m; "hilly" is derived from terrain
instead, because it is a property of relief rather than of surface type. The
two are therefore independent dimensions — an urban pixel can also be hilly,
and reporting them separately is more informative than forcing a single label.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

LANDSCAPE_CLASSES: tuple[str, ...] = ("urban", "sparse", "forested", "hilly")

# ESA WorldCover v200 class codes.
WORLDCOVER_TO_CLASS: dict[int, str] = {
    10: "forested",   # tree cover
    20: "sparse",     # shrubland
    30: "sparse",     # grassland
    40: "sparse",     # cropland
    50: "urban",      # built-up
    60: "sparse",     # bare / sparse vegetation
    70: "excluded",   # snow and ice
    80: "excluded",   # permanent water
    90: "excluded",   # herbaceous wetland
    95: "excluded",   # mangroves
    100: "sparse",    # moss and lichen
}

WATER_CODE = 80


def landcover_masks(worldcover: np.ndarray) -> dict[str, np.ndarray]:
    """Boolean mask per land-cover-derived landscape class.

    Codes mapped to "excluded" (water, snow, wetland) and codes absent from
    the mapping belong to no class, so they are dropped from every stratum
    rather than silently landing in one.
    """
    masks = {
        name: np.zeros(worldcover.shape, dtype=bool)
        for name in ("urban", "sparse", "forested")
    }
    for code, name in WORLDCOVER_TO_CLASS.items():
        if name in masks:
            masks[name] |= worldcover == code
    return masks


def water_mask(worldcover: np.ndarray) -> np.ndarray:
    """Permanent water. Heights here are meaningless and must be masked out
    of every metric (spec §6)."""
    return worldcover == WATER_CODE


def hilly_mask(
    dtm: np.ndarray,
    gsd_m: float,
    slope_std_threshold_deg: float = 5.0,
    window_px: int = 16,
) -> np.ndarray:
    """Terrain whose *slope varies* strongly within a local window.

    Slope variation rather than slope magnitude: a planar hillside is steep
    but geometrically simple, while a ridged, broken surface is what actually
    challenges height estimation.
    """
    filled = np.where(np.isfinite(dtm), dtm, np.nanmean(dtm)).astype(np.float64)
    dz_dy, dz_dx = np.gradient(filled, gsd_m)
    slope_deg = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy)))

    mean = ndimage.uniform_filter(slope_deg, size=window_px)
    mean_sq = ndimage.uniform_filter(slope_deg**2, size=window_px)
    local_std = np.sqrt(np.maximum(mean_sq - mean**2, 0.0))

    return local_std > slope_std_threshold_deg
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
.venv/bin/pytest tests/test_stratify.py -v
```

Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add src/altimap/eval/stratify.py tests/test_stratify.py
git commit -m "feat: add landscape stratification for per-class reporting

Hilly is derived from slope *variation*, not magnitude — a planar
hillside is steep but geometrically simple, while broken terrain is
what actually challenges height estimation."
```

---

### Task 7: Report assembly

Combines every scope and stratum into one structure, and renders it as JSON plus a markdown table. Both the CLI (Task 8) and the viewer's validation panel consume this, so the demo and the report can never disagree (spec §5.3).

**Files:**
- Create: `src/altimap/eval/report.py`
- Create: `tests/test_report.py`

**Interfaces:**
- Consumes: `core_metrics`, `CoreMetrics`, `building_mask`, `building_wise_rmse`, `height_balanced_rmse`, `estimate_alignment`, `apply_alignment`, `Alignment`, `landcover_masks`, `water_mask`, `hilly_mask`
- Produces:
  - `EvalReport` frozen dataclass with `all_pixel: CoreMetrics`, `building_pixel: CoreMetrics`, `building_wise_rmse: float`, `building_count: int`, `height_balanced_rmse: float`, `height_strata: dict[str, float]`, `per_landscape: dict[str, CoreMetrics]`, `alignment: Alignment`, `corrected_all_pixel: CoreMetrics | None`, `gsd_m: float`, `dataset: str`
  - `build_report(pred, ref, *, gsd_m, dataset, worldcover=None, dtm=None, apply_correction=True) -> EvalReport`
  - `report_to_dict(report: EvalReport) -> dict`
  - `report_to_markdown(report: EvalReport) -> str`

- [ ] **Step 1: Write the failing test**

Create `tests/test_report.py`:

```python
import json

import numpy as np
import pytest

from altimap.eval.report import build_report, report_to_dict, report_to_markdown
from tests.conftest import make_synthetic_ndsm


@pytest.fixture
def pred_ref() -> tuple[np.ndarray, np.ndarray]:
    ref = make_synthetic_ndsm((128, 128))
    rng = np.random.default_rng(7)
    pred = ref + rng.normal(0.0, 0.5, ref.shape).astype(np.float32)
    return pred, ref


def test_report_has_all_scopes(pred_ref) -> None:
    pred, ref = pred_ref
    r = build_report(pred, ref, gsd_m=2.0, dataset="synthetic")
    assert r.all_pixel.rmse > 0
    assert r.building_pixel.rmse > 0
    assert r.building_count == 3  # three synthetic buildings
    assert np.isfinite(r.building_wise_rmse)
    assert np.isfinite(r.height_balanced_rmse)


def test_report_records_gsd_and_dataset(pred_ref) -> None:
    """An RMSE without dataset and GSD is uninterpretable (spec §6.2)."""
    pred, ref = pred_ref
    r = build_report(pred, ref, gsd_m=2.0, dataset="dfc2023")
    assert r.gsd_m == 2.0
    assert r.dataset == "dfc2023"


def test_report_reports_both_raw_and_corrected(pred_ref) -> None:
    """The correction must be visible, never silent."""
    pred, ref = pred_ref
    shifted = pred + 3.0
    r = build_report(shifted, ref, gsd_m=2.0, dataset="synthetic")
    assert r.alignment.vertical_bias_m == pytest.approx(3.0, abs=0.2)
    assert r.corrected_all_pixel is not None
    assert r.corrected_all_pixel.rmse < r.all_pixel.rmse


def test_correction_can_be_disabled(pred_ref) -> None:
    pred, ref = pred_ref
    r = build_report(pred, ref, gsd_m=2.0, dataset="s", apply_correction=False)
    assert r.corrected_all_pixel is None


def test_water_is_excluded_from_metrics(pred_ref) -> None:
    pred, ref = pred_ref
    wc = np.full(ref.shape, 50, dtype=np.uint8)   # built-up
    wc[0:64, :] = 80                              # top half is water
    corrupted = pred.copy()
    corrupted[0:64, :] = 1000.0                   # nonsense over water

    r = build_report(
        corrupted, ref, gsd_m=2.0, dataset="s", worldcover=wc,
        apply_correction=False,
    )
    # Water is masked out, so the nonsense must not reach the metric
    assert r.all_pixel.rmse < 5.0


def test_per_landscape_populated_when_worldcover_given(pred_ref) -> None:
    pred, ref = pred_ref
    wc = np.full(ref.shape, 50, dtype=np.uint8)
    wc[64:, :] = 10  # bottom half tree cover
    r = build_report(pred, ref, gsd_m=2.0, dataset="s", worldcover=wc)
    assert "urban" in r.per_landscape
    assert "forested" in r.per_landscape
    assert r.per_landscape["urban"].n_pixels > 0
    assert r.per_landscape["forested"].n_pixels > 0


def test_report_serialises_to_json(pred_ref) -> None:
    pred, ref = pred_ref
    r = build_report(pred, ref, gsd_m=2.0, dataset="s")
    payload = report_to_dict(r)
    text = json.dumps(payload)  # must not raise
    assert "all_pixel" in json.loads(text)


def test_markdown_states_gsd_and_scope(pred_ref) -> None:
    pred, ref = pred_ref
    r = build_report(pred, ref, gsd_m=2.0, dataset="synthetic")
    md = report_to_markdown(r)
    assert "2.0" in md
    assert "synthetic" in md
    assert "Building-wise" in md
    assert "Height-balanced" in md
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
.venv/bin/pytest tests/test_report.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'altimap.eval.report'`

- [ ] **Step 3: Implement the report module**

Create `src/altimap/eval/report.py`:

```python
"""Assemble the metric matrix into a report.

Consumed by both the CLI and the viewer's validation panel, so the demo and
the written report are computed by the same code and cannot disagree.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from altimap.eval.align import Alignment, apply_alignment, estimate_alignment
from altimap.eval.metrics import CoreMetrics, core_metrics
from altimap.eval.scopes import (
    building_mask,
    building_wise_rmse,
    height_balanced_rmse,
)
from altimap.eval.stratify import hilly_mask, landcover_masks, water_mask


@dataclasses.dataclass(frozen=True)
class EvalReport:
    dataset: str
    gsd_m: float
    all_pixel: CoreMetrics
    building_pixel: CoreMetrics
    building_wise_rmse: float
    building_count: int
    height_balanced_rmse: float
    height_strata: dict[str, float]
    per_landscape: dict[str, CoreMetrics]
    alignment: Alignment
    corrected_all_pixel: CoreMetrics | None


def build_report(
    pred: np.ndarray,
    ref: np.ndarray,
    *,
    gsd_m: float,
    dataset: str,
    worldcover: np.ndarray | None = None,
    dtm: np.ndarray | None = None,
    apply_correction: bool = True,
) -> EvalReport:
    """Score `pred` against `ref` across every scope and stratum.

    `worldcover` enables per-landscape reporting and water masking; `dtm`
    enables the hilly stratum. Both optional so the harness stays usable on
    a bare pair of rasters.
    """
    if pred.shape != ref.shape:
        raise ValueError(
            f"prediction shape {pred.shape} does not match reference {ref.shape}"
        )

    base = np.ones(ref.shape, dtype=bool)
    if worldcover is not None:
        if worldcover.shape != ref.shape:
            raise ValueError(
                f"worldcover shape {worldcover.shape} does not match "
                f"reference {ref.shape}"
            )
        base &= ~water_mask(worldcover)

    buildings = building_mask(ref) & base
    bw_rmse, bw_count = building_wise_rmse(pred, ref, base)
    hb_rmse, strata = height_balanced_rmse(pred, ref, base)

    per_landscape: dict[str, CoreMetrics] = {}
    if worldcover is not None:
        for name, lc_mask in landcover_masks(worldcover).items():
            selection = lc_mask & base
            if selection.any():
                per_landscape[name] = core_metrics(pred, ref, selection)
    if dtm is not None:
        hilly = hilly_mask(dtm, gsd_m) & base
        if hilly.any():
            per_landscape["hilly"] = core_metrics(pred, ref, hilly)

    alignment = estimate_alignment(pred, ref, base)
    corrected = (
        core_metrics(apply_alignment(pred, alignment), ref, base)
        if apply_correction
        else None
    )

    return EvalReport(
        dataset=dataset,
        gsd_m=gsd_m,
        all_pixel=core_metrics(pred, ref, base),
        building_pixel=core_metrics(pred, ref, buildings),
        building_wise_rmse=bw_rmse,
        building_count=bw_count,
        height_balanced_rmse=hb_rmse,
        height_strata=strata,
        per_landscape=per_landscape,
        alignment=alignment,
        corrected_all_pixel=corrected,
    )


def report_to_dict(report: EvalReport) -> dict[str, Any]:
    """JSON-serialisable form. NaN becomes None so the output is valid JSON."""

    def clean(value: Any) -> Any:
        if isinstance(value, float) and not np.isfinite(value):
            return None
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return {k: clean(v) for k, v in dataclasses.asdict(value).items()}
        return value

    return {k: clean(v) for k, v in dataclasses.asdict(report).items()}


def _fmt(value: float) -> str:
    return "—" if not np.isfinite(value) else f"{value:.3f}"


def report_to_markdown(report: EvalReport) -> str:
    """Render the report. Every number carries dataset, GSD, and scope,
    because an unqualified RMSE is uninterpretable (spec §6.2)."""
    lines = [
        f"# Evaluation — {report.dataset} @ {report.gsd_m} m GSD",
        "",
        "All heights in metres. Scope is stated for every figure; an RMSE",
        "without dataset, GSD, and scope cannot be compared to anything.",
        "",
        "## Primary metrics",
        "",
        "| Scope | RMSE | MAE | Bias | Pearson r | P50 |abs| | P90 |abs| | N |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for label, m in (
        ("All pixels", report.all_pixel),
        ("Building pixels", report.building_pixel),
    ):
        lines.append(
            f"| {label} | {_fmt(m.rmse)} | {_fmt(m.mae)} | {_fmt(m.bias)} | "
            f"{_fmt(m.pearson_r)} | {_fmt(m.p50_abs_error)} | "
            f"{_fmt(m.p90_abs_error)} | {m.n_pixels} |"
        )

    lines += [
        "",
        f"- **Building-wise RMSE:** {_fmt(report.building_wise_rmse)} "
        f"({report.building_count} instances)",
        f"- **Height-balanced RMSE:** {_fmt(report.height_balanced_rmse)}",
        f"- **δ1 (all pixels, ref ≥ 1 m):** {_fmt(report.all_pixel.delta1)} "
        f"over {report.all_pixel.delta1_n_pixels} px",
        "",
        "## Height strata",
        "",
        "| Reference height (m) | RMSE |",
        "|---|---|",
    ]
    for label, value in report.height_strata.items():
        lines.append(f"| {label} | {_fmt(value)} |")

    if report.per_landscape:
        lines += [
            "",
            "## Per landscape class",
            "",
            "| Class | RMSE | MAE | N |",
            "|---|---|---|---|",
        ]
        for name, m in sorted(report.per_landscape.items()):
            lines.append(
                f"| {name} | {_fmt(m.rmse)} | {_fmt(m.mae)} | {m.n_pixels} |"
            )

    lines += [
        "",
        "## Co-registration",
        "",
        "Reported, not silently applied.",
        "",
        f"- Estimated shift: {report.alignment.row_shift:+.2f} rows, "
        f"{report.alignment.col_shift:+.2f} cols",
        f"- Estimated vertical bias: {report.alignment.vertical_bias_m:+.3f} m",
    ]
    if report.corrected_all_pixel is not None:
        lines.append(
            f"- All-pixel RMSE after correction: "
            f"{_fmt(report.corrected_all_pixel.rmse)} "
            f"(raw {_fmt(report.all_pixel.rmse)})"
        )
    return "\n".join(lines) + "\n"
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
.venv/bin/pytest tests/test_report.py -v
```

Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add src/altimap/eval/report.py tests/test_report.py
git commit -m "feat: assemble the evaluation metric matrix into a report

Same code path serves the CLI and the viewer's validation panel, so
the demo and the written report cannot disagree."
```

---

### Task 8: The `eval` CLI

**Files:**
- Create: `src/altimap/eval/cli.py`
- Create: `tests/test_eval_cli.py`

**Interfaces:**
- Consumes: `read_elevation`, `build_report`, `report_to_dict`, `report_to_markdown`
- Produces: `main(argv: list[str] | None = None) -> int`; console script `altimap-eval`

- [ ] **Step 1: Write the failing test**

Create `tests/test_eval_cli.py`:

```python
import json
from pathlib import Path

import numpy as np
import pytest
from rasterio.crs import CRS
from rasterio.transform import from_origin

from altimap.contract import Sidecar, write_elevation_cog
from altimap.eval.cli import main
from tests.conftest import make_synthetic_ndsm


def _write(path: Path, array: np.ndarray) -> None:
    write_elevation_cog(
        path,
        array,
        from_origin(500000.0, 4500000.0, 2.0, 2.0),
        CRS.from_epsg(32612),
        Sidecar(
            gsd_m=2.0,
            source_gsd_m=0.6,
            datum="ellipsoidal",
            vertical_unit="m",
            model_version="test",
            height_range_m=(float(np.nanmin(array)), float(np.nanmax(array))),
            tile_overlap_px=0,
            dtm_source=None,
        ),
    )


@pytest.fixture
def raster_pair(tmp_path: Path) -> tuple[Path, Path]:
    ref = make_synthetic_ndsm((128, 128))
    rng = np.random.default_rng(11)
    pred = ref + rng.normal(0.0, 0.4, ref.shape).astype(np.float32)
    pred_path, ref_path = tmp_path / "pred.tif", tmp_path / "ref.tif"
    _write(pred_path, pred)
    _write(ref_path, ref)
    return pred_path, ref_path


def test_cli_writes_json_and_markdown(raster_pair, tmp_path: Path) -> None:
    pred_path, ref_path = raster_pair
    out = tmp_path / "report.json"
    code = main(
        [str(pred_path), str(ref_path), "--out", str(out), "--dataset", "synthetic"]
    )
    assert code == 0
    assert out.exists()
    assert out.with_suffix(".md").exists()

    payload = json.loads(out.read_text())
    assert payload["dataset"] == "synthetic"
    assert payload["gsd_m"] == 2.0
    assert payload["all_pixel"]["rmse"] > 0


def test_cli_reads_gsd_from_sidecar(raster_pair, tmp_path: Path) -> None:
    pred_path, ref_path = raster_pair
    out = tmp_path / "report.json"
    main([str(pred_path), str(ref_path), "--out", str(out)])
    assert json.loads(out.read_text())["gsd_m"] == 2.0


def test_cli_rejects_shape_mismatch(tmp_path: Path) -> None:
    _write(tmp_path / "a.tif", make_synthetic_ndsm((64, 64)))
    _write(tmp_path / "b.tif", make_synthetic_ndsm((32, 32)))
    code = main(
        [
            str(tmp_path / "a.tif"),
            str(tmp_path / "b.tif"),
            "--out",
            str(tmp_path / "r.json"),
        ]
    )
    assert code == 1


def test_cli_missing_file_returns_error(tmp_path: Path) -> None:
    code = main(
        [
            str(tmp_path / "nope.tif"),
            str(tmp_path / "also-nope.tif"),
            "--out",
            str(tmp_path / "r.json"),
        ]
    )
    assert code == 1
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
.venv/bin/pytest tests/test_eval_cli.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'altimap.eval.cli'`

- [ ] **Step 3: Implement the CLI**

Create `src/altimap/eval/cli.py`:

```python
"""Command-line entry point: score a predicted elevation raster."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rasterio

from altimap.contract import Sidecar, read_elevation, sidecar_path_for
from altimap.eval.report import build_report, report_to_dict, report_to_markdown


def _read_aux(path: Path | None, shape: tuple[int, int]) -> np.ndarray | None:
    """Read an auxiliary raster (land cover, DTM) and check it aligns."""
    if path is None:
        return None
    with rasterio.open(path) as src:
        arr = src.read(1)
    if arr.shape != shape:
        raise ValueError(
            f"{path.name} has shape {arr.shape}, expected {shape}. "
            "Reproject it onto the reference grid first."
        )
    return arr


def _resolve_gsd(pred_path: Path, override: float | None) -> float:
    if override is not None:
        return override
    sidecar = sidecar_path_for(pred_path)
    if sidecar.exists():
        return Sidecar.from_json(sidecar).gsd_m
    with rasterio.open(pred_path) as src:
        return float(abs(src.transform.a))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="altimap-eval",
        description="Score a predicted elevation raster against a reference.",
    )
    parser.add_argument("predicted", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--out", type=Path, required=True, help="JSON output path")
    parser.add_argument("--dataset", default="unknown", help="Name for the report")
    parser.add_argument("--landcover", type=Path, help="ESA WorldCover raster")
    parser.add_argument("--dtm", type=Path, help="Bare-earth DEM, for the hilly stratum")
    parser.add_argument("--gsd", type=float, help="Override GSD in metres")
    parser.add_argument(
        "--no-correction",
        action="store_true",
        help="Skip the co-registration-corrected figures",
    )
    args = parser.parse_args(argv)

    try:
        pred, _, _ = read_elevation(args.predicted)
        ref, _, _ = read_elevation(args.reference)
        report = build_report(
            pred,
            ref,
            gsd_m=_resolve_gsd(args.predicted, args.gsd),
            dataset=args.dataset,
            worldcover=_read_aux(args.landcover, ref.shape),
            dtm=_read_aux(args.dtm, ref.shape),
            apply_correction=not args.no_correction,
        )
    except (rasterio.errors.RasterioIOError, OSError) as exc:
        print(f"error: cannot read input: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report_to_dict(report), indent=2) + "\n")
    markdown = report_to_markdown(report)
    args.out.with_suffix(".md").write_text(markdown)
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
.venv/bin/pytest tests/test_eval_cli.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Verify the console script is installed**

```bash
.venv/bin/altimap-eval --help
```

Expected: prints usage. If `command not found`, re-run `uv pip install -e ".[dev]"`.

- [ ] **Step 6: Run the whole suite**

```bash
.venv/bin/pytest -v
```

Expected: all tests pass (44 total across Tasks 2–8).

- [ ] **Step 7: Commit**

```bash
git add src/altimap/eval/cli.py tests/test_eval_cli.py
git commit -m "feat: add altimap-eval CLI

Phase 1 measurement foundation complete: the accuracy half of the
rubric is now measurable before any model exists."
```

---

### Task 9: Planetary Computer STAC client

Spec §7. Network-dependent, so tests are split: pure logic is unit-tested offline, live access is marked `network` and skipped by default.

**Files:**
- Create: `src/altimap/data/__init__.py`
- Create: `src/altimap/data/stac.py`
- Create: `tests/test_stac.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `STAC_URL: str`
  - `COLLECTIONS: dict[str, str]` — logical name → collection id
  - `ASSET_KEYS: dict[str, str]` — collection id → asset key
  - `AOI` frozen dataclass with `name: str`, `bbox: tuple[float, float, float, float]`, `landscape: str`
  - `open_client() -> pystac_client.Client`
  - `find_items(client, collection: str, bbox, limit: int = 50) -> list`
  - `coverage_counts(client, bbox, collections=None) -> dict[str, int]`
  - `is_viable(counts: dict[str, int]) -> bool`

- [ ] **Step 1: Write the failing test**

Create `tests/test_stac.py`:

```python
import pytest

from altimap.data.stac import (
    AOI,
    ASSET_KEYS,
    COLLECTIONS,
    coverage_counts,
    is_viable,
    open_client,
)


def test_collections_include_imagery_and_labels() -> None:
    assert COLLECTIONS["imagery"] == "naip"
    assert COLLECTIONS["ndsm"] == "3dep-lidar-hag"
    assert COLLECTIONS["dtm"] == "3dep-seamless"
    assert COLLECTIONS["landcover"] == "esa-worldcover"


def test_asset_keys_defined_for_every_collection() -> None:
    for cid in COLLECTIONS.values():
        assert cid in ASSET_KEYS


def test_aoi_rejects_malformed_bbox() -> None:
    with pytest.raises(ValueError, match="bbox"):
        AOI(name="bad", bbox=(1.0, 2.0, 0.0, 3.0), landscape="urban")


def test_aoi_rejects_unknown_landscape() -> None:
    with pytest.raises(ValueError, match="landscape"):
        AOI(name="bad", bbox=(-1.0, -1.0, 1.0, 1.0), landscape="tundra")


def test_aoi_accepts_valid_input() -> None:
    aoi = AOI(name="slc", bbox=(-111.95, 40.72, -111.85, 40.80), landscape="urban")
    assert aoi.name == "slc"


def test_is_viable_requires_both_imagery_and_labels() -> None:
    assert is_viable({"naip": 4, "3dep-lidar-hag": 2})
    assert not is_viable({"naip": 4, "3dep-lidar-hag": 0})
    assert not is_viable({"naip": 0, "3dep-lidar-hag": 3})
    assert not is_viable({"naip": 4})


@pytest.mark.network
def test_live_coverage_over_known_good_aoi() -> None:
    """Wasatch, Utah — confirmed to have both NAIP and HAG coverage."""
    client = open_client()
    counts = coverage_counts(client, (-111.95, 40.55, -111.85, 40.65))
    assert counts["naip"] > 0
    assert counts["3dep-lidar-hag"] > 0
    assert is_viable(counts)
```

- [ ] **Step 2: Run the offline tests to verify they fail**

```bash
.venv/bin/pytest tests/test_stac.py -v -m "not network"
```

Expected: FAIL — `ModuleNotFoundError: No module named 'altimap.data'`

- [ ] **Step 3: Implement the STAC client**

Create `src/altimap/data/__init__.py`:

```python
"""Training-data acquisition. Reads windows from remote COGs; never
downloads whole rasters."""
```

Create `src/altimap/data/stac.py`:

```python
"""Microsoft Planetary Computer STAC access.

Asset keys are hard-coded from the Task 1 spike rather than discovered at
runtime, so a change upstream fails loudly in one place instead of producing
mysterious KeyErrors deep in the extraction loop.
"""

from __future__ import annotations

import dataclasses

import planetary_computer
import pystac_client

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

COLLECTIONS: dict[str, str] = {
    "imagery": "naip",
    "ndsm": "3dep-lidar-hag",
    "dtm": "3dep-seamless",
    "landcover": "esa-worldcover",
}

# Verified in the Task 1 spike. Correct these if the spike findings differ.
ASSET_KEYS: dict[str, str] = {
    "naip": "image",
    "3dep-lidar-hag": "data",
    "3dep-seamless": "data",
    "esa-worldcover": "map",
}

VALID_LANDSCAPES = ("urban", "sparse", "forested", "hilly")

REQUIRED_FOR_TRAINING = (COLLECTIONS["imagery"], COLLECTIONS["ndsm"])


@dataclasses.dataclass(frozen=True)
class AOI:
    """A candidate training area."""

    name: str
    bbox: tuple[float, float, float, float]  # west, south, east, north (EPSG:4326)
    landscape: str

    def __post_init__(self) -> None:
        west, south, east, north = self.bbox
        if west >= east or south >= north:
            raise ValueError(
                f"bbox must be (west, south, east, north) with west<east and "
                f"south<north, got {self.bbox}"
            )
        if self.landscape not in VALID_LANDSCAPES:
            raise ValueError(
                f"landscape must be one of {VALID_LANDSCAPES}, "
                f"got {self.landscape!r}"
            )


def open_client() -> pystac_client.Client:
    """A signing client. Planetary Computer needs no API key, but assets
    require a short-lived SAS token that `sign_inplace` attaches."""
    return pystac_client.Client.open(
        STAC_URL, modifier=planetary_computer.sign_inplace
    )


def find_items(
    client: pystac_client.Client,
    collection: str,
    bbox: tuple[float, float, float, float],
    limit: int = 50,
) -> list:
    return list(client.search(collections=[collection], bbox=bbox, limit=limit).items())


def coverage_counts(
    client: pystac_client.Client,
    bbox: tuple[float, float, float, float],
    collections: list[str] | None = None,
) -> dict[str, int]:
    """Item count per collection over a bbox.

    Coverage for 3DEP is workunit-based, not seamless, so this must be checked
    per AOI before committing to it (spec §7.2).
    """
    targets = collections or list(COLLECTIONS.values())
    return {cid: len(find_items(client, cid, bbox)) for cid in targets}


def is_viable(counts: dict[str, int]) -> bool:
    """An AOI is usable for training only if both imagery and labels exist."""
    return all(counts.get(cid, 0) > 0 for cid in REQUIRED_FOR_TRAINING)
```

- [ ] **Step 4: Run the offline tests to verify they pass**

```bash
.venv/bin/pytest tests/test_stac.py -v -m "not network"
```

Expected: 6 passed, 1 deselected.

- [ ] **Step 5: Run the network test explicitly**

```bash
.venv/bin/pytest tests/test_stac.py -v -m network
```

Expected: 1 passed. If it fails, cross-check the asset keys against the Task 1 spike findings and correct `ASSET_KEYS`.

- [ ] **Step 6: Commit**

```bash
git add src/altimap/data/ tests/test_stac.py
git commit -m "feat: add Planetary Computer STAC client

Asset keys are hard-coded from the spike so an upstream change fails
loudly in one place. Live tests are marked 'network' and deselected
by default."
```

---

### Task 10: Patch extraction from remote COGs

The payoff of spec §7.1 — training patches read as windows, never downloading the 467 MB source rasters.

**Files:**
- Create: `src/altimap/data/patches.py`
- Create: `tests/test_patches.py`

**Interfaces:**
- Consumes: `altimap.data.stac.ASSET_KEYS`
- Produces:
  - `PatchSpec` frozen dataclass with `size_px: int`, `source_gsd_m: float`, `label_gsd_m: float`
  - `DEFAULT_PATCH_SPEC: PatchSpec`
  - `patch_windows(width: int, height: int, size_px: int, stride_px: int) -> Iterator[Window]`
  - `read_patch_pair(rgb_src, label_src, window, spec) -> tuple[np.ndarray, np.ndarray] | None`
  - `is_patch_usable(rgb: np.ndarray, label: np.ndarray, max_nodata_frac: float = 0.05) -> bool`
  - `save_patch_pair(out_dir: Path, patch_id: str, rgb: np.ndarray, label: np.ndarray) -> None`

- [ ] **Step 1: Write the failing test**

Create `tests/test_patches.py`:

```python
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin
from rasterio.windows import Window

from altimap.data.patches import (
    DEFAULT_PATCH_SPEC,
    PatchSpec,
    is_patch_usable,
    patch_windows,
    read_patch_pair,
    save_patch_pair,
)


def test_default_spec_matches_project_resolutions() -> None:
    """Global Constraints: 0.6 m input, 2.0 m labels."""
    assert DEFAULT_PATCH_SPEC.source_gsd_m == 0.6
    assert DEFAULT_PATCH_SPEC.label_gsd_m == 2.0
    assert DEFAULT_PATCH_SPEC.size_px == 518


def test_patch_windows_tile_without_overrun() -> None:
    windows = list(patch_windows(100, 100, size_px=40, stride_px=40))
    assert len(windows) == 4  # 2x2; the ragged remainder is dropped
    for w in windows:
        assert w.width == 40 and w.height == 40
        assert w.col_off + w.width <= 100
        assert w.row_off + w.height <= 100


def test_patch_windows_overlap_with_smaller_stride() -> None:
    windows = list(patch_windows(100, 100, size_px=40, stride_px=20))
    assert len(windows) == 16  # 4x4
    assert windows[1].col_off == 20


def test_patch_windows_empty_when_raster_too_small() -> None:
    assert list(patch_windows(10, 10, size_px=40, stride_px=40)) == []


def test_is_patch_usable_rejects_excess_nodata() -> None:
    rgb = np.zeros((3, 10, 10), dtype=np.uint8)
    label = np.zeros((10, 10), dtype=np.float32)
    label[0:5, :] = np.nan  # 50% nodata
    assert not is_patch_usable(rgb, label, max_nodata_frac=0.05)


def test_is_patch_usable_accepts_clean_patch() -> None:
    rgb = np.full((3, 10, 10), 128, dtype=np.uint8)
    label = np.zeros((10, 10), dtype=np.float32)
    assert is_patch_usable(rgb, label)


def test_is_patch_usable_rejects_all_black_imagery() -> None:
    """An all-zero RGB patch is off the edge of the orthophoto, not a scene."""
    rgb = np.zeros((3, 10, 10), dtype=np.uint8)
    label = np.zeros((10, 10), dtype=np.float32)
    assert not is_patch_usable(rgb, label)


@pytest.fixture
def local_pair(tmp_path: Path) -> tuple[Path, Path]:
    """A local RGB/label pair standing in for remote COGs, at the project's
    0.6 m / 2.0 m resolutions."""
    rgb_path = tmp_path / "rgb.tif"
    label_path = tmp_path / "label.tif"
    crs = CRS.from_epsg(32612)
    origin = (500000.0, 4500000.0)

    rng = np.random.default_rng(3)
    rgb = rng.integers(30, 220, (3, 600, 600), dtype=np.uint8)
    with rasterio.open(
        rgb_path, "w", driver="GTiff", height=600, width=600, count=3,
        dtype="uint8", crs=crs,
        transform=from_origin(*origin, 0.6, 0.6), tiled=True,
        blockxsize=256, blockysize=256,
    ) as dst:
        dst.write(rgb)

    label = (rng.random((180, 180)) * 20.0).astype(np.float32)
    with rasterio.open(
        label_path, "w", driver="GTiff", height=180, width=180, count=1,
        dtype="float32", crs=crs, nodata=np.nan,
        transform=from_origin(*origin, 2.0, 2.0), tiled=True,
        blockxsize=128, blockysize=128,
    ) as dst:
        dst.write(label, 1)

    return rgb_path, label_path


def test_read_patch_pair_returns_matching_scales(local_pair) -> None:
    rgb_path, label_path = local_pair
    spec = PatchSpec(size_px=120, source_gsd_m=0.6, label_gsd_m=2.0)
    with rasterio.open(rgb_path) as rgb_src, rasterio.open(label_path) as lbl_src:
        result = read_patch_pair(rgb_src, lbl_src, Window(0, 0, 120, 120), spec)

    assert result is not None
    rgb, label = result
    assert rgb.shape == (3, 120, 120)
    # 120 px at 0.6 m = 72 m; at 2.0 m that is 36 px
    assert label.shape == (36, 36)


def test_read_patch_pair_label_is_float32_metres(local_pair) -> None:
    rgb_path, label_path = local_pair
    spec = PatchSpec(size_px=120, source_gsd_m=0.6, label_gsd_m=2.0)
    with rasterio.open(rgb_path) as rgb_src, rasterio.open(label_path) as lbl_src:
        rgb, label = read_patch_pair(rgb_src, lbl_src, Window(0, 0, 120, 120), spec)
    assert label.dtype == np.float32
    assert rgb.dtype == np.uint8


def test_save_patch_pair_writes_both_arrays(tmp_path: Path) -> None:
    rgb = np.full((3, 8, 8), 42, dtype=np.uint8)
    label = np.full((4, 4), 3.5, dtype=np.float32)
    save_patch_pair(tmp_path, "patch_0001", rgb, label)

    loaded = np.load(tmp_path / "patch_0001.npz")
    np.testing.assert_array_equal(loaded["rgb"], rgb)
    np.testing.assert_allclose(loaded["label"], label)
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
.venv/bin/pytest tests/test_patches.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'altimap.data.patches'`

- [ ] **Step 3: Implement patch extraction**

Create `src/altimap/data/patches.py`:

```python
"""Extract training patches as windows from Cloud-Optimized GeoTIFFs.

Reading windows means only the internal COG tiles actually touched are
fetched, so a 467 MB NAIP scene costs kilobytes per patch. This is what makes
training viable on 32 GB of free disk (spec §7.1).

Imagery and labels sit on different grids — 0.6 m and 2.0 m — so a patch is
defined by its *ground footprint* and each source is read at its own native
resolution. Resampling the label up to imagery resolution would invent detail
that cannot be validated (spec §4.4).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.windows import Window


@dataclasses.dataclass(frozen=True)
class PatchSpec:
    size_px: int          # patch edge in *source imagery* pixels
    source_gsd_m: float
    label_gsd_m: float

    def __post_init__(self) -> None:
        if self.size_px <= 0:
            raise ValueError(f"size_px must be positive, got {self.size_px}")
        if self.label_gsd_m < self.source_gsd_m:
            raise ValueError(
                "label_gsd_m must be coarser than or equal to source_gsd_m; "
                "upsampling labels invents unvalidatable detail"
            )

    @property
    def footprint_m(self) -> float:
        return self.size_px * self.source_gsd_m

    @property
    def label_size_px(self) -> int:
        return int(round(self.footprint_m / self.label_gsd_m))


# 518 px is Depth Anything's native input size and a multiple of its patch
# size 14, so this avoids a resize at training time.
DEFAULT_PATCH_SPEC = PatchSpec(size_px=518, source_gsd_m=0.6, label_gsd_m=2.0)


def patch_windows(
    width: int, height: int, size_px: int, stride_px: int
) -> Iterator[Window]:
    """Tile a raster into square windows, dropping the ragged remainder.

    Partial edge patches are discarded rather than padded — padding
    introduces synthetic zero regions that the model would learn to predict.
    """
    for row in range(0, height - size_px + 1, stride_px):
        for col in range(0, width - size_px + 1, stride_px):
            yield Window(col, row, size_px, size_px)


def read_patch_pair(
    rgb_src: rasterio.DatasetReader,
    label_src: rasterio.DatasetReader,
    window: Window,
    spec: PatchSpec,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Read a co-located RGB patch and label patch.

    Returns None if the window's ground footprint falls outside the label
    raster, which happens routinely since imagery and label coverage differ.
    """
    rgb = rgb_src.read(
        indexes=[1, 2, 3], window=window, boundless=True, fill_value=0
    )

    bounds = rasterio.windows.bounds(window, rgb_src.transform)
    label_size = spec.label_size_px
    dst_transform = rasterio.transform.from_bounds(*bounds, label_size, label_size)
    label = np.full((label_size, label_size), np.nan, dtype=np.float32)

    reproject(
        source=rasterio.band(label_src, 1),
        destination=label,
        src_transform=label_src.transform,
        src_crs=label_src.crs,
        dst_transform=dst_transform,
        dst_crs=rgb_src.crs,
        dst_nodata=np.nan,
        resampling=Resampling.bilinear,
    )

    if not np.isfinite(label).any():
        return None
    return rgb.astype(np.uint8), label


def is_patch_usable(
    rgb: np.ndarray, label: np.ndarray, max_nodata_frac: float = 0.05
) -> bool:
    """Reject patches that would teach the model the wrong thing.

    Two failure modes, both common at scene edges: too much missing label
    data, and all-black imagery from outside the orthophoto footprint.
    """
    nodata_frac = float(np.mean(~np.isfinite(label)))
    if nodata_frac > max_nodata_frac:
        return False
    if not rgb.any():
        return False
    return True


def save_patch_pair(
    out_dir: Path, patch_id: str, rgb: np.ndarray, label: np.ndarray
) -> None:
    """Persist one patch as a compressed .npz.

    Compressed npz rather than PNG pairs: it holds float32 labels natively,
    where a 16-bit PNG would force a quantisation choice.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"{patch_id}.npz", rgb=rgb, label=label)
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
.venv/bin/pytest tests/test_patches.py -v
```

Expected: 10 passed.

- [ ] **Step 5: Run the whole suite**

```bash
.venv/bin/pytest -v -m "not network"
```

Expected: all tests pass, network tests deselected.

- [ ] **Step 6: Commit**

```bash
git add src/altimap/data/patches.py tests/test_patches.py
git commit -m "feat: extract training patches as windows from remote COGs

Imagery and labels stay on their native 0.6 m and 2.0 m grids; a patch
is defined by ground footprint. Upsampling labels would invent detail
that cannot be validated."
```

---

### Task 11: Dataset build CLI

Ties Tasks 9 and 10 into the tool that produces a training set, with the diverse-AOI sampling spec §7.2 calls for.

**Files:**
- Create: `src/altimap/data/build.py`
- Create: `src/altimap/data/aois.py`
- Create: `tests/test_build.py`

**Interfaces:**
- Consumes: `AOI`, `open_client`, `find_items`, `coverage_counts`, `is_viable`, `ASSET_KEYS`, `COLLECTIONS`, `PatchSpec`, `DEFAULT_PATCH_SPEC`, `patch_windows`, `read_patch_pair`, `is_patch_usable`, `save_patch_pair`
- Produces:
  - `TRAINING_AOIS: tuple[AOI, ...]` in `aois.py`, populated from the Task 1 spike
  - `held_out_aois(aois) -> tuple[tuple[AOI, ...], tuple[AOI, ...]]` — (train, holdout) split by region
  - `ManifestEntry` frozen dataclass and `write_manifest(out_dir, entries) -> None`
  - `main(argv=None) -> int`; console script `altimap-build-dataset`

- [ ] **Step 1: Write the failing test**

Create `tests/test_build.py`:

```python
import json
from pathlib import Path

import pytest

from altimap.data.aois import TRAINING_AOIS, held_out_aois
from altimap.data.build import ManifestEntry, write_manifest
from altimap.data.stac import AOI


def test_training_aois_cover_every_landscape_class() -> None:
    """The rubric grades stability across urban, sparse, hilly, forested."""
    covered = {aoi.landscape for aoi in TRAINING_AOIS}
    assert covered == {"urban", "sparse", "forested", "hilly"}


def test_training_aois_are_geographically_diverse() -> None:
    """Label diversity beats label quality for generalisation (spec §7.2):
    out-of-domain error 9.41 m -> 3.83 m. So we need spread, not depth."""
    assert len(TRAINING_AOIS) >= 6
    longitudes = {round(aoi.bbox[0]) for aoi in TRAINING_AOIS}
    assert len(longitudes) >= 4


def test_holdout_split_is_by_region_not_by_tile() -> None:
    """A random tile split leaks — adjacent tiles share buildings and
    lighting. Whole AOIs must be held out (spec §6)."""
    train, holdout = held_out_aois(TRAINING_AOIS)
    assert len(holdout) >= 1
    assert not ({a.name for a in train} & {a.name for a in holdout})
    assert len(train) + len(holdout) == len(TRAINING_AOIS)


def test_holdout_is_not_empty_for_small_input() -> None:
    aois = (
        AOI(name="a", bbox=(-1.0, 0.0, 0.0, 1.0), landscape="urban"),
        AOI(name="b", bbox=(1.0, 0.0, 2.0, 1.0), landscape="forested"),
    )
    train, holdout = held_out_aois(aois)
    assert len(train) == 1
    assert len(holdout) == 1


def test_manifest_records_provenance(tmp_path: Path) -> None:
    entries = [
        ManifestEntry(
            patch_id="p0001",
            aoi="wasatch_hilly",
            landscape="hilly",
            split="train",
            imagery_item="naip-item-abc",
            label_item="hag-item-xyz",
        )
    ]
    write_manifest(tmp_path, entries)

    payload = json.loads((tmp_path / "manifest.json").read_text())
    assert payload["n_patches"] == 1
    assert payload["patches"][0]["aoi"] == "wasatch_hilly"
    assert payload["patches"][0]["imagery_item"] == "naip-item-abc"


def test_manifest_counts_by_split_and_landscape(tmp_path: Path) -> None:
    entries = [
        ManifestEntry("p1", "a", "urban", "train", "i1", "l1"),
        ManifestEntry("p2", "a", "urban", "train", "i1", "l1"),
        ManifestEntry("p3", "b", "forested", "holdout", "i2", "l2"),
    ]
    write_manifest(tmp_path, entries)
    payload = json.loads((tmp_path / "manifest.json").read_text())
    assert payload["by_split"] == {"train": 2, "holdout": 1}
    assert payload["by_landscape"] == {"urban": 2, "forested": 1}
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
.venv/bin/pytest tests/test_build.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'altimap.data.aois'`

- [ ] **Step 3: Create the AOI registry**

Create `src/altimap/data/aois.py`. **Replace the placeholder list with the viable AOIs recorded in the Task 1 spike findings** — the values below are the candidates probed, and any that came back non-viable must be removed:

```python
"""Training areas of interest.

Populated from the Task 1 coverage spike. Every entry here was verified to
have both NAIP imagery and 3DEP HAG labels — coverage is workunit-based, not
seamless, so this cannot be assumed (spec §7.2).

Breadth is deliberate. Published evidence: a model trained on high-quality
labels from one region scored 4.33 m in-domain but 9.41 m out-of-domain,
while training on diverse imperfect labels reached 3.83 m out-of-domain. Wide
geographic spread matters more here than depth in any one area.
"""

from __future__ import annotations

from altimap.data.stac import AOI

TRAINING_AOIS: tuple[AOI, ...] = (
    AOI(name="salt_lake_city_urban", bbox=(-111.95, 40.72, -111.85, 40.80), landscape="urban"),
    AOI(name="wasatch_hilly", bbox=(-111.95, 40.55, -111.85, 40.65), landscape="hilly"),
    AOI(name="denver_urban", bbox=(-105.05, 39.70, -104.95, 39.78), landscape="urban"),
    AOI(name="portland_forested", bbox=(-122.75, 45.48, -122.65, 45.56), landscape="forested"),
    AOI(name="phoenix_sparse", bbox=(-112.10, 33.42, -112.00, 33.50), landscape="sparse"),
    AOI(name="asheville_forested", bbox=(-82.60, 35.55, -82.50, 35.63), landscape="forested"),
    AOI(name="iowa_cropland", bbox=(-93.70, 41.55, -93.60, 41.63), landscape="sparse"),
)

# Held out entirely from training. Whole regions, never random tiles —
# adjacent tiles share buildings and illumination, so a random split leaks
# and produces a validation number that will not survive scrutiny (spec §6).
HOLDOUT_NAMES: frozenset[str] = frozenset({"denver_urban", "asheville_forested"})


def held_out_aois(
    aois: tuple[AOI, ...] = TRAINING_AOIS,
) -> tuple[tuple[AOI, ...], tuple[AOI, ...]]:
    """Split into (train, holdout) by region.

    Falls back to holding out the last AOI when none of the configured
    holdout names are present, so the split is never empty.
    """
    holdout = tuple(a for a in aois if a.name in HOLDOUT_NAMES)
    train = tuple(a for a in aois if a.name not in HOLDOUT_NAMES)
    if not holdout and len(aois) >= 2:
        return aois[:-1], aois[-1:]
    return train, holdout
```

- [ ] **Step 4: Implement the dataset builder**

Create `src/altimap/data/build.py`:

```python
"""Build a patch dataset from Planetary Computer imagery and labels."""

from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import sys
from pathlib import Path

import rasterio

from altimap.data.aois import TRAINING_AOIS, held_out_aois
from altimap.data.patches import (
    DEFAULT_PATCH_SPEC,
    PatchSpec,
    is_patch_usable,
    patch_windows,
    read_patch_pair,
    save_patch_pair,
)
from altimap.data.stac import (
    ASSET_KEYS,
    COLLECTIONS,
    AOI,
    coverage_counts,
    find_items,
    is_viable,
    open_client,
)


@dataclasses.dataclass(frozen=True)
class ManifestEntry:
    patch_id: str
    aoi: str
    landscape: str
    split: str
    imagery_item: str
    label_item: str


def write_manifest(out_dir: Path, entries: list[ManifestEntry]) -> None:
    """Record provenance for every patch.

    Without this, a trained model cannot be traced back to its data, and a
    holdout claim cannot be audited.
    """
    by_split: collections.Counter[str] = collections.Counter()
    by_landscape: collections.Counter[str] = collections.Counter()
    for e in entries:
        by_split[e.split] += 1
        by_landscape[e.landscape] += 1

    payload = {
        "n_patches": len(entries),
        "by_split": dict(by_split),
        "by_landscape": dict(by_landscape),
        "patches": [dataclasses.asdict(e) for e in entries],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n")


def _extract_for_aoi(
    client,
    aoi: AOI,
    split: str,
    out_dir: Path,
    spec: PatchSpec,
    stride_px: int,
    max_patches: int,
) -> list[ManifestEntry]:
    counts = coverage_counts(
        client, aoi.bbox, [COLLECTIONS["imagery"], COLLECTIONS["ndsm"]]
    )
    if not is_viable(counts):
        print(f"  SKIP {aoi.name}: not viable ({counts})", file=sys.stderr)
        return []

    rgb_items = find_items(client, COLLECTIONS["imagery"], aoi.bbox)
    label_items = find_items(client, COLLECTIONS["ndsm"], aoi.bbox)
    entries: list[ManifestEntry] = []

    for rgb_item in rgb_items:
        rgb_href = rgb_item.assets[ASSET_KEYS[COLLECTIONS["imagery"]]].href
        for label_item in label_items:
            label_href = label_item.assets[ASSET_KEYS[COLLECTIONS["ndsm"]]].href
            try:
                with rasterio.open(rgb_href) as rgb_src, rasterio.open(
                    label_href
                ) as label_src:
                    for window in patch_windows(
                        rgb_src.width, rgb_src.height, spec.size_px, stride_px
                    ):
                        if len(entries) >= max_patches:
                            return entries
                        pair = read_patch_pair(rgb_src, label_src, window, spec)
                        if pair is None:
                            continue
                        rgb, label = pair
                        if not is_patch_usable(rgb, label):
                            continue
                        patch_id = f"{aoi.name}_{len(entries):05d}"
                        save_patch_pair(out_dir / split, patch_id, rgb, label)
                        entries.append(
                            ManifestEntry(
                                patch_id=patch_id,
                                aoi=aoi.name,
                                landscape=aoi.landscape,
                                split=split,
                                imagery_item=rgb_item.id,
                                label_item=label_item.id,
                            )
                        )
            except rasterio.errors.RasterioIOError as exc:
                print(f"  WARN {aoi.name}: {exc}", file=sys.stderr)
                continue
    return entries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="altimap-build-dataset",
        description="Extract training patches from Planetary Computer COGs.",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SPEC.size_px)
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Window stride in px; defaults to patch size (no overlap)",
    )
    parser.add_argument(
        "--max-per-aoi",
        type=int,
        default=200,
        help="Cap per AOI. Breadth across AOIs beats depth in one (spec §7.2)",
    )
    args = parser.parse_args(argv)

    spec = PatchSpec(
        size_px=args.patch_size,
        source_gsd_m=DEFAULT_PATCH_SPEC.source_gsd_m,
        label_gsd_m=DEFAULT_PATCH_SPEC.label_gsd_m,
    )
    stride = args.stride or args.patch_size

    client = open_client()
    train_aois, holdout_aois = held_out_aois(TRAINING_AOIS)
    entries: list[ManifestEntry] = []

    for split, aois in (("train", train_aois), ("holdout", holdout_aois)):
        for aoi in aois:
            print(f"{split}: {aoi.name} ({aoi.landscape})")
            got = _extract_for_aoi(
                client, aoi, split, args.out, spec, stride, args.max_per_aoi
            )
            print(f"  extracted {len(got)} patches")
            entries.extend(got)

    write_manifest(args.out, entries)
    print(f"\nTotal: {len(entries)} patches -> {args.out / 'manifest.json'}")
    return 0 if entries else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Register the console script**

In `pyproject.toml`, replace the `[project.scripts]` block:

```toml
[project.scripts]
altimap-eval = "altimap.eval.cli:main"
altimap-build-dataset = "altimap.data.build:main"
```

Reinstall so the new script appears:

```bash
uv pip install -e ".[dev]"
```

- [ ] **Step 6: Run the test to verify it passes**

```bash
.venv/bin/pytest tests/test_build.py -v
```

Expected: 6 passed. `test_training_aois_cover_every_landscape_class` will fail if the spike found no viable hilly or forested AOI — that is a real finding, not a test bug; update `aois.py` from the spike results and re-probe for a replacement.

- [ ] **Step 7: Smoke-test the builder against live data**

```bash
.venv/bin/altimap-build-dataset --out data/patches --patch-size 518 --max-per-aoi 3
```

Expected: prints per-AOI progress and writes `data/patches/manifest.json`. Verify disk stayed bounded:

```bash
du -sh data/patches && df -h /home | tail -1
```

Expected: patches directory in the low tens of MB. **If it is gigabytes, windowed reads are not working and Task 10 needs revisiting** — that is the central assumption of spec §7.1.

- [ ] **Step 8: Run the full suite**

```bash
.venv/bin/pytest -v -m "not network"
```

Expected: all pass.

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml src/altimap/data/aois.py src/altimap/data/build.py tests/test_build.py
git commit -m "feat: add dataset build CLI with region-level holdout

AOIs are held out whole, never by random tile — adjacent tiles share
buildings and lighting, so a random split leaks. Breadth across AOIs
is capped per-AOI because diversity beats depth for generalisation."
```

---

### Task 12: Phase 1 documentation

**Files:**
- Create: `docs/eval-harness.md`
- Modify: `README.md`

- [ ] **Step 1: Write the evaluation harness guide**

Create `docs/eval-harness.md`:

```markdown
# Evaluation harness

Scores a predicted elevation raster against reference LiDAR. Built before any
model, because half the project's grade is a number this produces.

## Usage

```bash
altimap-eval predicted.tif reference.tif \
  --out reports/run01.json \
  --dataset "3dep-utah" \
  --landcover worldcover.tif \
  --dtm dtm.tif
```

Writes `run01.json` and `run01.md`. Auxiliary rasters must already be on the
reference grid; the tool refuses mismatched shapes rather than guessing.

## Why the metric matrix

A single all-pixel RMSE flatters an nDSM model badly, because most pixels are
near-zero ground. Published illustration: one model scored **4.89 m**
all-pixel and **37.47 m** height-balanced on the same data.

| Metric | Scope | Reads as |
|---|---|---|
| RMSE, MAE | all pixels | Comparability with prior work |
| RMSE, MAE | building pixels | The honest difficulty measure |
| Building-wise RMSE | per instance | Instance-level utility — "how tall is that building" |
| Height-balanced RMSE | equal weight per height stratum | Exposes tall-building underestimation |
| δ1 | pixels with reference ≥ 1 m | Official DFC2023 Track 2 metric |

δ1 is restricted to pixels above a height floor because nDSM background is
exactly zero, where its ratio is undefined. The report states how many pixels
were used.

## Co-registration

A one-pixel horizontal misalignment dominates RMSE. The harness estimates
planar shift by phase correlation and vertical bias by median difference, then
reports raw and corrected figures **side by side**. Corrections are never
applied silently.

## Targets

DFC2023 is close to this project's configuration (0.5 m optical with 2 m nDSM
labels, versus 0.6 m NAIP with 2 m HAG), so its results are the relevant bar:

| Level | Building-wise RMSE | δ1 |
|---|---|---|
| Weak | > 6.0 m | < 0.5 |
| Plain U-Net reference | ~4.93 m | — |
| Depth Anything V2 S reference | ~6.40 m | — |
| Best published pipeline | ~4.17 m | — |
| DFC2023 Track 2 winner | — | 0.8012 |

**Always quote dataset, GSD, and scope.** An unqualified RMSE cannot be
compared to anything: the same architecture family scores 1.30 m at 0.09 m
GSD, 2.12 m at 1.3 m, and 4.49 m at 3 m.
```

- [ ] **Step 2: Update the README**

Replace `README.md`:

```markdown
# AltiMap

Single-view optical remote-sensing imagery to metric elevation models, with an
interactive 3D flythrough viewer.

## Status

Phase 1 complete: evaluation harness and data pipeline.

## Setup

Requires Python 3.12 — the system Python 3.14 has no PyTorch wheels.

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
```

## Commands

| Command | Purpose |
|---|---|
| `altimap-eval` | Score a predicted elevation raster (see `docs/eval-harness.md`) |
| `altimap-build-dataset` | Extract training patches from Planetary Computer |

## Tests

```bash
.venv/bin/pytest -m "not network"   # offline
.venv/bin/pytest -m network         # live Planetary Computer access
```

## Documentation

- Design: `docs/superpowers/specs/2026-08-23-single-view-dsm-design.md`
- Plan: `docs/superpowers/plans/2026-08-23-eval-harness-and-data-pipeline.md`
- Evaluation: `docs/eval-harness.md`
- Spike findings: `docs/superpowers/spikes/`

## Approach

Elevation is predicted as **nDSM** — height above ground, in metres — and
absolute elevation is composed as `DSM = nDSM + DTM`, where the DTM comes from
a public bare-earth DEM. The learned model handles high-frequency structure
where LiDAR supervision exists; the public DEM supplies the absolute datum
where 10–30 m resolution is sufficient. Scale ambiguity therefore never
arises: the network emits metres by construction.
```

- [ ] **Step 3: Commit**

```bash
git add README.md docs/eval-harness.md
git commit -m "docs: document the evaluation harness and update README"
```

---

## Self-Review

**Spec coverage.** Checked each spec section against the plan:

| Spec section | Covered by |
|---|---|
| §3.1 contract | Task 2 |
| §6 co-registration, masking, percentiles | Tasks 5, 7 |
| §6.1 metric matrix | Tasks 3, 4, 7 |
| §6.2 targets | Task 12 docs |
| §7 Planetary Computer access | Tasks 1, 9 |
| §7.1 windowed COG reads | Task 10, verified Task 11 step 7 |
| §7.2 AOI coverage gate, diverse sampling | Tasks 1, 11 |
| §8 Python 3.12 pin | Task 2 |
| §12 testing | Every task |

**Deferred to later plans, by design:** §2 DTM composition, §4 model and losses, §5 visualization, §9 packaging, §10 phases 3–6. Each needs its own plan.

**One gap accepted:** the spec's §12 "tiling round-trip identity" test belongs with the inference tiling code (spec §4.4), which is a Phase 3 concern. Task 10 tests patch *extraction*; stitching does not exist yet.

**Type consistency verified.** `CoreMetrics` fields are consistent between `metrics.py`, `report.py` and the CLI test. `Alignment` field names match across `align.py`, `report.py` and tests. `PatchSpec.label_size_px` is used by `read_patch_pair` and asserted in tests. `ASSET_KEYS` is keyed by collection id in both `stac.py` and `build.py`.

**One known ambiguity, flagged inline:** the phase-correlation argument order in Task 5. `skimage`'s shift-sign convention is easy to invert, so the plan identifies `test_applying_alignment_reduces_error` as the authoritative check — it asserts an end-to-end property and is therefore immune to which convention is right.

---

## Plan complete

**Plan saved to `docs/superpowers/plans/2026-08-23-eval-harness-and-data-pipeline.md`.**

12 tasks. Task 1 is a spike with a hard STOP gate; Tasks 2–12 are TDD with a commit each.
