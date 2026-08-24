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
