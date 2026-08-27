import numpy as np
import pytest

from viewer.export_dem_direct import elevation_to_fake_depth
from viewer.geo import decode_rg16, encode_rg16


def test_fake_depth_round_trips_to_correct_height_ordering():
    """The whole trick this module relies on: encoding elev_max - elevation
    through the SAME shader math the depth pipeline uses must recover the
    correct height ordering, with no changes to terrain.js or the shader.
    Getting the sign backwards here would render every mountain as a pit."""
    rng = np.random.default_rng(0)
    elev = rng.uniform(200.0, 350.0, size=(40, 40)).astype(np.float32)

    fake_depth, elev_min, elev_max = elevation_to_fake_depth(elev)
    encoded, lo, hi = encode_rg16(fake_depth)
    decoded = decode_rg16(encoded, lo, hi)

    # This is exactly terrain.js's heightAt() / shaders.js's heightAt(): h = (hi - d) / (hi - lo)
    height = (hi - decoded) / (hi - lo)

    assert height[np.unravel_index(np.argmax(elev), elev.shape)] == pytest.approx(1.0, abs=1e-3)
    assert height[np.unravel_index(np.argmin(elev), elev.shape)] == pytest.approx(0.0, abs=1e-3)
    assert np.corrcoef(height.ravel(), elev.ravel())[0, 1] > 0.999


def test_fake_depth_handles_flat_elevation_without_nan():
    elev = np.full((8, 8), 42.0, dtype=np.float32)
    fake_depth, elev_min, elev_max = elevation_to_fake_depth(elev)
    assert elev_min == elev_max == 42.0
    assert np.all(fake_depth == 0.0)
    encoded, lo, hi = encode_rg16(fake_depth)
    assert np.isfinite([lo, hi]).all()
