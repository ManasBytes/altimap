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
    # The peak lands inside the bump -- but not necessarily dead centre: the
    # least-squares plane tilts slightly to absorb part of the bump, so the
    # residual minimum sits at one of its corners.
    peak_row, peak_col = np.unravel_index(np.argmax(h), h.shape)
    assert 10 <= peak_row < 14
    assert 10 <= peak_col < 14
    # And the ramp is gone: without detrending, the whole left edge would win.
    assert h[10:14, 10:14].min() > h[:, 0].max()


def test_terrain_grid_uv_puts_image_top_on_row_zero():
    """Regression: glTF UV origin is the image's UPPER-left, so the height
    field's row 0 (top of the source image) must get v = 0. The mirrored
    convention paints the texture upside-down -- invisible on a symmetric scene,
    glaring on a road."""
    from viewer.terrain import terrain_grid

    res = 8
    xx, zz, faces, uv = terrain_grid(res)
    v = uv[:, 1].reshape(res, res)
    z = zz.reshape(res, res)

    assert z[0, 0] == pytest.approx(-0.5)   # row 0 sits at the far edge
    assert v[0, 0] == pytest.approx(0.0)    # ...and must sample the image top
    assert v[-1, 0] == pytest.approx(1.0)   # last row samples the image bottom

    u = uv[:, 0].reshape(res, res)
    assert u[0, 0] == pytest.approx(0.0)
    assert u[0, -1] == pytest.approx(1.0)
    assert faces.shape == (2 * (res - 1) ** 2, 3)
