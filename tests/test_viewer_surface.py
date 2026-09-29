"""CPU tests: tiled inference must preserve corners, aspect and class identity."""
import numpy as np
import pytest

from viewer.surface import origins, regularize_surface, infer_tiled


@pytest.mark.parametrize("size,tile,stride", [(17, 32, 24), (32, 32, 24), (87, 32, 24), (100, 32, 32)])
def test_tile_origins_cover_every_pixel(size, tile, stride):
    covered = np.zeros(size, dtype=bool)
    for x in origins(size, tile, stride):
        covered[x:x+tile] = True
    assert covered.all()


@pytest.mark.parametrize("shape", [(13, 19), (47, 83), (64, 64)])
def test_tiled_blending_is_seam_free_and_preserves_dimensions(shape):
    torch = pytest.importorskip("torch")
    from viewer.classify import IMAGENET_MEAN, IMAGENET_STD

    class PixelModel(torch.nn.Module):
        def surface(self, x):
            # Pointwise signal means expected outputs do not depend on tile edges.
            intensity = x[:, 0] * float(IMAGENET_STD[0]) + float(IMAGENET_MEAN[0])
            logits = torch.zeros((x.shape[0], 7, *x.shape[2:]), device=x.device)
            logits[:, 3] = 4 * intensity + 2
            return logits, torch.log1p(intensity * 10)

    rgb = np.random.default_rng(4).integers(0, 256, (*shape, 3), dtype=np.uint8)
    labels, height, confidence = infer_tiled(rgb, PixelModel(), "cpu", tile=32, overlap=8)
    assert labels.shape == height.shape == confidence.shape == shape
    assert (labels == 3).all()
    np.testing.assert_allclose(height, rgb[:, :, 0]/255*10, atol=1e-5)
    expected = np.exp(4.0*rgb[:, :, 0]/255+2)
    np.testing.assert_allclose(confidence, expected/(expected+6), atol=1e-6)


def test_regularization_does_not_flatten_distinct_roofs_or_tree_interiors():
    y, x = np.mgrid[:32, :48]
    height = (2 + x*.1 + y*.1).astype(np.float32)
    classes = np.full(height.shape, 6, np.uint8)
    classes[3:13, 3:13] = 3
    classes[18:28, 30:40] = 3
    height[3:13, 3:13] = 8
    height[18:28, 30:40] = 22
    height[7, 7] = 100
    result = regularize_surface(height, classes)
    assert result[7, 7] == 8
    assert result[23, 35] == 22
    assert result[15, 20] == height[15, 20]
    assert height[7, 7] == 100  # original prediction remains untouched


def test_regularization_handles_nonfinite_input():
    height = np.array([[np.nan, np.inf, -np.inf], [-1., 1., 9.], [3., 4., 5.]])
    result = regularize_surface(height, np.zeros(height.shape, np.uint8))
    assert np.isfinite(result).all()
    assert result.min() >= 0 and result.max() <= 150


@pytest.mark.parametrize("tile,overlap", [(0, 0), (33, 8), (32, 32), (32, -1)])
def test_invalid_tile_configuration_is_rejected(tile, overlap):
    pytest.importorskip("torch")
    with pytest.raises(ValueError):
        infer_tiled(np.zeros((32, 32, 3), np.uint8), None, "cpu", tile, overlap)
