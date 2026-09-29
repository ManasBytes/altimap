import pytest

torch = pytest.importorskip("torch")

from viewer.height_train import losses  # noqa: E402


def _batch(ref_val):
    ref = torch.zeros(1, 4, 4)
    ref[0, 0, 0] = ref_val  # one tall pixel
    height = torch.zeros(1, 4, 4)  # predicts 0 everywhere: error = ref at that pixel
    logits = torch.zeros(1, 8, 4, 4)
    labels = torch.full((1, 4, 4), 255, dtype=torch.long)
    return height, logits, ref, labels


def test_unweighted_is_plain_mean():
    l1, _, _ = losses(*_batch(40.0))
    assert l1.item() == pytest.approx(40.0 / 16)


def test_height_weighting_upweights_tall_pixels():
    plain, _, _ = losses(*_batch(40.0))
    weighted, _, _ = losses(*_batch(40.0), height_weight_m=10.0)
    # tall pixel weight 1 + 40/10 = 5, others 1 -> 5*40 / (15 + 5)
    assert weighted.item() == pytest.approx(5 * 40.0 / 20)
    assert weighted > plain
