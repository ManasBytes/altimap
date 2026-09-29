import sys

import numpy as np

from viewer.height_model import GAMUS_TO_OEM, OEM_TO_GAMUS, clean_height, feather, window_starts


def test_windows_cover_length_and_end_flush() -> None:
    starts = window_starts(1024, 518, 259)
    assert starts[0] == 0
    assert starts[-1] + 518 == 1024
    covered = np.zeros(1024, bool)
    for s in starts:
        covered[s:s + 518] = True
    assert covered.all()


def test_windows_exact_fit_is_single_window() -> None:
    assert window_starts(518, 518, 259) == [0]


def test_windows_shorter_than_patch_is_single_window() -> None:
    # caller pads the image up to the patch size
    assert window_starts(300, 518, 259) == [0]


def test_feather_positive_and_peaks_centrally() -> None:
    w = feather(518)
    assert w.shape == (518, 518) and w.dtype == np.float32
    assert (w > 0).all()
    assert w[259, 259] == w.max()
    assert w[0, 0] < w[259, 259]


def test_class_maps_roundtrip_every_gamus_class() -> None:
    for gamus_class in range(1, 7):  # 0 = background, ignored in training
        assert OEM_TO_GAMUS[GAMUS_TO_OEM[gamus_class]] == gamus_class
    assert GAMUS_TO_OEM[0] == 255


def test_stray_label_values_map_to_ignore() -> None:
    assert GAMUS_TO_OEM[255] == 255 and GAMUS_TO_OEM[7] == 255


def test_clean_height_masks_sentinels_keeps_small_negatives() -> None:
    agl = np.array([-9999.0, -0.5, 0.0, 42.0, 5000.0], np.float32)
    out = clean_height(agl)
    assert np.isnan(out[0]) and np.isnan(out[4])
    np.testing.assert_array_equal(out[1:4], [-0.5, 0.0, 42.0])


def test_import_does_not_load_torch() -> None:
    assert "torch" not in sys.modules
