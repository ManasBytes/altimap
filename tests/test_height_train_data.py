import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
pytest.importorskip("PIL")

from viewer.height_model import PATCH  # noqa: E402
from viewer.height_train import SynCrops, augment, block_index, degrade, find_synrs3d  # noqa: E402


def _write(path, arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = arr if arr.ndim == 3 else arr[None]
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[1], width=arr.shape[2],
                       count=arr.shape[0], dtype=arr.dtype) as dst:
        dst.write(arr)


def _syn_folder(root, name, n=2, size=512):
    rng = np.random.default_rng(0)
    for i in range(n):
        stem = f"{i:04d}.tif"
        _write(root / name / "opt" / stem, rng.integers(0, 255, (3, size, size), dtype=np.uint8))
        nd = np.zeros((size, size), np.float32)
        nd[100:200, 100:200] = 40.0
        _write(root / name / "gt_nDSM" / stem, nd)
        ss = np.full((size, size), 3, np.uint8)
        ss[100:200, 100:200] = 8  # building
        ss[0:10, 0:10] = 0  # unlabelled
        _write(root / name / "gt_ss_mask" / stem, ss)


def test_find_synrs3d_reads_gsd_range_from_folder_name(tmp_path):
    _syn_folder(tmp_path, "terrain_g1_high_v1")
    _syn_folder(tmp_path, "grid_g05_high_v1")
    items = find_synrs3d(tmp_path)
    assert len(items) == 4
    gsd = {str(i[0].parent.parent.name): i[3] for i in items}
    assert gsd == {"terrain_g1_high_v1": (0.6, 1.0), "grid_g05_high_v1": (0.3, 0.6)}


def test_syn_crops_are_patch_sized_with_oem_labels(tmp_path):
    _syn_folder(tmp_path, "terrain_g05_mid_v1", n=1)
    x, h, labels = SynCrops(find_synrs3d(tmp_path))[0]
    assert x.shape == (3, PATCH, PATCH) and x.dtype == np.float32
    assert h.shape == (PATCH, PATCH) and labels.shape == (PATCH, PATCH)
    assert set(np.unique(labels)) <= {2, 7, 255}  # developed (3-1), building (8-1), ignore
    assert np.nanmax(h) == pytest.approx(40.0, abs=0.5)


def test_augment_output_shapes_and_label_passthrough():
    rng = np.random.default_rng(1)
    rgb = rng.integers(0, 255, (600, 600, 3), dtype=np.uint8)
    labels = np.full((600, 600), 7, np.uint8)
    agl = np.full((600, 600), 12.0, np.float32)
    x, h, lab = augment(rgb, labels, agl, rng, degrade_p=1.0)
    assert x.shape == (3, PATCH, PATCH) and h.shape == lab.shape == (PATCH, PATCH)
    assert (lab == 7).all() and np.allclose(h, 12.0)  # degradation touches pixels only


def test_degrade_keeps_shape_and_dtype():
    rng = np.random.default_rng(2)
    rgb = rng.integers(0, 255, (PATCH, PATCH, 3), dtype=np.uint8)
    for _ in range(5):
        out = degrade(rgb, rng)
        assert out.shape == rgb.shape and out.dtype == np.uint8


def test_block_index_handles_plain_and_chunked_names():
    assert block_index("pretrained.blocks.5.attn.qkv.weight") == 5
    assert block_index("pretrained.blocks.0.17.mlp.fc1.weight") == 17
    assert block_index("pretrained.patch_embed.proj.weight") is None
    assert block_index("heads.regression.conv.weight") is None
