"""Real upload endpoint, synthetic predictions: no GPU, model download or network."""
import io
import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from PIL import Image
from rasterio.transform import from_origin

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

import viewer.estimate as est
import viewer.server as server


@pytest.fixture
def upload_client(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "SCENES_DIR", tmp_path / "scenes")
    monkeypatch.setattr(server, "GRID_SIDE", 65)
    monkeypatch.setattr(server, "_get_height_model", lambda: {"model": None, "device": "cpu"})
    monkeypatch.setattr(server, "_height_ckpt", lambda: Path("best.pth"))
    monkeypatch.setattr(est, "predict_scene", lambda rgb, *a, **kw:
                        (rgb[..., 0].astype(np.float32) / 100, np.zeros(rgb.shape[:2], np.uint8)))
    with TestClient(server.app, raise_server_exceptions=False) as client:
        yield client


def imagery_bytes(suffix):
    rgb = np.full((3, 60, 60), 100, np.uint8)
    rgb[0] += np.arange(60, dtype=np.uint8)[None, :]
    if suffix == ".png":
        buf = io.BytesIO()
        Image.fromarray(rgb.transpose(1, 2, 0)).save(buf, format="PNG")
        return buf.getvalue()
    with rasterio.MemoryFile() as mem:
        with mem.open(driver="GTiff", height=60, width=60, count=3, dtype="uint8",
                      crs="EPSG:32643", transform=from_origin(700000, 3100000, 1, 1)) as dst:
            dst.write(rgb)
        return mem.read()


@pytest.mark.parametrize("base_dem", ["glo30", "srtm"])
@pytest.mark.parametrize("compare_dems", [False, True])
@pytest.mark.parametrize("secondary", ["offline", "nodata"])
def test_flat_dem_upload_succeeds_and_secondary_fetch_is_opt_in(
        upload_client, monkeypatch, base_dem, compare_dems, secondary):
    reads = []

    def dem(geo, name="glo30"):
        reads.append(name)
        if name != base_dem:
            if secondary == "offline":
                raise OSError("secondary DEM offline")
            return np.full((22, 22), np.nan, np.float32), (300, 300)
        return np.full((22, 22), 500.0, np.float32), (300, 300)

    monkeypatch.setattr(est, "padded_dem", dem)
    data = {"base_dem": base_dem}
    if compare_dems:
        data["compare_dems"] = "true"
    response = upload_client.post("/api/estimate", files={"file": ("scene.tif", imagery_bytes(".tif"))},
                                  data=data)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["dsm"]["base_dem"] == base_dem
    assert "dsm.tif" in result["downloads"]
    assert (server.SCENES_DIR / result["id"] / "dsm.tif").is_file()
    json.dumps(result, allow_nan=False)
    if compare_dems:
        assert result["dem_agreement"][base_dem]["r"] is None
        other = "srtm" if base_dem == "glo30" else "glo30"
        if secondary == "offline":
            assert result["dem_agreement"][other] is None
        else:
            assert all(v is None for v in result["dem_agreement"][other].values())
        assert reads == [base_dem, other]
    else:
        assert "dem_agreement" not in result
        assert reads == [base_dem]


def test_varying_dem_upload_keeps_finite_agreement(upload_client, monkeypatch):
    monkeypatch.setattr(est, "padded_dem", lambda *args:
                        (np.broadcast_to(np.arange(22, dtype=np.float32), (22, 22)).copy() + 500,
                         (300, 300)))
    response = upload_client.post("/api/estimate", files={"file": ("scene.tif", imagery_bytes(".tif"))},
                                  data={"compare_dems": "true"})
    assert response.status_code == 200, response.text
    metrics = response.json()["dem_agreement"]["glo30"]
    assert all(v is not None and np.isfinite(v) for v in metrics.values())


def test_png_ignores_dem_comparison_without_fetching(upload_client, monkeypatch):
    def unexpected_fetch(*args):
        pytest.fail("PNG must not fetch a DEM")

    monkeypatch.setattr(est, "padded_dem", unexpected_fetch)
    response = upload_client.post("/api/estimate", files={"file": ("scene.png", imagery_bytes(".png"))},
                                  data={"compare_dems": "true"})
    assert response.status_code == 200, response.text
    assert response.json()["dsm"] is None
    assert "dem_agreement" not in response.json()
