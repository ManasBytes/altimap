import base64
import json
from pathlib import Path

import pytest

from viewer.demo_bundle import bundle_scene


def test_bundle_is_independent_of_vm_and_preserves_geometry_and_raster_bytes(tmp_path):
    source = tmp_path / "upload"
    source.mkdir()
    raster = b"original metric raster"
    (source / "ndsm.tif").write_bytes(raster)
    (source / "source.tif").write_bytes(b"private original upload")
    png = "data:image/png;base64," + base64.b64encode(b"png bytes").decode()
    data = {"id": "city__abc", "rgb": png, "height": png, "classes": png,
            "city": {"buildings": [{"h": 22, "rings": [[[0, 0], [1, 0], [1, 1]]]}]},
            "grids": {"side": 3, "ndsm": {"data": "original grid"}},
            "terrain": {"ground": {"png": png, "min_m": 100}, "dsm": {"png": png}},
            "error": {"png": png, "limit_m": 10}, "validation": {"rmse": 3.1},
            "downloads": {"ndsm.tif": "/data-uploads/scenes/city__abc/ndsm.tif"}}
    original = json.dumps(data)
    result = source / "result.json"
    result.write_text(original)
    output = tmp_path / "bundle"
    scene = bundle_scene(result, output, "City demo")
    target = output / scene["id"]
    saved = json.loads((target / "result.json").read_text())
    assert scene["label"] == "City demo"
    assert saved["downloads"] == {"ndsm.tif": "ndsm.tif"}
    assert (target / "ndsm.tif").read_bytes() == raster
    assert (target / saved["rgb"]).read_bytes() == b"png bytes"
    assert saved["terrain"]["ground"]["min_m"] == 100
    assert saved["city"] == data["city"] and saved["grids"] == data["grids"]
    assert saved["validation"] == data["validation"]
    assert not (target / "source.tif").exists()
    assert result.read_text() == original


@pytest.mark.parametrize("scene_id, downloads", [("../escape", {}), ("ok", {"../source.tif": "x"})])
def test_bundle_rejects_paths_outside_scene(tmp_path, scene_id, downloads):
    result = tmp_path / "result.json"
    result.write_text(json.dumps({"id": scene_id, "rgb": "x", "city": {"buildings": []},
                                  "grids": {"side": 3}, "downloads": downloads}))
    with pytest.raises(ValueError):
        bundle_scene(result, tmp_path / "output")
