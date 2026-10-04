"""Package completed /api/estimate results for a static demo dataset (no inference).

    python -m viewer.demo_bundle scene/result.json --out output/hf-demo

Exports display images, complete geometry/grids and original raster downloads.
Uploads themselves and credentials are never copied.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import shutil
from pathlib import Path


def bundle_scene(result_path: Path, output: Path, label: str | None = None) -> dict:
    data = json.loads(result_path.read_text())
    scene_id = data["id"]
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", scene_id) or scene_id in (".", ".."):
        raise ValueError("invalid scene id")
    if not data.get("city") or not data.get("grids") or not data.get("rgb"):
        raise ValueError("expected a complete result.json from /api/estimate, not meta.json")
    if data.get("building_geometry", {}).get("mapped_parts"):
        raise ValueError("use a new image-derived result; map-assisted reconstruction was reverted")
    target = output / scene_id
    if target.resolve() == result_path.parent.resolve():
        raise ValueError("bundle output must differ from the original upload directory")
    downloads = {}
    for name in data.get("downloads", {}):
        if Path(name).name != name or name in (".", "..", "result.json"):
            raise ValueError("invalid download filename")
        source = result_path.parent / name
        if not source.is_file():
            raise ValueError(f"missing raster/export: {name}")
        downloads[name] = source
    target.mkdir(parents=True, exist_ok=True)

    def image(value: str | None, name: str):
        if value and value.startswith("data:image/png;base64,"):
            (target / name).write_bytes(base64.b64decode(value.split(",", 1)[1], validate=True))
            return name
        return value

    for key in ("rgb", "height", "classes"):
        data[key] = image(data.get(key), f"{key}.png")
    if data.get("terrain"):
        for key in ("ground", "dsm"):
            data["terrain"][key]["png"] = image(data["terrain"][key]["png"], f"{key}.png")
    if data.get("error"):
        data["error"]["png"] = image(data["error"]["png"], "error.png")
    for name, source in downloads.items():
        shutil.copy2(source, target / name)
    data["downloads"] = {name: name for name in downloads}
    data["demo"] = {"kind": "prepared model output", "live_inference": False}
    (target / "result.json").write_text(json.dumps(data))
    return {"id": scene_id, "label": label or scene_id.split("__")[0].replace("_", " "),
            "result": f"{scene_id}/result.json"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--label", help="display label for a single result")
    args = parser.parse_args()
    if args.label and len(args.results) != 1:
        parser.error("--label requires a single result")
    index_path = args.out / "index.json"
    existing = json.loads(index_path.read_text())["scenes"] if index_path.exists() else []
    scenes = {scene["id"]: scene for scene in existing}
    for path in args.results:
        scene = bundle_scene(path, args.out, args.label)
        scenes[scene["id"]] = scene
        print(f'Prepared {scene["label"]}')
    index_path.write_text(json.dumps({"scenes": list(scenes.values())}, indent=2))


if __name__ == "__main__":
    main()
