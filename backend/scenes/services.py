"""Django-facing glue between the pure pipeline (pipeline.py) and the Scene
model -- shared by the upload view's background thread and the seed_scenes
management command, so the record -> row field mapping only lives in one place.
"""

from __future__ import annotations

from pathlib import Path

from .models import Scene
from .pipeline import ProcessingError, clean_metric, process_scene


def run_pipeline_for_scene(scene: Scene, source_path: Path, scene_dir: Path) -> None:
    """Runs process_scene and updates `scene` in place to DONE or FAILED.

    Always deletes `source_path` afterward, mirroring the pipeline's "the
    source is not served; the derived assets are what the viewer needs"
    convention. Re-raises on failure after recording it on the row, so a
    caller (a background thread, a management command) can still notice.
    """
    scene.status = Scene.STATUS_PROCESSING
    scene.save(update_fields=["status"])
    try:
        record = process_scene(source_path, scene.id, scene.original_filename, scene_dir)
    except Exception as exc:  # noqa: BLE001 -- must record on the row, not just propagate
        scene.status = Scene.STATUS_FAILED
        scene.error_message = str(exc) if isinstance(exc, ProcessingError) else "processing failed"
        scene.save(update_fields=["status", "error_message"])
        raise
    finally:
        source_path.unlink(missing_ok=True)

    scene.status = Scene.STATUS_DONE
    scene.width = record.get("width")
    scene.height = record.get("height")
    scene.seconds = record.get("seconds")
    scene.plane_r2 = clean_metric(record.get("plane_r2"))
    scene.residual_relief = clean_metric(record.get("residual_relief"))
    scene.structure_alignment = clean_metric(record.get("structure_alignment"))
    scene.georeferenced = bool(record.get("geo", {}).get("georeferenced"))
    scene.calibrated = bool((record.get("absolute") or {}).get("usable"))
    scene.has_glb = bool(record.get("has_glb"))
    scene.metrics_json = record
    scene.save()
