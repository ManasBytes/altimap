"""Seeds the database with a couple of demo scenes so `make yolo` has
something to show in the viewer on a first run, without requiring the user to
find and upload a real image first.

Runs the exact same pipeline (scenes.services.run_pipeline_for_scene ->
scenes.pipeline.process_scene -> viewer/geo.py + viewer/metrics.py +
viewer/terrain.py) an upload would, just against a synthetic image generated
in-process instead of a network fetch -- so it needs no bundled test fixtures
and works offline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from django.conf import settings
from django.core.management.base import BaseCommand
from PIL import Image

from scenes.models import Scene
from scenes.services import run_pipeline_for_scene

# (id suffix, filename, rng seed, description) -- two different synthetic
# fields so the demo shows some variety rather than one look-alike scene.
DEMO_SCENES = [
    ("demo-ridgeline", "demo-ridgeline.png", 1),
    ("demo-basin", "demo-basin.png", 2),
]


def _synthetic_terrain_image(seed: int, size: int = 256) -> Image.Image:
    """A smooth, ridged synthetic height field rendered as a fake satellite
    RGB tile -- not real imagery, just something with structure for DA3 (and
    the plane-fit/detrend metrics) to have an opinion about, rather than pure
    noise."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    ridges = (
        35 * np.sin(xx / 22 + rng.uniform(0, 6))
        + 25 * np.cos(yy / 17 + rng.uniform(0, 6))
        + 12 * np.sin((xx + yy) / 9)
    )
    field = 110 + ridges + rng.normal(0, 5, (size, size))
    field = np.clip(field, 0, 255)

    # Slight per-channel variation so it reads as a colour photo, not a
    # greyscale height map.
    r = field
    g = np.clip(field * 0.92 + 8, 0, 255)
    b = np.clip(field * 0.80 + 4, 0, 255)
    rgb = np.stack([r, g, b], axis=-1).astype(np.uint8)
    return Image.fromarray(rgb)


class Command(BaseCommand):
    help = "Seed the database with demo scenes (idempotent -- skips if scenes already exist)."

    def add_arguments(self, parser):
        parser.add_argument(
            "--force", action="store_true",
            help="Reseed even if scenes already exist (re-runs the pipeline and overwrites the demo rows).",
        )

    def handle(self, *args, **options):
        if Scene.objects.exists() and not options["force"]:
            self.stdout.write("Scenes already exist -- skipping (pass --force to reseed).")
            return

        for suffix, filename, seed in DEMO_SCENES:
            scene_id = f"{suffix}__seed"
            self.stdout.write(f"Seeding {scene_id}...")

            scene_dir = Path(settings.MEDIA_ROOT) / "scenes" / scene_id
            scene_dir.mkdir(parents=True, exist_ok=True)
            source_path = scene_dir / "source.png"
            _synthetic_terrain_image(seed).save(source_path)

            Scene.objects.filter(id=scene_id).delete()
            scene = Scene.objects.create(id=scene_id, original_filename=filename, status=Scene.STATUS_PENDING)

            try:
                run_pipeline_for_scene(scene, source_path, scene_dir)
            except Exception as exc:  # noqa: BLE001 -- keep seeding the rest even if one scene fails
                self.stderr.write(self.style.ERROR(f"  failed: {exc}"))
                continue

            self.stdout.write(self.style.SUCCESS(f"  done in {scene.seconds}s"))
