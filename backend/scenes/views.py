import shutil
import threading
import uuid
from pathlib import Path

from django.conf import settings
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from .models import Scene
from .pipeline import ALLOWED_SUFFIXES, MAX_UPLOAD_BYTES, safe_stem
from .serializers import SceneSerializer
from .services import run_pipeline_for_scene

# Uploads are processed on a background thread (see upload() below) so the
# request returns immediately with status="pending" and the frontend polls
# scene_detail() for completion, instead of blocking on multi-second DA3
# inference. This lock keeps two uploads from running the model at once --
# there's one shared model instance (pipeline._get_model), not one per request.
_PROCESSING_LOCK = threading.Lock()


def _scene_dir(scene_id: str) -> Path:
    return Path(settings.MEDIA_ROOT) / "scenes" / scene_id


def _run_in_background(scene: Scene, source_path: Path, scene_dir: Path) -> None:
    def run():
        with _PROCESSING_LOCK:
            try:
                run_pipeline_for_scene(scene, source_path, scene_dir)
            except Exception:  # noqa: BLE001 -- already recorded on the row; just stop the thread quietly
                pass

    threading.Thread(target=run, daemon=True).start()


@api_view(["POST"])
def upload(request):
    file = request.FILES.get("file")
    if not file:
        return Response({"detail": "no file provided"}, status=status.HTTP_400_BAD_REQUEST)

    suffix = Path(file.name or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        return Response(
            {"detail": f"unsupported type {suffix or '(none)'}; expected one of {sorted(ALLOWED_SUFFIXES)}"},
            status=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
        )
    if file.size == 0:
        return Response({"detail": "empty upload"}, status=status.HTTP_400_BAD_REQUEST)
    if file.size > MAX_UPLOAD_BYTES:
        return Response(
            {"detail": f"file is {file.size / 1e6:.0f} MB; limit is {MAX_UPLOAD_BYTES / 1e6:.0f} MB"},
            status=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
        )

    scene_id = f"{safe_stem(file.name or 'upload')}__{uuid.uuid4().hex[:8]}"
    scene_dir = _scene_dir(scene_id)
    scene_dir.mkdir(parents=True, exist_ok=True)
    staged = scene_dir / f"source{suffix}"
    with open(staged, "wb") as fh:
        for chunk in file.chunks():
            fh.write(chunk)

    scene = Scene.objects.create(
        id=scene_id,
        original_filename=Path(file.name or "upload").name,
        status=Scene.STATUS_PENDING,
    )
    _run_in_background(scene, staged, scene_dir)

    return Response(SceneSerializer(scene).data, status=status.HTTP_202_ACCEPTED)


@api_view(["GET"])
def list_scenes(request):
    return Response(SceneSerializer(Scene.objects.all(), many=True).data)


@api_view(["GET", "DELETE"])
def scene_detail(request, scene_id):
    try:
        scene = Scene.objects.get(id=scene_id)
    except Scene.DoesNotExist:
        return Response({"detail": "no such scene"}, status=status.HTTP_404_NOT_FOUND)

    if request.method == "DELETE":
        shutil.rmtree(_scene_dir(scene_id), ignore_errors=True)
        scene.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)

    return Response(SceneSerializer(scene).data)
