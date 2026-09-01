from django.conf import settings
from rest_framework import serializers

from .models import Scene


class SceneSerializer(serializers.ModelSerializer):
    base_url = serializers.SerializerMethodField()
    depth_url = serializers.SerializerMethodField()
    rgb_url = serializers.SerializerMethodField()
    glb_url = serializers.SerializerMethodField()

    class Meta:
        model = Scene
        fields = [
            "id", "original_filename", "uploaded_at", "status", "error_message",
            "width", "height", "seconds", "plane_r2", "residual_relief",
            "structure_alignment", "georeferenced", "calibrated", "has_glb",
            "metrics_json", "base_url", "depth_url", "rgb_url", "glb_url",
        ]

    def _asset(self, obj, name):
        return f"{settings.MEDIA_URL}scenes/{obj.id}/{name}"

    def get_base_url(self, obj):
        return f"{settings.MEDIA_URL}scenes/{obj.id}"

    def get_depth_url(self, obj):
        return self._asset(obj, "depth.png") if obj.status == Scene.STATUS_DONE else None

    def get_rgb_url(self, obj):
        return self._asset(obj, "rgb.jpg") if obj.status == Scene.STATUS_DONE else None

    def get_glb_url(self, obj):
        return self._asset(obj, "terrain.glb") if obj.has_glb else None
