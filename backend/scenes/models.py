from django.db import models


class Scene(models.Model):
    """One uploaded image and the DA3/terrain pipeline's output for it.

    `id` reuses the scene_id scheme viewer/server.py used (safe filename stem
    + a short uuid suffix) so it doubles as the directory name under
    MEDIA_ROOT/scenes/<id>/ that holds meta.json, depth.png, rgb.jpg and
    terrain.glb.
    """

    STATUS_PENDING = "pending"
    STATUS_PROCESSING = "processing"
    STATUS_DONE = "done"
    STATUS_FAILED = "failed"
    STATUS_CHOICES = [
        (STATUS_PENDING, "pending"),
        (STATUS_PROCESSING, "processing"),
        (STATUS_DONE, "done"),
        (STATUS_FAILED, "failed"),
    ]

    id = models.SlugField(primary_key=True, max_length=64)
    original_filename = models.CharField(max_length=255)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=16, choices=STATUS_CHOICES, default=STATUS_PENDING)
    error_message = models.TextField(blank=True, default="")

    width = models.IntegerField(null=True, blank=True)
    height = models.IntegerField(null=True, blank=True)
    seconds = models.FloatField(null=True, blank=True)
    plane_r2 = models.FloatField(null=True, blank=True)
    residual_relief = models.FloatField(null=True, blank=True)
    structure_alignment = models.FloatField(null=True, blank=True)
    georeferenced = models.BooleanField(default=False)
    calibrated = models.BooleanField(default=False)
    has_glb = models.BooleanField(default=False)

    # Full pipeline record (metrics, geo, absolute-calibration block) --
    # same shape as the meta.json written alongside it, kept here too so the
    # scene list/detail API doesn't need to read the file back off disk.
    metrics_json = models.JSONField(null=True, blank=True)

    class Meta:
        ordering = ["-uploaded_at"]

    def __str__(self) -> str:
        return f"{self.id} ({self.status})"
