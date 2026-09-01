from django.contrib import admin

from .models import Scene


@admin.register(Scene)
class SceneAdmin(admin.ModelAdmin):
    list_display = ("id", "original_filename", "status", "uploaded_at", "georeferenced", "calibrated", "has_glb")
    list_filter = ("status", "georeferenced", "calibrated")
    readonly_fields = [f.name for f in Scene._meta.fields]
