from django.urls import path

from . import views

urlpatterns = [
    path("", views.list_scenes, name="scene-list"),
    path("upload/", views.upload, name="scene-upload"),
    path("<str:scene_id>/", views.scene_detail, name="scene-detail"),
]
