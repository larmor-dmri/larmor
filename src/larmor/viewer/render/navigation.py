"""Keeps the views' cameras on the view state."""

from larmor.viewer.model.camera import Camera2D, Camera3D, Cameras
from larmor.viewer.model.events import Event, ViewStateChanged
from larmor.viewer.model.item import ViewKind
from larmor.viewer.model.view_state import ViewField
from larmor.viewer.model.viewer import Viewer
from larmor.viewer.render.view import CAMERA_DIRECTIONS_UP, ViewRenderer
from larmor.viewer.render.view_3d import light, light_glyphs, place_camera


class Navigation:
    """Points each view's camera as the view state says."""

    def __init__(self, viewer: Viewer, views: dict[ViewKind, ViewRenderer]) -> None:
        """Points the cameras of views and lights them."""
        self.viewer = viewer
        self.views = {k: v for k, v in views.items() if k in CAMERA_DIRECTIONS_UP}
        self.view_3d = views.get(ViewKind.THREE_D)
        state = viewer.scene.view
        for kind, view in self.views.items():
            camera = state.cameras.of(kind)
            assert isinstance(camera, Camera2D)
            view.set_camera(state.field_of_view, camera, state.convention)
            light_glyphs(view)  # only FOD glyphs are lit in the 2D views
        if self.view_3d is not None:
            light(self.view_3d)
            self.place_3d()

    def place_3d(self, camera: Camera3D | None = None) -> None:
        """Points the 3D camera at a pose, or as the view state says if None."""
        if self.view_3d is not None:
            state = self.viewer.scene.view
            pose = state.cameras.three_d if camera is None else camera
            place_camera(self.view_3d, state.field_of_view, pose)

    def on_event(self, event: Event) -> None:
        """Follows the field of view, the cameras and the convention."""
        if not isinstance(event, ViewStateChanged):
            return
        state = self.viewer.scene.view
        if event.name == ViewField.FIELD_OF_VIEW:
            for view in self.views.values():
                view.set_camera(field=state.field_of_view)
            self.place_3d()
        elif event.name == ViewField.CAMERAS:
            assert isinstance(event.old, Cameras) and isinstance(event.new, Cameras)
            for kind, view in self.views.items():
                old, new = event.old.of(kind), event.new.of(kind)
                if old != new:
                    assert isinstance(new, Camera2D)
                    view.set_camera(camera=new)
            if event.old.three_d != event.new.three_d:
                self.place_3d()
        elif event.name == ViewField.CONVENTION:
            for view in self.views.values():
                view.set_camera(convention=state.convention)
