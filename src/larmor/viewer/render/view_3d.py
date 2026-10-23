"""Places the 3D view's camera from the view state, and lights the views."""

import math

import numpy as np
import pygfx as gfx

from larmor.viewer.model.camera import Camera3D
from larmor.viewer.render.view import Vec3, ViewRenderer

# Render orders that draw the 3D view's transparent surfaces before the volume
# rendering that encloses them; both come after every opaque object.
ORDER_SURFACE = 1.0
ORDER_RENDERING = 2.0
# How much narrower the angle of view at zoom 1 is than the one that fits the
# sphere round the field, so a head fills most of the view.
FRAMING = 1.5
# Position of the glyphs' key lamp in the camera's frame, above and left of it.
KEY_LAMP = (-0.6, 0.8, 0.0)


def light(view: ViewRenderer) -> None:
    """Lights a view's surfaces with soft light all round and a lamp at the camera."""
    add_lights(view.scene, view.camera)


def add_lights(scene: gfx.Scene, camera: gfx.Camera) -> None:
    """Adds soft light all round a scene, and a lamp at its camera."""
    scene.add(gfx.AmbientLight(intensity=0.4))
    camera.add(gfx.DirectionalLight(intensity=2.0))
    scene.add(camera)  # so the camera's lamp is in the scene


def light_glyphs(view: ViewRenderer) -> None:
    """Lights a view's FOD glyphs, as `add_glyph_lights` does."""
    add_glyph_lights(view.scene, view.camera)


def add_glyph_lights(scene: gfx.Scene, camera: gfx.Camera) -> None:
    """Adds soft light all round, a key lamp at the camera's upper left and a weak one.

    Both lamps turn with the camera.
    """
    scene.add(gfx.AmbientLight(intensity=0.35))
    key = gfx.DirectionalLight(intensity=2.2)
    key.local.position = KEY_LAMP
    camera.add(key)
    camera.add(gfx.DirectionalLight(intensity=0.6))
    scene.add(camera)  # so the camera's lamps are in the scene


def place_camera(
    view: ViewRenderer, field: tuple[Vec3, Vec3], camera: Camera3D
) -> None:
    """Points the 3D camera at the field's centre, framing the field at zoom 1."""
    lo, hi = (np.asarray(p, dtype=float) for p in field)
    centre = (lo + hi) / 2
    # The camera stays outside the sphere around the field.
    radius = max(float(np.linalg.norm(hi - lo)) / 2, 1.0)
    cam = view.camera
    distance = radius / math.sin(math.radians(cam.fov) / 2)
    cam.zoom = camera.zoom * FRAMING
    cam.depth_range = (0.1 * distance, distance + 2 * radius)
    cam.world.reference_up = camera.up
    cam.world.position = centre - distance * np.asarray(camera.forward)
    cam.world.forward = camera.forward
    view.request_draw()
