"""The cameras of the views, and where each 2D view looks from."""

import math
from dataclasses import dataclass, replace
from enum import StrEnum

import numpy as np

from larmor.types import Grid
from larmor.viewer.model.item import ViewKind

type Vec3 = tuple[float, float, float]


class Convention(StrEnum):
    """How the 2D views show left and right."""

    RAS = "RAS"  # neurological: the subject's right on the right
    LAS = "LAS"  # radiological: the subject's right on the left


@dataclass(frozen=True)
class Camera2D:
    """A 2D view's camera: its zoom and its pan, in mm in the view's plane."""

    zoom: float = 1.0
    pan: tuple[float, float] = (0.0, 0.0)  # along the screen's right and up


class CameraPreset(StrEnum):
    """The standard positions of the 3D camera."""

    AXIAL = "axial"  # from above
    CORONAL = "coronal"  # from the front
    SAGITTAL = "sagittal"  # from the subject's right


# The direction the 3D camera looks along and its screen's up, for each preset.
PRESETS: dict[CameraPreset, tuple[Vec3, Vec3]] = {
    CameraPreset.AXIAL: ((0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
    CameraPreset.CORONAL: ((0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
    CameraPreset.SAGITTAL: ((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
}


def _unit(v: np.ndarray) -> Vec3:
    """Returns v scaled to unit length, as a tuple."""
    u = v / np.linalg.norm(v)
    return (float(u[0]), float(u[1]), float(u[2]))


# The direction and up of the reset 3D camera: from the front, left and above.
_HOME_FORWARD = np.array([0.6, -0.65, -0.45])
_HOME_UP = np.array([0.0, 0.0, 1.0]) - _HOME_FORWARD * (
    _HOME_FORWARD[2] / (_HOME_FORWARD @ _HOME_FORWARD)
)


@dataclass(frozen=True)
class Camera3D:
    """The 3D camera, which looks at the centre of the field of view.

    At zoom 1 it frames the whole field of view.

    `forward` and `up` are unit and at right angles; zoom narrows the angle of view.
    """

    forward: Vec3 = _unit(_HOME_FORWARD)
    up: Vec3 = _unit(_HOME_UP)
    zoom: float = 1.0

    def rotated(self, right: float, up: float) -> Camera3D:
        """Returns the camera after turning the scene right and up on the screen.

        The scene turns about the screen's up axis, then about its right axis.

        Args:
            right: Degrees to turn the scene to the right.
            up: Degrees to turn the scene upwards.

        Returns:
            Camera3D: The new camera.
        """
        f, u = np.asarray(self.forward), np.asarray(self.up)
        # Turning the scene one way is turning the camera the other way.
        f = _rotate(f, u, -math.radians(right))
        r = np.cross(f, u)
        a = math.radians(up)
        f, u = _rotate(f, r, a), _rotate(u, r, a)
        # Keep the pair exactly orthonormal as rotations accumulate.
        u = u - f * (u @ f) / (f @ f)
        return replace(self, forward=_unit(f), up=_unit(u))


def _rotate(v: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotates a vector by angle, in radians, about an axis (Rodrigues' formula)."""
    k = axis / np.linalg.norm(axis)
    c, s = math.cos(angle), math.sin(angle)
    rotated: np.ndarray = v * c + np.cross(k, v) * s + k * (k @ v) * (1 - c)
    return rotated


@dataclass(frozen=True)
class Cameras:
    """The camera of each view."""

    axial: Camera2D = Camera2D()
    coronal: Camera2D = Camera2D()
    sagittal: Camera2D = Camera2D()
    three_d: Camera3D = Camera3D()

    def of(self, view: ViewKind) -> Camera2D | Camera3D:
        """Returns the camera of a view."""
        if view == ViewKind.THREE_D:
            return self.three_d
        camera: Camera2D = getattr(self, view)
        return camera

    def with_camera(self, view: ViewKind, camera: Camera2D | Camera3D) -> Cameras:
        """Returns a copy with the camera of one view replaced."""
        if isinstance(camera, Camera3D):
            if view != ViewKind.THREE_D:
                raise ValueError(f"The {view} view has a 2D camera.")
            return replace(self, three_d=camera)
        if view == ViewKind.AXIAL:
            return replace(self, axial=camera)
        if view == ViewKind.CORONAL:
            return replace(self, coronal=camera)
        if view == ViewKind.SAGITTAL:
            return replace(self, sagittal=camera)
        raise ValueError("The 3D view has a 3D camera.")


# The direction each 2D view looks along and its screen's up, per convention. In
# RAS the axial and coronal views look from above and from behind, so the
# subject's right is on the right; in LAS from below and from the front. The
# sagittal view is the same in both.
_DIRECTIONS: dict[tuple[ViewKind, Convention], tuple[Vec3, Vec3]] = {
    (ViewKind.AXIAL, Convention.RAS): ((0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
    (ViewKind.AXIAL, Convention.LAS): ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
    (ViewKind.CORONAL, Convention.RAS): ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    (ViewKind.CORONAL, Convention.LAS): ((0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
    (ViewKind.SAGITTAL, Convention.RAS): ((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    (ViewKind.SAGITTAL, Convention.LAS): ((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
}

# The field of view before any item gives a grid, a 200 mm cube around the origin.
EMPTY_FIELD: tuple[Vec3, Vec3] = ((-100.0, -100.0, -100.0), (100.0, 100.0, 100.0))


def view_directions(view: ViewKind, convention: Convention) -> tuple[Vec3, Vec3]:
    """Returns the direction a 2D view looks along and its screen's up, in world."""
    return _DIRECTIONS[(view, convention)]


def screen_axes(view: ViewKind, convention: Convention) -> tuple[Vec3, Vec3]:
    """Returns the world directions of a 2D view's screen right and screen up."""
    forward, up = view_directions(view, convention)
    right = np.cross(forward, up)
    return (float(right[0]), float(right[1]), float(right[2])), up


def field_of_view(grid: Grid | None) -> tuple[Vec3, Vec3]:
    """Returns the world box around a grid's voxels; the 2D views show it at zoom 1."""
    if grid is None:
        return EMPTY_FIELD
    shape = np.array(grid.shape[:3], dtype=float)
    # The voxels' outer edges, half a voxel beyond the centres of the end voxels.
    corners = np.array(
        [[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], dtype=float
    )
    voxels = corners * shape - 0.5
    world = voxels @ grid.affine[:3, :3].T + grid.affine[:3, 3]
    lo, hi = world.min(axis=0), world.max(axis=0)
    return (
        (float(lo[0]), float(lo[1]), float(lo[2])),
        (float(hi[0]), float(hi[1]), float(hi[2])),
    )
