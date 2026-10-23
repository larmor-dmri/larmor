"""Helpers and fixtures shared by the render tests."""

import math

import numpy as np
import pygfx as gfx

from larmor.types import Tractogram, Volume
from larmor.viewer import (
    Category,
    Colormap,
    Item,
    Viewer,
    ViewKind,
)
from larmor.viewer.render.scene_renderer import SceneRenderer
from larmor.viewer.render.tractogram_adapter import (
    TractogramAdapter,
)
from larmor.viewer.render.view import ViewRenderer
from larmor.viewer.render.volume_adapter import (
    COLORMAP_MAPS,
    VolumeAdapter,
    _label_colours,
)

VIEWS_2D: list[ViewKind] = [ViewKind.AXIAL, ViewKind.CORONAL, ViewKind.SAGITTAL]


AXIS: dict[ViewKind, int] = {
    ViewKind.SAGITTAL: 0,
    ViewKind.CORONAL: 1,
    ViewKind.AXIAL: 2,
}


# An overlay's threshold: entries of its colormap below it are transparent


def _table(cmap: gfx.TextureMap) -> np.ndarray:
    return np.asarray(cmap.texture.data)


def _thresholded(name: Colormap, hidden: int) -> np.ndarray:
    table = _table(COLORMAP_MAPS[name]).copy()
    table[:hidden, 3] = 0.0
    return table


def _volume(
    slope: float = 1.0, intercept: float = 0.0, affine: np.ndarray | None = None
) -> Volume:
    data = np.zeros((8, 8, 8), np.float32, order="F")
    data[1:3, 4:7, 2:4] = 5.0
    return Volume(
        data, np.eye(4) if affine is None else affine, slope=slope, intercept=intercept
    )


def _adapter(
    viewer: Viewer, views: dict[ViewKind, ViewRenderer], item: Item
) -> VolumeAdapter:
    adapter = VolumeAdapter(item, views, viewer, 0.0)
    viewer.scene.subscribe(adapter.on_event)
    return adapter


def _objects(views: dict[ViewKind, ViewRenderer]) -> dict[ViewKind, list[object]]:
    # The background is the first child of every scene; lights and the camera are
    # the view's, not the items'.
    return {
        k: [
            c
            for c in v.scene.children[1:]
            if not isinstance(c, (gfx.Light, gfx.Camera))
        ]
        for k, v in views.items()
    }


def _in_3d(adapter: VolumeAdapter) -> list[object]:
    three_d = adapter.three_d
    assert three_d is not None
    shown = [*three_d.slices.values(), three_d.rendering, three_d.surface]
    return [obj for obj in shown if obj is not None and obj.parent is not None]


def _maps(adapter: VolumeAdapter) -> list[np.ndarray]:
    return [_table(obj.material.map) for obj in adapter.slices.values()]


def _same_map_on_every_slice(adapter: VolumeAdapter) -> np.ndarray:
    first, *others = _maps(adapter)
    assert all(np.array_equal(first, other) for other in others)
    return first


# The volume shown from a 4D image


def _series(count: int = 3) -> Volume:
    # Volume k holds k + 1 in a block that starts at x = k: each has its own values.
    data = np.zeros((8, 8, 8, count), np.float32, order="F")
    for k in range(count):
        data[k : k + 2, 4:7, 2:4, k] = k + 1.0
    return Volume(data, np.eye(4))


def _one_grid(adapter: VolumeAdapter) -> gfx.Texture:
    grid, *others = [obj.geometry.grid for obj in adapter.slices.values()]
    assert all(other is grid for other in others)
    return grid


# The views' cameras


def _ndc(view: ViewRenderer, world: np.ndarray) -> np.ndarray:
    point = view.camera.camera_matrix @ np.array([*world, 1.0])
    ndc: np.ndarray = point[:2] / point[3]
    return ndc


def _mesh(centre: tuple[float, float, float]) -> gfx.Mesh:
    mesh = gfx.Mesh(gfx.box_geometry(10, 10, 10), gfx.MeshBasicMaterial())
    mesh.local.position = centre
    return mesh


WHITE = 255


def _image(view: ViewRenderer) -> np.ndarray:
    return np.asarray(view.canvas.draw())


# The block of _volume() is voxels 1-2, 4-6, 2-3; (6, 1, 6) is outside it.
BLOCK = (1.5, 5.0, 2.5)


def _rotated() -> np.ndarray:
    c, s = math.cos(math.radians(30)), math.sin(math.radians(30))
    affine = np.eye(4)
    affine[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) * 2.0
    affine[:3, 3] = [10.0, -20.0, 5.0]
    return affine


def _palette(label: float) -> np.ndarray:
    colour: np.ndarray = _table(_label_colours(np.array([float(label)]), {}))[0]
    return colour


def _rgb(view: ViewRenderer, image: np.ndarray, world: np.ndarray) -> list[int]:
    ndc = view.camera.camera_matrix @ np.array([*world, 1.0])
    x, y = ndc[:2] / ndc[3]
    height, width = image.shape[:2]
    row, col = int((1 - y) / 2 * height), int((x + 1) / 2 * width)
    return [int(c) for c in image[row, col, :3]]


def _streamlines(count: int = 200, points: int = 30) -> Tractogram:
    rng = np.random.default_rng(0)
    start = rng.uniform(-20, 20, (count, 1, 3))
    step = rng.normal(0, 1, (count, 1, 3))
    walk = start + np.arange(points)[None, :, None] * step
    return Tractogram(walk.reshape(-1, 3), np.arange(count + 1) * points)


def _tracts(
    viewer: Viewer, renderer: SceneRenderer, data: Tractogram
) -> TractogramAdapter:
    item = viewer.add(data, Category.TRACTOGRAMS)
    adapter = renderer.adapters[item.id]
    assert isinstance(adapter, TractogramAdapter)
    return adapter
