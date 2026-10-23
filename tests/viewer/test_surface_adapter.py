from collections.abc import Callable

import numpy as np
import pygfx as gfx
import pytest
from scipy.spatial import ConvexHull

from larmor.surfaces import vertex_normals
from larmor.types import Surface
from larmor.viewer import Category, Colouring, Item, OptionName, Viewer, ViewKind
from larmor.viewer.model.options import SurfaceOptions
from larmor.viewer.model.surface import SurfaceData
from larmor.viewer.render import surface_adapter
from larmor.viewer.render.colour_space import physical_colours
from larmor.viewer.render.geometry import PlaneCut, colormap_colours, plane_cut
from larmor.viewer.render.surface_adapter import SurfaceAdapter
from larmor.viewer.render.view import ViewRenderer
from larmor.viewer.render.volume_adapter import COLORMAP_MAPS

TWO_D = (ViewKind.SAGITTAL, ViewKind.CORONAL, ViewKind.AXIAL)
# An octahedron of radius 4 about the origin: +x, -x, +y, -y, +z, -z.
VERTICES = 4 * np.array(
    [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], np.float32
)
FACES = np.array(
    [
        [0, 2, 4],
        [2, 1, 4],
        [1, 3, 4],
        [3, 0, 4],
        [2, 0, 5],
        [1, 2, 5],
        [3, 1, 5],
        [0, 3, 5],
    ]
)
UPPER = FACES[:4]
# The medial wall is the bottom vertex, so the four faces below the equator.
CORTEX = np.array([True, True, True, True, True, False])


@pytest.fixture
def cuts(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, float]]:
    # The (world axis, position) of every cut made.
    made: list[tuple[int, float]] = []

    def record(
        vertices: np.ndarray, faces: np.ndarray, axis: int, position: float
    ) -> PlaneCut:
        made.append((axis, position))
        return plane_cut(vertices, faces, axis, position)

    monkeypatch.setattr(surface_adapter, "plane_cut", record)
    return made


def _item(viewer: Viewer, cortex: np.ndarray | None = None) -> Item:
    values = {
        "curv": np.linspace(-1, 1, 6, dtype=np.float32),
        "thickness": np.linspace(1, 4, 6, dtype=np.float32),
    }
    surface = Surface(VERTICES, FACES, vertex_data=values)
    return viewer.add(SurfaceData(surface, cortex), Category.SURFACES, name="lh.white")


def _adapter(
    viewer: Viewer, views: dict[ViewKind, ViewRenderer], item: Item
) -> SurfaceAdapter:
    adapter = SurfaceAdapter(item, views, viewer, 0.0)
    viewer.scene.subscribe(adapter.on_event)
    return adapter


def _expected_colours(item: Item) -> np.ndarray:
    o = item.options
    assert isinstance(o, SurfaceOptions) and o.colour_by is not None
    assert isinstance(item.data, SurfaceData)
    values = item.data.surface.vertex_data[o.colour_by]
    table = COLORMAP_MAPS[o.colormap].texture.data
    # The mesh's vertex colours are linear light: the colormap's, converted.
    return physical_colours(colormap_colours(values, table, o.minimum, o.maximum))


def test_a_surface_is_a_lit_mesh_with_its_normals_in_the_3d_view(
    viewer: Viewer, views: dict[ViewKind, ViewRenderer]
) -> None:
    item = _item(viewer)
    adapter = _adapter(viewer, views, item)
    assert adapter.mesh is not None
    assert adapter.mesh in views[ViewKind.THREE_D].scene.children
    assert isinstance(adapter.mesh.material, gfx.MeshPhongMaterial)
    assert isinstance(item.data, SurfaceData)
    normals = adapter.mesh.geometry.normals.data
    np.testing.assert_allclose(normals, vertex_normals(item.data.surface))


def test_colouring_by_other_values_rewrites_only_the_colours(
    viewer: Viewer, views: dict[ViewKind, ViewRenderer]
) -> None:
    item = _item(viewer)
    adapter = _adapter(viewer, views, item)
    assert adapter.mesh is not None
    geometry = adapter.mesh.geometry
    positions, colours = geometry.positions, geometry.colors
    viewer.set(item.id, OptionName.COLOUR_BY, "thickness")
    item = viewer.scene.item(item.id)
    assert adapter.mesh.geometry is geometry
    assert geometry.positions is positions and geometry.colors is colours
    assert adapter.mesh.material.color_mode == "vertex"
    np.testing.assert_allclose(colours.data, _expected_colours(item))


@pytest.mark.parametrize(
    ("hide", "cortex", "hidden"),
    [(False, CORTEX, False), (True, CORTEX, True), (True, None, False)],
    ids=["shown", "hidden", "no label"],
)
def test_hiding_the_medial_wall_leaves_its_faces_out_in_3d_and_2d(
    hide: bool,
    cortex: np.ndarray | None,
    hidden: bool,
    viewer: Viewer,
    views: dict[ViewKind, ViewRenderer],
) -> None:
    item = _item(viewer, cortex)
    adapter = _adapter(viewer, views, item)
    viewer.move_cursor((0.0, 0.0, -2.0))  # an axial plane through the wall only
    viewer.set(Category.SURFACES, OptionName.HIDE_MEDIAL_WALL, hide)
    assert adapter.mesh is not None
    indices = adapter.mesh.geometry.indices.data
    np.testing.assert_array_equal(indices, UPPER if hidden else FACES)
    cut = adapter.cuts[ViewKind.AXIAL]
    assert cut is not None
    assert len(cut.positions) == (0 if hidden else 8)


def test_turning_cut_lines_off_hides_the_2d_lines_only(
    viewer: Viewer, views: dict[ViewKind, ViewRenderer]
) -> None:
    item = _item(viewer)
    adapter = _adapter(viewer, views, item)
    assert all(adapter.lines[k].visible for k in TWO_D)
    viewer.set(item.id, OptionName.CUT_LINES, False)
    assert not any(adapter.lines[k].visible for k in TWO_D)
    assert adapter.mesh is not None and adapter.mesh.visible


@pytest.mark.parametrize(
    ("cursor", "made"),
    [
        ((1.0, 0.0, 0.0), [(0, 1.0)]),
        ((0.0, 3.0, 0.0), [(1, 3.0)]),
        ((1.0, 0.0, 2.0), [(0, 1.0), (2, 2.0)]),
    ],
    ids=["x", "y", "x and z"],
)
def test_a_cursor_move_cuts_again_only_the_views_whose_plane_moved(
    cursor: tuple[float, float, float],
    made: list[tuple[int, float]],
    viewer: Viewer,
    views: dict[ViewKind, ViewRenderer],
    cuts: list[tuple[int, float]],
) -> None:
    _adapter(viewer, views, _item(viewer))
    assert sorted(cuts) == [(0, 0.0), (1, 0.0), (2, 0.0)]
    cuts.clear()
    viewer.move_cursor(cursor)
    assert sorted(cuts) == made


def _sphere(inward: bool) -> Surface:
    # About 400 vertices of radius 10 around the origin, wound outward unless asked.
    n = 400
    golden = np.pi * (3 - np.sqrt(5))
    z = 1 - 2 * (np.arange(n) + 0.5) / n
    r = np.sqrt(1 - z**2)
    points = (
        10
        * np.c_[r * np.cos(golden * np.arange(n)), r * np.sin(golden * np.arange(n)), z]
    )
    faces = ConvexHull(points).simplices
    a, b, c = (points[faces[:, i]] for i in range(3))
    flip = np.sum(np.cross(b - a, c - a) * a, axis=1) < 0
    faces[flip] = faces[flip][:, ::-1]
    if inward:
        faces = faces[:, ::-1]
    return Surface(points.astype(np.float32), faces)


def _segments(line: gfx.Line) -> np.ndarray:
    positions = line.geometry.positions.data
    return np.asarray(positions[~np.isnan(positions).any(axis=1)]).reshape(-1, 2, 3)


@pytest.mark.parametrize("inward", [False, True], ids=["outward", "reversed faces"])
def test_a_spheres_normals_point_out_or_in_with_reversed_faces_in_3d_and_2d(
    inward: bool, viewer: Viewer, views: dict[ViewKind, ViewRenderer]
) -> None:
    item = viewer.add(
        SurfaceData(_sphere(inward), None), Category.SURFACES, name="sphere"
    )
    adapter = _adapter(viewer, views, item)
    viewer.set(item.id, OptionName.SHOW_NORMALS, True)
    viewer.set(item.id, OptionName.NORMALS_LENGTH, 1.5)
    sign = -1 if inward else 1
    assert adapter.normals_3d is not None
    lines = _segments(adapter.normals_3d)
    assert len(lines) == 400  # the default count, capped at the vertex count
    along = lines[:, 1] - lines[:, 0]
    assert np.all(sign * np.sum(along * lines[:, 0], axis=1) > 0)
    np.testing.assert_allclose(np.linalg.norm(along, axis=1), 1.5, rtol=1e-5)
    for kind, axis in zip(TWO_D, range(3), strict=True):
        lines = _segments(adapter.cut_normals[kind])
        assert len(lines) > 10
        np.testing.assert_allclose(lines[:, :, axis], 0.0, atol=1e-5)
        along = lines[:, 1] - lines[:, 0]
        # The cut is a circle about the origin, so outward is away from it.
        assert np.all(sign * np.sum(along * lines[:, 0], axis=1) > 0)


def test_hiding_the_medial_wall_hides_its_vertices_normals(
    viewer: Viewer, views: dict[ViewKind, ViewRenderer]
) -> None:
    item = _item(viewer, CORTEX)
    adapter = _adapter(viewer, views, item)
    viewer.set(item.id, OptionName.SHOW_NORMALS, True)
    viewer.set(item.id, OptionName.HIDE_MEDIAL_WALL, True)
    assert adapter.normals_3d is not None
    starts = _segments(adapter.normals_3d)[:, 0]
    assert sorted(map(tuple, starts.tolist())) == sorted(
        map(tuple, VERTICES[CORTEX].tolist())
    )


@pytest.mark.parametrize("show", [False, True])
@pytest.mark.parametrize("cut_lines", [False, True])
def test_normals_show_in_3d_when_on_and_in_2d_only_with_the_cut_lines(
    show: bool, cut_lines: bool, viewer: Viewer, views: dict[ViewKind, ViewRenderer]
) -> None:
    item = _item(viewer)
    adapter = _adapter(viewer, views, item)
    viewer.set(item.id, OptionName.SHOW_NORMALS, True)
    viewer.set(item.id, OptionName.CUT_LINES, cut_lines)
    viewer.set(item.id, OptionName.SHOW_NORMALS, show)
    assert adapter.normals_3d is not None
    assert adapter.normals_3d.visible == show
    assert all(adapter.cut_normals[k].visible == (show and cut_lines) for k in TWO_D)


@pytest.mark.parametrize("colouring", [Colouring.DIRECTION, Colouring.SINGLE])
def test_normals_are_coloured_by_direction_or_in_the_chosen_colour(
    colouring: Colouring, viewer: Viewer, views: dict[ViewKind, ViewRenderer]
) -> None:
    item = _item(viewer)
    adapter = _adapter(viewer, views, item)
    viewer.set(item.id, OptionName.SHOW_NORMALS, True)
    viewer.set(item.id, OptionName.NORMALS_COLOUR, "#ff8000")
    viewer.set(item.id, OptionName.NORMALS_COLOURING, colouring)
    assert adapter.normals_3d is not None
    for line in [adapter.normals_3d, *(adapter.cut_normals[k] for k in TWO_D)]:
        material = line.material
        if colouring == Colouring.SINGLE:
            assert material.color_mode == "uniform"
            np.testing.assert_allclose(material.color.rgb, (1.0, 0.5, 0.0), atol=0.01)
        else:
            assert material.color_mode == "vertex"
            # On the octahedron each vertex's normal is its own direction.
            ends = line.geometry.positions.data[::2]
            unit = ends / np.linalg.norm(ends, axis=1, keepdims=True)
            np.testing.assert_allclose(
                line.geometry.colors.data[::2], np.abs(unit), atol=1e-6
            )


class _Gated:
    """A worker that runs requests at once while open, and holds them while shut."""

    open = True
    held: list[Callable[[], None]] = []

    def __init__(self, *args: object) -> None:
        pass

    def submit(
        self, work: Callable[[], object], done: Callable[[object], None], *_: object
    ) -> None:
        if _Gated.open:
            done(work())
        else:
            _Gated.held.append(lambda: done(work()))

    def close(self) -> None:
        pass


def test_2d_normals_of_an_old_cut_are_not_shown_on_a_new_one(
    viewer: Viewer, views: dict[ViewKind, ViewRenderer], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(surface_adapter, "Worker", _Gated)
    monkeypatch.setattr(_Gated, "held", [])
    item = _item(viewer)
    adapter = _adapter(viewer, views, item)
    line = adapter.cut_normals[ViewKind.AXIAL]
    viewer.set(item.id, OptionName.SHOW_NORMALS, True)
    assert len(_segments(line)) > 0
    viewer.set(item.id, OptionName.SHOW_NORMALS, False)
    viewer.move_cursor((0.0, 0.0, 1.0))
    monkeypatch.setattr(_Gated, "open", False)
    viewer.set(item.id, OptionName.SHOW_NORMALS, True)
    assert len(_Gated.held) > 0
    assert np.isnan(line.geometry.positions.data).all()
    for deliver in _Gated.held:
        deliver()
    starts = _segments(line)[:, 0]
    assert len(starts) > 0
    np.testing.assert_allclose(starts[:, 2], 1.0, atol=1e-6)
