"""The regions, detail boxes and sketches drawn in the views, and 3D picking."""

import numpy as np
import pygfx as gfx
import pytest

from larmor.types import Tractogram, Volume
from larmor.viewer import Category, RegionRole, Viewer, ViewKind
from larmor.viewer.render.regions_guides import Sketch
from larmor.viewer.render.scene_renderer import SceneRenderer
from larmor.viewer.render.view import ViewRenderer


def _lines(renderer: SceneRenderer, kind: ViewKind) -> list[gfx.Line]:
    group = renderer.region_guides.groups[kind]
    return [c for c in group.children if isinstance(c, gfx.Line)]


def _radius(line: gfx.Line, centre: tuple[float, float, float]) -> float:
    points = np.asarray(line.geometry.positions.data, dtype=float)
    return float(np.linalg.norm(points[0] - np.asarray(centre)))


def test_a_sphere_is_drawn_as_its_true_cross_section_with_each_slice(
    scene_renderer: tuple[Viewer, SceneRenderer],
) -> None:
    viewer, renderer = scene_renderer
    viewer.move_cursor((0.0, 0.0, 0.0))
    viewer.add_sphere((0.0, 0.0, 0.0), 5.0, RegionRole.EXCLUDE)
    (line,) = _lines(renderer, ViewKind.AXIAL)
    assert _radius(line, (0.0, 0.0, 0.0)) == pytest.approx(5.0)
    assert line.material.dash_pattern == ()
    viewer.move_cursor((0.0, 0.0, 3.0))
    (line,) = _lines(renderer, ViewKind.AXIAL)
    assert _radius(line, (0.0, 0.0, 3.0)) == pytest.approx(4.0)
    assert line.material.dash_pattern != ()  # its centre is off this slice
    viewer.move_cursor((0.0, 0.0, 6.0))
    assert _lines(renderer, ViewKind.AXIAL) == []
    # The 3D view draws the whole sphere, translucent.
    (mesh,) = renderer.region_guides.groups[ViewKind.THREE_D].children
    assert isinstance(mesh, gfx.Mesh) and mesh.material.opacity < 1


def test_a_box_is_drawn_as_its_rectangle_and_a_mask_not_at_all(
    scene_renderer: tuple[Viewer, SceneRenderer],
) -> None:
    viewer, renderer = scene_renderer
    mask = viewer.add(Volume(np.ones((2, 2, 2)), np.eye(4)), Category.OVERLAYS)
    viewer.add_mask(mask.id)
    viewer.move_cursor((0.0, 0.0, 0.0))
    viewer.add_box((1.0, 2.0, 0.0), (4.0, 6.0, 2.0))
    (line,) = _lines(renderer, ViewKind.AXIAL)
    points = np.asarray(line.geometry.positions.data)
    np.testing.assert_allclose(points.min(axis=0), (-1.0, -1.0, 0.0))
    np.testing.assert_allclose(points.max(axis=0), (3.0, 5.0, 0.0))


def test_the_selected_region_shows_its_handles(
    scene_renderer: tuple[Viewer, SceneRenderer],
) -> None:
    viewer, renderer = scene_renderer
    sphere = viewer.add_sphere(viewer.scene.view.cursor, 5.0)
    group = renderer.region_guides.groups[ViewKind.CORONAL]
    assert not any(isinstance(c, gfx.Points) for c in group.children)
    viewer.select(sphere)
    group = renderer.region_guides.groups[ViewKind.CORONAL]
    assert any(isinstance(c, gfx.Points) for c in group.children)


def test_a_detail_is_outlined_in_its_view_only(
    scene_renderer: tuple[Viewer, SceneRenderer],
) -> None:
    viewer, renderer = scene_renderer
    viewer.add(Volume(np.zeros((20, 20, 20)), np.eye(4)), Category.VOLUMES)
    viewer.open_detail(ViewKind.CORONAL, (2, 3, 4, 5))
    (line,) = renderer.detail_boxes.groups[ViewKind.CORONAL].children
    points = np.asarray(line.geometry.positions.data)
    # Voxels 2 to 5 along x and 3 to 7 along z, to their outer edges.
    np.testing.assert_allclose(points.min(axis=0)[[0, 2]], (1.5, 2.5))
    np.testing.assert_allclose(points.max(axis=0)[[0, 2]], (5.5, 7.5))
    assert renderer.detail_boxes.groups[ViewKind.AXIAL].children == ()


def test_a_sketch_draws_a_rectangle_and_hides_again(
    views: dict[ViewKind, ViewRenderer],
) -> None:
    sketch = Sketch(views[ViewKind.AXIAL])
    sketch.show_box((0.0, 0.0, 1.0), (2.0, 3.0, 1.0), 2)
    points = np.asarray(sketch.line.geometry.positions.data)
    assert sketch.line.visible and len(points) == 5
    np.testing.assert_allclose(points.max(axis=0), (2.0, 3.0, 1.0))
    sketch.clear()
    assert not sketch.line.visible


def test_the_3d_view_picks_the_nearest_slice_or_streamline_under_the_pointer(
    scene_renderer: tuple[Viewer, SceneRenderer],
) -> None:
    viewer, renderer = scene_renderer
    viewer.add(Volume(np.ones((21, 21, 21)), np.eye(4)), Category.VOLUMES)
    view = renderer.views[ViewKind.THREE_D]
    view.canvas.draw()
    width, height = view.logical_size()
    picked = view.pick(width / 2, height / 2)
    assert picked is not None
    # On one of the slices through the cursor, the volume's centre voxel.
    assert np.isclose(np.asarray(picked) - 10.0, 0.0, atol=0.6).any()
    assert view.pick(0.0, 0.0) is None
    # A streamline through the point under the pointer, nearer the camera.
    origin, direction = view.ray(width / 2, height / 2)
    near = origin + direction * 5.0
    points = np.array([near - (0.0, 0.0, 1.0), near + (0.0, 0.0, 1.0)], np.float32)
    viewer.add(Tractogram(points, np.array([0, 2])), Category.TRACTOGRAMS)
    view.canvas.draw()
    picked = view.pick(width / 2, height / 2)
    assert picked is not None
    np.testing.assert_allclose(picked, near, atol=0.5)


def test_a_label_region_fills_its_voxels_on_the_slice_and_has_a_surface_in_3d(
    scene_renderer: tuple[Viewer, SceneRenderer],
) -> None:
    viewer, renderer = scene_renderer
    labels = np.zeros((8, 8, 8), np.int32)
    labels[2:4, 2:5, 1:6] = 3
    labels[5:7, 5:7, 1:6] = 4
    parcellation = viewer.add(Volume(labels, np.eye(4)), Category.OVERLAYS)
    viewer.move_cursor((0.0, 0.0, 3.0))
    viewer.add_label_region(parcellation.id, labels_a=[3], labels_b=[4])
    group = renderer.label_regions.groups[ViewKind.AXIAL]
    fills = [c for c in group.children if isinstance(c, gfx.Mesh)]
    # Set A's six voxels and set B's four, each in its own colour.
    assert [len(f.geometry.positions.data) // 4 for f in fills] == [6, 4]
    assert fills[0].material.color != fills[1].material.color
    assert len(renderer.label_regions.groups[ViewKind.THREE_D].children) == 2
