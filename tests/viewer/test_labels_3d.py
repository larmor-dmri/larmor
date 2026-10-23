from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import pygfx as gfx
import pytest

from larmor.types import Volume
from larmor.viewer import Category, ItemId, OptionName, Viewer, ViewKind
from larmor.viewer.render import labels_3d
from larmor.viewer.render.colour_space import physical_colours
from larmor.viewer.render.labels_3d import Labels3D
from larmor.viewer.render.surfaces import (
    LabelMesh,
    label_surfaces,
    merge_surfaces,
)
from larmor.viewer.render.view import ViewRenderer
from larmor.viewer.render.volume_3d import Volume3D
from larmor.viewer.render.volume_adapter import VolumeAdapter

# 2 mm voxels, moved off the origin: a mesh left in voxel units would miss its box.
AFFINE = np.array(
    [[2.0, 0, 0, -10.0], [0, 2.0, 0, 4.0], [0, 0, 2.0, 7.0], [0, 0, 0, 1.0]]
)
# Label 4 fills voxels 2-4, 3-5, 1-3; label 9 is the single voxel (6, 6, 6).
BLOCK = (slice(2, 5), slice(3, 6), slice(1, 4))
LUT = "4 Four 200 30 60 0\n9 Nine 10 220 90 0\n"


def _parcellation() -> Volume:
    data = np.zeros((8, 8, 8), np.int16)
    data[BLOCK] = 4
    data[6, 6, 6] = 9
    return Volume(data, AFFINE)


def _world(vertices: np.ndarray, affine: np.ndarray) -> np.ndarray:
    world: np.ndarray = np.asarray(vertices, np.float64) @ affine[:3, :3].T
    world += affine[:3, 3]
    return world


def _surface(label: int) -> LabelMesh:
    surface = label_surfaces(_parcellation(), 0, [label])[label]
    assert surface is not None
    return surface


# The surfaces: no GPU needed


def test_a_labels_mesh_lies_within_its_box_in_world_mm() -> None:
    world = _world(_surface(4).vertices, AFFINE)
    # The box's faces are half a voxel out from its outer voxels' centres.
    low = _world(np.array([[b.start - 0.5 for b in BLOCK]]), AFFINE)[0]
    high = _world(np.array([[b.stop - 0.5 for b in BLOCK]]), AFFINE)[0]
    tolerance = 1e-4  # float32 vertices of a few mm
    assert np.all(world >= low - tolerance)
    assert np.all(world <= high + tolerance)
    # Not shrunk into a corner: it reaches across the box's middle.
    centre = (low + high) / 2
    assert np.all(world.min(axis=0) < centre)
    assert np.all(world.max(axis=0) > centre)


def test_a_one_voxel_label_still_has_a_surface_around_its_voxel() -> None:
    vertices = _surface(9).vertices
    assert len(vertices) > 0
    assert np.all(np.abs(vertices - 6.0) <= 0.5 + 1e-6)


def test_a_label_the_volume_lacks_has_no_surface() -> None:
    assert label_surfaces(_parcellation(), 0, [5]) == {5: None}


def test_merged_surfaces_give_each_vertex_its_labels_colour() -> None:
    four, nine = _surface(4), _surface(9)
    red, green = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)
    merged = merge_surfaces([four, nine], [red, green])
    assert merged is not None
    n = len(four.vertices)
    assert np.array_equal(merged.colours[:n], np.tile(red, (n, 1)))
    assert np.array_equal(merged.colours[n:], np.tile(green, (len(nine.vertices), 1)))
    # The second surface's faces point at its own vertices, after the first's.
    assert merged.surface.faces[len(four.faces) :].min() == n


# Labels3D in the 3D view


@pytest.fixture
def built(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    made: list[int] = []

    def record(
        volume: Volume, index: int, labels: Iterable[int]
    ) -> dict[int, LabelMesh | None]:
        wanted = list(labels)
        made.extend(wanted)
        return label_surfaces(volume, index, wanted)

    monkeypatch.setattr(labels_3d, "label_surfaces", record)
    return made


def _shown(
    views: dict[ViewKind, ViewRenderer], tmp_path: Path
) -> tuple[Viewer, ItemId, Volume3D]:
    viewer = Viewer()
    item = viewer.add(_parcellation(), Category.OVERLAYS)
    lut = tmp_path / "lut.txt"
    lut.write_text(LUT)
    viewer.set(item.id, OptionName.LOOKUP_TABLE, str(lut))
    adapter = VolumeAdapter(item, views, viewer, 0.0)
    viewer.scene.subscribe(adapter.on_event)
    viewer.set(item.id, OptionName.LABELS, True)
    assert adapter.three_d is not None
    return viewer, item.id, adapter.three_d


def _labels(three_d: Volume3D) -> Labels3D:
    assert three_d.labels is not None
    return three_d.labels


def _mesh(three_d: Volume3D) -> gfx.Mesh:
    mesh = _labels(three_d).mesh
    assert mesh is not None
    return mesh


def _has_colours(mesh: gfx.Mesh, *rgbs: tuple[int, int, int]) -> bool:
    found = np.unique(np.asarray(mesh.geometry.colors.data), axis=0)
    # A mesh's vertex colours are linear light: the sRGB colours, converted.
    expected = np.unique(physical_colours(np.array(rgbs) / 255), axis=0)
    # Colours are float32: equal to about 1e-7.
    return found.shape == expected.shape and np.allclose(found, expected, atol=1e-6)


def test_no_label_is_built_until_one_is_chosen(
    views: dict[ViewKind, ViewRenderer], tmp_path: Path, built: list[int]
) -> None:
    _, _, three_d = _shown(views, tmp_path)
    assert _labels(three_d).mesh is None
    assert built == []


def test_choosing_labels_shows_their_mesh_in_world_mm_and_their_colours(
    views: dict[ViewKind, ViewRenderer], tmp_path: Path, built: list[int]
) -> None:
    viewer, item_id, three_d = _shown(views, tmp_path)
    viewer.set(item_id, OptionName.SURFACE_LABELS, [4, 9])
    mesh = _mesh(three_d)
    assert sorted(built) == [4, 9]
    assert mesh.visible and mesh.parent is not None
    assert np.allclose(mesh.local.matrix, AFFINE)
    assert _has_colours(mesh, (200, 30, 60), (10, 220, 90))


def test_adding_a_label_builds_only_that_label(
    views: dict[ViewKind, ViewRenderer], tmp_path: Path, built: list[int]
) -> None:
    viewer, item_id, three_d = _shown(views, tmp_path)
    viewer.set(item_id, OptionName.SURFACE_LABELS, [4])
    first = len(_mesh(three_d).geometry.positions.data)
    viewer.set(item_id, OptionName.SURFACE_LABELS, [4, 9])
    assert built == [4, 9]
    assert len(_mesh(three_d).geometry.positions.data) > first


def test_unchoosing_a_label_drops_its_vertices_without_building_again(
    views: dict[ViewKind, ViewRenderer], tmp_path: Path, built: list[int]
) -> None:
    viewer, item_id, three_d = _shown(views, tmp_path)
    viewer.set(item_id, OptionName.SURFACE_LABELS, [4, 9])
    viewer.set(item_id, OptionName.SURFACE_LABELS, [9])
    mesh = _mesh(three_d)
    assert sorted(built) == [4, 9]
    assert len(mesh.geometry.positions.data) == len(_surface(9).vertices)
    assert _has_colours(mesh, (10, 220, 90))
    # Chosen again, a dropped label is built again.
    viewer.set(item_id, OptionName.SURFACE_LABELS, [4, 9])
    assert sorted(built) == [4, 4, 9]


def test_surface_3d_off_hides_the_labels_mesh_and_keeps_the_slices(
    views: dict[ViewKind, ViewRenderer], tmp_path: Path, built: list[int]
) -> None:
    viewer, item_id, three_d = _shown(views, tmp_path)
    viewer.set(item_id, OptionName.SURFACE_LABELS, [4])
    mesh = _mesh(three_d)
    viewer.set(item_id, OptionName.SURFACE_3D, False)
    assert not mesh.visible
    assert all(s.visible for s in three_d.slices.values())
    viewer.set(item_id, OptionName.SURFACE_3D, True)
    assert _mesh(three_d) is mesh and mesh.visible
    assert built == [4]


def test_the_labels_mesh_is_blended_only_below_full_opacity(
    views: dict[ViewKind, ViewRenderer], tmp_path: Path, built: list[int]
) -> None:
    viewer, item_id, three_d = _shown(views, tmp_path)
    viewer.set(item_id, OptionName.SURFACE_LABELS, [4])
    material = _mesh(three_d).material
    assert (material.alpha_mode, material.depth_write) == ("solid", True)
    viewer.set(item_id, OptionName.OPACITY_3D, 0.5)
    assert (material.alpha_mode, material.depth_write) == ("blend", False)
    assert material.opacity == pytest.approx(0.5)


class _Held:
    """A worker that only queues requests, so a test runs and delivers them."""

    def __init__(self, *args: object) -> None:
        self.queue: list[tuple[Callable[[], object], Callable[[object], None]]] = []

    def submit(
        self, work: Callable[[], object], done: Callable[[object], None], *_: object
    ) -> None:
        self.queue.append((work, done))

    def close(self) -> None:
        pass


def test_a_request_replaced_after_it_ran_still_saves_its_labels_work(
    views: dict[ViewKind, ViewRenderer],
    tmp_path: Path,
    built: list[int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(labels_3d, "Worker", _Held)
    viewer, item_id, three_d = _shown(views, tmp_path)
    worker = _labels(three_d)._worker
    assert isinstance(worker, _Held)
    viewer.set(item_id, OptionName.SURFACE_LABELS, [4])
    viewer.set(item_id, OptionName.SURFACE_LABELS, [4, 9])
    (first, _), (second, deliver) = worker.queue
    first()  # ran to the end, but replaced: its result is never handed back
    deliver(second())
    assert built == [4, 9]  # label 4 was not computed a second time
    assert _has_colours(_mesh(three_d), (200, 30, 60), (10, 220, 90))
