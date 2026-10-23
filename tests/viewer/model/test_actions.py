import importlib.util
import math
import re
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from viewer_helpers import record_events

from larmor.io import load_volume, save_tractogram, save_volume
from larmor.types import Surface, Tractogram, Volume
from larmor.viewer import (
    Category,
    OptionError,
    OptionName,
    Viewer,
    ViewerUnavailableError,
    ViewKind,
)
from larmor.viewer.model.events import (
    Event,
    ItemRemoved,
    OptionChanged,
    ViewStateChanged,
)
from larmor.viewer.model.options import OverlayOptions, VolumeOptions
from larmor.viewer.model.view_state import ViewField
from larmor.viewer.model.viewer import MIXED
from larmor.viewer.system import INSTALL_HINT


def _volume(affine: np.ndarray | None = None) -> Volume:
    return Volume(
        np.zeros((2, 3, 4), np.float32), np.eye(4) if affine is None else affine
    )


def _tractogram(points: list[list[float]]) -> Tractogram:
    return Tractogram(np.array(points), np.array([0, len(points)]))


def _visible(viewer: Viewer, category: Category = Category.VOLUMES) -> list[str]:
    return [i.name for i in viewer.scene.in_category(category) if i.options.visible]


def _main(viewer: Viewer) -> str | None:
    main = viewer.scene.view.main_volume
    return None if main is None else viewer.scene.item(main).name


def _viewer_with(*names: str) -> Viewer:
    viewer = Viewer()
    for name in names:
        viewer.add(_volume(), Category.VOLUMES, name=name)
    return viewer


def _checked(viewer: Viewer) -> list[str]:
    # Events are synchronous: a subscriber (a render adapter, the Layers panel) acts on
    # each one as it arrives, so the rules must hold at every event, not only after
    # the action ends.
    problems: list[str] = []

    def check(event: Event) -> None:
        view = viewer.scene.view
        ids = {i.id for i in viewer.scene.items}
        visible = [
            i for i in viewer.scene.in_category(Category.VOLUMES) if i.options.visible
        ]
        if len(visible) > 1:
            problems.append(f"{type(event).__name__}: {len(visible)} visible volumes")
        if view.main_volume is not None and view.main_volume not in ids:
            problems.append(
                f"{type(event).__name__}: main volume {view.main_volume} gone"
            )
        if isinstance(view.selected, int) and view.selected not in ids:
            problems.append(f"{type(event).__name__}: selected {view.selected} gone")
        viewer._reference_grid()  # must not raise

    viewer.scene.subscribe(check)
    return problems


# Adding


@pytest.mark.parametrize(
    ("category", "data"),
    [
        (Category.VOLUMES, "tractogram"),
        (Category.OVERLAYS, "tractogram"),
        (Category.TRACTOGRAMS, "volume"),
    ],
)
def test_data_of_the_wrong_type_for_the_category_is_refused(
    category: Category, data: str
) -> None:
    obj = _tractogram([[0, 0, 0]]) if data == "tractogram" else _volume()
    viewer = Viewer()
    with pytest.raises(ValueError, match=f"hold .* objects, not {type(obj).__name__}"):
        viewer.add(obj, category)
    assert viewer.scene.items == []


@pytest.mark.parametrize("shape", [(2, 3, 4), (2, 3, 4, 4)])
def test_a_peaks_image_without_3_volumes_per_peak_is_refused(
    shape: tuple[int, ...],
) -> None:
    viewer = Viewer()
    with pytest.raises(ValueError, match="3 volumes"):
        viewer.add(Volume(np.zeros(shape, np.float32), np.eye(4)), Category.PEAKS)
    assert viewer.scene.items == []


def test_items_without_a_name_are_numbered_per_category() -> None:
    viewer = Viewer()
    names = [
        viewer.add(_volume(), Category.VOLUMES).name,
        viewer.add(_volume(), Category.VOLUMES).name,
        viewer.add(_tractogram([[0, 0, 0]]), Category.TRACTOGRAMS).name,
    ]
    assert names == ["Volume 1", "Volume 2", "Tractogram 1"]


# One visible volume, the main volume


def test_the_first_volume_is_the_visible_main_volume() -> None:
    viewer = _viewer_with("a")
    assert _main(viewer) == "a"
    assert _visible(viewer) == ["a"]


def test_a_new_volume_becomes_main_and_hides_the_others() -> None:
    viewer = _viewer_with("a", "b")
    assert _main(viewer) == "b"
    assert _visible(viewer) == ["b"]


def test_showing_a_volume_hides_the_others_and_makes_it_main() -> None:
    viewer = _viewer_with("a", "b", "c")
    a = viewer.scene.in_category(Category.VOLUMES)[0]
    viewer.set(a.id, OptionName.VISIBLE, True)
    assert _main(viewer) == "a"
    assert _visible(viewer) == ["a"]


def test_hiding_the_main_volume_keeps_it_main() -> None:
    viewer = _viewer_with("a", "b")
    b = viewer.scene.in_category(Category.VOLUMES)[1]
    viewer.set(b.id, OptionName.VISIBLE, False)
    assert _main(viewer) == "b"
    assert _visible(viewer) == []


def test_several_overlays_can_be_visible() -> None:
    viewer = _viewer_with("v")
    viewer.add(_volume(), Category.OVERLAYS, name="o1")
    viewer.add(_volume(), Category.OVERLAYS, name="o2")
    o1 = viewer.scene.in_category(Category.OVERLAYS)[0]
    viewer.set(o1.id, OptionName.VISIBLE, True)
    assert _visible(viewer, Category.OVERLAYS) == ["o1", "o2"]
    assert _visible(viewer) == ["v"]
    assert _main(viewer) == "v"


# Removing


def test_removing_the_visible_main_volume_shows_the_next() -> None:
    viewer = _viewer_with("a", "b")
    viewer.remove(viewer.scene.in_category(Category.VOLUMES)[1].id)
    assert _main(viewer) == "a"
    assert _visible(viewer) == ["a"]


def test_removing_a_hidden_main_volume_keeps_the_next_hidden() -> None:
    viewer = _viewer_with("a", "b")
    b = viewer.scene.in_category(Category.VOLUMES)[1]
    viewer.set(b.id, OptionName.VISIBLE, False)
    viewer.remove(b.id)
    assert _main(viewer) == "a"
    assert _visible(viewer) == []


def test_removing_another_volume_keeps_the_main_volume() -> None:
    viewer = _viewer_with("a", "b")
    viewer.remove(viewer.scene.in_category(Category.VOLUMES)[0].id)
    assert _main(viewer) == "b"
    assert _visible(viewer) == ["b"]


def test_removing_the_last_volume_leaves_no_main_volume() -> None:
    viewer = _viewer_with("a")
    viewer.remove(viewer.scene.in_category(Category.VOLUMES)[0].id)
    assert viewer.scene.view.main_volume is None


def test_removing_the_selected_item_clears_the_selection() -> None:
    viewer = _viewer_with("a", "b")
    a, b = viewer.scene.in_category(Category.VOLUMES)
    viewer.select(a.id)
    viewer.remove(b.id)
    assert viewer.scene.view.selected == a.id
    viewer.remove(a.id)
    assert viewer.scene.view.selected is None


def _mixed_viewer() -> Viewer:
    viewer = _viewer_with("v1", "v2")
    viewer.add(_volume(), Category.OVERLAYS, name="o1")
    viewer.add(_volume(), Category.OVERLAYS, name="o2")
    viewer.add(_tractogram([[0, 0, 0]]), Category.TRACTOGRAMS, name="t1")
    viewer.add(_tractogram([[1, 1, 1]]), Category.TRACTOGRAMS, name="t2")
    return viewer


def _summary(events: list[Event], names: dict[int, str]) -> list[tuple[object, ...]]:
    # Items by name and view fields by value, so the expected order reads plainly.
    out: list[tuple[object, ...]] = []
    for e in events:
        if isinstance(e, ItemRemoved):
            out.append(("removed", e.item.name))
        elif isinstance(e, OptionChanged):
            out.append(("option", e.item.name, e.name, e.new))
        elif isinstance(e, ViewStateChanged):
            new = names.get(e.new, e.new) if isinstance(e.new, int) else e.new
            out.append(("view", e.name, new))
        else:
            out.append((type(e).__name__,))
    return out


@pytest.mark.parametrize("category", list(Category))
def test_removing_a_category_removes_its_items_and_only_them(
    category: Category,
) -> None:
    viewer = _mixed_viewer()
    others = [
        (i.id, i.category, i.name, i.options)
        for i in viewer.scene.items
        if i.category != category
    ]
    viewer.remove(category)
    assert viewer.scene.in_category(category) == []
    assert [(i.id, i.category, i.name, i.options) for i in viewer.scene.items] == others


@pytest.mark.parametrize("selected", [None, 0, 1])
@pytest.mark.parametrize("main_visible", [True, False])
@pytest.mark.parametrize("main", [0, 1, 2])
def test_removing_the_volumes_category_removes_the_main_volume_last(
    main: int, main_visible: bool, selected: int | None
) -> None:
    viewer = _viewer_with("a", "b", "c")
    volumes = viewer.scene.in_category(Category.VOLUMES)
    viewer.set(volumes[main].id, OptionName.VISIBLE, True)
    viewer.set(volumes[main].id, OptionName.VISIBLE, main_visible)
    if selected is not None:
        viewer.select(volumes[selected].id)
    names = {i.id: i.name for i in viewer.scene.items}
    problems = _checked(viewer)
    events = record_events(viewer)
    viewer.remove(Category.VOLUMES)
    assert problems == []
    assert viewer.scene.view.main_volume is None
    assert viewer.scene.view.selected is None
    assert viewer.scene.items == []
    # No volume is shown or hidden on the way: the others go first, in Layers order.
    expected: list[tuple[object, ...]] = []
    for i, name in enumerate(["a", "b", "c"]):
        if i == main:
            continue
        if i == selected:
            expected.append(("view", ViewField.SELECTED, None))
        expected.append(("removed", name))
    if main == selected:
        expected.append(("view", ViewField.SELECTED, None))
    expected += [("view", ViewField.MAIN_VOLUME, None), ("removed", "abc"[main])]
    assert _summary(events, names) == expected


@pytest.mark.parametrize("selected", [0, 1])
def test_removing_a_category_clears_the_selection_before_its_item_goes(
    selected: int,
) -> None:
    viewer = _mixed_viewer()
    overlays = viewer.scene.in_category(Category.OVERLAYS)
    viewer.select(overlays[selected].id)
    names = {i.id: i.name for i in viewer.scene.items}
    problems = _checked(viewer)
    events = record_events(viewer)
    viewer.remove(Category.OVERLAYS)
    assert problems == []
    assert viewer.scene.view.selected is None
    expected: list[tuple[object, ...]] = [("removed", "o1"), ("removed", "o2")]
    expected.insert(selected, ("view", ViewField.SELECTED, None))
    assert _summary(events, names) == expected


@pytest.mark.parametrize("category", list(Category))
def test_removing_an_empty_category_sends_nothing(category: Category) -> None:
    viewer = _mixed_viewer()
    for item in viewer.scene.in_category(category):
        viewer.remove(item.id)
    # The category's row stays, so a selected category stays selected.
    viewer.select(category)
    before = [(i.id, i.options) for i in viewer.scene.items]
    view = viewer.scene.view
    events = record_events(viewer)
    viewer.remove(category)
    assert events == []
    assert [(i.id, i.options) for i in viewer.scene.items] == before
    assert viewer.scene.view == view


# Category changes


def test_a_category_change_sets_the_field_on_every_item() -> None:
    viewer = _viewer_with("a", "b")
    events = record_events(viewer)
    viewer.set(Category.VOLUMES, OptionName.OPACITY_2D, 0.5)
    a, b = viewer.scene.in_category(Category.VOLUMES)
    assert [i.options.opacity_2d for i in (a, b)] == [0.5, 0.5]
    assert [i.options.opacity_3d for i in (a, b)] == [1.0, 1.0]
    assert events == [
        OptionChanged(a, OptionName.OPACITY_2D, 1.0, 0.5),
        OptionChanged(b, OptionName.OPACITY_2D, 1.0, 0.5),
    ]


def test_a_category_change_leaves_other_categories_alone() -> None:
    viewer = _viewer_with("a")
    viewer.add(_volume(), Category.OVERLAYS, name="o")
    viewer.set(Category.VOLUMES, OptionName.OPACITY_2D, 0.5)
    assert viewer.scene.in_category(Category.OVERLAYS)[0].options.opacity_2d == 1.0


@pytest.mark.parametrize(OptionName.VISIBLE, [True, False])
def test_volumes_category_visible_acts_on_the_main_volume_only(visible: bool) -> None:
    viewer = _viewer_with("a", "b", "c")
    b = viewer.scene.in_category(Category.VOLUMES)[1]
    viewer.set(b.id, OptionName.VISIBLE, True)
    viewer.set(b.id, OptionName.VISIBLE, False)
    viewer.set(Category.VOLUMES, OptionName.VISIBLE, visible)
    assert _visible(viewer) == (["b"] if visible else [])
    assert _main(viewer) == "b"


def test_overlays_category_visible_acts_on_every_overlay() -> None:
    viewer = Viewer()
    viewer.add(_volume(), Category.OVERLAYS, name="o1")
    viewer.add(_volume(), Category.OVERLAYS, name="o2")
    viewer.set(Category.OVERLAYS, OptionName.VISIBLE, False)
    assert _visible(viewer, Category.OVERLAYS) == []
    viewer.set(Category.OVERLAYS, OptionName.VISIBLE, True)
    assert _visible(viewer, Category.OVERLAYS) == ["o1", "o2"]


def test_a_category_value_is_mixed_where_items_differ() -> None:
    viewer = _viewer_with("a", "b")
    viewer.set(
        viewer.scene.in_category(Category.VOLUMES)[0].id, OptionName.OPACITY_2D, 0.5
    )
    assert viewer._category_value(Category.VOLUMES, OptionName.OPACITY_2D) is MIXED


def test_volumes_visible_is_the_main_volumes_like_the_action() -> None:
    # Only one volume can be visible, so the category's value follows the main volume
    # (the one set(Category.VOLUMES, OptionName.VISIBLE, ...) changes), never MIXED.
    viewer = _viewer_with("a", "b", "c")
    assert viewer._category_value(Category.VOLUMES, OptionName.VISIBLE) is True
    viewer.set(Category.VOLUMES, OptionName.VISIBLE, False)
    assert viewer._category_value(Category.VOLUMES, OptionName.VISIBLE) is False
    assert (
        Viewer()._category_value(Category.VOLUMES, OptionName.VISIBLE) is True
    )  # the default


def test_a_category_value_is_shared_where_items_agree() -> None:
    viewer = _viewer_with("a", "b")
    viewer.set(Category.VOLUMES, OptionName.OPACITY_3D, 0.25)
    assert viewer._category_value(Category.VOLUMES, OptionName.OPACITY_3D) == 0.25


def test_an_empty_category_reports_the_default() -> None:
    assert Viewer()._category_value(Category.TRACTOGRAMS, OptionName.VISIBLE) is True
    assert Viewer()._category_value(Category.OVERLAYS, OptionName.OPACITY_2D) == 1.0


def test_a_category_value_of_an_option_the_category_lacks_raises() -> None:
    with pytest.raises(OptionError, match="no option 'threshold'"):
        _viewer_with("a")._category_value(Category.VOLUMES, OptionName.THRESHOLD)


@pytest.mark.parametrize("target", ["item", "category"])
@pytest.mark.parametrize(
    ("name", "value"),
    [
        (OptionName.OPACITY_2D, 2.0),
        (OptionName.VISIBLE, "yes"),
        (OptionName.THRESHOLD, 0.5),
    ],
)
def test_a_refused_value_changes_nothing_and_sends_nothing(
    target: str, name: OptionName, value: object
) -> None:
    viewer = _viewer_with("a", "b")
    before = [i.options for i in viewer.scene.items]
    events = record_events(viewer)
    goal = viewer.scene.items[0].id if target == "item" else Category.VOLUMES
    with pytest.raises(OptionError):
        viewer.set(goal, name, value)
    assert events == []
    assert [i.options for i in viewer.scene.items] == before


def test_option_an_unknown_item_raises() -> None:
    with pytest.raises(ValueError, match="There is no item 9"):
        Viewer().set(9, OptionName.VISIBLE, True)


def _overlays(n: int) -> Viewer:
    viewer = Viewer()
    for i in range(n):
        viewer.add(_volume(), Category.OVERLAYS, name=f"O{i}")
    return viewer


@pytest.mark.parametrize("target", ["item", "category"])
@pytest.mark.parametrize("text", [None, "1 One 1 2\n"], ids=["missing", "garbled"])
def test_a_bad_lookup_table_changes_nothing_and_sends_nothing(
    target: str, text: str | None, tmp_path: Path
) -> None:
    path = tmp_path / "lut.txt"
    if text is not None:
        path.write_text(text)
    viewer = _overlays(2)
    before = [i.options for i in viewer.scene.items]
    events = record_events(viewer)
    goal = viewer.scene.items[0].id if target == "item" else Category.OVERLAYS
    with pytest.raises(OptionError, match=re.escape(str(path))):
        viewer.set(goal, OptionName.LOOKUP_TABLE, str(path))
    assert events == []
    assert [i.options for i in viewer.scene.items] == before


def test_a_lookup_table_set_on_the_overlays_reaches_every_overlay(
    tmp_path: Path,
) -> None:
    path = tmp_path / "lut.txt"
    path.write_text("1 One 1 2 3 0\n")
    viewer = _overlays(2)
    viewer.set(Category.OVERLAYS, OptionName.LOOKUP_TABLE, str(path))
    viewer.set(Category.OVERLAYS, OptionName.LABELS, True)
    for item in viewer.scene.items:
        assert isinstance(item.options, OverlayOptions)
        assert item.options.lookup_table == str(path)
        assert item.options.labels is True


# Defaults for new items


@pytest.mark.parametrize(
    ("category", "other"),
    [
        (Category.VOLUMES, Category.OVERLAYS),
        (Category.OVERLAYS, Category.VOLUMES),
    ],
)
def test_a_default_applies_to_items_added_later_and_not_to_existing_ones(
    category: Category, other: Category
) -> None:
    viewer = Viewer()
    existing = viewer.add(_volume(), category).id
    viewer.set_default(category, OptionName.INTERPOLATE, False)
    added = viewer.add(_volume(), category).id
    elsewhere = viewer.add(_volume(), other).id
    interpolate = {
        i.id: getattr(i.options, OptionName.INTERPOLATE) for i in viewer.scene.items
    }
    assert interpolate == {existing: True, added: False, elsewhere: True}


def test_an_asymmetric_peaks_default_applies_to_peaks_added_later() -> None:
    peaks = Volume(np.zeros((2, 2, 2, 3), np.float32), np.eye(4))
    viewer = Viewer()
    existing = viewer.add(peaks, Category.PEAKS)
    viewer.set_default(Category.PEAKS, OptionName.ASYMMETRIC_PEAKS, True)
    added = viewer.add(peaks, Category.PEAKS)
    assert getattr(existing.options, OptionName.ASYMMETRIC_PEAKS) is False
    assert getattr(added.options, OptionName.ASYMMETRIC_PEAKS) is True


def test_a_bad_default_is_refused() -> None:
    viewer = Viewer()
    with pytest.raises(OptionError):
        viewer.set_default(Category.VOLUMES, OptionName.INTERPOLATE, "no")
    options = viewer.add(_volume(), Category.VOLUMES).options
    assert isinstance(options, VolumeOptions)
    assert options.interpolate is True


# Slices and the reference grid


@pytest.mark.parametrize(
    ("view", "axis"),
    [(ViewKind.SAGITTAL, 0), (ViewKind.CORONAL, 1), (ViewKind.AXIAL, 2)],
)
def test_a_slice_step_is_the_main_volume_voxel_size_along_the_view_axis(
    view: ViewKind, axis: int
) -> None:
    viewer = Viewer()
    viewer.add(_volume(np.diag([2.0, 3.0, 4.0, 1.0])), Category.VOLUMES)
    viewer.move_cursor((1.0, 1.0, 1.0))
    viewer.step_slice(view, -2)
    expected = [1.0, 1.0, 1.0]
    expected[axis] -= 2 * (2.0, 3.0, 4.0)[axis]
    assert viewer.scene.view.cursor == pytest.approx(tuple(expected))


@pytest.mark.parametrize(
    ("view", "axis", "size"),
    [(ViewKind.SAGITTAL, 0, 2.0), (ViewKind.CORONAL, 1, 3.0), (ViewKind.AXIAL, 2, 4.0)],
)
def test_a_slice_step_on_an_oblique_grid_is_the_voxel_size(
    view: ViewKind, axis: int, size: float
) -> None:
    # 30 degrees about z: the voxel x axis is still closest to world x, but the
    # diagonal of the affine is 2 cos 30 and 3 cos 30, not the voxel sizes.
    c, s = math.cos(math.radians(30)), math.sin(math.radians(30))
    affine = np.eye(4)
    affine[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ np.diag([2, 3, 4])
    viewer = Viewer()
    viewer.add(_volume(affine), Category.VOLUMES)
    expected = list(viewer.scene.view.cursor)
    viewer.step_slice(view, 1)
    expected[axis] += size
    assert viewer.scene.view.cursor == pytest.approx(tuple(expected))


def test_a_slice_step_is_1_mm_without_a_grid() -> None:
    viewer = Viewer()
    viewer.step_slice(ViewKind.CORONAL, 3)
    assert viewer.scene.view.cursor == (0.0, 3.0, 0.0)


def test_the_3d_view_has_no_slices() -> None:
    with pytest.raises(ValueError, match="3d view has no slices"):
        Viewer().step_slice(ViewKind.THREE_D, 1)


def test_the_reference_grid_is_the_main_volume_grid() -> None:
    viewer = Viewer()
    viewer.add(_volume(np.diag([5.0, 5.0, 5.0, 1.0])), Category.OVERLAYS)
    viewer.add(_volume(np.diag([2.0, 2.0, 2.0, 1.0])), Category.VOLUMES)
    viewer.add(_volume(np.diag([3.0, 3.0, 3.0, 1.0])), Category.VOLUMES, name="main")
    grid = viewer._reference_grid()
    assert grid is not None
    assert np.array_equal(grid.affine, np.diag([3.0, 3.0, 3.0, 1.0]))
    assert grid.shape == (2, 3, 4)


def test_without_a_volume_the_reference_grid_is_an_overlay_grid() -> None:
    viewer = Viewer()
    viewer.add(_tractogram([[0, 0, 0], [9, 9, 9]]), Category.TRACTOGRAMS)
    viewer.add(_volume(np.diag([5.0, 5.0, 5.0, 1.0])), Category.OVERLAYS)
    grid = viewer._reference_grid()
    assert grid is not None
    assert np.array_equal(grid.affine, np.diag([5.0, 5.0, 5.0, 1.0]))


@pytest.mark.parametrize("overlay", [False, True])
def test_without_a_volume_or_overlay_the_reference_grid_is_a_peaks_grid(
    overlay: bool,
) -> None:
    viewer = Viewer()
    viewer.add(_tractogram([[0, 0, 0], [9, 9, 9]]), Category.TRACTOGRAMS)
    peaks = Volume(np.zeros((2, 3, 4, 6), np.float32), np.diag([4.0, 4.0, 4.0, 1.0]))
    viewer.add(peaks, Category.PEAKS)
    if overlay:  # added after the peaks, yet an overlay comes first
        viewer.add(_volume(np.diag([5.0, 5.0, 5.0, 1.0])), Category.OVERLAYS)
    grid = viewer._reference_grid()
    assert grid is not None
    size = 5.0 if overlay else 4.0
    assert np.array_equal(grid.affine, np.diag([size, size, size, 1.0]))
    assert grid.shape == (2, 3, 4)


def test_with_only_tractograms_the_reference_grid_is_1_mm_around_the_points() -> None:
    viewer = Viewer()
    viewer.add(_tractogram([[0.5, -1.2, 2.0], [1.0, 0.0, 3.0]]), Category.TRACTOGRAMS)
    viewer.add(_tractogram([[3.7, 1.0, 4.9]]), Category.TRACTOGRAMS)
    grid = viewer._reference_grid()
    assert grid is not None
    expected = np.eye(4)
    expected[:3, 3] = [0.0, -2.0, 2.0]
    assert np.array_equal(grid.affine, expected)
    assert grid.shape == (5, 4, 4)


def test_an_empty_scene_has_no_reference_grid() -> None:
    assert Viewer()._reference_grid() is None


# The value range of images, and the cursor on the first image


def _ranged(slope: float = 1.0, intercept: float = 0.0) -> Volume:
    data = np.array([-2, 0, 5, 1], np.int16).reshape(1, 2, 2)
    return Volume(data, np.eye(4), slope=slope, intercept=intercept)


@pytest.mark.parametrize("category", [Category.VOLUMES, Category.OVERLAYS])
@pytest.mark.parametrize(
    ("slope", "intercept", "expected"),
    [
        (1.0, 0.0, (-2.0, 5.0)),
        (2.0, 10.0, (6.0, 20.0)),
        # A negative slope turns the stored maximum into the scaled minimum.
        (-0.5, 1.0, (-1.5, 2.0)),
    ],
)
def test_an_image_value_range_is_its_scaled_minimum_and_maximum(
    category: Category, slope: float, intercept: float, expected: tuple[float, float]
) -> None:
    item = Viewer().add(_ranged(slope, intercept), category)
    assert item.value_range == expected
    assert all(type(v) is float for v in expected)


def test_nan_voxels_are_left_out_of_the_value_range() -> None:
    data = np.array([np.nan, 1.5, -0.5, np.nan], np.float32).reshape(1, 2, 2)
    item = Viewer().add(Volume(data, np.eye(4)), Category.OVERLAYS)
    assert item.value_range == (-0.5, 1.5)


def test_a_tractogram_has_no_value_range() -> None:
    item = Viewer().add(_tractogram([[0, 0, 0], [1, 1, 1]]), Category.TRACTOGRAMS)
    assert item.value_range is None


@pytest.mark.parametrize("category", [Category.VOLUMES, Category.OVERLAYS])
def test_an_image_starts_with_minimum_and_maximum_at_its_value_range(
    category: Category,
) -> None:
    options = Viewer().add(_ranged(-0.5, 1.0), category).options
    assert isinstance(options, VolumeOptions)
    assert (options.minimum, options.maximum) == (-1.5, 2.0)


@pytest.mark.parametrize(
    ("slope", "intercept", "minimum"), [(1.0, 0.0, -2.0), (-0.5, 1.0, -1.5)]
)
def test_an_overlay_threshold_starts_at_its_value_range_minimum(
    slope: float, intercept: float, minimum: float
) -> None:
    options = Viewer().add(_ranged(slope, intercept), Category.OVERLAYS).options
    assert isinstance(options, OverlayOptions)
    assert options.threshold == minimum


def test_a_volume_has_no_threshold() -> None:
    options = Viewer().add(_ranged(), Category.VOLUMES).options
    assert not isinstance(options, OverlayOptions)
    assert not hasattr(options, OptionName.THRESHOLD)


def test_an_overlays_category_threshold_change_sets_only_the_threshold() -> None:
    viewer = Viewer()
    a = viewer.add(_ranged(), Category.OVERLAYS)
    b = viewer.add(_ranged(2.0, 10.0), Category.OVERLAYS)
    volume = viewer.add(_ranged(), Category.VOLUMES)
    before = {i.id: i.options for i in viewer.scene.items}
    events = record_events(viewer)
    viewer.set(Category.OVERLAYS, OptionName.THRESHOLD, 3.0)
    assert [
        (e.item.id, e.name, e.new) for e in events if isinstance(e, OptionChanged)
    ] == [
        (a.id, OptionName.THRESHOLD, 3.0),
        (b.id, OptionName.THRESHOLD, 3.0),
    ]
    assert len(events) == 2
    for item in (a, b):
        options = before[item.id]
        assert isinstance(options, OverlayOptions)
        assert viewer.scene.item(item.id).options == replace(options, threshold=3.0)
    assert viewer.scene.item(volume.id).options == before[volume.id]


def _oblique() -> np.ndarray:
    affine = np.eye(4)
    c, s = math.cos(math.radians(30)), math.sin(math.radians(30))
    affine[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ np.diag([2, 3, 4])
    affine[:3, 3] = [10.0, -20.0, 5.0]
    return affine


@pytest.mark.parametrize("category", [Category.VOLUMES, Category.OVERLAYS])
def test_the_first_image_centres_the_cursor_on_its_grid(category: Category) -> None:
    affine = _oblique()
    viewer = Viewer()
    viewer.add(_volume(affine), category)
    # The volume is (2, 3, 4): its centre voxel is (1, 1, 2).
    centre = affine[:3, :3] @ [1.0, 1.0, 2.0] + affine[:3, 3]
    assert viewer.scene.view.cursor == pytest.approx(tuple(centre))


@pytest.mark.parametrize("affine", [np.eye(4), _oblique()], ids=["identity", "oblique"])
@pytest.mark.parametrize("shape", [(8, 8, 8), (7, 7, 7), (2, 5, 1), (1, 6, 9)])
def test_the_centred_cursor_is_a_whole_voxel(
    shape: tuple[int, int, int], affine: np.ndarray
) -> None:
    # Off a whole voxel, a slice at the cursor on an even axis would blend two planes.
    viewer = Viewer()
    viewer.add(Volume(np.zeros(shape, np.float32), affine), Category.VOLUMES)
    voxel = np.linalg.inv(affine) @ [*viewer.scene.view.cursor, 1.0]
    assert voxel[:3] == pytest.approx(np.array(shape) // 2, abs=1e-9)


def test_a_first_tractogram_centres_the_cursor_on_its_grid() -> None:
    viewer = Viewer()
    viewer.add(_tractogram([[0.5, -1.2, 2.0], [1.0, 0.0, 3.0]]), Category.TRACTOGRAMS)
    # The 1 mm grid from (0, -2, 2), shape (2, 3, 2): voxel (1, 1, 1).
    assert viewer.scene.view.cursor == (1.0, -1.0, 3.0)


def test_an_empty_tractogram_gives_no_grid_so_the_next_image_centres() -> None:
    viewer = Viewer()
    viewer.add(Tractogram(np.zeros((0, 3)), np.array([0])), Category.TRACTOGRAMS)
    assert viewer.scene.view.cursor == (0.0, 0.0, 0.0)
    viewer.add(_volume(np.diag([2.0, 2.0, 2.0, 1.0])), Category.OVERLAYS)
    assert viewer.scene.view.cursor == (2.0, 2.0, 4.0)


@pytest.mark.parametrize("first", list(Category))
@pytest.mark.parametrize("second", list(Category))
def test_a_later_item_keeps_the_cursor(first: Category, second: Category) -> None:
    def data(category: Category) -> object:
        if category == Category.TRACTOGRAMS:
            return _tractogram([[30.0, 30.0, 30.0], [40.0, 41.0, 42.0]])
        if category == Category.PEAKS:
            return Volume(np.zeros((2, 3, 4, 3), np.float32), _oblique())
        if category == Category.SURFACES:
            return Surface(np.eye(3) * 30.0, np.array([[0, 1, 2]]))
        return _volume(_oblique())

    viewer = Viewer()
    viewer.add(data(first), first)
    viewer.move_cursor((1.0, 2.0, 3.0))
    viewer.add(data(second), second)
    assert viewer.scene.view.cursor == (1.0, 2.0, 3.0)


def test_a_scene_emptied_and_filled_again_centres_the_cursor_again() -> None:
    viewer = Viewer()
    item = viewer.add(_volume(), Category.VOLUMES)
    viewer.remove(item.id)
    viewer.move_cursor((9.0, 9.0, 9.0))
    viewer.add(_volume(np.diag([2.0, 2.0, 2.0, 1.0])), Category.VOLUMES)
    assert viewer.scene.view.cursor == (2.0, 2.0, 4.0)


def _kind(event: Event) -> tuple[str, str | None]:
    name = event.name if isinstance(event, (OptionChanged, ViewStateChanged)) else None
    return (type(event).__name__, name)


@pytest.mark.parametrize(
    ("category", "expected"),
    [
        (
            Category.VOLUMES,
            [
                ("ItemAdded", None),
                ("OptionChanged", OptionName.VISIBLE),
                ("ViewStateChanged", ViewField.MAIN_VOLUME),
                ("ViewStateChanged", ViewField.FIELD_OF_VIEW),
                ("ViewStateChanged", ViewField.CURSOR),
            ],
        ),
        (
            Category.OVERLAYS,
            [
                ("ItemAdded", None),
                ("ViewStateChanged", ViewField.FIELD_OF_VIEW),
                ("ViewStateChanged", ViewField.CURSOR),
            ],
        ),
    ],
)
def test_the_cursor_is_centred_after_the_item_is_in_place(
    category: Category, expected: list[tuple[str, str | None]]
) -> None:
    viewer = Viewer()
    problems = _checked(viewer)
    events = record_events(viewer)
    seen_at_cursor: list[object] = []

    def at_cursor(event: Event) -> None:
        if isinstance(event, ViewStateChanged) and event.name == ViewField.CURSOR:
            seen_at_cursor.append(viewer._reference_grid())

    viewer.scene.subscribe(at_cursor)
    viewer.add(_volume(np.diag([2.0, 2.0, 2.0, 1.0])), category)
    assert [_kind(e) for e in events] == expected
    # A subscriber moving its slices on the cursor event already finds the image.
    assert len(seen_at_cursor) == 1 and seen_at_cursor[0] is not None
    assert problems == []


# View state


def test_toggle_maximised_toggles_and_switches() -> None:
    viewer = Viewer()
    seen = []
    for view in (ViewKind.AXIAL, ViewKind.THREE_D, ViewKind.THREE_D):
        viewer.toggle_maximised(view)
        seen.append(viewer.scene.view.maximised)
    assert seen == [ViewKind.AXIAL, ViewKind.THREE_D, None]


def test_a_view_enlarged_while_another_is_becomes_the_selected_view() -> None:
    viewer = Viewer()
    viewer.toggle_maximised(ViewKind.AXIAL)
    viewer.toggle_maximised(ViewKind.CORONAL)
    assert viewer.scene.view.selected_view == ViewKind.CORONAL
    viewer.toggle_maximised(ViewKind.CORONAL)  # back to four views: it stays selected
    assert viewer.scene.view.selected_view == ViewKind.CORONAL


def test_select_takes_an_item_a_category_or_nothing() -> None:
    viewer = _viewer_with("a")
    item_id = viewer.scene.items[0].id
    seen = []
    for target in (item_id, Category.TRACTOGRAMS, None):
        viewer.select(target)
        seen.append(viewer.scene.view.selected)
    assert seen == [item_id, Category.TRACTOGRAMS, None]


def test_select_refuses_an_unknown_item() -> None:
    viewer = Viewer()
    with pytest.raises(ValueError, match="no item 42"):
        viewer.select(42)
    assert viewer.scene.view.selected is None


def test_select_view_changes_the_selected_view() -> None:
    viewer = Viewer()
    viewer.select_view(ViewKind.SAGITTAL)
    assert viewer.scene.view.selected_view == ViewKind.SAGITTAL


def test_move_cursor_stores_floats() -> None:
    viewer = Viewer()
    viewer.move_cursor((1, np.float32(2.5), 3))  # type: ignore[arg-type]
    cursor = viewer.scene.view.cursor
    assert cursor == (1.0, 2.5, 3.0)
    assert all(type(x) is float for x in cursor)


# Loading


def test_load_reads_a_volume_named_after_its_file(tmp_path: Path) -> None:
    path = tmp_path / "t1.nii.gz"
    save_volume(_volume(np.diag([2.0, 2.0, 2.0, 1.0])), path)
    viewer = Viewer()
    item = viewer.load(path, Category.VOLUMES)
    assert item.name == "t1.nii.gz"
    assert item.source == path
    assert isinstance(item.data, Volume)
    assert np.array_equal(item.data.affine, load_volume(path).affine)
    assert _main(viewer) == "t1.nii.gz"


def test_load_reads_a_tractogram_named_after_its_file(tmp_path: Path) -> None:
    path = tmp_path / "cst.tck"
    save_tractogram(_tractogram([[0, 0, 0], [1, 2, 3]]), path)
    item = Viewer().load(str(path), Category.TRACTOGRAMS)
    assert item.name == "cst.tck"
    assert item.source == path
    assert isinstance(item.data, Tractogram)
    assert np.allclose(item.data.points, [[0, 0, 0], [1, 2, 3]])


# Names from plain Python: text is refused, even text that names a member

# A call with one argument left open: the viewer, its volume's id, a volume file, and
# the value under test.
type _Call = Callable[[Viewer, int, Path, Any], object]

_ENUM_CALLS: list[tuple[str, str, _Call]] = [
    ("load", "category", lambda v, i, p, x: v.load(p, x)),
    ("add", "category", lambda v, i, p, x: v.add(_volume(), x)),
    (
        "_category_value",
        "category",
        lambda v, i, p, x: v._category_value(x, OptionName.VISIBLE),
    ),
    (
        "_category_value",
        "name",
        lambda v, i, p, x: v._category_value(Category.VOLUMES, x),
    ),
    ("set", "name", lambda v, i, p, x: v.set(i, x, False)),
    ("set", "target", lambda v, i, p, x: v.set(x, OptionName.VISIBLE, False)),
    ("remove", "target", lambda v, i, p, x: v.remove(x)),
    ("select", "target", lambda v, i, p, x: v.select(x)),
    ("step_slice", "view", lambda v, i, p, x: v.step_slice(x, 1)),
    ("select_view", "view", lambda v, i, p, x: v.select_view(x)),
    ("toggle_maximised", "view", lambda v, i, p, x: v.toggle_maximised(x)),
    ("zoom", "view", lambda v, i, p, x: v.zoom(x, 2.0)),
    ("pan", "view", lambda v, i, p, x: v.pan(x, 1.0, 1.0)),
    ("reset_camera", "view", lambda v, i, p, x: v.reset_camera(x)),
    ("view_centre", "view", lambda v, i, p, x: v.view_centre(x)),
    ("set_convention", "convention", lambda v, i, p, x: v.set_convention(x)),
]
_EXPECTED: dict[str, tuple[str, list[object]]] = {
    "category": (
        "Expected a Category, such as Category.VOLUMES",
        ["volumes", "overlays", "bogus"],
    ),
    "name": (
        "Expected an OptionName, such as OptionName.VISIBLE",
        ["visible", "bogus"],
    ),
    "view": ("Expected a ViewKind, such as ViewKind.AXIAL", ["axial", "bogus"]),
    "convention": (
        "Expected a Convention, such as Convention.RAS",
        ["RAS", "LAS", "bogus"],
    ),
    "target": (
        "Expected an item id or a Category, such as Category.VOLUMES",
        ["volumes", "overlays", "bogus", "1", True, 1.5],
    ),
}
_REFUSED = [
    pytest.param(call, value, _EXPECTED[kind][0], id=f"{action}-{kind}-{value!r}")
    for action, kind, call in _ENUM_CALLS
    for value in _EXPECTED[kind][1]
]


def _state(viewer: Viewer) -> object:
    return [
        (i.id, i.category, i.options) for i in viewer.scene.items
    ], viewer.scene.view


@pytest.mark.parametrize(("call", "value", "expected"), _REFUSED)
def test_an_action_refuses_what_is_not_its_enum_and_changes_nothing(
    call: _Call, value: object, expected: str, tmp_path: Path
) -> None:
    path = tmp_path / "t1.nii.gz"
    save_volume(_volume(), path)
    viewer = Viewer()
    item = viewer.add(_volume(), Category.VOLUMES)
    before = _state(viewer)
    events = record_events(viewer)
    message = re.escape(f"{expected}; got {value!r}.")
    with pytest.raises(TypeError, match=f"^{message}$"):
        call(viewer, item.id, path, value)
    assert events == []
    assert _state(viewer) == before


def test_every_action_still_takes_its_enums(tmp_path: Path) -> None:
    path = tmp_path / "t1.nii.gz"
    save_volume(_volume(), path)
    viewer = Viewer()
    volume = viewer.load(path, Category.VOLUMES)
    overlay = viewer.add(_volume(), Category.OVERLAYS)
    viewer.set(overlay.id, OptionName.VISIBLE, False)
    assert viewer._category_value(Category.OVERLAYS, OptionName.VISIBLE) is False
    viewer.set(Category.OVERLAYS, OptionName.VISIBLE, True)
    viewer.select(Category.OVERLAYS)
    viewer.select(overlay.id)
    viewer.select_view(ViewKind.CORONAL)
    viewer.toggle_maximised(ViewKind.CORONAL)
    viewer.step_slice(ViewKind.CORONAL, 1)
    viewer.remove(overlay.id)
    viewer.remove(Category.OVERLAYS)
    view = viewer.scene.view
    assert view.selected_view == view.maximised == ViewKind.CORONAL
    assert [i.id for i in viewer.scene.items] == [volume.id]


# Values from NumPy and non-finite values


def test_numpy_scalars_are_accepted_and_stored_as_python_numbers() -> None:
    viewer = _viewer_with("a")
    item = viewer.scene.items[0]
    viewer.set(item.id, OptionName.OPACITY_2D, np.float32(0.5))
    assert item.options.opacity_2d == 0.5
    assert type(item.options.opacity_2d) is float
    viewer.set(item.id, OptionName.OPACITY_3D, np.int64(0))
    assert type(item.options.opacity_3d) is float


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_values_are_refused(value: float) -> None:
    viewer = _viewer_with("a")
    item = viewer.scene.items[0]
    with pytest.raises(OptionError, match="finite"):
        viewer.set(item.id, OptionName.OPACITY_2D, value)
    assert item.options.opacity_2d == 1.0


# The cursor


@pytest.mark.parametrize(
    "position",
    [
        (1.0, 2.0),
        (1.0, 2.0, 3.0, 4.0),
        (math.nan, 0.0, 0.0),
        (0.0, math.inf, 0.0),
        ("1", 2.0, 3.0),
        (True, 0.0, 0.0),
    ],
)
def test_a_cursor_position_must_be_three_finite_numbers(
    position: tuple[object, ...],
) -> None:
    viewer = Viewer()
    events = record_events(viewer)
    with pytest.raises(ValueError, match="three finite numbers"):
        viewer.move_cursor(position)  # type: ignore[arg-type]  # wrong on purpose
    assert events == []
    assert viewer.scene.view.cursor == (0.0, 0.0, 0.0)


def test_a_cursor_position_from_numpy_is_stored_as_floats() -> None:
    viewer = Viewer()
    viewer.move_cursor(tuple(np.array([1, 2.5, -3], np.float32)))
    assert viewer.scene.view.cursor == (1.0, 2.5, -3.0)
    assert all(type(v) is float for v in viewer.scene.view.cursor)


# Subscribers never see a broken state


def test_every_event_finds_the_scene_consistent() -> None:
    viewer = Viewer()
    problems = _checked(viewer)
    a = viewer.add(_volume(), Category.VOLUMES, name="a")
    b = viewer.add(_volume(), Category.VOLUMES, name="b")
    c = viewer.add(_volume(), Category.VOLUMES, name="c")
    viewer.set(a.id, OptionName.VISIBLE, True)
    viewer.select(a.id)
    viewer.remove(a.id)
    viewer.set(c.id, OptionName.VISIBLE, False)
    viewer.select(b.id)
    viewer.remove(c.id)
    viewer.remove(b.id)
    assert problems == []
    assert viewer.scene.view.main_volume is None


# The volume shown from 4D data


def _series(count: int | None) -> Volume:
    shape = (2, 3, 4) if count is None else (2, 3, 4, count)
    return Volume(np.zeros(shape, np.float32), np.eye(4))


def _index(viewer: Viewer, item_id: int) -> object:
    return getattr(viewer.scene.item(item_id).options, OptionName.VOLUME_INDEX)


@pytest.mark.parametrize("category", [Category.VOLUMES, Category.OVERLAYS])
@pytest.mark.parametrize(("count", "expected"), [(5, 5), (1, 1), (None, 1)])
def test_an_image_volume_count_is_its_fourth_dimension_or_1(
    category: Category, count: int | None, expected: int
) -> None:
    item = Viewer().add(_series(count), category)
    assert item.volume_count == expected
    assert type(item.volume_count) is int


def test_a_tractogram_has_no_volume_count() -> None:
    item = Viewer().add(_tractogram([[0, 0, 0], [1, 1, 1]]), Category.TRACTOGRAMS)
    assert item.volume_count is None


@pytest.mark.parametrize("category", [Category.VOLUMES, Category.OVERLAYS])
@pytest.mark.parametrize("index", [0, 4, np.int64(3)])
def test_a_volume_index_within_a_4d_image_is_set_as_an_int(
    category: Category, index: int
) -> None:
    viewer = Viewer()
    item = viewer.add(_series(5), category)
    viewer.set(item.id, OptionName.VOLUME_INDEX, 1)
    viewer.set(item.id, OptionName.VOLUME_INDEX, index)
    assert _index(viewer, item.id) == index
    assert type(_index(viewer, item.id)) is int


@pytest.mark.parametrize("category", [Category.VOLUMES, Category.OVERLAYS])
@pytest.mark.parametrize(("count", "index"), [(5, 5), (5, 9), (None, 1), (1, 1)])
def test_a_volume_index_past_the_last_volume_changes_nothing_and_sends_nothing(
    category: Category, count: int | None, index: int
) -> None:
    viewer = Viewer()
    item = viewer.add(_series(count), category)
    viewer.set(item.id, OptionName.VOLUME_INDEX, 0)
    before = viewer.scene.item(item.id).options
    events = record_events(viewer)
    with pytest.raises(OptionError, match="Volume"):
        viewer.set(item.id, OptionName.VOLUME_INDEX, index)
    assert events == []
    assert viewer.scene.item(item.id).options == before


def test_a_3d_image_accepts_volume_index_0() -> None:
    viewer = Viewer()
    item = viewer.add(_series(None), Category.OVERLAYS)
    viewer.set(item.id, OptionName.VOLUME_INDEX, 0)
    assert _index(viewer, item.id) == 0


@pytest.mark.parametrize("category", [Category.VOLUMES, Category.OVERLAYS])
@pytest.mark.parametrize("index", [0, 1])
def test_a_volume_index_on_a_category_is_refused_and_changes_nothing(
    category: Category, index: int
) -> None:
    viewer = Viewer()
    ids = [viewer.add(_series(3), category).id for _ in range(2)]
    viewer.set(ids[0], OptionName.VOLUME_INDEX, 2)
    before = [i.options for i in viewer.scene.items]
    events = record_events(viewer)
    with pytest.raises(OptionError, match="Volume index"):
        viewer.set(category, OptionName.VOLUME_INDEX, index)
    assert events == []
    assert [i.options for i in viewer.scene.items] == before
    assert [_index(viewer, i) for i in ids] == [2, 0]


@pytest.mark.parametrize("category", [Category.VOLUMES, Category.OVERLAYS])
def test_other_options_on_a_category_of_4d_images_reach_every_item(
    category: Category,
) -> None:
    viewer = Viewer()
    ids = [viewer.add(_series(3), category).id for _ in range(2)]
    viewer.set(ids[1], OptionName.VOLUME_INDEX, 2)
    viewer.set(category, OptionName.OPACITY_2D, 0.5)
    options = [viewer.scene.item(i).options for i in ids]
    assert [o.opacity_2d for o in options] == [0.5, 0.5]
    assert [_index(viewer, i) for i in ids] == [0, 2]


# The information overlay


def test_toggle_overlay_flips_it_with_one_event_each_time() -> None:
    viewer = Viewer()
    assert viewer.scene.view.overlay_visible
    events = record_events(viewer)
    seen = []
    for _ in range(2):
        viewer.toggle_overlay()
        seen.append(viewer.scene.view.overlay_visible)
    assert seen == [False, True]
    assert events == [
        ViewStateChanged(ViewField.OVERLAY_VISIBLE, True, False),
        ViewStateChanged(ViewField.OVERLAY_VISIBLE, False, True),
    ]


def test_a_screenshot_without_pygfx_raises_with_the_install_hint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *args: None if name == "pygfx" else find_spec(name, *args),
    )
    viewer = Viewer()
    with pytest.raises(ViewerUnavailableError, match=re.escape(INSTALL_HINT)):
        viewer.screenshot(ViewKind.AXIAL, tmp_path / "a.png")
    assert not (tmp_path / "a.png").exists()
