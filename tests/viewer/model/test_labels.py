import numpy as np
import pytest

from larmor.types import Volume
from larmor.viewer import Category, ItemId, OptionError, OptionName, Viewer
from larmor.viewer.model.events import Event, OptionChanged
from larmor.viewer.model.labels import MAX_LABELS
from larmor.viewer.model.options import OverlayOptions, validate


def _series() -> Volume:
    # Volume 0 holds labels 2, 5 and 7; volume 1 only 5 and 9.
    data = np.zeros((3, 3, 3, 2), np.int16)
    data[0, 0, 0, 0], data[1, 1, 1, 0], data[2, 2, 2, 0] = 2, 5, 7
    data[1, 1, 1, 1], data[2, 2, 2, 1] = 5, 9
    return Volume(data, np.eye(4))


def _overlay(volume: Volume) -> tuple[Viewer, ItemId]:
    viewer = Viewer()
    return viewer, viewer.add(volume, Category.OVERLAYS).id


def _surface_labels(viewer: Viewer, item_id: ItemId) -> tuple[int, ...]:
    o = viewer.scene.item(item_id).options
    assert isinstance(o, OverlayOptions)
    return o.surface_labels


@pytest.mark.parametrize(
    ("value", "stored"),
    [([7, 2, 7, 5], (2, 5, 7)), ({3}, (3,)), ((np.int16(4), 1), (1, 4)), ([], ())],
)
def test_surface_labels_are_stored_sorted_without_repeats(
    value: object, stored: tuple[int, ...]
) -> None:
    result = validate(OverlayOptions, OptionName.SURFACE_LABELS, value)
    assert isinstance(result, tuple)
    assert result == stored
    assert all(type(v) is int for v in result)


@pytest.mark.parametrize("value", [[1.5], [True], ["3"], "12", 3, None])
def test_surface_labels_refuse_anything_but_whole_numbers(value: object) -> None:
    with pytest.raises(OptionError, match="whole numbers"):
        validate(OverlayOptions, OptionName.SURFACE_LABELS, value)


def test_labels_are_the_non_zero_values_of_the_shown_volume() -> None:
    viewer, item_id = _overlay(_series())
    assert viewer.labels(item_id) == (2, 5, 7)
    viewer.set(item_id, OptionName.VOLUME_INDEX, 1)
    assert viewer.labels(item_id) == (5, 9)


def test_labels_are_in_scaled_units_rounded() -> None:
    data = np.array([[[0.0, 1.0, 2.0]]], np.float32)
    # Stored 0, 1 and 2 are -2.5, 0.5 and 3.5: rounded half to even, -2, 0 and 4.
    viewer, item_id = _overlay(Volume(data, np.eye(4), slope=3.0, intercept=-2.5))
    assert viewer.labels(item_id) == (-2, 4)


def test_an_image_of_too_many_values_has_no_labels() -> None:
    data = np.arange(MAX_LABELS + 1, dtype=np.float32).reshape(1, 1, -1)
    viewer, item_id = _overlay(Volume(data, np.eye(4)))
    assert viewer.labels(item_id) == ()


def test_only_an_overlay_has_labels() -> None:
    viewer = Viewer()
    item = viewer.add(_series(), Category.VOLUMES)
    with pytest.raises(ValueError, match="not an overlay"):
        viewer.labels(item.id)


def test_set_keeps_the_chosen_labels_sorted() -> None:
    viewer, item_id = _overlay(_series())
    viewer.set(item_id, OptionName.SURFACE_LABELS, [7, 2])
    assert _surface_labels(viewer, item_id) == (2, 7)


def test_set_refuses_a_label_the_image_lacks() -> None:
    viewer, item_id = _overlay(_series())
    with pytest.raises(OptionError, match="no label 9"):
        viewer.set(item_id, OptionName.SURFACE_LABELS, [2, 9])
    assert _surface_labels(viewer, item_id) == ()


def test_surface_labels_cannot_be_set_on_a_category() -> None:
    viewer, _ = _overlay(_series())
    with pytest.raises(OptionError, match="per item"):
        viewer.set(Category.OVERLAYS, OptionName.SURFACE_LABELS, [5])


def test_a_new_volume_index_drops_its_missing_labels_before_it_changes() -> None:
    viewer, item_id = _overlay(_series())
    viewer.set(item_id, OptionName.SURFACE_LABELS, [2, 5])
    changes: list[tuple[OptionName, object, int]] = []

    def record(event: Event) -> None:
        if isinstance(event, OptionChanged):
            # The index the item had when the change was sent.
            o = viewer.scene.item(item_id).options
            assert isinstance(o, OverlayOptions)
            changes.append((event.name, event.new, o.volume_index))

    viewer.scene.subscribe(record)
    viewer.set(item_id, OptionName.VOLUME_INDEX, 1)
    assert changes == [
        (OptionName.SURFACE_LABELS, (5,), 0),
        (OptionName.VOLUME_INDEX, 1, 1),
    ]


def test_a_new_volume_index_holding_every_chosen_label_keeps_them() -> None:
    viewer, item_id = _overlay(_series())
    viewer.set(item_id, OptionName.SURFACE_LABELS, [5])
    names: list[OptionName] = []
    viewer.scene.subscribe(
        lambda e: names.append(e.name) if isinstance(e, OptionChanged) else None
    )
    viewer.set(item_id, OptionName.VOLUME_INDEX, 1)
    assert names == [OptionName.VOLUME_INDEX]
    assert _surface_labels(viewer, item_id) == (5,)


def test_surface_labels_cannot_be_every_new_overlays_default() -> None:
    viewer, _ = _overlay(_series())
    with pytest.raises(OptionError, match="per item"):
        viewer.set_default(Category.OVERLAYS, OptionName.SURFACE_LABELS, [99])
