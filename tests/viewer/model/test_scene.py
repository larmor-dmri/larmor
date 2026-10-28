import json
from pathlib import Path

import pytest

from larmor.viewer import Category, OptionName, ViewKind
from larmor.viewer.model.events import (
    Event,
    ItemAdded,
    ItemRemoved,
    OptionChanged,
    ViewStateChanged,
)
from larmor.viewer.model.options import ItemOptions
from larmor.viewer.model.scene import Scene
from larmor.viewer.model.view_state import ViewField


def _recorded(scene: Scene) -> list[Event]:
    events: list[Event] = []
    scene.subscribe(events.append)
    return events


def test_adding_an_item_sends_item_added() -> None:
    scene = Scene()
    events = _recorded(scene)
    item = scene.add(Category.VOLUMES, "a", None, object(), ItemOptions())
    assert events == [ItemAdded(item)]


def test_removing_an_item_sends_item_removed() -> None:
    scene = Scene()
    item = scene.add(Category.VOLUMES, "a", None, object(), ItemOptions())
    events = _recorded(scene)
    assert scene.remove(item.id) is item
    assert events == [ItemRemoved(item)]
    assert scene.items == []


def test_changing_an_option_sends_old_and_new() -> None:
    scene = Scene()
    item = scene.add(Category.VOLUMES, "a", None, object(), ItemOptions())
    events = _recorded(scene)
    scene.set_option(item.id, OptionName.OPACITY_2D, 0.25)
    assert events == [OptionChanged(item, OptionName.OPACITY_2D, 1.0, 0.25)]
    assert item.options.opacity_2d == 0.25


def test_changing_the_view_sends_old_and_new() -> None:
    scene = Scene()
    events = _recorded(scene)
    scene.set_view(ViewField.SELECTED_VIEW, ViewKind.THREE_D)
    assert events == [
        ViewStateChanged(ViewField.SELECTED_VIEW, ViewKind.AXIAL, ViewKind.THREE_D)
    ]
    assert scene.view.selected_view == ViewKind.THREE_D


def test_setting_the_current_value_sends_nothing() -> None:
    scene = Scene()
    item = scene.add(Category.VOLUMES, "a", None, object(), ItemOptions())
    events = _recorded(scene)
    scene.set_option(item.id, OptionName.OPACITY_2D, 1.0)
    scene.set_view(ViewField.SELECTED_VIEW, ViewKind.AXIAL)
    assert events == []


def test_an_unsubscribed_callback_gets_nothing() -> None:
    scene = Scene()
    events: list[Event] = []
    stop = scene.subscribe(events.append)
    stop()
    scene.set_view(ViewField.SELECTED_VIEW, ViewKind.THREE_D)
    assert events == []


def test_unsubscribing_during_an_event_still_reaches_the_others() -> None:
    scene = Scene()
    first: list[Event] = []
    stop_first = None

    def once(event: Event) -> None:
        first.append(event)
        assert stop_first is not None
        stop_first()

    stop_first = scene.subscribe(once)
    second = _recorded(scene)
    scene.set_view(ViewField.SELECTED_VIEW, ViewKind.THREE_D)
    scene.set_view(ViewField.SELECTED_VIEW, ViewKind.AXIAL)
    assert len(first) == 1
    assert len(second) == 2


def test_a_failing_subscriber_is_logged_and_the_others_still_get_the_event(
    caplog: pytest.LogCaptureFixture,
) -> None:
    scene = Scene()

    def fail(event: Event) -> None:
        raise RuntimeError("cannot draw")

    scene.subscribe(fail)
    later = _recorded(scene)
    scene.set_view(ViewField.SELECTED_VIEW, ViewKind.THREE_D)
    assert len(later) == 1
    assert "cannot draw" in caplog.text


def test_an_unknown_item_is_a_value_error() -> None:
    with pytest.raises(ValueError, match="There is no item 7"):
        Scene().item(7)


def test_items_keep_insertion_order_within_and_across_categories() -> None:
    scene = Scene()
    a = scene.add(Category.VOLUMES, "a", None, object(), ItemOptions())
    b = scene.add(Category.TRACTOGRAMS, "b", None, object(), ItemOptions())
    c = scene.add(Category.VOLUMES, "c", None, object(), ItemOptions())
    assert scene.items == [a, b, c]
    assert scene.in_category(Category.VOLUMES) == [a, c]
    assert scene.in_category(Category.OVERLAYS) == []


def test_to_dict_is_plain_data_without_the_items_data() -> None:
    scene = Scene()
    scene.add(Category.VOLUMES, "a", Path("/data/t1.nii"), object(), ItemOptions())
    scene.add(Category.TRACTOGRAMS, "b", None, object(), ItemOptions(visible=False))
    plain = scene.to_dict()
    # A session file holds text: compare the plain values, not the enums.
    json.dumps(plain)
    items = plain["items"]
    assert isinstance(items, list)
    assert items[0] == {
        "id": 1,
        "category": "volumes",
        "name": "a",
        "source": str(Path("/data/t1.nii")),
        "part": None,
        "registration": None,
        "options": {"visible": True, "opacity_2d": 1.0, "opacity_3d": 1.0},
    }
    assert items[1]["source"] is None
    assert items[1]["options"]["visible"] is False
    assert "data" not in items[0]
    view = plain["view"]
    assert isinstance(view, dict)
    assert view["selected_view"] == "axial"
    assert view["cameras"]["axial"] == {"zoom": 1.0, "pan": (0.0, 0.0)}
    assert set(view["cameras"]["three_d"]) == {"forward", "up", "zoom"}


def test_to_dict_lets_the_view_state_find_its_items_after_removals() -> None:
    # The view state names items by id; ids are not reused, so after removals they
    # have gaps, and a session file must keep them to find the main volume again.
    scene = Scene()
    for name in ("a", "b", "c"):
        scene.add(Category.VOLUMES, name, None, object(), ItemOptions())
    scene.remove(1)
    scene.remove(2)
    scene.set_view(ViewField.MAIN_VOLUME, 3)
    plain = scene.to_dict()
    items = plain["items"]
    view = plain["view"]
    assert isinstance(items, list) and isinstance(view, dict)
    main = [item for item in items if item["id"] == view["main_volume"]]
    assert [item["name"] for item in main] == ["c"]


def test_an_item_keeps_the_value_range_it_is_added_with() -> None:
    scene = Scene()
    image = scene.add(Category.VOLUMES, "a", None, object(), ItemOptions(), (-1.0, 2.0))
    other = scene.add(Category.TRACTOGRAMS, "b", None, object(), ItemOptions())
    assert image.value_range == (-1.0, 2.0)
    assert other.value_range is None


def test_renaming_an_item_sends_one_event_and_an_empty_name_is_refused() -> None:
    import numpy as np

    from larmor.types import Volume
    from larmor.viewer import Viewer
    from larmor.viewer.model.events import ItemRenamed

    viewer = Viewer()
    item = viewer.add(
        Volume(np.zeros((2, 2, 2), np.float32), np.eye(4)), Category.VOLUMES
    )
    events: list[object] = []
    viewer.scene.subscribe(events.append)
    viewer.rename(item.id, "  T1 brain  ")
    assert viewer.scene.item(item.id).name == "T1 brain"
    assert [type(e) for e in events] == [ItemRenamed]
    with pytest.raises(ValueError, match="name"):
        viewer.rename(item.id, "   ")
    assert viewer.scene.item(item.id).name == "T1 brain"
