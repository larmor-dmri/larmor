import numpy as np
import pytest

from larmor.types import Tractogram, Volume
from larmor.viewer import Category, Item, OptionError, OptionName, Viewer
from larmor.viewer.model.events import Event, ItemRemoved, OptionChanged

SHAPE = (2, 3, 4)


def _image(
    volumes: int | None = None,
    shape: tuple[int, ...] = SHAPE,
    affine: np.ndarray | None = None,
) -> Volume:
    full = shape if volumes is None else (*shape, volumes)
    return Volume(np.zeros(full, np.float32), np.eye(4) if affine is None else affine)


def _option(viewer: Viewer, item_id: int, name: OptionName) -> object:
    return getattr(viewer.scene.item(item_id).options, name)


@pytest.fixture
def dti(viewer: Viewer) -> Item:
    return viewer.add(_image(3), Category.VOLUMES, name="v1")


@pytest.mark.parametrize("volumes", [None, 1, 2, 4])
def test_dti_colours_need_a_4d_image_with_3_volumes(
    volumes: int | None, viewer: Viewer
) -> None:
    item = viewer.add(_image(volumes), Category.VOLUMES)
    with pytest.raises(OptionError, match="3 volumes"):
        viewer.set(item.id, OptionName.DTI, True)
    assert _option(viewer, item.id, OptionName.DTI) is False


@pytest.mark.parametrize("category", [Category.VOLUMES, Category.OVERLAYS])
def test_dti_colours_can_be_turned_on_and_off_for_3_volumes(
    category: Category, viewer: Viewer
) -> None:
    item = viewer.add(_image(3), category)
    viewer.set(item.id, OptionName.DTI, True)
    assert _option(viewer, item.id, OptionName.DTI) is True
    viewer.set(item.id, OptionName.DTI, False)
    assert _option(viewer, item.id, OptionName.DTI) is False


def test_item_choices_are_the_other_3d_images_on_the_same_grid(
    viewer: Viewer, dti: Item
) -> None:
    shifted = np.eye(4)
    shifted[0, 3] = 1.0
    fa = viewer.add(_image(), Category.OVERLAYS, name="fa")
    md = viewer.add(_image(), Category.VOLUMES, name="md")
    viewer.add(_image(shape=(2, 3, 5)), Category.OVERLAYS, name="other shape")
    viewer.add(_image(affine=shifted), Category.OVERLAYS, name="other affine")
    viewer.add(_image(3), Category.OVERLAYS, name="4d")
    viewer.add(_image(3), Category.PEAKS, name="peaks")
    tract = Tractogram(np.zeros((2, 3), np.float32), np.array([0, 2]))
    viewer.add(tract, Category.TRACTOGRAMS, name="tract")
    choices = viewer.item_choices(dti.id, OptionName.MODULATE)
    assert sorted(i.id for i in choices) == sorted([fa.id, md.id])


def test_a_3d_image_is_not_its_own_choice(viewer: Viewer) -> None:
    a = viewer.add(_image(), Category.OVERLAYS, name="a")
    b = viewer.add(_image(), Category.OVERLAYS, name="b")
    assert [i.id for i in viewer.item_choices(a.id, OptionName.MODULATE)] == [b.id]


def test_item_choices_refuse_an_option_that_names_no_item(
    viewer: Viewer, dti: Item
) -> None:
    with pytest.raises(OptionError):
        viewer.item_choices(dti.id, OptionName.COLORMAP)


def test_a_modulating_image_on_the_grid_is_set(viewer: Viewer, dti: Item) -> None:
    fa = viewer.add(_image(), Category.OVERLAYS, name="fa")
    viewer.set(dti.id, OptionName.MODULATE, fa.id)
    assert _option(viewer, dti.id, OptionName.MODULATE) == fa.id
    viewer.set(dti.id, OptionName.MODULATE, None)
    assert _option(viewer, dti.id, OptionName.MODULATE) is None


@pytest.mark.parametrize("bad", ["missing", "itself", "other grid", "4d"])
def test_a_modulating_item_it_cannot_name_is_refused(
    bad: str, viewer: Viewer, dti: Item
) -> None:
    other = {
        "other grid": lambda: viewer.add(_image(shape=(2, 3, 5)), Category.OVERLAYS),
        "4d": lambda: viewer.add(_image(3), Category.OVERLAYS),
    }
    target = {"missing": 99, "itself": dti.id}.get(bad)
    if target is None:
        target = other[bad]().id
    with pytest.raises(OptionError, match="Modulate by"):
        viewer.set(dti.id, OptionName.MODULATE, target)
    assert _option(viewer, dti.id, OptionName.MODULATE) is None


def test_removing_the_modulating_item_resets_modulate_before_it_goes(
    viewer: Viewer, dti: Item
) -> None:
    fa = viewer.add(_image(), Category.OVERLAYS, name="fa")
    viewer.set(dti.id, OptionName.MODULATE, fa.id)
    events: list[Event] = []
    named_at_removal: list[object] = []

    def record(event: Event) -> None:
        events.append(event)
        if isinstance(event, ItemRemoved):
            named_at_removal.append(_option(viewer, dti.id, OptionName.MODULATE))

    viewer.scene.subscribe(record)
    viewer.remove(fa.id)
    changes = [
        (e.item.id, e.name, e.old, e.new)
        for e in events
        if isinstance(e, OptionChanged)
    ]
    assert changes == [(dti.id, OptionName.MODULATE, fa.id, None)]
    assert isinstance(events[-1], ItemRemoved)
    assert named_at_removal == [None]


def test_removing_another_item_keeps_modulate(viewer: Viewer, dti: Item) -> None:
    fa = viewer.add(_image(), Category.OVERLAYS, name="fa")
    other = viewer.add(_image(), Category.OVERLAYS, name="md")
    viewer.set(dti.id, OptionName.MODULATE, fa.id)
    viewer.remove(other.id)
    assert _option(viewer, dti.id, OptionName.MODULATE) == fa.id


@pytest.mark.parametrize(
    ("name", "value"), [(OptionName.DTI, True), (OptionName.MODULATE, 5)]
)
def test_a_per_item_option_cannot_be_every_new_items_default(
    name: OptionName, value: object
) -> None:
    viewer = Viewer()
    with pytest.raises(OptionError, match="per item"):
        viewer.set_default(Category.VOLUMES, name, value)
    item = viewer.add(
        Volume(np.zeros((2, 2, 2), np.float32), np.eye(4)), Category.VOLUMES
    )
    assert getattr(item.options, name) in (False, None)
