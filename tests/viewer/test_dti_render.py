import numpy as np
import pytest

from larmor.types import Volume
from larmor.viewer import Category, Colormap, OptionName, Viewer, ViewKind
from larmor.viewer.render.view import ViewRenderer
from larmor.viewer.render.volume_adapter import (
    COLORMAP_MAPS,
    VolumeAdapter,
    dti_colours,
)

NAN = float("nan")


@pytest.fixture
def view_size() -> int:
    return 32


def _vectors(rows: list[list[float]], slope: float = 1.0, inter: float = 0.0) -> Volume:
    # One voxel per row, along x.
    data = np.array(rows, np.float32).reshape(len(rows), 1, 1, 3)
    return Volume(data, np.eye(4), slope=slope, intercept=inter)


def _rgb(colours: np.ndarray) -> np.ndarray:
    # The texture is (z, y, x, RGBA); back to one row per voxel along x.
    return colours[0, 0, :, :3]


def test_dti_colours_are_absolute_components_in_voxel_order() -> None:
    data = np.arange(2 * 3 * 4 * 3, dtype=np.float32).reshape(2, 3, 4, 3) / -100
    colours = dti_colours(Volume(data, np.eye(4)), None)
    assert colours.shape == (4, 3, 2, 4)
    assert colours.dtype == np.float32
    np.testing.assert_array_equal(colours[..., :3], np.abs(data).transpose(2, 1, 0, 3))
    assert np.all(colours[..., 3] == 1)


def test_dti_colours_count_nan_as_0_and_clip_to_1() -> None:
    colours = dti_colours(_vectors([[NAN, -0.5, 2.0], [-3.0, 0.25, NAN]]), None)
    np.testing.assert_array_equal(_rgb(colours), [[0, 0.5, 1], [1, 0.25, 0]])


def test_dti_colours_are_in_scaled_units() -> None:
    colours = dti_colours(_vectors([[0.1, -0.2, 0.3]], slope=2.0, inter=-0.1), None)
    np.testing.assert_allclose(_rgb(colours), [[0.1, 0.5, 0.5]], atol=1e-7)


def test_the_modulating_image_scales_the_colours_clipped_to_0_and_1() -> None:
    vectors = _vectors([[1.0, 0.5, 0.0]] * 4)
    stored = np.array([0.25, 3.0, -1.0, NAN], np.float32).reshape(4, 1, 1)
    weights = Volume(stored, np.eye(4), slope=2.0)
    colours = dti_colours(vectors, weights)
    expected = np.array([[0.5], [1.0], [0.0], [0.0]]) * [1.0, 0.5, 0.0]
    np.testing.assert_allclose(_rgb(colours), expected, atol=1e-7)


def test_a_4d_modulating_image_of_one_volume_weighs_as_its_volume() -> None:
    vectors = _vectors([[1.0, 1.0, 1.0]] * 2)
    weights = Volume(np.array([0.5, 1.0], np.float32).reshape(2, 1, 1, 1), np.eye(4))
    colours = dti_colours(vectors, weights)
    np.testing.assert_allclose(_rgb(colours), [[0.5] * 3, [1.0] * 3])


def test_an_overlay_is_transparent_where_it_has_no_colour() -> None:
    colours = dti_colours(_vectors([[0.0, 0.0, 0.0], [0.0, 0.2, 0.0]]), None, True)
    np.testing.assert_array_equal(colours[0, 0, :, 3], [0, 1])


# On the GPU


def test_dti_colours_replace_the_colormap_until_turned_off(
    views: dict[ViewKind, ViewRenderer],
) -> None:
    viewer = Viewer()
    data = np.zeros((2, 3, 4, 3), np.float32)
    data[..., 0] = -1.0  # along x: red
    item = viewer.add(Volume(data, np.eye(4)), Category.VOLUMES)
    viewer.set(item.id, OptionName.COLORMAP, Colormap.HOT)
    adapter = VolumeAdapter(item, views, viewer, 0.0)
    viewer.scene.subscribe(adapter.on_event)
    three_d = adapter.three_d
    assert three_d is not None and three_d.rendering is not None
    slices = [*adapter.slices.values(), *three_d.slices.values()]
    assert three_d.rendering.visible

    viewer.set(item.id, OptionName.DTI, True)
    for obj in slices:
        texture = np.asarray(obj.geometry.grid.data)
        assert texture.shape == (4, 3, 2, 4)
        np.testing.assert_array_equal(texture[0, 0, 0], [1, 0, 0, 1])
        assert obj.material.map is None
        assert tuple(obj.material.clim) == (0.0, 1.0)
    assert not three_d.rendering.visible

    viewer.set(item.id, OptionName.DTI, False)
    for obj in slices:
        assert np.asarray(obj.geometry.grid.data).ndim == 3
        assert obj.material.map is COLORMAP_MAPS[Colormap.HOT]
    assert three_d.rendering.visible
    adapter.close()


def test_choosing_a_modulating_image_while_dti_is_on_recolours_the_slices(
    views: dict[ViewKind, ViewRenderer],
) -> None:
    viewer = Viewer()
    data = np.zeros((2, 3, 4, 3), np.float32)
    data[..., 0] = 1.0  # along x: red
    item = viewer.add(Volume(data, np.eye(4)), Category.VOLUMES)
    fa = viewer.add(
        Volume(np.full((2, 3, 4), 0.5, np.float32), np.eye(4)), Category.OVERLAYS
    )
    adapter = VolumeAdapter(item, views, viewer, 0.0)
    viewer.scene.subscribe(adapter.on_event)
    viewer.set(item.id, OptionName.DTI, True)

    def red() -> float:
        (obj, *_) = adapter.slices.values()
        return float(np.asarray(obj.geometry.grid.data)[0, 0, 0, 0])

    assert red() == 1.0
    viewer.set(item.id, OptionName.MODULATE, fa.id)
    assert red() == 0.5
    viewer.set(item.id, OptionName.MODULATE, None)
    assert red() == 1.0
    adapter.close()
