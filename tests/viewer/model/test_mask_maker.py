from pathlib import Path

import numpy as np
import pytest

from larmor.dwi import (
    combine_masks,
    create_brain_mask,
    dilate_mask,
    erode_mask,
    fill_mask_holes,
    label_mask,
    largest_component,
    threshold_mask,
)
from larmor.io import load_volume
from larmor.types import Volume
from larmor.viewer import Category, Viewer
from larmor.viewer.model.mask_maker import (
    Combination,
    MaskMaker,
    MaskRecipe,
    MaskSource,
    b0_of,
)

AFFINE = np.array([[2.0, 0, 0, -20], [0, 2.0, 0, -22], [0, 0, 2.5, -10], [0, 0, 0, 1]])


def _ball(shape: tuple[int, int, int] = (24, 24, 24)) -> np.ndarray:
    """A bright noisy ball with a dark hole, in a noisy background."""
    rng = np.random.default_rng(3)
    x, y, z = np.indices(shape)
    image = 50 * rng.random(shape)
    image[(x - 12) ** 2 + (y - 12) ** 2 + (z - 12) ** 2 <= 64] += 1000
    image[11:13, 11:13, 11:13] = 0
    return image


@pytest.fixture
def maker() -> MaskMaker:
    viewer = Viewer()
    dwi = np.stack([_ball(), 0.5 * _ball(), _ball()], axis=-1)
    viewer.add(Volume(dwi.astype(np.float32), AFFINE), Category.VOLUMES, name="dwi")
    labels = np.zeros((24, 24, 24), np.int16)
    labels[4:10, 4:10, 4:10] = 3
    labels[14:20, 14:20, 14:20] = 8
    labels[0, 0, 0] = 8
    viewer.add(Volume(labels, AFFINE), Category.OVERLAYS, name="parc")
    return MaskMaker(viewer)


def _id(maker: MaskMaker, name: str) -> int:
    return next(i.id for i in maker.viewer.scene.items if i.name == name)


def _labels(maker: MaskMaker) -> np.ndarray:
    data = maker.viewer.scene.item(_id(maker, "parc")).data
    assert isinstance(data, Volume)
    return np.asarray(data.data)


def test_each_source_gives_the_library_s_voxels(
    maker: MaskMaker, tmp_path: Path
) -> None:
    dwi = _id(maker, "dwi")
    bvals = tmp_path / "bvals"
    bvals.write_text("0 1000 5\n")
    b0 = (_ball() + _ball()) / 2
    np.testing.assert_allclose(b0_of(maker._volume(dwi), bvals), b0, rtol=1e-6)
    brain = maker.make(MaskRecipe(MaskSource.DWI, dwi, bvals=str(bvals)))
    np.testing.assert_array_equal(brain, create_brain_mask(b0))
    threshold = MaskRecipe(MaskSource.THRESHOLD, dwi, lower=100.0, upper=900.0)
    np.testing.assert_array_equal(
        maker.make(threshold), threshold_mask(_ball(), 100.0, 900.0)
    )
    parc = _id(maker, "parc")
    labels = maker.make(MaskRecipe(MaskSource.LABELS, parc, labels=(8,)))
    np.testing.assert_array_equal(labels, label_mask(_labels(maker), [8]))


def test_the_clean_up_and_combination_give_the_library_s_voxels(
    maker: MaskMaker,
) -> None:
    parc = _id(maker, "parc")
    recipe = MaskRecipe(
        MaskSource.LABELS,
        parc,
        labels=(3, 8),
        dilate=2,
        erode=1,
        fill_holes=True,
        largest=True,
        combine_with=parc,
        combination=Combination.SUBTRACT,
    )
    expected = label_mask(_labels(maker), [3, 8])
    expected = erode_mask(dilate_mask(expected, 2), 1)
    expected = largest_component(fill_mask_holes(expected))
    expected = combine_masks(expected, _labels(maker) != 0, "subtract")
    np.testing.assert_array_equal(maker.make(recipe), expected)


def test_a_new_mask_saves_and_reads_back_the_same(
    maker: MaskMaker, tmp_path: Path
) -> None:
    parc = _id(maker, "parc")
    mask_id = maker.create(MaskRecipe(MaskSource.LABELS, parc, labels=(3,)), "M")
    item = maker.viewer.scene.item(mask_id)
    assert (item.name, item.category) == ("M", Category.OVERLAYS)
    data = item.data
    assert isinstance(data, Volume)
    assert data.data.dtype == np.uint8
    for name in ("m.nii.gz", "m.mif"):
        maker.viewer.save_item(mask_id, tmp_path / name)
        read = load_volume(tmp_path / name)
        np.testing.assert_array_equal(read.data, data.data)
        np.testing.assert_allclose(read.affine, AFFINE, atol=1e-5)
    # It can be a mask region at once.
    maker.viewer.add_mask(mask_id)


def test_only_masks_on_the_same_grid_can_be_combined(maker: MaskMaker) -> None:
    other = maker.viewer.add(
        Volume(np.ones((24, 24, 24), np.uint8), np.eye(4)), Category.OVERLAYS
    )
    dwi = _id(maker, "dwi")
    assert [i.name for i in maker.masks_like(dwi)] == ["parc"]
    with pytest.raises(ValueError, match="another grid"):
        maker.make(MaskRecipe(MaskSource.THRESHOLD, dwi, combine_with=other.id))
