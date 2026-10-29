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


def _blob(radius: int) -> np.ndarray:
    """A bright ball of the given radius in a 24^3 noisy background."""
    rng = np.random.default_rng(5)
    x, y, z = np.mgrid[:24, :24, :24]
    image = 50 * rng.random((24, 24, 24))
    image[(x - 12) ** 2 + (y - 12) ** 2 + (z - 12) ** 2 <= radius**2] += 1000
    return image


def test_object_smaller_than_the_erosion_element_raises() -> None:
    # A ball of radius 2 on a zero background: the 7-voxel erosion removes it.
    rng = np.random.default_rng(5)
    x, y, z = np.mgrid[:24, :24, :24]
    image = np.zeros((24, 24, 24))
    ball = (x - 12) ** 2 + (y - 12) ** 2 + (z - 12) ** 2 <= 4
    image[ball] = 1000 + 100 * rng.random(int(ball.sum()))
    with pytest.raises(ValueError, match="empty after erosion"):
        create_brain_mask(image)


def test_threshold_factor_around_one_gives_the_same_mask() -> None:
    image = _blob(8)
    masks = [create_brain_mask(image, threshold=t) for t in (0.999, 1.0, 1.001)]
    np.testing.assert_array_equal(masks[1], masks[0])
    np.testing.assert_array_equal(masks[1], masks[2])


def _cube(size: int = 9) -> np.ndarray:
    """A 3^3 cube of True in the middle of a size^3 grid."""
    mask = np.zeros((size, size, size), dtype=bool)
    c = size // 2
    mask[c - 1 : c + 2, c - 1 : c + 2, c - 1 : c + 2] = True
    return mask


def test_threshold_keeps_values_between_the_bounds_inclusive() -> None:
    image = np.array([[[0.1, 0.2, 0.5, 0.9, np.nan]]])
    np.testing.assert_array_equal(
        threshold_mask(image, 0.2), [[[False, True, True, True, False]]]
    )
    np.testing.assert_array_equal(
        threshold_mask(image, 0.2, 0.5), [[[False, True, True, False, False]]]
    )


def test_label_mask_keeps_the_rounded_labels_given() -> None:
    labels = np.array([[[0.0, 1.0, 2.0000001, 3.0, 2.0]]])
    np.testing.assert_array_equal(
        label_mask(labels, [2, 3]), [[[False, False, True, True, True]]]
    )


def test_dilating_and_eroding_by_one_voxel_moves_the_faces() -> None:
    cube = _cube()
    grown = dilate_mask(cube, 1)
    # The cube plus one voxel on each face, not the edges or corners.
    assert grown.sum() == 27 + 6 * 9
    np.testing.assert_array_equal(erode_mask(grown, 1), cube)
    assert erode_mask(cube, 1).sum() == 1
    np.testing.assert_array_equal(dilate_mask(cube, 0), cube)
    with pytest.raises(ValueError, match="0 or more"):
        erode_mask(cube, -1)


def test_fill_holes_fills_an_enclosed_cavity_only() -> None:
    shell = dilate_mask(_cube(), 1)
    shell[4, 4, 4] = False
    filled = fill_mask_holes(shell)
    assert filled[4, 4, 4]
    assert filled.sum() == shell.sum() + 1


def test_largest_component_keeps_the_biggest_piece() -> None:
    mask = _cube(11)
    mask[0, 0, 0] = True
    np.testing.assert_array_equal(largest_component(mask), _cube(11))
    assert not largest_component(np.zeros((3, 3, 3), dtype=bool)).any()


def test_combine_masks() -> None:
    a = np.array([True, True, False, False])
    b = np.array([True, False, True, False])
    np.testing.assert_array_equal(combine_masks(a, b, "union"), [1, 1, 1, 0])
    np.testing.assert_array_equal(combine_masks(a, b, "intersect"), [1, 0, 0, 0])
    np.testing.assert_array_equal(combine_masks(a, b, "subtract"), [0, 1, 0, 0])
    with pytest.raises(ValueError, match="same shape"):
        combine_masks(a, b[:2], "union")
