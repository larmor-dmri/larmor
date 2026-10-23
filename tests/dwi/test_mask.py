import numpy as np
import pytest

from larmor.dwi import create_brain_mask


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
