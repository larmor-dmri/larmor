import numpy as np

from larmor.viewer.render.colour_space import physical_colours


def test_srgb_values_become_linear_light_and_alpha_is_kept() -> None:
    srgb = np.array([[0.0, 0.02, 0.5, 0.3], [1.0, 1.5, -0.1, 1.0]], np.float32)
    linear = physical_colours(srgb)
    expected = [0.0, 0.02 / 12.92, ((0.5 + 0.055) / 1.055) ** 2.4, 0.3]
    assert linear.dtype == np.float32
    assert np.allclose(linear[0], expected, atol=1e-6)
    assert np.allclose(linear[1], [1.0, 1.0, 0.0, 1.0])  # clipped to 0..1
    assert srgb[0, 2] == np.float32(0.5)  # the input is left as it was
