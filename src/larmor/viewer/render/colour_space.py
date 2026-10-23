"""Converts sRGB colours to the linear values of a mesh's vertex colours."""

import numpy as np

# Constants of the sRGB transfer function, its linear segment and its curve.
_KNEE = np.float32(0.04045)
_SLOPE = np.float32(12.92)
_OFFSET = np.float32(0.055)
_SCALE = np.float32(1.055)
_GAMMA = np.float32(2.4)


def physical_colours(colours: np.ndarray) -> np.ndarray:
    """Turns sRGB colours (0 to 1) into the linear values of a mesh's vertex colours.

    Every mesh with vertex colours needs this: pygfx reads them as linear light,
    so sRGB values come out lighter and paler. Lines, uniform colours and
    textures take sRGB as they are.

    Args:
        colours: (N, 3) or (N, 4) sRGB values; an alpha column is kept as it is.

    Returns:
        A new float32 array of the same shape.
    """
    out = np.array(colours, dtype=np.float32, copy=True)
    rgb = out[..., :3]
    np.clip(rgb, 0.0, 1.0, out=rgb)
    low = rgb <= _KNEE
    low_values = rgb[low] / _SLOPE
    rgb += _OFFSET
    rgb /= _SCALE
    np.power(rgb, _GAMMA, out=rgb)
    rgb[low] = low_values
    return out
