"""Finds the labels of a parcellation and gives their names and colours."""

import colorsys
from collections.abc import Mapping

import numpy as np

from larmor._log import LarmorLogger
from larmor.io import read_lookup_table
from larmor.types import Label, Volume

# The most distinct values an overlay can have to be shown as labels. WebGPU
# guarantees 1D textures of this size, which the label table in the volume
# adapter uses; a larger table fails on some GPUs.
MAX_LABELS = 8192


def labels_in(volume: Volume, index: int) -> tuple[int, ...]:
    """Returns the sorted labels other than 0 in one volume of an image, scaled.

    Empty for an image with more than MAX_LABELS distinct values.

    Args:
        volume: The overlay.
        index: Which volume of the image.

    Returns:
        The labels.
    """
    data = volume.data[..., index] if volume.data.ndim == 4 else volume.data
    values = np.unique(data)
    values = values[np.isfinite(values)]
    if len(values) > MAX_LABELS:
        return ()
    scaled = np.rint(values * volume.slope + volume.intercept).astype(np.int64)
    return tuple(int(v) for v in np.unique(scaled) if v != 0)


def label_rgb(label: int, table: Mapping[int, Label]) -> tuple[float, float, float]:
    """Returns a label's RGB colour, 0 to 1, from the table or the built-in colours."""
    if label in table:
        r, g, b = (c / 255 for c in table[label].rgb)
        return (r, g, b)
    # Hues a golden-ratio turn apart: neighbouring labels look different.
    r, g, b = colorsys.hsv_to_rgb((label * 0.618033988749895) % 1.0, 0.7, 0.95)
    return (r, g, b)


def label_name(label: int, table: Mapping[int, Label]) -> str:
    """Returns a label's name from the lookup table, else "Label N"."""
    return table[label].name if label in table else f"Label {label}"


def lookup_table(path: str) -> dict[int, Label]:
    """Reads a lookup-table file; returns no labels for "" or an unreadable file."""
    if not path:
        return {}
    try:
        return read_lookup_table(path)
    except ValueError as exc:  # the file changed since it was chosen
        LarmorLogger.warning("%s; using the built-in colours.", exc)
        return {}
