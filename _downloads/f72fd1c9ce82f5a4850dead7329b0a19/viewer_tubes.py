"""Streamlines as tubes
====================

This example builds two crossing bundles of streamlines and saves the 3D view with
the streamlines drawn as lines, as tubes coloured by direction, and as tubes in one
colour.

The 3D view can draw a tractogram's streamlines as lit tubes instead of lines, which
shows better which streamline passes in front of which. The 2D views keep drawing
lines. Tubes take far more triangles than lines, so they are drawn only up to a
number of streamlines, the tube limit in the settings; above it the 3D view draws
lines.
"""

import numpy as np

from larmor.types import Tractogram
from larmor.viewer import (
    CameraPreset,
    Category,
    Colouring,
    OptionName,
    TractStyle,
    Viewer,
    ViewKind,
)

# %%
# Two crossing bundles
# --------------------
# One bundle is an arch from left to right, and the other a column from top to bottom
# through its middle. Each streamline is shifted at random by a few millimetres. The
# coordinates are world coordinates in mm.

rng = np.random.default_rng(0)
n, length = 300, 60
t = np.linspace(0, 1, length, dtype=np.float32)


def bundle(curve: np.ndarray, spread: float) -> np.ndarray:
    """n copies of a curve, each shifted at random by about spread mm."""
    shifts = rng.normal(scale=spread, size=(n, 1, 3)).astype(np.float32)
    return curve[None] + shifts


arch = np.stack([80 * t - 40, np.zeros_like(t), 25 * np.sin(np.pi * t)], -1)
column = np.stack([np.zeros_like(t), 4 * np.sin(2 * np.pi * t), 40 - 70 * t], -1)
streamlines = np.concatenate([bundle(arch, 2.5), bundle(column, 2.0)])
points = streamlines.reshape(-1, 3)
offsets = np.arange(0, len(points) + 1, length, dtype=np.int64)
tracts = Tractogram(points, offsets)

viewer = Viewer()
item = viewer.add(tracts, Category.TRACTOGRAMS, name="bundles")
viewer.set_camera(ViewKind.THREE_D, CameraPreset.CORONAL)
viewer.rotate_camera(30.0, 20.0)

# %%
# Lines, then tubes
# -----------------
# The style is a 3D option, lines by default or tubes. A tube's radius is in mm, 0.25
# by default; here it is 0.4, as the bundles are small.

viewer.screenshot(ViewKind.THREE_D, "viewer_tubes_lines.png", size=(600, 600))
viewer.set(item.id, OptionName.STYLE, TractStyle.TUBES)
viewer.set(item.id, OptionName.TUBE_RADIUS, 0.4)
viewer.screenshot(ViewKind.THREE_D, "viewer_tubes_direction.png", size=(600, 600))

# %%
# One colour
# ----------
# Tubes take the lines' colouring, opacity and visibility. In one colour, the
# lighting alone tells the streamlines apart.

viewer.set(item.id, OptionName.COLOURING, Colouring.SINGLE)
viewer.set(item.id, OptionName.COLOUR, "#ffb000")
viewer.screenshot(ViewKind.THREE_D, "viewer_tubes_single.png", size=(600, 600))
print(f"{tracts.num_streamlines} streamlines as tubes of 8 sides")
