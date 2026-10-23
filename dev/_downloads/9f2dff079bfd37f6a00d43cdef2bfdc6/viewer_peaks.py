"""Peaks in the viewer
===================

This example builds a synthetic head and an image of the peaks of fibre orientation
distributions (FODs) over it. Most of the brain has one peak per voxel, and a band
through the middle has two crossing peaks. The example saves screenshots of the axial
view with the line lengths set by amplitude and with one length for all.

The peaks of an FOD, as `lar peaks` writes them, are drawn in the 2D views as short
lines through each voxel of the slice, one per peak, pointing along the peak's
direction and coloured by it: red left to right, green front to back, blue up and
down. A peak's length follows its amplitude, or is the same for every peak.
"""

import numpy as np

from larmor.types import Volume
from larmor.viewer import Category, OptionName, Viewer, ViewKind

# %%
# A synthetic head
# ----------------
# The head is a 4 mm image of an ellipsoid, coarse so that each voxel's lines show,
# centred at the world origin. World coordinates are in mm, RAS+.

shape = (45, 54, 45)
affine = np.diag([4.0, 4.0, 4.0, 1.0])
affine[:3, 3] = -2 * np.array(shape)  # half the extent in mm: the centre at 0
i, j, k = np.indices(shape, dtype=np.float32)
x, y, z = (2 * i / shape[0] - 1, 2 * j / shape[1] - 1, 2 * k / shape[2] - 1)
r = np.sqrt((x / 0.8) ** 2 + (y / 0.9) ** 2 + (z / 0.85) ** 2)
brain = r < 0.8
head = Volume(np.where(r < 0.9, 600 - 300 * r, 0).astype(np.float32), affine)

# %%
# A peaks image
# -------------
# A peaks image holds, for each peak, the x, y and z of its direction (in world
# coordinates) scaled by its amplitude: 3 volumes per peak, NaN where a voxel has
# fewer peaks. Here every voxel of the brain has a front-to-back peak whose
# amplitude falls towards the edge; a band in the middle has a second, weaker peak
# running left to right, which crosses it.

peaks = np.full((*shape, 6), np.nan, np.float32)
peaks[brain, 0:3] = [0.0, 1.0, 0.0]
peaks[brain, 0:3] *= (1 - r[brain] / 0.8)[:, None] * 0.8 + 0.2
band = brain & (np.abs(y) < 0.25)
peaks[band, 3:6] = [0.6, 0.0, 0.0]
peaks_image = Volume(peaks, affine)

# %%
# The viewer and its items
# ------------------------
# Peaks are added like volumes; `load` reads a file written by `lar peaks`. The
# image must hold 3 volumes per peak.

viewer = Viewer()
# The screenshots hide the crosshair and the text, so they show only the data.
viewer.toggle_crosshair()
viewer.toggle_overlay()
viewer.add(head, Category.VOLUMES, name="head")
item = viewer.add(peaks_image, Category.PEAKS, name="peaks")

# %%
# Length by amplitude
# -------------------
# By default a peak's length follows its amplitude, and at scale 1 the largest peak
# of the image spans one voxel. The front-to-back peaks shorten towards the edge of
# the brain, and the crossing peaks in the band are shorter than the largest.

viewer.screenshot(ViewKind.AXIAL, "viewer_peaks_amplitude.png", size=(600, 600))

# %%
# The same length
# ---------------
# With the amplitude switch off, every peak is drawn the same length, which shows the
# directions alone. Here the lines are 0.6 of a voxel long, so that neighbours do not
# touch.

viewer.set(item.id, OptionName.AMPLITUDE, False)
viewer.set(item.id, OptionName.SCALE, 0.6)
viewer.screenshot(ViewKind.AXIAL, "viewer_peaks_directions.png", size=(600, 600))
print(f"{int(brain.sum())} voxels with peaks, {int(band.sum())} with two")
