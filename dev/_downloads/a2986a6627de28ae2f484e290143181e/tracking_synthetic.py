"""Tracking streamlines through crossing bundles
=============================================

This example builds an image of fibre orientation distributions (FODs) of two
straight bundles that cross at right angles, seeds one point in each of their voxels,
tracks streamlines from the seeds, prints how many streamlines it kept and how long
they are, and draws them as tubes in 3D.

Tractography follows the fibre directions of the FODs from seed points, step by
step, into streamlines. At each step the deterministic algorithm moves along the
FOD's peak closest to the current direction; it stops where the FOD becomes too
small or the streamline turns too sharply, and tracks both ways from the seed.
"""

import numpy as np

from larmor.sd import fit_sh
from larmor.tractography import Tracker, seeds_per_voxel, streamline_lengths
from larmor.types import Volume
from larmor.viewer import (
    CameraPreset,
    Category,
    OptionName,
    TractStyle,
    Viewer,
    ViewKind,
)

# %%
# An FOD image of two bundles
# ---------------------------
# The image is a 2 mm grid of 30 x 30 x 6 voxels. One bundle runs left to right
# through the middle rows, the other front to back through the middle columns; where
# they cross, the FOD has both lobes. Each FOD is fitted with spherical harmonics
# (SH) up to degree 8 from its amplitudes on 600 directions.

n = 600
z = 1 - (2 * np.arange(n) + 1) / n
azimuth = np.pi * (3 - np.sqrt(5)) * np.arange(n)
radius = np.sqrt(1 - z**2)
dirs = np.stack([radius * np.cos(azimuth), radius * np.sin(azimuth), z], axis=1)


def fod(*fibres: np.ndarray) -> np.ndarray:
    """The SH coefficients of narrow lobes along the fibres."""
    cosines = dirs @ np.array(fibres).T
    return fit_sh(np.exp(-8 * (1 - cosines**2)).sum(axis=1), dirs, 8)


x, y = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
shape = (30, 30, 6)
affine = np.diag([2.0, 2.0, 2.0, 1.0])
affine[:3, 3] = -(np.array(shape) - 1)  # the centre at the world origin
across = np.zeros(shape, bool)
across[:, 12:18] = True  # left to right
along = np.zeros(shape, bool)
along[12:18, :] = True  # front to back
fods = np.zeros((*shape, 45), np.float32)
fods[across] = fod(x)
fods[along] = fod(y)
fods[across & along] = fod(x, y)
bundles = Volume((across | along).astype(np.uint8), affine)

# %%
# Tracking
# --------
# There is one random seed in each voxel of the bundles. The mask keeps the
# streamlines inside the bundles, and with no output file they stay in memory.

seeds = seeds_per_voxel(bundles, 1, rng=0)
tracker = Tracker("det")
run = tracker.track(Volume(fods, affine), None, seeds, mask=bundles, seed=0)
assert run.tractogram is not None
lengths = streamline_lengths(run.tractogram)
print(f"{run.num_streamlines} streamlines from {run.num_seeds} seeds")
print(f"lengths from {lengths.min():.0f} to {lengths.max():.0f} mm")

# %%
# The streamlines in 3D
# ---------------------
# The streamlines are coloured by direction: red left to right, green front to back.
# The two bundles pass through each other where they cross.

viewer = Viewer()
tracts = viewer.add(run.tractogram, Category.TRACTOGRAMS, name="bundles")
viewer.set(tracts.id, OptionName.STYLE, TractStyle.TUBES)
viewer.set_camera(ViewKind.THREE_D, CameraPreset.AXIAL)
viewer.rotate_camera(20.0, -35.0)
viewer.screenshot(
    ViewKind.THREE_D, "tracking_synthetic.png", size=(600, 450), overlay=False
)
