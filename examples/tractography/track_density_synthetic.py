"""Track density images
====================

This example builds a synthetic bundle, maps its track density on a 2 mm grid and on
a 0.5 mm one, weights the streamlines, maps their endpoints, and shows the maps in
the viewer. `lar tract-density` does the same from files.

A track density image counts, in each voxel of a grid, the streamlines that pass
through it. On the grid of the diffusion data it shows where tractography went; on
a finer grid it gives the super-resolution maps of track-density imaging (Calamante
et al. 2010), because each streamline is followed more finely than the voxels it
was tracked in.
"""

import numpy as np

from larmor.tractography import density_grid, track_density
from larmor.types import Grid, Tractogram
from larmor.viewer import Category, Colormap, OptionName, Viewer, ViewKind

# %%
# A synthetic bundle
# ------------------
# The bundle has 200 streamlines with 1 mm steps. They fan out from a narrow stem
# near the origin along +y, each bending sideways by its own amount. The points are
# in world coordinates (mm).

rng = np.random.default_rng(1)
steps = np.arange(60, dtype=np.float32)
streamlines = []
for _ in range(200):
    start = rng.normal(0.0, 1.0, 3).astype(np.float32)
    bend = rng.uniform(-0.01, 0.01)
    x = start[0] + bend * steps**2
    streamlines.append(np.stack([x, start[1] + steps, start[2] + 0 * steps], axis=1))
offsets = np.concatenate([[0], np.cumsum([len(s) for s in streamlines])])
bundle = Tractogram(np.concatenate(streamlines), offsets)
print(f"{bundle.num_streamlines} streamlines, {bundle.num_points} points")

# %%
# On the diffusion grid
# ---------------------
# The first grid has 2 mm voxels and covers the bundle. Each streamline adds 1 to
# every voxel it passes through, once, however many of its points fall in the voxel.
# Before the points are mapped, the streamlines are upsampled along a smooth curve,
# so that no voxel between two points is missed.

affine = np.diag([2.0, 2.0, 2.0, 1.0])
affine[:3, 3] = [-40.0, -10.0, -10.0]
grid = Grid((40, 40, 10), affine)
density = track_density(bundle, grid)
print(
    f"2 mm: {np.count_nonzero(density.data)} voxels reached, at most "
    f"{density.data.max()} streamlines in one"
)
assert density.data.max() <= bundle.num_streamlines

# %%
# Super-resolution
# ----------------
# `density_grid` makes a grid of 0.5 mm voxels over the same field of view. The
# fan's stem stays dense while its spread-out end thins to single streamlines.

fine = track_density(bundle, density_grid(grid, 0.5))
print(f"0.5 mm: grid {fine.data.shape}, {np.count_nonzero(fine.data)} voxels reached")

# %%
# Weights and endpoints
# ---------------------
# With per-streamline weights, such as those SIFT2 computes, each streamline adds its
# weight instead of 1. The endpoint density counts only where streamlines start and
# end.

weights = rng.uniform(0.5, 1.5, bundle.num_streamlines)
weighted = track_density(bundle, grid, weights=weights)
ends = track_density(bundle, grid, ends_only=True)
print(f"total weight in the densest voxel: {weighted.data.max():.1f}")
print(f"endpoints counted: {ends.data.sum()}")
assert ends.data.sum() == 2 * bundle.num_streamlines

# %%
# In the viewer
# -------------
# The result is a `Volume`, which `save_volume` saves and the viewer shows. Here the
# viewer shows the 2 mm and the 0.5 mm maps on the axial slice through the bundle, in
# the same colormap. The colormap saturates at half of each map's maximum, so that
# the thin branches show. The screenshots hide the crosshair and the text.

for name, image in (("tdi_2mm.png", density), ("tdi_05mm.png", fine)):
    viewer = Viewer()
    item = viewer.add(image, Category.VOLUMES, name="density")
    viewer.set(item.id, OptionName.COLORMAP, Colormap.INFERNO)
    viewer.set(item.id, OptionName.MAXIMUM, float(image.data.max()) / 2)
    viewer.move_cursor((0.0, 20.0, 0.0))
    viewer.toggle_crosshair()
    viewer.screenshot(ViewKind.AXIAL, name, size=(500, 500), overlay=False)
