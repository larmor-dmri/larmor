"""DTI colours in the viewer
=========================

This example builds a synthetic head with a field of directions and a fractional
anisotropy (FA) map over it, and saves screenshots of the axial and coronal views in
direction-encoded colours, first plain and then modulated by the FA.

The principal eigenvector of a DTI fit is the main direction of diffusion in each
voxel. `lar dti` writes it as `<prefix>_pe.nii.gz`, a 4D image with the x, y and z of
that direction. The viewer can show it as a direction-encoded colour map, with red,
green and blue for the size of the direction's x, y and z, so red runs left to right,
green front to back and blue up and down. The colours can be modulated by another
image on the same grid, usually the FA, which darkens the voxels with no clear
direction.
"""

import numpy as np

from larmor.types import Volume
from larmor.viewer import Category, OptionName, Viewer, ViewKind

# %%
# A synthetic head
# ----------------
# The head is a 2 mm image of an ellipsoid, centred at the world origin. World
# coordinates are in mm, RAS+.

shape = (80, 96, 80)
affine = np.diag([2.0, 2.0, 2.0, 1.0])
affine[:3, 3] = -np.array(shape)  # half the extent in mm: the centre at 0
i, j, k = np.indices(shape, dtype=np.float32)
x, y, z = (2 * i / shape[0] - 1, 2 * j / shape[1] - 1, 2 * k / shape[2] - 1)
r = np.sqrt((x / 0.8) ** 2 + (y / 0.9) ** 2 + (z / 0.85) ** 2)
brain = r < 0.8

# %%
# Directions and FA
# -----------------
# Three kinds of bundle lie over a background of random directions. A band across the
# middle runs left to right, like the corpus callosum. Two lateral bundles run front
# to back, like the cingulum or the superior longitudinal fasciculus. Two columns run
# up and down, like the corticospinal tracts. The FA is high in the bundles and low
# elsewhere. Each direction is a unit vector, as `lar dti` writes it; its sign does
# not matter.

rng = np.random.default_rng(0)
directions = rng.standard_normal((*shape, 3)).astype(np.float32)
fa = np.full(shape, 0.15, np.float32)
bundles = {  # in this order, so the columns cross the band
    (0.0, 1.0, 0.0): brain & (np.abs(x) > 0.45),  # front to back
    (1.0, 0.0, 0.0): brain & (np.abs(z) < 0.1) & (np.abs(y) < 0.4) & (np.abs(x) < 0.45),
    (0.0, 0.0, 1.0): brain & (np.abs(np.abs(x) - 0.25) < 0.08) & (np.abs(y) < 0.2),
}
for direction, where in bundles.items():
    directions[where] = direction
    fa[where] = 0.8
directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
directions[~brain] = 0
fa[~brain] = 0
pe = Volume(directions, affine)
fa_map = Volume(fa, affine)

# %%
# The viewer and its items
# ------------------------
# The eigenvector is added as a volume and the FA as an overlay. The FA overlay is
# hidden, because it is only used to modulate the colours. DTI colours are an option
# of the image, available for a 4D image with exactly 3 volumes.

viewer = Viewer()
# The screenshots hide the crosshair and the text, so they show only the data.
viewer.toggle_crosshair()
viewer.toggle_overlay()
vectors = viewer.add(pe, Category.VOLUMES, name="pe")
fa_item = viewer.add(fa_map, Category.OVERLAYS, name="fa")
viewer.set(fa_item.id, OptionName.VISIBLE, False)
viewer.set(vectors.id, OptionName.DTI, True)

# %%
# The plain colours
# -----------------
# Each channel is the size of the direction's component, clipped to [0, 1], so a
# unit vector is at full brightness: the background's random directions are as
# bright as the bundles.

viewer.screenshot(ViewKind.AXIAL, "viewer_dti_axial.png", size=(600, 600))
viewer.screenshot(ViewKind.CORONAL, "viewer_dti_coronal.png", size=(600, 600))

# %%
# Modulated by FA
# ---------------
# Any 3D volume or overlay on the same grid can modulate the colours; its values,
# clipped to [0, 1], multiply them. With the FA, the bundles stand out and the
# background fades. An eigenvector already multiplied by the FA (as `mrcalc pe fa
# -mult` makes it) gives the same picture with no modulation.

viewer.set(vectors.id, OptionName.MODULATE, fa_item.id)
viewer.screenshot(ViewKind.AXIAL, "viewer_dti_fa_axial.png", size=(600, 600))
viewer.screenshot(ViewKind.CORONAL, "viewer_dti_fa_coronal.png", size=(600, 600))
print(f"{int(brain.sum())} voxels in the brain")
