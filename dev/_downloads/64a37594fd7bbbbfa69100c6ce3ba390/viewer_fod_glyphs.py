"""FOD glyphs in the viewer
========================

This example builds a small synthetic image of fibre orientation distributions (FODs)
with three regions: fibres crossing at right angles, a single bundle, and an
asymmetric FOD with a single lobe at the end of the bundle, around a rim of round,
isotropic FODs. It saves screenshots of the axial view as the viewer starts and with
the round FODs hidden, and of one asymmetric FOD drawn large.

An FOD is stored as spherical harmonic (SH) coefficients, one 4D image with the
coefficients along its last axis. The viewer draws it in the 2D views as a glyph per
voxel on the current slice: the FOD's amplitude in every direction, as a surface
coloured by direction (red left to right, green front to back, blue up and down). Its
options set the SH basis, the glyphs' size and whether each is scaled on its own or
by the whole image, an amplitude threshold, a minimum anisotropy that hides round
FODs, the detail of the sphere and the colouring.
"""

import numpy as np

from larmor.sd import fit_sh
from larmor.types import Volume
from larmor.viewer import Category, FodInspector, OptionName, Viewer, ViewKind

# %%
# FODs from lobes
# ---------------
# An FOD is made from its amplitudes on many directions over the whole sphere,
# fitted with SH. A symmetric FOD has lobes on both ends of each fibre direction
# and needs only the even degrees: (lmax + 1)(lmax + 2) / 2 coefficients, 45 for
# lmax 8. An asymmetric FOD uses all degrees, (lmax + 1)^2 coefficients, 81 for
# lmax 8, and can have a lobe on one end only. The viewer tells the two apart by
# the number of coefficients.

LMAX = 8
# 600 directions spread evenly over the sphere (a Fibonacci spiral).
n = 600
z = 1 - (2 * np.arange(n) + 1) / n
azimuth = np.pi * (3 - np.sqrt(5)) * np.arange(n)
radius = np.sqrt(1 - z**2)
dirs = np.stack([radius * np.cos(azimuth), radius * np.sin(azimuth), z], axis=1)
x_axis, y_axis = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])


def lobes(*fibres: np.ndarray, one_sided: bool = False) -> np.ndarray:
    """Amplitudes with a narrow lobe along each fibre; one end only if one_sided."""
    cosines = dirs @ np.array(fibres).T
    spread = 1 - cosines if one_sided else 1 - cosines**2
    amplitudes: np.ndarray = np.exp(-6 * spread).sum(axis=1)
    return amplitudes


crossing = fit_sh(lobes(x_axis, y_axis), dirs, LMAX)
bundle = fit_sh(lobes(x_axis), dirs, LMAX)
ending = fit_sh(lobes(x_axis, one_sided=True), dirs, LMAX, full=True)
isotropic = fit_sh(np.ones(len(dirs)), dirs, LMAX)

# %%
# The images
# ----------
# The images share a 2 mm grid of 18 x 12 voxels and 3 slices, centred at the world
# origin. The symmetric FOD image holds the crossing, the bundle and the isotropic
# rim. The asymmetric one holds the end of the bundle, with its odd degrees from the
# same fit. A plain image goes under them, as a T1 would.

shape = (18, 12, 3)
affine = np.diag([2.0, 2.0, 2.0, 1.0])
affine[:3, 3] = -(np.array(shape) - 1)  # the centre at 0
symmetric = np.zeros((*shape, len(crossing)), np.float32)
symmetric[:, :] = isotropic * 0.3
symmetric[2:8, 2:10] = crossing
symmetric[8:12, 2:10] = bundle
asymmetric = np.zeros((*shape, len(ending)), np.float32)
asymmetric[12:16, 2:10] = ending
symmetric[12:16, 2:10] = 0  # the asymmetric FOD is drawn there
i, j, _ = np.indices(shape)
background = Volume((100 + 10 * i + 5 * j).astype(np.float32), affine)

# %%
# The viewer
# ----------
# FODs are added in their own category; `load` reads an FOD file instead. The
# coefficients' basis is not in the file, so it is an option: the tournier basis by
# default, or descoteaux or leemans. By default all glyphs are scaled together, so
# that a lobe of the image's high amplitude, the 90th percentile of the voxels'
# largest amplitudes, reaches the edge of its voxel. Weak FODs, such as the round
# ones of the rim, are then drawn smaller than strong ones. The Scaling option can
# instead scale each glyph on its own.

viewer = Viewer()
# The screenshots hide the crosshair and the text, so they show only the data.
viewer.toggle_crosshair()
viewer.toggle_overlay()
viewer.add(background, Category.VOLUMES, name="background")
viewer.add(Volume(symmetric, affine), Category.FODS, name="symmetric")
ends = viewer.add(Volume(asymmetric, affine), Category.FODS, name="asymmetric")
viewer.screenshot(ViewKind.AXIAL, "viewer_fod_glyphs.png", size=(600, 600))

# %%
# Hiding round FODs
# -----------------
# The anisotropy of an FOD, from its coefficients, is 0 for a sphere and close to 1
# for a sharp FOD. Raising the minimum anisotropy hides the round FODs of grey
# matter and CSF, here the rim, and leaves the fibres.

viewer.set(Category.FODS, OptionName.ANISOTROPY, 0.5)
viewer.screenshot(ViewKind.AXIAL, "viewer_fod_glyphs_anisotropic.png", size=(600, 600))

# %%
# One FOD, large
# --------------
# In the window, the Inspect FOD button in an FOD's options, or the shortcut Ctrl+4
# or F, opens a panel that draws the FOD at the crosshair as one large glyph on a
# finer sphere. The glyph turns with the mouse and follows the crosshair. The panel
# draws off screen too; here it shows the end of the bundle, whose single lobe points
# to the subject's right.

viewer.move_cursor((9.0, -1.0, 0.0))
with FodInspector(viewer, ends.id) as inspector:
    inspector.save("viewer_fod_inspector.png")
print(f"{len(crossing)} symmetric and {len(ending)} asymmetric coefficients")
