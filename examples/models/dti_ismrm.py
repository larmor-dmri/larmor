"""DTI of a simulated brain
========================

This example fits the diffusion tensor (DTI) to the simulated brain of the ISMRM
2015 Tractography Challenge, prints the mean fractional anisotropy (FA) of its white
matter, and shows the principal direction of the tensor as colours on an axial
slice, brighter where the FA is high: red left to right, green front to back, blue
up and down.

The data is simulated: diffusion-weighted images at 2 mm with 32 directions at
b=1000 s/mm^2, and a T1-like image, from the challenge's Zenodo record
(https://doi.org/10.5281/zenodo.572345, CC BY 4.0). Please cite: Maier-Hein KH et al.
The challenge of mapping the human connectome based on diffusion tractography.
Nature Communications 8 (2017) 1349.
"""

# needs: example-data (the tests run it only when LARMOR_DATA_DIR is set)

import numpy as np

from larmor.data import fetch_ismrm2015
from larmor.dwi import DTIModel, create_brain_mask, mean_b0
from larmor.io import load_dwi
from larmor.viewer import Category, OptionName, Viewer, ViewKind

# %%
# Fitting DTI
# -----------
# `fetch_ismrm2015` downloads the data once (about 41 MB) and returns the paths of
# its files. The tensor is fitted in a brain mask made from the b=0 image.

files = fetch_ismrm2015()
dwi, gtab = load_dwi(files["dwi"], files["bvals"], files["bvecs"])
mask = create_brain_mask(mean_b0(dwi, gtab))
fit = DTIModel().fit(dwi, gtab, mask=mask)
white_matter = fit.fa > 0.3
print(f"Mean FA where it is above 0.3: {fit.fa[white_matter].mean():.2f}")

# %%
# Direction colours
# -----------------
# The principal eigenvector, the tensor's main direction, is shown with the DTI
# option, which draws it as colours by direction. Modulated by the FA map, the
# colours are dark where diffusion is isotropic. The FA map itself is hidden. The
# screenshot hides the crosshair and the text, so it shows only the data.

shape = np.array(dwi.data.shape[:3])
centre = dwi.affine[:3, :3] @ ((shape - 1) / 2) + dwi.affine[:3, 3]

viewer = Viewer()
directions = viewer.add(dwi.with_data(fit.pe), Category.VOLUMES, name="direction")
fa = viewer.add(dwi.with_data(fit.fa), Category.OVERLAYS, name="FA")
viewer.set(fa.id, OptionName.VISIBLE, False)
viewer.set(directions.id, OptionName.DTI, True)
viewer.set(directions.id, OptionName.MODULATE, fa.id)
viewer.move_cursor(tuple(centre + [0.0, 0.0, 4.0]))
viewer.toggle_crosshair()
viewer.zoom(ViewKind.AXIAL, 1.3)
viewer.screenshot(
    ViewKind.AXIAL, "dti_ismrm_colours.png", size=(900, 600), overlay=False
)
