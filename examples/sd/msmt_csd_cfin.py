"""Multi-tissue FODs of a multi-shell brain scan
=============================================

This example fits multi-shell multi-tissue constrained spherical deconvolution
(MSMT-CSD) to a multi-shell scan of a healthy volunteer. The responses of white
matter, grey matter and CSF, the signal each pure tissue gives, are estimated from
the data itself. Each voxel then gets a white matter fibre orientation distribution
(FOD) and the fraction of each tissue. The example shows the FODs and the three
tissue fractions on the same coronal slice.

The data is the CFIN multi-shell dataset (Hansen B, Jespersen SN. Data for
evaluation of fast kurtosis strategies, b-value optimization and exploration of
diffusion MRI contrast. Scientific Data 3 (2016) 160072).

MSMT-CSD follows Jeurissen B et al. NeuroImage 103 (2014) 411-426, and the tissue
responses are estimated as by `dwi2response dhollander` of MRtrix3 (Dhollander T et
al., 2016 and 2019; Tournier JD et al. NeuroImage 202 (2019) 116137).
"""

# needs: example-data (the tests run it only when LARMOR_DATA_DIR is set)

import numpy as np

from larmor.data import fetch_cfin_multishell
from larmor.dwi import create_brain_mask, mean_b0
from larmor.io import load_dwi
from larmor.sd import MSMTCSDModel
from larmor.viewer import Category, Viewer, ViewKind

# %%
# The data and the fit
# --------------------
# `fetch_cfin_multishell` downloads the data once and returns the paths of its
# files. With no responses given, the model estimates them from the data in the
# brain mask, then fits every voxel of the mask.

files = fetch_cfin_multishell()
dwi, gtab = load_dwi(files["dwi"], files["bvals"], files["bvecs"])
b0 = mean_b0(dwi, gtab)
mask = create_brain_mask(b0)
fit = MSMTCSDModel().fit(dwi, gtab, mask)
fod = dwi.with_data(fit.fods[0].astype(np.float32))
fractions = fit.tissue_fractions
assert fractions is not None
for name, i in (("white matter", 0), ("grey matter", 1), ("CSF", 2)):
    print(
        f"Mean {name} fraction in the brain: {np.nanmean(fractions[..., i][mask]):.2f}"
    )

# %%
# White matter FODs
# -----------------
# The glyphs are drawn over the mean b=0 image, on a coronal slice through the
# centre of the brain.

shape = np.array(dwi.data.shape[:3])
centre = tuple(dwi.affine[:3, :3] @ ((shape - 1) / 2) + dwi.affine[:3, 3])

viewer = Viewer()
viewer.add(dwi.with_data(b0.astype(np.float32)), Category.VOLUMES, name="mean b=0")
viewer.add(fod, Category.FODS, name="WM FOD")
viewer.move_cursor(centre)
viewer.toggle_crosshair()
viewer.zoom(ViewKind.CORONAL, 2.0, about=centre)
viewer.screenshot(
    ViewKind.CORONAL, "msmt_csd_cfin_fods.png", size=(900, 600), overlay=False
)

# %%
# Tissue fractions
# ----------------
# Each of the three images shows the fraction of one tissue, white matter, grey
# matter or CSF, in each voxel of the same slice.

for name, i, image in (
    ("WM fraction", 0, "msmt_csd_cfin_wm.png"),
    ("GM fraction", 1, "msmt_csd_cfin_gm.png"),
    ("CSF fraction", 2, "msmt_csd_cfin_csf.png"),
):
    viewer = Viewer()
    viewer.add(
        dwi.with_data(fractions[..., i].astype(np.float32)), Category.VOLUMES, name=name
    )
    viewer.move_cursor(centre)
    viewer.toggle_crosshair()
    viewer.screenshot(ViewKind.CORONAL, image, size=(600, 600), overlay=False)
