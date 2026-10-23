"""Fitting DKI to a real brain
===========================

This example fits diffusion kurtosis imaging (DKI) to a real multi-shell scan: the
CFIN dataset of Hansen and Jespersen, one healthy volunteer scanned at 3 T with 33
directions at each of 15 b-values up to 3000 s/mm^2. `larmor.data` downloads it
once (about 197 MB) into a cache folder and returns the paths of its files; later
runs use the cached copy.

The data is shared under CC0 1.0. Please cite: Hansen B, Jespersen SN. Data for
evaluation of fast kurtosis strategies, b-value optimization and exploration of
diffusion MRI contrast. Scientific Data 3 (2016) 160072.
"""

# needs: example-data (the tests run it only when LARMOR_DATA_DIR is set)

import numpy as np

from larmor.data import fetch_cfin_multishell
from larmor.dwi import DKIModel, create_brain_mask, mean_b0
from larmor.io import load_dwi, save_volume
from larmor.types import GradientTable, Volume

# %%
# Downloading and reading the data
# --------------------------------
# The fetcher returns a dictionary of paths; `load_dwi` reads the image with its
# b-values and b-vectors.

files = fetch_cfin_multishell()
dwi, gtab = load_dwi(files["dwi"], files["bvals"], files["bvecs"])
print(f"DWI: {dwi.data.shape}, b-values {np.unique(gtab.bvals.round(-2))}")

# %%
# Keeping the b=0, 1000 and 2000 shells
# -------------------------------------
# DKI is usually fitted to b-values up to about 2000 to 2500 s/mm^2: at higher
# b-values the signal departs from the model's quadratic in b.

keep = np.isin(gtab.bvals.round(-2), [0, 1000, 2000])
dwi = dwi.with_data(dwi.data[..., keep])
gtab = GradientTable(gtab.bvals[keep], gtab.bvecs[keep])

# %%
# A brain mask and the fit
# ------------------------
# The mask comes from the mean b=0 image. The fit runs in parallel over the voxels.

mask = create_brain_mask(mean_b0(dwi, gtab))
fit = DKIModel().fit(dwi, gtab, mask=mask)

# %%
# Mean kurtosis in white and grey matter
# --------------------------------------
# White matter, with its tightly packed axons, has a higher mean kurtosis (MK)
# than grey matter. A rough split by fractional anisotropy (FA) shows it.

white = mask & (fit.fa > 0.4)
grey = mask & (fit.fa > 0.05) & (fit.fa < 0.2) & (fit.md < 1.2e-3)
mk_white = float(np.median(fit.mk[white]))
mk_grey = float(np.median(fit.mk[grey]))
print(f"Median MK: {mk_white:.2f} where FA > 0.4, {mk_grey:.2f} where FA < 0.2")
assert mk_white > mk_grey

# %%
# Saving the maps
# ---------------
# The maps are on the DWI's grid; saved as NIfTI, they open in the viewer
# (`lar bv dki_mk.nii.gz dki_fa.nii.gz`) or any other tool.

for name, values in (("mk", fit.mk), ("fa", fit.fa)):
    save_volume(Volume(values, dwi.affine), f"dki_{name}.nii.gz")
