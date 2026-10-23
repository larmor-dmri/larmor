"""Fitting DKI to a synthetic signal
=================================

This example builds a two-shell diffusion image whose signal comes from a known
diffusivity and kurtosis, fits diffusion kurtosis imaging (DKI) with Larmor, and
checks that the fitted mean diffusivity (MD) and mean kurtosis (MK) match them.

DKI extends DTI with a second tensor, the kurtosis tensor W, which measures how far
the diffusion in each direction departs from a Gaussian. Restricted diffusion, as in
tissue with many membranes, gives a positive kurtosis; free Gaussian diffusion gives
zero. MK averages the apparent kurtosis over directions. DKI needs at least two
non-zero b-values, because the kurtosis shows up as the curvature of the log signal
in b.
"""

import numpy as np

from larmor.dwi import DKIModel
from larmor.types import GradientTable, Volume

# %%
# Isotropic diffusion with kurtosis
# ---------------------------------
# The signal has the same diffusivity (1e-3 mm^2/s) and the same kurtosis (0.8) in
# every direction. The isotropic kurtosis tensor has W_iiii = k and W_iijj = k / 3,
# so that its quartic form W(n) equals k for every unit vector n.

diffusivity = 1e-3
kurtosis = 0.8

# %%
# The acquisition: one b=0 image and 30 directions at each of b=1000 and 2000
# ---------------------------------------------------------------------------

rng = np.random.default_rng(0)
directions = rng.normal(size=(60, 3))
directions /= np.linalg.norm(directions, axis=1, keepdims=True)
bvals = np.concatenate([[0.0], np.full(30, 1000.0), np.full(30, 2000.0)])
bvecs = np.vstack([[0.0, 0.0, 0.0], directions])

# %%
# The signal follows the DKI model, ln S = ln S0 - b D(n) + b^2 MD^2 W(n) / 6,
# where D(n) is the diffusivity and W(n) the kurtosis along the direction n. Here
# D(n) = MD = the diffusivity and W(n) = the kurtosis in every direction. The signal
# is the same in every voxel of a small 3x3x3 image.

s0 = 1000.0
signal = s0 * np.exp(-bvals * diffusivity + bvals**2 * diffusivity**2 * kurtosis / 6)
data = np.broadcast_to(signal, (3, 3, 3, len(bvals))).astype(np.float32)

# %%
# Fitting DKI
# -----------
# The default fit is iterated weighted least squares (`fit_method="wls"`); "ols"
# gives ordinary least squares. Real data would be read from files with
# `load_volume("dwi.nii.gz")` and `read_bvals_bvecs("bvals", "bvecs", "dwi.nii.gz")`.

volume = Volume(data, np.eye(4))
gtab = GradientTable(bvals, bvecs)
fit = DKIModel().fit(volume, gtab, mask=np.ones((3, 3, 3), dtype=bool))

print(f"MD: {fit.md[1, 1, 1]:.6f} mm^2/s (expected {diffusivity})")
print(f"MK: {fit.mk[1, 1, 1]:.4f} (expected {kurtosis})")
assert np.allclose(fit.md, diffusivity, rtol=1e-3)
assert np.allclose(fit.mk, kurtosis, atol=1e-3)
