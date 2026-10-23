"""Denoising a DWI with MP-PCA
===========================

This example builds a noisy diffusion-weighted image (DWI) from known tensors,
denoises it with MP-PCA (Marchenko-Pastur principal component analysis), and checks
the estimated noise level and the error before and after denoising.

Thermal noise lowers the precision of everything fitted from a diffusion image, most
of all at high b-values where the signal is low. MP-PCA (Veraart et al. 2016) removes
much of it. In a small patch around each voxel, the signals of all volumes form a
matrix whose principal components split into a few that carry the signal and many
that carry only noise. The eigenvalues of pure noise follow the Marchenko-Pastur
distribution, which tells where the split is and gives the noise level sigma as a
by-product.
"""

import numpy as np

from larmor.dwi import denoise_mppca
from larmor.types import Volume

# %%
# A synthetic acquisition
# -----------------------
# The acquisition has two b0 images and 60 directions at b = 1000 s/mm^2, on a
# 12x12x12 grid. Each voxel has a prolate tensor (eigenvalues 1.7e-3 and 0.3e-3
# mm^2/s) whose direction turns slowly along x, so that neighbouring voxels have
# similar signals, as in tissue.

rng = np.random.default_rng(0)
directions = rng.normal(size=(60, 3))
directions /= np.linalg.norm(directions, axis=1, keepdims=True)
bvals = np.concatenate([[0.0, 0.0], np.full(60, 1000.0)])
bvecs = np.vstack([[[0.0, 0.0, 0.0]] * 2, directions])

shape = (12, 12, 12)
angle = np.linspace(0, np.pi / 2, shape[0])[:, None, None] * np.ones(shape)
fibre = np.stack([np.cos(angle), np.sin(angle), np.zeros(shape)], axis=-1)
along = (bvecs @ fibre[..., :, None])[..., 0] ** 2  # cos^2 to the fibre, per volume
diffusivity = 0.3e-3 + (1.7e-3 - 0.3e-3) * along
s0 = 1000.0
truth = s0 * np.exp(-bvals * diffusivity)

# %%
# Gaussian noise of sigma = 20 is added, which gives a signal-to-noise ratio of 50 in
# the b0 images.

sigma = 20.0
noisy = truth + rng.normal(scale=sigma, size=truth.shape)
volume = Volume(noisy.astype(np.float32), np.eye(4))

# %%
# Denoising
# ---------
# The patch defaults to the smallest odd cube with at least as many voxels as
# volumes: 5x5x5 for these 62 volumes. Real data would be read from a file with
# `load_volume("dwi.nii.gz")`, and a brain mask would limit the work to the brain.

result = denoise_mppca(volume)

print(f"estimated sigma: {np.median(result.noise):.2f} (true {sigma})")
print(f"signal components per patch: {np.median(result.num_signal):.0f} of 62")
error_before = np.sqrt(np.mean((noisy - truth) ** 2))
error_after = np.sqrt(np.mean((result.volume.data - truth) ** 2))
print(f"RMS error: {error_before:.2f} before, {error_after:.2f} after")
assert abs(np.median(result.noise) - sigma) < 0.05 * sigma
assert error_after < error_before / 2
