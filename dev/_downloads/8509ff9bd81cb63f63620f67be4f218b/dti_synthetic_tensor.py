"""Fitting DTI to a synthetic tensor
=================================

This example builds a small diffusion image whose signal comes from a known tensor,
fits diffusion tensor imaging (DTI) with Larmor, and checks that the fitted
fractional anisotropy (FA) matches the FA of that tensor.

DTI describes the diffusion of water in each voxel with a 3x3 tensor. Its eigenvalues
give maps such as FA, which is 0 for diffusion that is the same in every direction
and approaches 1 when diffusion follows a single direction, as along a fibre bundle.
"""

import numpy as np

from larmor.dwi import DTIModel
from larmor.types import GradientTable, Volume

# %%
# A tensor that looks like white matter
# -------------------------------------
# The tensor has fast diffusion along x (1.7e-3 mm^2/s) and slow diffusion across it
# (0.3e-3 mm^2/s). Its FA follows from the eigenvalues.

eigenvalues = np.array([1.7e-3, 0.3e-3, 0.3e-3])
tensor = np.diag(eigenvalues)
mean = eigenvalues.mean()
expected_fa = np.sqrt(1.5 * np.sum((eigenvalues - mean) ** 2) / np.sum(eigenvalues**2))

# %%
# The acquisition: one b=0 image and 30 directions at b=1000 s/mm^2
# -----------------------------------------------------------------

rng = np.random.default_rng(0)
directions = rng.normal(size=(30, 3))
directions /= np.linalg.norm(directions, axis=1, keepdims=True)
bvals = np.concatenate([[0.0], np.full(30, 1000.0)])
bvecs = np.vstack([[0.0, 0.0, 0.0], directions])

# %%
# The signal of each measurement is S0 exp(-b g^T D g), the same in every voxel of a
# small 3x3x3 image.

s0 = 1000.0
signal = s0 * np.exp(-bvals * np.einsum("ni,ij,nj->n", bvecs, tensor, bvecs))
data = np.broadcast_to(signal, (3, 3, 3, len(bvals))).astype(np.float32)

# %%
# Fitting DTI
# -----------
# The image is a `Volume` (its data and a voxel-to-world affine, here 1 mm voxels)
# and the acquisition a `GradientTable`. Real data would be read from files instead,
# with `load_volume("dwi.nii.gz")` and
# `read_bvals_bvecs("bvals", "bvecs", "dwi.nii.gz")`.

volume = Volume(data, np.eye(4))
gtab = GradientTable(bvals, bvecs)
fit = DTIModel().fit(volume, gtab, mask=np.ones((3, 3, 3), dtype=bool))

print(f"expected FA: {expected_fa:.4f}")
print(f"fitted FA:   {fit.fa[1, 1, 1]:.4f}")
assert np.allclose(fit.fa, expected_fa, atol=1e-3)
