"""Asymmetric FOD for Larmor's tests, from MRIToolkit's fods_mrtrix.nii.gz.

The input is in the tournier basis, lmax 8. scilpy 1.5 (dipy 1.7) works in the legacy
tournier07 basis, whose m != 0 functions are Larmor's divided by sqrt(2): coefficients
are multiplied by sqrt(2) going in and divided by it coming out. scilpy's
local_asym_filtering (sigma 1, sharpness 1, repulsion724) runs in z slabs with one
voxel of overlap, since the filter only reaches the first neighbours.
"""

import sys

import nibabel as nib
import numpy as np
from dipy.reconst.shm import sph_harm_ind_list
from scilpy.denoise.asym_averaging import local_asym_filtering

src, dst = sys.argv[1], sys.argv[2]
img = nib.load(src)
sym = img.get_fdata(dtype=np.float32)
sym = np.nan_to_num(sym)
m_sym, _ = sph_harm_ind_list(8)
m_full, _ = sph_harm_ind_list(8, full_basis=True)
legacy_in = sym * np.where(m_sym != 0, np.sqrt(2), 1.0)

nz = sym.shape[2]
out = np.zeros(sym.shape[:3] + (81,), dtype=np.float32)
step = 6
for z0 in range(0, nz, step):
    z1 = min(z0 + step, nz)
    a, b = max(z0 - 1, 0), min(z1 + 1, nz)
    part = local_asym_filtering(
        legacy_in[:, :, a:b],
        sh_order=8,
        sh_basis="tournier07",
        in_full_basis=False,
        dot_sharpness=1.0,
        sphere_str="repulsion724",
        sigma=1.0,
    )
    out[:, :, z0:z1] = part[:, :, z0 - a : z0 - a + (z1 - z0)]
    print(f"z {z0}-{z1} of {nz}", flush=True)  # noqa: T201 (script output)
out /= np.where(m_full != 0, np.sqrt(2), 1.0).astype(np.float32)
nib.save(nib.Nifti1Image(out, img.affine, img.header), dst)
print("saved", dst, out.shape)  # noqa: T201 (script output)
