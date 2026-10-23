"""Generates tests/data/sh_tournier_ref.npz: MRtrix sh2amp evaluation of every basis
function up to lmax 8 at 60 random unit directions."""

import pathlib
import subprocess
import tempfile
from typing import cast

import nibabel as nib
import numpy as np

LMAX, N_COEFFS = 8, 45
rng = np.random.default_rng(0)
dirs = rng.normal(size=(60, 3))
dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

tmp = pathlib.Path(tempfile.mkdtemp())
coeffs = np.eye(N_COEFFS, dtype=np.float32).reshape(N_COEFFS, 1, 1, N_COEFFS)
nib.save(nib.Nifti1Image(coeffs, np.eye(4)), tmp / "eye.nii.gz")
np.savetxt(tmp / "dirs.txt", dirs)
subprocess.run(
    ["sh2amp", tmp / "eye.nii.gz", tmp / "dirs.txt", tmp / "amp.nii.gz", "-quiet"],
    check=True,
)
# nib.load is typed as returning any image type; sh2amp writes NIfTI.
amp_img = cast(nib.Nifti1Image, nib.load(tmp / "amp.nii.gz"))
amps = amp_img.get_fdata()[:, 0, 0, :].T  # (n_dirs, num_coeffs)
np.savez("tests/data/sh_tournier_ref.npz", dirs=dirs, basis=amps, lmax=LMAX)
print(  # noqa: T201 (script output)
    "saved",
    amps.shape,
    "mrtrix",
    subprocess.run(["mrinfo", "-version"], capture_output=True, text=True).stdout.split(
        "\n"
    )[0],
)
