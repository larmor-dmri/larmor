"""Generates the MRtrix-written reference files in tests/data/mif/ with mrconvert.

Run from the repository root: uv run python tests/data/make_mif_ref.py
"""

import math
import pathlib
import subprocess

import nibabel as nib
import numpy as np

OUT = pathlib.Path("tests/data/mif")
OUT.mkdir(exist_ok=True)

# Axes swapped and flipped, and slightly oblique, so that MRtrix reorders them in
# the .mif header and every layout below is relative to a non-trivial frame.
c, s = math.cos(0.3), math.sin(0.3)
rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
affine = np.eye(4)
affine[:3, :3] = rotation @ np.array([[0, -3.0, 0], [2.0, 0, 0], [0, 0, -2.5]])
affine[:3, 3] = [10.5, -20.25, 5.0]

# Distinct values wider than one byte, so that a wrong voxel or byte order shows.
data = (np.arange(2 * 3 * 4 * 5) * 250 - 15000).astype(np.int16).reshape(2, 3, 4, 5)
nib.save(nib.Nifti1Image(data, affine), OUT / "source.nii")
mask = (np.arange(2 * 3 * 4) * 7 % 5 == 0).astype(np.uint8).reshape(2, 3, 4)
nib.save(nib.Nifti1Image(mask, affine), OUT / "mask.nii")

rng = np.random.default_rng(0)
bvecs = rng.standard_normal((3, 5))
bvecs /= np.linalg.norm(bvecs, axis=0)
bvecs[:, 0] = 0
np.savetxt(OUT / "dwi.bvec", bvecs, fmt="%.17g")
np.savetxt(OUT / "dwi.bval", [[0, 1000, 2000, 1000, 3000]], fmt="%d")

COMMANDS = [
    ["source.nii", "-strides", "-1,3,2,4", "layout_permuted.mif"],
    ["source.nii", "-datatype", "float32be", "-strides", "2,3,4,1", "float32be.mif"],
    ["source.nii", "-datatype", "int16be", "int16be.mif.gz"],
    ["mask.nii", "-datatype", "bit", "-strides", "-1,2,3", "mask_bit.mif"],
    ["source.nii", "split.mih"],
    ["source.nii", "-fslgrad", "dwi.bvec", "dwi.bval", "dwi.mif"],
]
for args in COMMANDS:
    subprocess.run(["mrconvert", "-quiet", "-force", *args], cwd=OUT, check=True)

version = subprocess.run(["mrconvert", "-version"], capture_output=True, text=True)
print("saved", OUT, version.stdout.splitlines()[0])  # noqa: T201 (script output)
