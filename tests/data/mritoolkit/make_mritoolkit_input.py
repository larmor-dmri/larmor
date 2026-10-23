"""Makes the input of ../../matlab_scripts/make_mritoolkit_ref.m in this folder.

A 6x7x5 crop of the DKI block in ../dki, given an oblique affine with voxels of
1.5 x 2.0 x 2.5 mm, so that a reader which swaps two axes, two voxel sizes or
ignores the rotation gives wrong world positions. The determinant stays negative,
as the CFIN affine's, so the FSL b-vectors keep their meaning (MRIToolkit's
`grad_perm` 2, `grad_flip` 1). Run from the repository root:

    uv run python tests/data/mritoolkit/make_mritoolkit_input.py
"""

import pathlib
import shutil

import numpy as np

from larmor.io import load_volume, save_volume
from larmor.types import Volume

SOURCE = pathlib.Path("tests/data/dki")
OUT = pathlib.Path("tests/data/mritoolkit")
CROP = (slice(1, 7), slice(0, 7), slice(2, 7))

dwi = load_volume(SOURCE / "dwi.nii.gz")
mask = load_volume(SOURCE / "mask.nii.gz")

axis = np.array([1.0, 2.0, 3.0]) / np.sqrt(14.0)
angle = np.deg2rad(15.0)
cross = np.array(
    [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
)
rotation = np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * cross @ cross
affine = np.eye(4)
affine[:3, :3] = rotation @ np.diag([-1.5, 2.0, 2.5])
affine[:3, 3] = [20.0, -30.0, 10.0]

save_volume(Volume(np.ascontiguousarray(dwi.data[CROP]), affine), OUT / "dwi.nii.gz")
save_volume(Volume(np.ascontiguousarray(mask.data[CROP]), affine), OUT / "mask.nii.gz")
shutil.copy(SOURCE / "bvals", OUT / "bvals")
shutil.copy(SOURCE / "bvecs", OUT / "bvecs")
for path in sorted(OUT.iterdir()):
    print(path, path.stat().st_size, "bytes")  # noqa: T201 (script output)
