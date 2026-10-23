"""Generates the sheared-image gradient reference in tests/data/mif/.

Writes a 2x2x2x6 int16 image whose affine is oblique, sheared and has a positive
determinant (only an sform, so that every reader takes the same affine), with an
FSL table of five directions and one b0. Run from the repository root, with
MRtrix 3.0.8 on the PATH:

    uv run python tests/data/make_sheared_grad_ref.py

which runs, in tests/data/mif/ (the image MRtrix writes is deleted):

    mrconvert sheared.nii -fslgrad sheared.bvec sheared.bval
        -export_grad_mrtrix sheared.b
"""

import pathlib
import subprocess

import nibabel as nib
import numpy as np

OUT = pathlib.Path("tests/data/mif")

c, s = np.cos(0.3), np.sin(0.3)
rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
shear = np.array([[2.0, 0.3, 0.0], [0.0, 2.0, 0.3], [0.0, 0.0, 2.5]])
affine = np.eye(4)
affine[:3, :3] = rotation @ shear
affine[:3, 3] = (-10.0, 20.0, 5.0)

header = nib.Nifti1Header()
image = nib.Nifti1Image(np.ones((2, 2, 2, 6), np.int16), None, header)
image.set_sform(affine, code=1)
image.set_qform(None, code=0)
nib.save(image, OUT / "sheared.nii")

bvecs = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.6, 0.0, 0.8],
        [0.48, 0.6, 0.64],
    ]
)
np.savetxt(OUT / "sheared.bval", [[0, 1000, 1000, 1000, 2000, 2000]], fmt="%d")
np.savetxt(OUT / "sheared.bvec", bvecs.T, fmt="%.17g")

subprocess.run(
    [
        "mrconvert",
        "-quiet",
        "-force",
        "sheared.nii",
        "-fslgrad",
        "sheared.bvec",
        "sheared.bval",
        "-export_grad_mrtrix",
        "sheared.b",
        "unused.mif",
    ],
    cwd=OUT,
    check=True,
)
(OUT / "unused.mif").unlink()

version = subprocess.run(["mrconvert", "-version"], capture_output=True, text=True)
print("saved", OUT, version.stdout.splitlines()[0])  # noqa: T201 (script output)
