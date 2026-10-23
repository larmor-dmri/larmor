"""Generates the MRtrix gradient table (.b) references in tests/data/mif/.

Uses the image and FSL table of make_mif_ref.py (swapped, flipped, slightly
oblique axes, so the FSL to world conversion is not trivial). Run from the
repository root, with MRtrix 3.0.8 on the PATH:

    uv run python tests/data/make_mrtrix_grad_ref.py

which runs, in tests/data/mif/ (each command also writes an image, which is
deleted):

    mrconvert source.nii -fslgrad dwi.bvec dwi.bval -export_grad_mrtrix dwi.b
    mrconvert source.nii -grad grad_odd.b -export_grad_mrtrix grad_odd_export.b
        -export_grad_fsl grad_odd_export.bvec grad_odd_export.bval
    mrconvert source.nii -grad larmor_dwi.b \
        -export_grad_fsl larmor_dwi.bvec larmor_dwi.bval

grad_odd.b is written by this script: comments, commas, semicolons and tabs, a
b = 0 row, a b0 stored as b = 5, and directions that are not unit vectors (one of
length 0.5 at b = 2800, which makes MRtrix scale every b-value by the squared
length). larmor_dwi.b is the FSL table read and written by Larmor, so that the last
command checks MRtrix reads Larmor's files.
"""

import pathlib
import subprocess

from larmor.io import read_bvals_bvecs, write_mrtrix_gradients

OUT = pathlib.Path("tests/data/mif")

GRAD_ODD = """\
# A gradient table as people write them by hand.
0 0 0 0
0.5 0.5 0.7071067811865476 1000   # a comment after a row
0,0.6,0.8,2000
0;0;0.5;2800
1\t0\t0\t5
"""
(OUT / "grad_odd.b").write_text(GRAD_ODD)

table = read_bvals_bvecs(
    OUT / "dwi.bval", OUT / "dwi.bvec", ref_path=OUT / "source.nii"
)
write_mrtrix_gradients(table, OUT / "larmor_dwi.b", ref_path=OUT / "source.nii")

COMMANDS = [
    ["-fslgrad", "dwi.bvec", "dwi.bval", "-export_grad_mrtrix", "dwi.b"],
    [
        "-grad",
        "grad_odd.b",
        "-export_grad_mrtrix",
        "grad_odd_export.b",
        "-export_grad_fsl",
        "grad_odd_export.bvec",
        "grad_odd_export.bval",
    ],
    ["-grad", "larmor_dwi.b", "-export_grad_fsl", "larmor_dwi.bvec", "larmor_dwi.bval"],
]
# The image MRtrix writes is not kept; only the gradient files are.
for args in COMMANDS:
    subprocess.run(
        ["mrconvert", "-quiet", "-force", "source.nii", *args, "unused.mif"],
        cwd=OUT,
        check=True,
    )
(OUT / "unused.mif").unlink()

version = subprocess.run(["mrconvert", "-version"], capture_output=True, text=True)
print("saved", OUT, version.stdout.splitlines()[0])  # noqa: T201 (script output)
