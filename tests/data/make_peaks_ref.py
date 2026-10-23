"""Generates the FOD block and its MRtrix sh2peaks reference in tests/data/peaks/.

The input is the ISMRM FOD (90x108x90x45, lmax 8, tournier basis) and its brain
mask, which are not in the repository. Run from the repository root:

    uv run python tests/data/make_peaks_ref.py FOD MASK

for example with FOD = .../ISMRM/Data/fod.nii.gz and MASK =
.../ISMRM/Data/dwi_brain_mask.nii.gz.
"""

import pathlib
import subprocess
import sys

import numpy as np

from larmor.io import load_volume, save_volume
from larmor.types import Volume

OUT = pathlib.Path("tests/data/peaks")
OUT.mkdir(exist_ok=True)

# An 8x8x8 block in the white matter, entirely inside the brain mask, with no
# near-degenerate voxels (lobes of almost equal amplitude), where Larmor and
# sh2peaks can stop at slightly different points of a flat ridge.
START = (42, 55, 38)
SIZE = 8
block = tuple(slice(s, s + SIZE) for s in START)

fod = load_volume(sys.argv[1])
mask = load_volume(sys.argv[2])
if not mask.data[block].astype(bool).all():
    raise SystemExit("The block is not entirely inside the brain mask.")

shift = np.eye(4)
shift[:3, 3] = START
crop = Volume(
    data=np.ascontiguousarray(fod.as_float(np.float32)[block]),
    affine=fod.affine @ shift,
)
save_volume(crop, OUT / "fod.mif")

subprocess.run(
    [
        "sh2peaks",
        "fod.mif",
        "peaks_ref.mif",
        "-num",
        "3",
        "-nthreads",
        "0",
        "-quiet",
        "-force",
    ],
    cwd=OUT,
    check=True,
)

version = subprocess.run(["sh2peaks", "-version"], capture_output=True, text=True)
print("saved", OUT, version.stdout.splitlines()[0])  # noqa: T201 (script output)
