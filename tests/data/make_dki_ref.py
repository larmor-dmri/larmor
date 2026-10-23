"""Generates the DKI test block in tests/data/dki/.

The input is the CFIN multi-shell dataset (Hansen and Jespersen, 2016, CC0 1.0),
which `larmor.data.fetch_cfin_multishell` downloads (about 197 MB) into the folder
in LARMOR_DATA_DIR, or the default cache folder. Run from the repository root:

    uv run python tests/data/make_dki_ref.py
"""

import pathlib

import numpy as np

from larmor.data import fetch_cfin_multishell
from larmor.dwi import create_brain_mask
from larmor.io import load_volume, save_volume
from larmor.types import Volume

OUT = pathlib.Path("tests/data/dki")
OUT.mkdir(exist_ok=True)

# An 8x8x8 block (2.5 mm voxels) entirely inside the brain mask, with white
# matter (about 90 voxels of FA > 0.6), grey matter (about 100 of FA < 0.2 and
# MD < 1.2e-3 mm^2/s) and some cerebrospinal fluid (about 15 of MD > 2e-3
# mm^2/s). Chosen from a DTI fit of the b0 and b=1000 volumes, among the blocks
# where every measurement is positive and at most the b0 in all but one voxel.
START = (30, 28, 7)
SIZE = 8
block = tuple(slice(s, s + SIZE) for s in START)

# CFIN has one b0 and 33 directions at each of 15 b-values from 200 to 3000.
# These shells give two shells for DKI (1000 and 2000) and more volumes (133)
# than a 5x5x5 denoising patch has voxels.
SHELLS = (0, 600, 1000, 1400, 2000)

files = fetch_cfin_multishell()
bvals = np.loadtxt(files["bvals"])
bvecs = np.loadtxt(files["bvecs"])  # (3, N), FSL frame

# MRIToolkit sorts the volumes by b-value and expects the b0s first; the other
# volumes stay in acquisition order.
kept = np.flatnonzero(np.isin(np.round(bvals), SHELLS))
order = np.concatenate([kept[bvals[kept] < 50], kept[bvals[kept] >= 50]])

dwi = load_volume(files["dwi"])
mask = create_brain_mask(dwi.data[..., 0].astype(np.float64))
if not mask[block].all():
    raise SystemExit("The block is not entirely inside the brain mask.")

shift = np.eye(4)
shift[:3, 3] = START
affine = dwi.affine @ shift
data = np.ascontiguousarray(dwi.data[(*block, order)], dtype=np.float32)
save_volume(Volume(data, affine), OUT / "dwi.nii.gz")
save_volume(Volume(mask[block].astype(np.uint8), affine), OUT / "mask.nii.gz")
np.savetxt(OUT / "bvals", np.round(bvals[order])[None, :], fmt="%d")
np.savetxt(OUT / "bvecs", bvecs[:, order], fmt="%.6f")

for path in sorted(OUT.iterdir()):
    print(path, path.stat().st_size, "bytes")  # noqa: T201 (script output)
print(len(order), "volumes,", int(mask[block].sum()), "masked voxels")  # noqa: T201
