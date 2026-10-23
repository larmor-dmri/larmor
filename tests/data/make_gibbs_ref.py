"""Generates the Gibbs-ringing test images in tests/data/gibbs/ and MRtrix3's result.

Two inputs, each written as float32 NIfTI:

- `dwi.nii.gz`: a 95x96x3 block of the CFIN multi-shell dataset (Hansen and
  Jespersen, 2016, CC0 1.0; not in the repository), its b0 and its first b=1000
  volume, as acquired. One in-plane size is odd and one even.
- `phantom.nii.gz`: a 64x62x2 synthetic image with ringing: an ellipse and a
  square of constant intensity drawn on a grid four times finer, then cut to the
  central 64x62 frequencies of their Fourier transform (no window), as an
  acquisition with full k-space sampling records them; both sizes are even, and
  the second slice has the shapes shifted by a third of a voxel.

`mrdegibbs` (MRtrix3, defaults: axes 0,1, 20 shifts, window 1 to 3) is run on
both, and its results are saved in `mrdegibbs_ref.npz`. The CFIN data comes from
`larmor.data.fetch_cfin_multishell` (about 197 MB, into the folder in
LARMOR_DATA_DIR or the default cache folder). Run from the repository root:

    uv run python tests/data/make_gibbs_ref.py
"""

import pathlib
import subprocess
import tempfile
from typing import cast

import nibabel as nib
import numpy as np

from larmor.data import fetch_cfin_multishell
from larmor.io import load_volume, save_volume
from larmor.types import Volume

OUT = pathlib.Path("tests/data/gibbs")
OUT.mkdir(exist_ok=True)

START = (0, 0, 8)
SIZE = (95, 96, 3)
VOLUMES = [0, 133]  # the b0 and the first b=1000 volume of CFIN's order

PHANTOM_SIZE = (64, 62)
FINE = 4  # the phantom is drawn on a grid this many times finer


def cfin_block() -> Volume:
    """The CFIN block, float32, with its affine."""
    block = tuple(slice(s, s + n) for s, n in zip(START, SIZE, strict=True))
    dwi = load_volume(fetch_cfin_multishell()["dwi"])
    data = np.asarray(dwi.data[(*block, VOLUMES)], dtype=np.float32)
    shift = np.eye(4)
    shift[:3, 3] = START
    return Volume(np.ascontiguousarray(data), dwi.affine @ shift)


def phantom_slice(centre_shift: float) -> np.ndarray:
    """One phantom slice: shapes on the fine grid, low-passed to PHANTOM_SIZE."""
    n_fine = [n * FINE for n in PHANTOM_SIZE]
    x, y = np.meshgrid(
        np.arange(n_fine[0]) / FINE, np.arange(n_fine[1]) / FINE, indexing="ij"
    )
    centre = (PHANTOM_SIZE[0] / 2 + centre_shift, PHANTOM_SIZE[1] / 2)
    ellipse = ((x - centre[0]) / 20.3) ** 2 + ((y - centre[1]) / 14.7) ** 2 <= 1
    square = (np.abs(x - centre[0] - 4.2) <= 6.1) & (np.abs(y - centre[1]) <= 5.3)
    fine = 100.0 * ellipse + 60.0 * square
    spectrum = np.fft.fftshift(np.fft.fft2(fine))
    # After fftshift the zero frequency is at index nf // 2.
    low = [
        slice(nf // 2 - n // 2, nf // 2 - n // 2 + n)
        for nf, n in zip(n_fine, PHANTOM_SIZE, strict=True)
    ]
    kept = np.fft.ifftshift(spectrum[low[0], low[1]])
    return np.real(np.fft.ifft2(kept)) / FINE**2


def phantom() -> Volume:
    """Two phantom slices, the second with the shapes shifted by a third of a voxel."""
    data = np.stack([phantom_slice(0.0), phantom_slice(1 / 3)], axis=-1)
    return Volume(data.astype(np.float32), np.eye(4))


def mrdegibbs(path: pathlib.Path, tmp: pathlib.Path) -> np.ndarray:
    """mrdegibbs's result for the image at path, as float32."""
    out = tmp / f"{path.stem}_degibbs.nii"
    subprocess.run(
        ["mrdegibbs", str(path), str(out), "-nthreads", "0", "-quiet", "-force"],
        check=True,
    )
    result = cast(nib.Nifti1Image, nib.load(out))
    return np.asarray(result.get_fdata(), dtype=np.float32)


save_volume(cfin_block(), OUT / "dwi.nii.gz")
save_volume(phantom(), OUT / "phantom.nii.gz")

arrays: dict[str, np.ndarray] = {}
with tempfile.TemporaryDirectory() as tmp:
    for name in ("dwi", "phantom"):
        arrays[name] = mrdegibbs(OUT / f"{name}.nii.gz", pathlib.Path(tmp))

version = subprocess.run(["mrdegibbs", "-version"], capture_output=True, text=True)
arrays["mrtrix_version"] = np.array(version.stdout.splitlines()[0])
ref = OUT / "mrdegibbs_ref.npz"
np.savez_compressed(ref, **arrays)  # type: ignore[arg-type]
for path in sorted(OUT.iterdir()):
    print(path, path.stat().st_size, "bytes")  # noqa: T201 (script output)
print(arrays["mrtrix_version"])  # noqa: T201 (script output)
