"""Generates MRtrix3's DTI fit of the test block in tests/data/dki/.

Runs `dwi2tensor` (MRtrix3, without `-dkt`) on `tests/data/dki/dwi.nii.gz` with
its FSL gradient files and mask, twice: ordinary least squares (`-ols -iter 0`)
and weighted least squares with weights from the predicted signal
(`-ols -iter 10`).

On this noisy, unprocessed block a few voxels have not settled after MRtrix's
largest number of reweightings (10), so the WLS fit is also run with `-iter 9`
and `wls_settled` marks the voxels whose tensor changed by at most 1e-6
(relative to the largest element) between the two.

Run from the repository root:

    uv run python tests/data/make_dti_mrtrix_ref.py
"""

import pathlib
import subprocess
import tempfile
from typing import cast

import nibabel as nib
import numpy as np

DKI = pathlib.Path("tests/data/dki")
OUT = DKI / "mrtrix_dt_ref.npz"

# MRtrix stores D11, D22, D33, D12, D13, D23; Larmor xx, xy, xz, yy, yz, zz.
DT_ORDER = [0, 3, 4, 1, 5, 2]
ITERATIONS = {"ols": 0, "wls": 10, "wls9": 9}


def load(path: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    """Data and affine of a NIfTI file."""
    image = cast(nib.Nifti1Image, nib.load(path))
    return np.asarray(image.get_fdata()), image.affine


dwi_path = DKI / "dwi.nii.gz"
_, affine = load(dwi_path)
mask_data, _ = load(DKI / "mask.nii.gz")
mask = mask_data > 0

arrays: dict[str, np.ndarray] = {"mask": mask}
with tempfile.TemporaryDirectory() as tmp:
    for name, iterations in ITERATIONS.items():
        dt_path = pathlib.Path(tmp) / f"dt_{name}.nii"
        subprocess.run(
            [
                "dwi2tensor",
                str(dwi_path),
                str(dt_path),
                "-fslgrad",
                str(DKI / "bvecs"),
                str(DKI / "bvals"),
                "-mask",
                str(DKI / "mask.nii.gz"),
                "-ols",
                "-iter",
                str(iterations),
                "-nthreads",
                "0",
                "-quiet",
            ],
            check=True,
        )
        dt, dt_affine = load(dt_path)
        # The tensors must be on the input grid, voxel for voxel.
        if not np.allclose(dt_affine, affine):
            raise SystemExit("dwi2tensor wrote its output on another voxel layout.")
        arrays[f"dt_{name}"] = dt[mask][:, DT_ORDER].astype(np.float32)


def change(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Largest change of each row, relative to its largest element."""
    a, b = a.astype(np.float64), b.astype(np.float64)
    return np.abs(a - b).max(axis=1) / np.abs(b).max(axis=1)


arrays["wls_settled"] = change(arrays.pop("dt_wls9"), arrays["dt_wls"]) <= 1e-6
version = subprocess.run(["dwi2tensor", "-version"], capture_output=True, text=True)
arrays["mrtrix_version"] = np.array(version.stdout.splitlines()[0])
np.savez_compressed(OUT, **arrays)  # type: ignore[arg-type]
print("saved", OUT, OUT.stat().st_size, "bytes,", arrays["mrtrix_version"])  # noqa: T201 (script output)
print(int(arrays["wls_settled"].sum()), "of", int(mask.sum()), "voxels settled")  # noqa: T201 (script output)
