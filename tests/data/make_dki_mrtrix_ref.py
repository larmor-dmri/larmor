"""Generates MRtrix3's DKI fit of the test block in tests/data/dki/.

Runs `dwi2tensor -dkt` (MRtrix3) on `tests/data/dki/dwi.nii.gz` with its FSL
gradient files and mask, twice: ordinary least squares (`-ols -iter 0`) and
weighted least squares with weights from the predicted signal (`-ols -iter 10`).

On this noisy, unprocessed block a few voxels have not settled after MRtrix's
largest number of reweightings (10), so the WLS fit is also run with `-iter 9`
and `wls_settled` marks the voxels whose tensors changed by at most 1e-6
(relative to the largest element) between the two.

Run from the repository root:

    uv run python tests/data/make_dki_mrtrix_ref.py
"""

import pathlib
import subprocess
import tempfile
from typing import cast

import nibabel as nib
import numpy as np

DKI = pathlib.Path("tests/data/dki")
OUT = DKI / "mrtrix_dkt_ref.npz"

# MRtrix stores D11, D22, D33, D12, D13, D23; Larmor xx, xy, xz, yy, yz, zz.
DT_ORDER = [0, 3, 4, 1, 5, 2]
# MRtrix stores W1111, W2222, W3333, W1112, W1113, W1222, W1333, W2223, W2333,
# W1122, W1133, W2233, W1123, W1223, W1233; Larmor's order differs only in that
# W2223 comes before W1333 (W3331).
KT_ORDER = [0, 1, 2, 3, 4, 5, 7, 6, 8, 9, 10, 11, 12, 13, 14]
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
        kt_path = pathlib.Path(tmp) / f"kt_{name}.nii"
        subprocess.run(
            [
                "dwi2tensor",
                str(dwi_path),
                str(dt_path),
                "-dkt",
                str(kt_path),
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
        kt, kt_affine = load(kt_path)
        # The tensors must be on the input grid, voxel for voxel.
        if not (np.allclose(dt_affine, affine) and np.allclose(kt_affine, affine)):
            raise SystemExit("dwi2tensor wrote its output on another voxel layout.")
        arrays[f"dt_{name}"] = dt[mask][:, DT_ORDER].astype(np.float32)
        arrays[f"kt_{name}"] = kt[mask][:, KT_ORDER].astype(np.float32)


def change(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Largest change of each row, relative to its largest element."""
    a, b = a.astype(np.float64), b.astype(np.float64)
    return np.abs(a - b).max(axis=1) / np.abs(b).max(axis=1)


dt_change = change(arrays.pop("dt_wls9"), arrays["dt_wls"])
kt_change = change(arrays.pop("kt_wls9"), arrays["kt_wls"])
arrays["wls_settled"] = (dt_change <= 1e-6) & (kt_change <= 1e-6)
version = subprocess.run(["dwi2tensor", "-version"], capture_output=True, text=True)
arrays["mrtrix_version"] = np.array(version.stdout.splitlines()[0])
np.savez_compressed(OUT, **arrays)  # type: ignore[arg-type]
print("saved", OUT, OUT.stat().st_size, "bytes,", arrays["mrtrix_version"])  # noqa: T201 (script output)
print(int(arrays["wls_settled"].sum()), "of", int(mask.sum()), "voxels settled")  # noqa: T201 (script output)
