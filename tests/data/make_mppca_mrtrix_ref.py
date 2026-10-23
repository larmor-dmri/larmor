"""Generates MRtrix3's MP-PCA denoising of the test block in tests/data/dki/.

Runs `dwidenoise` (MRtrix3) on `tests/data/dki/dwi.nii.gz` with the estimator of
Veraart et al. (2016) (`-estimator Exp1`) in double precision, twice: on the whole
block with the default patch (7x7x7 for its 133 volumes), and with a 5x5x5 patch
(fewer voxels than volumes) inside a box mask `[2:6, 1:7, 3:8]`. Run from the
repository root:

    uv run python tests/data/make_mppca_mrtrix_ref.py
"""

import pathlib
import subprocess
import tempfile
from typing import cast

import nibabel as nib
import numpy as np

DKI = pathlib.Path("tests/data/dki")
OUT = DKI / "mrtrix_mppca_ref.npz"
BOX = (slice(2, 6), slice(1, 7), slice(3, 8))


def load(path: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    """Data and affine of a NIfTI file."""
    image = cast(nib.Nifti1Image, nib.load(path))
    return np.asarray(image.get_fdata()), image.affine


def dwidenoise(tmp: pathlib.Path, name: str, *options: str) -> tuple[np.ndarray, ...]:
    """The denoised image and noise map of one dwidenoise run."""
    denoised_path, noise_path = tmp / f"{name}.nii", tmp / f"{name}_noise.nii"
    subprocess.run(
        [
            "dwidenoise",
            str(dwi_path),
            str(denoised_path),
            "-noise",
            str(noise_path),
            "-estimator",
            "Exp1",
            "-datatype",
            "float64",
            *options,
            "-nthreads",
            "0",
            "-quiet",
        ],
        check=True,
    )
    denoised, denoised_affine = load(denoised_path)
    noise, noise_affine = load(noise_path)
    if not (np.allclose(denoised_affine, affine) and np.allclose(noise_affine, affine)):
        raise SystemExit("dwidenoise wrote its output on another voxel layout.")
    return denoised, noise


dwi_path = DKI / "dwi.nii.gz"
data, affine = load(dwi_path)
box = np.zeros(data.shape[:3], dtype=bool)
box[BOX] = True

arrays: dict[str, np.ndarray] = {"box": box}
with tempfile.TemporaryDirectory() as tmp_name:
    tmp = pathlib.Path(tmp_name)
    mask_path = tmp / "box_mask.nii"
    nib.save(nib.Nifti1Image(box.astype(np.uint8), affine), mask_path)
    denoised, noise = dwidenoise(tmp, "default")
    arrays["denoised"] = denoised.astype(np.float32)
    arrays["noise"] = noise.astype(np.float32)
    denoised, noise = dwidenoise(tmp, "box", "-extent", "5", "-mask", str(mask_path))
    if np.any(denoised[~box] != 0) or np.any(noise[~box] != 0):
        raise SystemExit("dwidenoise wrote values outside the mask.")
    arrays["box_denoised"] = denoised[box].astype(np.float32)
    arrays["box_noise"] = noise[box].astype(np.float32)

version = subprocess.run(["dwidenoise", "-version"], capture_output=True, text=True)
arrays["mrtrix_version"] = np.array(version.stdout.splitlines()[0])
np.savez_compressed(OUT, **arrays)  # type: ignore[arg-type]
print("saved", OUT, OUT.stat().st_size, "bytes,", arrays["mrtrix_version"])  # noqa: T201 (script output)
