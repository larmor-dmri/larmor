"""MSMT-CSD against MRtrix3's dwi2fod msmt_csd on the CFIN block of tests/data/dki."""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from larmor.io import load_dwi, load_volume, read_mrtrix_response
from larmor.sd import MSMTCSDModel, ZonalSHRF, response_shells
from larmor.sd.sphere import DIRS_300, get_theta_phi

DATA = Path(__file__).parents[1] / "data"
DKI, MSMT = DATA / "dki", DATA / "msmt"

needs_mrtrix = pytest.mark.skipif(
    shutil.which("dwi2fod") is None, reason="MRtrix3 is not on PATH"
)


@needs_mrtrix
def test_matches_dwi2fod_msmt_csd_with_the_same_constraint_directions(
    tmp_path: Path,
) -> None:
    # MRtrix3 is given Larmor's 300 directions, its other settings are its defaults.
    theta, phi = get_theta_phi(DIRS_300)
    directions = tmp_path / "dirs.txt"
    np.savetxt(directions, np.column_stack([phi, theta]), fmt="%.17g")
    outputs = {t: tmp_path / f"{t}.nii" for t in ("wm", "gm", "csf")}
    subprocess.run(
        [
            "dwi2fod",
            "msmt_csd",
            str(DKI / "dwi.nii.gz"),
            str(MSMT / "wm.txt"),
            str(outputs["wm"]),
            str(MSMT / "gm.txt"),
            str(outputs["gm"]),
            str(MSMT / "csf.txt"),
            str(outputs["csf"]),
            "-fslgrad",
            str(DKI / "bvecs"),
            str(DKI / "bvals"),
            "-mask",
            str(DKI / "mask.nii.gz"),
            "-directions",
            str(directions),
            "-nthreads",
            "0",
            "-quiet",
        ],
        check=True,
    )
    volume, gtab = load_dwi(
        DKI / "dwi.nii.gz", bvals=DKI / "bvals", bvecs=DKI / "bvecs"
    )
    mask = load_volume(DKI / "mask.nii.gz").data > 0
    shells = response_shells(gtab)
    rfs = [
        ZonalSHRF(bvals=shells, coefficients=read_mrtrix_response(MSMT / f"{t}.txt"))
        for t in ("wm", "gm", "csf")
    ]
    fit = MSMTCSDModel(rfs).fit(volume, gtab, mask=mask, n_jobs=2)

    wm = load_volume(outputs["wm"]).data[mask]
    ours = fit.fods[0][mask]
    # Both add a ridge of 1e-10 of the normal matrix's largest diagonal entry; what
    # is left is MRtrix3's float32 output and its regularised dual (-neg_lambda),
    # which move a few coefficients by about 1e-5 of the largest.
    scale = np.abs(wm[:, 0]).max()
    np.testing.assert_allclose(ours, wm, rtol=0, atol=1e-4 * scale)
    shares = np.column_stack(
        [wm[:, 0], *(load_volume(outputs[t]).data[mask].ravel() for t in ("gm", "csf"))]
    )
    fractions = shares / shares.sum(axis=1, keepdims=True)
    assert fit.tissue_fractions is not None
    np.testing.assert_allclose(fit.tissue_fractions[mask], fractions, atol=1e-5)
