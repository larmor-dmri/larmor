"""`lar fod --method MSMT-CSD` on the multi-shell CFIN block of tests/data/dki."""

from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner, Result

import larmor.sd.msmt_csd as msmt_csd
from larmor.io import load_volume, read_mrtrix_response, save_volume
from larmor.main import app
from larmor.sd import TissueResponses, build_shell_response
from larmor.types import GradientTable, Volume

DATA = Path(__file__).parents[1] / "data"
DKI, MSMT = DATA / "dki", DATA / "msmt"
GRADIENTS = ("--bvals", str(DKI / "bvals"), "--bvecs", str(DKI / "bvecs"))
RESPONSES = tuple(
    item
    for tissue in ("wm", "gm", "csf")
    for item in (f"--{tissue}-response", str(MSMT / f"{tissue}.txt"))
)


def _run(*args: str) -> Result:
    return CliRunner().invoke(app, list(args), env={"TERM": "dumb"})


def test_writes_the_wm_fod_and_three_fractions(tmp_path: Path) -> None:
    out = tmp_path / "msmt"
    result = _run(
        "fod",
        str(DKI / "dwi.nii.gz"),
        str(DKI / "mask.nii.gz"),
        str(out),
        *GRADIENTS,
        "--method",
        "msmt-csd",
        *RESPONSES,
        "--nt",
        "1",
    )
    assert result.exit_code == 0, result.output
    mask = load_volume(DKI / "mask.nii.gz").data > 0
    fod = load_volume(tmp_path / "msmt.nii.gz").data
    fractions = load_volume(tmp_path / "msmt_tissue_fraction.nii.gz").data
    assert fod.shape == (*mask.shape, 45) and fractions.shape == (*mask.shape, 3)
    np.testing.assert_allclose(fractions[mask].sum(axis=1), 1.0, atol=1e-6)


def test_without_responses_the_three_estimated_ones_are_fitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The block is too small for the estimate's eroded mask, so the estimator (tested
    # on its own) returns the stored dhollander responses; this checks the wiring.
    seen: list[np.ndarray] = []

    def estimate(
        volume: Volume, gtab: GradientTable, mask: np.ndarray, **_: object
    ) -> TissueResponses:
        seen.append(mask)
        wm, gm, csf = (
            build_shell_response(read_mrtrix_response(MSMT / f"{t}.txt"), gtab)
            for t in ("wm", "gm", "csf")
        )
        empty = np.zeros(mask.shape, dtype=bool)
        return TissueResponses(wm, gm, csf, empty, empty, empty)

    monkeypatch.setattr(msmt_csd, "estimate_tissue_responses", estimate)
    result = _run(
        "fod", str(DKI / "dwi.nii.gz"), str(DKI / "mask.nii.gz"),
        str(tmp_path / "auto"), *GRADIENTS, "--method", "msmt-csd", "--nt", "1",
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    mask = load_volume(DKI / "mask.nii.gz").data > 0
    assert len(seen) == 1 and (seen[0] == mask).all()
    fod = load_volume(tmp_path / "auto.nii.gz").data
    fractions = load_volume(tmp_path / "auto_tissue_fraction.nii.gz").data
    assert fod.shape == (*mask.shape, 45) and fractions.shape == (*mask.shape, 3)
    given = tmp_path / "given"
    _run(
        "fod", str(DKI / "dwi.nii.gz"), str(DKI / "mask.nii.gz"), str(given),
        *GRADIENTS, "--method", "msmt-csd", *RESPONSES, "--nt", "1",
    )  # fmt: skip
    # The estimated responses come in the order WM, GM, CSF, as given ones do.
    np.testing.assert_array_equal(load_volume(tmp_path / "given.nii.gz").data, fod)


def test_a_wm_response_alone_gives_no_fractions_and_nan_stays_nan(
    tmp_path: Path,
) -> None:
    dwi = load_volume(DKI / "dwi.nii.gz")
    mask = load_volume(DKI / "mask.nii.gz").data > 0
    data = np.array(dwi.data)
    voxel = tuple(np.argwhere(mask)[0])
    data[voxel][3] = np.nan
    volume = tmp_path / "dwi.nii.gz"
    save_volume(dwi.with_data(data), volume)
    result = _run(
        "fod", str(volume), str(DKI / "mask.nii.gz"), str(tmp_path / "wm"),
        *GRADIENTS, "--method", "msmt-csd", "--wm-response", str(MSMT / "wm.txt"),
        "--nt", "1",
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert not (tmp_path / "wm_tissue_fraction.nii.gz").exists()
    fod = load_volume(tmp_path / "wm.nii.gz").data
    assert np.isnan(fod[voxel]).all() and np.isfinite(fod[mask].sum(axis=1)).sum() == (
        mask.sum() - 1
    )


def test_responses_need_msmt_csd_and_a_wm_response(tmp_path: Path) -> None:
    out = str(tmp_path / "out")
    inputs = (str(DKI / "dwi.nii.gz"), str(DKI / "mask.nii.gz"), out)
    grl = _run("fod", *inputs, *GRADIENTS, "--wm-response", str(MSMT / "wm.txt"))
    assert grl.exit_code == 2 and "MSMT-CSD" in grl.output
    no_wm = _run(
        "fod",
        *inputs,
        *GRADIENTS,
        "--method",
        "msmt-csd",
        "--csf-response",
        str(MSMT / "csf.txt"),
    )
    assert no_wm.exit_code == 2 and "--wm-response" in no_wm.output


@pytest.mark.parametrize(
    "option",
    [
        ["--csf-rf", "none"],
        ["--solver", "drl"],
        ["--response", str(MSMT / "wm.txt")],
        ["--wm-response", "missing.txt"],
    ],
)
def test_refuses_options_it_does_not_use_and_missing_files(
    tmp_path: Path, option: list[str]
) -> None:
    result = _run(
        "fod", "missing.nii.gz", "m", str(tmp_path / "out"), *GRADIENTS,
        "--method", "msmt-csd", *option,
    )  # fmt: skip
    # A usage error, before the (missing) input is read.
    assert result.exit_code == 2, result.output
    assert option[0] in result.output and "missing.nii.gz" not in result.output
