from pathlib import Path

import numpy as np
from typer.testing import CliRunner, Result

from larmor.dwi import denoise_mppca
from larmor.io import load_dwi, load_volume
from larmor.main import app

DATA = Path(__file__).parents[1] / "data"
VOLUME, MASK = DATA / "dki" / "dwi.nii.gz", DATA / "dki" / "mask.nii.gz"


def _run(*args: str | Path) -> Result:
    return CliRunner().invoke(app, ["denoise", *map(str, args)], env={"TERM": "dumb"})


def test_denoise_writes_the_library_result_and_noise_map(tmp_path: Path) -> None:
    out, noise = tmp_path / "den.nii.gz", tmp_path / "noise.nii.gz"
    result = _run(
        VOLUME, out, "--noise", noise, "--mask", MASK, "--patch-size", "5", "--nt", "2"
    )
    assert result.exit_code == 0, result.output

    expected = denoise_mppca(
        load_volume(VOLUME), load_volume(MASK), patch_size=5, progress=False
    )
    np.testing.assert_array_equal(load_volume(out).data, expected.volume.data)
    np.testing.assert_array_equal(load_volume(noise).data, expected.noise)


def test_mif_output_keeps_the_gradient_table(tmp_path: Path) -> None:
    out = tmp_path / "den.mif"
    result = _run(DATA / "mif" / "dwi.mif", out)
    assert result.exit_code == 0, result.output

    _, gtab_in = load_dwi(DATA / "mif" / "dwi.mif")
    _, gtab_out = load_dwi(out)
    np.testing.assert_array_equal(gtab_out.bvals, gtab_in.bvals)
    np.testing.assert_allclose(gtab_out.bvecs, gtab_in.bvecs, atol=1e-12)
