from pathlib import Path

import numpy as np
from typer.testing import CliRunner, Result

from larmor.io import load_volume, save_volume
from larmor.main import app
from larmor.sd import filter_afod
from larmor.types import Volume

AFFINE = np.diag([-1.25, 1.25, 1.25, 1.0])


def _run(*args: str | Path) -> Result:
    return CliRunner().invoke(app, ["afod", *map(str, args)], env={"TERM": "dumb"})


def _fod(tmp_path: Path, n_coeffs: int = 45) -> tuple[Path, np.ndarray]:
    data = np.random.default_rng(0).standard_normal((5, 4, 3, n_coeffs))
    data = data.astype(np.float32)
    data[0] = 0
    path = tmp_path / "fod.nii.gz"
    save_volume(Volume(data, AFFINE), path)
    return path, data


def test_writes_what_the_library_gives(tmp_path: Path) -> None:
    fod, data = _fod(tmp_path)
    mask = np.ones(data.shape[:3], dtype=np.uint8)
    mask[1] = 0
    save_volume(Volume(mask, AFFINE), tmp_path / "mask.nii.gz")
    out = tmp_path / "afod.nii.gz"

    result = _run(fod, out, "-m", tmp_path / "mask.nii.gz", "--sigma", "2")

    assert result.exit_code == 0, result.output
    written = load_volume(out)
    expected = filter_afod(data, mask > 0, AFFINE, sigma=2.0)
    assert written.data.shape == (5, 4, 3, 81)
    np.testing.assert_allclose(written.data, expected, atol=1e-6)
    np.testing.assert_allclose(written.affine, AFFINE)


def test_without_a_mask_skips_voxels_with_an_empty_fod(tmp_path: Path) -> None:
    fod, data = _fod(tmp_path)
    out = tmp_path / "afod.nii.gz"

    assert _run(fod, out).exit_code == 0

    assert np.all(load_volume(out).data[0] == 0)
    assert np.any(load_volume(out).data[1:] != 0)


def test_an_asymmetric_fod_is_refused(tmp_path: Path) -> None:
    fod, _ = _fod(tmp_path, n_coeffs=81)

    result = _run(fod, tmp_path / "afod.nii.gz")

    assert result.exit_code == 2
    assert "already asymmetric" in result.output


def test_an_existing_output_needs_force(tmp_path: Path) -> None:
    fod, _ = _fod(tmp_path)
    out = tmp_path / "afod.nii.gz"
    out.write_text("")

    assert _run(fod, out).exit_code == 1
    assert _run(fod, out, "--force").exit_code == 0
