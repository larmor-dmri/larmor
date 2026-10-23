import logging
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from typer.testing import CliRunner, Result

from larmor.cli.options import SHBasis
from larmor.io import load_volume, save_volume
from larmor.main import app
from larmor.sd import convert_sh_basis
from larmor.sd.sh import BASES
from larmor.types import Volume

AFFINE = np.array(
    [[-2.0, 0.2, 0, 30], [0.1, -2.0, 0, 40], [0, 0, 2.5, -10], [0, 0, 0, 1]]
)


def _run(*args: str | Path) -> Result:
    return CliRunner().invoke(
        app, ["sh-convert", *map(str, args)], env={"TERM": "dumb"}
    )


def _coeffs(n: int) -> npt.NDArray[np.float32]:
    return np.random.default_rng(0).normal(size=(3, 4, 5, n)).astype(np.float32)


@pytest.fixture
def fod(tmp_path: Path) -> Path:
    """A 3x4x5 float32 FOD with lmax 4 (15 coefficients)."""
    path = tmp_path / "fod.nii.gz"
    save_volume(Volume(_coeffs(15), AFFINE), path)
    return path


def test_there_and_back_gives_the_input_and_keeps_the_affine(
    tmp_path: Path, fod: Path
) -> None:
    there, back = tmp_path / "leemans.nii.gz", tmp_path / "back.nii.gz"
    first = _run(fod, there, "--from", "tournier", "--to", "leemans")
    second = _run(there, back, "--from", "leemans", "--to", "tournier")

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    out = load_volume(back)
    np.testing.assert_array_equal(out.data, _coeffs(15))
    np.testing.assert_allclose(out.affine, AFFINE, rtol=0, atol=1e-5)


@pytest.mark.parametrize("dst", ["out.nii.gz", "out.mif"])
def test_output_is_the_library_conversion_of_the_input(
    tmp_path: Path, fod: Path, dst: str
) -> None:
    out = tmp_path / dst
    result = _run(fod, out, "--from", "descoteaux", "--to", "leemans")

    assert result.exit_code == 0, result.output
    expected = convert_sh_basis(_coeffs(15), "descoteaux", "leemans")
    written = load_volume(out)
    assert written.data.dtype == np.float32
    np.testing.assert_array_equal(written.data, expected)
    np.testing.assert_allclose(written.affine, AFFINE, rtol=0, atol=1e-5)


@pytest.mark.parametrize("dtype", [np.int16, np.float32])
def test_scaled_input_is_converted_from_its_scaled_values(
    tmp_path: Path, dtype: type[np.generic]
) -> None:
    # A sign flip of stored values with an intercept would give wrong values.
    stored = (_coeffs(15) * 100).astype(dtype)
    source = tmp_path / "scaled.nii"
    save_volume(Volume(stored, AFFINE, slope=0.01, intercept=-0.5), source)
    result = _run(
        source, tmp_path / "out.nii", "--from", "tournier", "--to", "descoteaux"
    )

    assert result.exit_code == 0, result.output
    expected = convert_sh_basis(
        load_volume(source).as_float(), "tournier", "descoteaux"
    )
    np.testing.assert_array_equal(
        load_volume(tmp_path / "out.nii").as_float(), expected
    )


def test_full_basis_converts_without_any_flag(tmp_path: Path) -> None:
    # 81 coefficients are lmax 8 with every degree; the count alone decides.
    source, there, back = (tmp_path / f"{n}.nii.gz" for n in ("full", "l", "back"))
    save_volume(Volume(_coeffs(81), AFFINE), source)
    first = _run(source, there, "--from", "tournier", "--to", "leemans")
    second = _run(there, back, "--from", "leemans", "--to", "tournier")

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    expected = convert_sh_basis(_coeffs(81), "tournier", "leemans")
    np.testing.assert_array_equal(load_volume(there).data, expected)
    np.testing.assert_array_equal(load_volume(back).data, _coeffs(81))


@pytest.mark.parametrize(("num_coeffs", "warned"), [(81, True), (45, False)])
def test_only_an_asymmetric_input_is_warned_about(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, num_coeffs: int, warned: bool
) -> None:
    source = tmp_path / "fod.nii.gz"
    save_volume(Volume(_coeffs(num_coeffs), AFFINE), source)
    result = _run(
        source, tmp_path / "out.nii.gz", "--from", "tournier", "--to", "leemans"
    )

    assert result.exit_code == 0, result.output
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    expected = f"{source} has 81 coefficients: read as an asymmetric FOD"
    assert warnings == ([f"{expected} (all degrees, lmax 8)."] if warned else [])


def test_full_option_is_gone(tmp_path: Path, fod: Path) -> None:
    out = tmp_path / "out.nii.gz"
    result = _run(fod, out, "--from", "tournier", "--to", "leemans", "--full")

    assert result.exit_code == 2
    assert "--full" in result.output
    assert not out.exists()


def test_3d_input_is_a_usage_error(tmp_path: Path) -> None:
    source = tmp_path / "map.nii.gz"
    save_volume(Volume(_coeffs(1)[..., 0], AFFINE), source)
    result = _run(
        source, tmp_path / "out.nii.gz", "--from", "tournier", "--to", "leemans"
    )

    assert result.exit_code == 2
    assert "4D" in result.output
    assert not (tmp_path / "out.nii.gz").exists()


def test_existing_output_needs_force(tmp_path: Path, fod: Path) -> None:
    out = tmp_path / "out.nii.gz"
    out.write_bytes(b"old")

    refused = _run(fod, out, "--from", "tournier", "--to", "leemans")
    assert refused.exit_code == 1
    assert "exists" in refused.output
    assert out.read_bytes() == b"old"

    forced = _run(fod, out, "--from", "tournier", "--to", "leemans", "--force")
    assert forced.exit_code == 0, forced.output
    expected = convert_sh_basis(_coeffs(15), "tournier", "leemans")
    np.testing.assert_array_equal(load_volume(out).data, expected)


def test_existing_output_is_refused_before_reading(tmp_path: Path) -> None:
    out = tmp_path / "out.nii.gz"
    out.write_bytes(b"old")
    result = _run(
        tmp_path / "missing.nii.gz", out, "--from", "tournier", "--to", "leemans"
    )

    assert result.exit_code == 1
    assert "exists" in result.output


@pytest.mark.parametrize(
    ("given", "missing"),
    [(["--from", "leemans"], "--to"), (["--to", "tournier"], "--from")],
)
def test_both_bases_are_required(
    tmp_path: Path, given: list[str], missing: str
) -> None:
    # The file does not record its basis: a default --from would silently leave
    # an MRIToolkit FOD in leemans when only --to tournier is given.
    result = _run(tmp_path / "missing.nii.gz", tmp_path / "out.nii.gz", *given)
    assert result.exit_code == 2
    assert missing in result.output


@pytest.mark.parametrize("option", ["--to", "--from"])
def test_unknown_basis_is_a_usage_error_naming_the_choices(
    tmp_path: Path, option: str
) -> None:
    args = ["--from", "tournier", "--to", "leemans", option, "mrtrix"]
    result = _run(tmp_path / "missing.nii.gz", tmp_path / "out.nii.gz", *args)

    assert result.exit_code == 2
    assert "mrtrix" in result.output
    assert "tournier" in result.output


def test_invalid_number_of_coefficients_is_a_one_line_error(tmp_path: Path) -> None:
    source = tmp_path / "ten.nii.gz"
    save_volume(Volume(_coeffs(10), AFFINE), source)
    result = _run(
        source, tmp_path / "out.nii.gz", "--from", "tournier", "--to", "leemans"
    )

    assert result.exit_code == 1
    assert result.output.startswith("ERROR: ")
    assert len(result.output.strip().splitlines()) == 1
    assert "10 SH coefficients fit no layout" in result.output
    assert not (tmp_path / "out.nii.gz").exists()


def test_float64_input_stays_float64(tmp_path: Path) -> None:
    # The conversion is exact: casting to float32 would be the only loss.
    source, out = tmp_path / "f64.nii", tmp_path / "out.nii"
    coeffs = _coeffs(15).astype(np.float64) / 3
    save_volume(Volume(coeffs, AFFINE), source)
    result = _run(source, out, "--from", "leemans", "--to", "tournier")

    assert result.exit_code == 0, result.output
    written = load_volume(out)
    assert written.data.dtype == np.float64
    np.testing.assert_array_equal(
        written.data, convert_sh_basis(coeffs, "leemans", "tournier")
    )


def test_basis_choices_are_the_library_bases() -> None:
    assert tuple(basis.value for basis in SHBasis) == BASES
