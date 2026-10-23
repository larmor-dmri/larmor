import logging
import math
import os
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from typer.testing import CliRunner, Result

from larmor.io import load_volume, save_volume
from larmor.main import app
from larmor.sd import Peaks, convert_sh_basis, find_peaks
from larmor.sd.sh import build_basis_matrix, get_sh_indices
from larmor.types import Volume

AFFINE = np.array(
    [[-2.0, 0.2, 0, 30], [0.1, -2.0, 0, 40], [0, 0, 2.5, -10], [0, 0, 0, 1]]
)
SHAPE = (3, 2, 2)
LMAX = 8
X, Y, Z = np.eye(3)
OBLIQUE = np.array([0.3, 0.5, -1.0]) / math.sqrt(1.34)
# Output values are float32: directions and amplitudes keep ~1e-7 relative.
RTOL = 1e-6


def _run(*args: str | Path) -> Result:
    return CliRunner().invoke(app, ["peaks", *map(str, args)], env={"TERM": "dumb"})


def _fibre(direction: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    ls, _ = get_sh_indices(LMAX)
    theta = math.acos(float(direction[2]))
    phi = math.atan2(float(direction[1]), float(direction[0]))
    b = build_basis_matrix(np.array([theta]), np.array([phi]), LMAX)
    sh: npt.NDArray[np.float64] = b[0] * np.exp(-0.02 * ls * (ls + 1))
    return sh


def _fod(*fibres: tuple[float, npt.NDArray[np.float64]]) -> npt.NDArray[np.float64]:
    sh: npt.NDArray[np.float64] = np.zeros(len(get_sh_indices(LMAX)[0]))
    for weight, direction in fibres:
        sh = sh + weight * _fibre(direction)
    return sh


def _coeffs() -> npt.NDArray[np.float32]:
    """One FOD per voxel: single fibres, crossings of two and three, and zero."""
    voxels = [
        _fod((1.0, X)),
        _fod((1.0, X), (0.8, Y)),
        _fod((1.0, X), (0.9, Y), (0.7, Z)),
        np.zeros(len(get_sh_indices(LMAX)[0])),
        _fod((1.0, OBLIQUE)),
        _fod(
            (0.6, OBLIQUE),
            (1.0, np.cross(OBLIQUE, X) / np.linalg.norm(np.cross(OBLIQUE, X))),
        ),
    ]
    flat = [voxels[i % len(voxels)] * (1 + 0.1 * i) for i in range(math.prod(SHAPE))]
    return np.array(flat).reshape(*SHAPE, -1).astype(np.float32)


def _expected(path: Path, **kwargs: object) -> Peaks:
    return find_peaks(load_volume(path).as_float(dtype=np.float64), **kwargs)  # type: ignore[arg-type]


def _decode(
    data: npt.NDArray[np.floating],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    vectors = data.astype(np.float64).reshape(*data.shape[:3], -1, 3)
    amplitudes = np.linalg.norm(vectors, axis=-1)
    return vectors / amplitudes[..., None], amplitudes


def _assert_matches(data: npt.NDArray[np.floating], expected: Peaks) -> None:
    directions, amplitudes = _decode(data)
    np.testing.assert_allclose(amplitudes, expected.amplitudes, rtol=RTOL, atol=0)
    np.testing.assert_allclose(directions, expected.directions, rtol=0, atol=RTOL)


@pytest.fixture
def fod(tmp_path: Path) -> Path:
    path = tmp_path / "fod.nii.gz"
    save_volume(Volume(_coeffs(), AFFINE), path)
    return path


def test_output_has_three_volumes_per_peak_as_float32_on_the_input_grid(
    tmp_path: Path, fod: Path
) -> None:
    out = tmp_path / "peaks.nii.gz"
    result = _run(fod, out)

    assert result.exit_code == 0, result.output
    written = load_volume(out)
    assert written.data.shape == (*SHAPE, 9)
    assert written.data.dtype == np.float32
    np.testing.assert_allclose(written.affine, AFFINE, rtol=0, atol=1e-5)


def test_each_triple_is_a_peak_direction_scaled_by_its_amplitude(
    tmp_path: Path, fod: Path
) -> None:
    out = tmp_path / "peaks.nii.gz"
    result = _run(fod, out)

    assert result.exit_code == 0, result.output
    expected = _expected(fod)
    # The test volume must exercise one, two and three peaks and empty voxels.
    counts = set(np.sum(~np.isnan(expected.amplitudes), axis=-1).ravel().tolist())
    assert counts == {0, 1, 2, 3}
    _assert_matches(load_volume(out).data, expected)


def test_missing_peaks_are_nan_in_all_three_components(
    tmp_path: Path, fod: Path
) -> None:
    out = tmp_path / "peaks.nii.gz"
    result = _run(fod, out)

    assert result.exit_code == 0, result.output
    triples = load_volume(out).data.reshape(*SHAPE, 3, 3)
    missing = np.isnan(_expected(fod).amplitudes)
    assert np.all(np.isnan(triples[missing]))
    assert not np.any(np.isnan(triples[~missing]))
    # The zero FOD is voxel 3 in C order.
    assert np.all(np.isnan(triples.reshape(-1, 3, 3)[3]))


def _in_full_layout(coeffs: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    ls, ms = get_sh_indices(LMAX)
    full = np.zeros((*coeffs.shape[:-1], (LMAX + 1) ** 2), dtype=coeffs.dtype)
    full[..., ls * ls + ls + ms] = coeffs
    return full


@pytest.mark.parametrize(
    ("asymmetric", "num", "volumes"),
    [(False, None, 9), (True, None, 18), (True, 2, 6)],
)
def test_asymmetric_fod_is_read_with_one_warning_and_six_peaks_by_default(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    asymmetric: bool,
    num: int | None,
    volumes: int,
) -> None:
    coeffs = _in_full_layout(_coeffs()) if asymmetric else _coeffs()
    source, out = tmp_path / "fod.nii.gz", tmp_path / "peaks.nii.gz"
    save_volume(Volume(coeffs, AFFINE), source)

    result = _run(source, out, *([] if num is None else ["--num", str(num)]))

    assert result.exit_code == 0, result.output
    data = load_volume(out).data
    assert data.shape == (*SHAPE, volumes)
    _assert_matches(data, _expected(source, num_peaks=volumes // 3))
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    message = (
        f"{source} has 81 coefficients: read as an asymmetric FOD (all degrees, "
        "lmax 8)."
    )
    assert warnings == ([message] if asymmetric else [])


@pytest.mark.parametrize("num", [1, 5])
def test_num_sets_the_number_of_peaks_and_keeps_the_largest_first(
    tmp_path: Path, fod: Path, num: int
) -> None:
    default, out = tmp_path / "default.nii", tmp_path / "out.nii"
    assert _run(fod, default).exit_code == 0
    result = _run(fod, out, "--num", str(num))

    assert result.exit_code == 0, result.output
    data = load_volume(out).data
    assert data.shape == (*SHAPE, 3 * num)
    _assert_matches(data, _expected(fod, num_peaks=num))
    np.testing.assert_array_equal(data[..., :3], load_volume(default).data[..., :3])


def test_outside_the_mask_is_nan_and_inside_is_the_unmasked_result(
    tmp_path: Path, fod: Path
) -> None:
    mask = np.zeros(SHAPE, dtype=np.uint8)
    mask[0, :, 1] = 1
    mask[2, 1, 0] = 1
    mask_path = tmp_path / "mask.nii.gz"
    save_volume(Volume(mask, AFFINE), mask_path)
    full, masked = tmp_path / "full.nii", tmp_path / "masked.nii"
    assert _run(fod, full).exit_code == 0
    result = _run(fod, masked, "--mask", mask_path)

    assert result.exit_code == 0, result.output
    data = load_volume(masked).data
    inside = mask.astype(bool)
    assert np.all(np.isnan(data[~inside]))
    np.testing.assert_array_equal(data[inside], load_volume(full).data[inside])


def test_mask_of_the_wrong_shape_is_a_one_line_error(tmp_path: Path, fod: Path) -> None:
    mask_path, out = tmp_path / "mask.nii.gz", tmp_path / "out.nii"
    save_volume(Volume(np.ones((3, 2, 3), dtype=np.uint8), AFFINE), mask_path)
    result = _run(fod, out, "--mask", mask_path)

    assert result.exit_code == 1
    assert result.output.startswith("ERROR: ")
    assert len(result.output.strip().splitlines()) == 1
    assert "mask" in result.output
    assert not out.exists()


@pytest.mark.parametrize("basis", ["descoteaux", "leemans"])
def test_fod_in_another_basis_gives_the_same_peaks(
    tmp_path: Path, fod: Path, basis: str
) -> None:
    other = tmp_path / f"{basis}.nii.gz"
    coeffs = convert_sh_basis(_coeffs().astype(np.float64), "tournier", basis)
    save_volume(Volume(coeffs, AFFINE), other)
    reference, out = tmp_path / "reference.nii", tmp_path / "out.nii"
    assert _run(fod, reference).exit_code == 0
    result = _run(other, out, "--basis", basis)

    assert result.exit_code == 0, result.output
    # The float32 input is rounded differently in each basis.
    directions, amplitudes = _decode(load_volume(out).data)
    ref_directions, ref_amplitudes = _decode(load_volume(reference).data)
    np.testing.assert_allclose(amplitudes, ref_amplitudes, rtol=1e-5, atol=0)
    np.testing.assert_allclose(directions, ref_directions, rtol=0, atol=1e-5)


def test_unknown_basis_is_a_usage_error_naming_the_choices(tmp_path: Path) -> None:
    result = _run(
        tmp_path / "missing.nii.gz", tmp_path / "out.nii", "--basis", "mrtrix"
    )

    assert result.exit_code == 2
    assert "mrtrix" in result.output
    for choice in ("tournier", "descoteaux", "leemans"):
        assert choice in result.output


@pytest.fixture
def weak_second_fibre(tmp_path: Path) -> Path:
    # Weights 1 and 0.05 at 90 degrees barely overlap: the weak peak is ~5%.
    path = tmp_path / "weak.nii"
    sh = _fod((1.0, X), (0.05, Y))
    save_volume(Volume(np.tile(sh, (*SHAPE, 1)), AFFINE), path)
    return path


def _n_found(path: Path) -> set[int]:
    triples = load_volume(path).data.reshape(*SHAPE, -1, 3)
    counts = np.sum(~np.isnan(triples[..., 0]), axis=-1)
    return set(counts.ravel().tolist())


@pytest.mark.parametrize(
    ("args", "n"), [([], 1), (["--relative-threshold", "0.04"], 2)]
)
def test_relative_threshold_drops_peaks_below_a_fraction_of_the_largest(
    tmp_path: Path, weak_second_fibre: Path, args: list[str], n: int
) -> None:
    out = tmp_path / "out.nii"
    result = _run(weak_second_fibre, out, *args)

    assert result.exit_code == 0, result.output
    assert _n_found(out) == {n}


@pytest.mark.parametrize(
    ("args", "n"),
    [([], 1), (["--fod-method", "mfod"], 1), (["--fod-method", "csd"], 2)],
)
def test_fod_method_sets_the_default_relative_threshold(
    tmp_path: Path, args: list[str], n: int
) -> None:
    # A second fibre at about 15% of the first: above the 0.1 of CSD and MSMT-CSD
    # FODs, below the 0.2 of GRL and mFOD ones.
    path, out = tmp_path / "fod.nii", tmp_path / "out.nii"
    sh = _fod((1.0, X), (0.15, Y))
    save_volume(Volume(np.tile(sh, (*SHAPE, 1)), AFFINE), path)
    result = _run(path, out, *args)

    assert result.exit_code == 0, result.output
    assert _n_found(out) == {n}


@pytest.mark.parametrize(("factor", "n"), [(0.9, 2), (1.1, 1)])
def test_threshold_drops_peaks_below_an_amplitude(
    tmp_path: Path, weak_second_fibre: Path, factor: float, n: int
) -> None:
    weak = float(
        _expected(weak_second_fibre, relative_threshold=0.0).amplitudes[0, 0, 0, 1]
    )
    out = tmp_path / "out.nii"
    result = _run(
        weak_second_fibre,
        out,
        "--relative-threshold",
        "0",
        "--threshold",
        str(factor * weak),
    )

    assert result.exit_code == 0, result.output
    assert _n_found(out) == {n}


@pytest.mark.parametrize("dtype", [np.int16, np.float32])
def test_scaled_input_gives_the_peaks_of_its_scaled_values(
    tmp_path: Path, dtype: type[np.generic]
) -> None:
    source, out = tmp_path / "scaled.nii", tmp_path / "out.nii"
    stored = (_coeffs() * 1000).astype(dtype)
    save_volume(Volume(stored, AFFINE, slope=0.002, intercept=0.0005), source)
    result = _run(source, out)

    assert result.exit_code == 0, result.output
    expected = _expected(source)
    raw = find_peaks(stored.astype(np.float64))
    # Otherwise the test could not tell scaled from raw values.
    assert not np.allclose(expected.amplitudes, raw.amplitudes, equal_nan=True)
    _assert_matches(load_volume(out).data, expected)


def test_existing_output_needs_force(tmp_path: Path, fod: Path) -> None:
    out = tmp_path / "out.nii.gz"
    out.write_bytes(b"old")

    refused = _run(fod, out)
    assert refused.exit_code == 1
    assert "exists" in refused.output
    assert out.read_bytes() == b"old"

    forced = _run(fod, out, "--force")
    assert forced.exit_code == 0, forced.output
    _assert_matches(load_volume(out).data, _expected(fod))


def test_existing_output_is_refused_before_reading(tmp_path: Path) -> None:
    out = tmp_path / "out.nii.gz"
    out.write_bytes(b"old")
    result = _run(tmp_path / "missing.nii.gz", out)

    assert result.exit_code == 1
    assert "exists" in result.output
    assert out.read_bytes() == b"old"


def test_3d_input_is_a_usage_error(tmp_path: Path) -> None:
    source, out = tmp_path / "map.nii.gz", tmp_path / "out.nii"
    save_volume(Volume(_coeffs()[..., 0], AFFINE), source)
    result = _run(source, out)

    assert result.exit_code == 2
    assert "4D" in result.output
    assert not out.exists()


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--num", "0"),
        ("--num", "-1"),
        ("--nt", "0"),
        ("--nt", "-2"),
        ("--relative-threshold", "1.5"),
        ("--relative-threshold", "-0.1"),
        ("--relative-threshold", "nan"),
    ],
)
def test_invalid_count_is_a_usage_error(
    tmp_path: Path, option: str, value: str
) -> None:
    result = _run(tmp_path / "missing.nii.gz", tmp_path / "out.nii", option, value)

    assert result.exit_code == 2
    assert option in result.output


@pytest.mark.parametrize("threads", [1, (os.cpu_count() or 1) + 5])
def test_thread_count_does_not_change_the_output(
    tmp_path: Path, fod: Path, threads: int
) -> None:
    default, out = tmp_path / "default.nii", tmp_path / "out.nii"
    assert _run(fod, default).exit_code == 0
    result = _run(fod, out, "--nt", str(threads))

    assert result.exit_code == 0, result.output
    np.testing.assert_array_equal(load_volume(out).data, load_volume(default).data)


@pytest.mark.parametrize("threads", [1, 3, -1])
def test_nt_is_passed_to_find_peaks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fod: Path, threads: int
) -> None:
    import larmor.sd

    calls: list[int] = []

    def recording(
        coeffs: npt.NDArray[np.float64],
        mask: npt.NDArray[np.bool_] | None = None,
        basis: str = "tournier",
        num_peaks: int = 3,
        relative_threshold: float = 0.1,
        threshold: float = 0.0,
        n_jobs: int = -1,
    ) -> Peaks:
        calls.append(n_jobs)
        return find_peaks(
            coeffs,
            mask=mask,
            basis=basis,
            num_peaks=num_peaks,
            relative_threshold=relative_threshold,
            threshold=threshold,
            n_jobs=n_jobs,
        )

    monkeypatch.setattr(larmor.sd, "find_peaks", recording)
    result = _run(fod, tmp_path / "out.nii", "--nt", str(threads))

    assert result.exit_code == 0, result.output
    assert calls == [threads]


@pytest.mark.parametrize("name", ["out.nii", "out.nii.gz", "out.mif"])
def test_every_output_format_holds_the_same_peaks(
    tmp_path: Path, fod: Path, name: str
) -> None:
    out = tmp_path / name
    result = _run(fod, out)

    assert result.exit_code == 0, result.output
    written = load_volume(out)
    _assert_matches(written.data, _expected(fod))
    np.testing.assert_allclose(written.affine, AFFINE, rtol=0, atol=1e-5)
