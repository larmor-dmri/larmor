from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
import pytest
import typer
from typer.testing import CliRunner, Result

import larmor.sd
from larmor._log import LarmorLogger
from larmor.cli.command import larmor_command
from larmor.cli.fod import Method
from larmor.dwi import DTIModel
from larmor.io import load_dwi, load_volume, read_bvals_bvecs, save_volume
from larmor.main import app
from larmor.sd import DRL, L2NNLS, Lasso, SDMethod
from larmor.types import Volume

COMMANDS = ["dti", "mask", "fod"]
# Commands that need a multi-shell DWI, which the single-shell fixtures here are
# not; only their checks before any work are tested in this file.
MULTI_SHELL_COMMANDS = ["dki", "noddi"]
# Gradient options for runs whose inputs do not exist.
NO_GRADIENTS = ("--bvals", "a", "--bvecs", "b")


def _read(path: Path) -> npt.NDArray[np.floating]:
    """The values of a volume a command wrote."""
    return load_volume(path).as_float()


def _run(*args: str) -> Result:
    # Avoid coloring for CI.
    return CliRunner().invoke(app, list(args), env={"TERM": "dumb"})


@pytest.fixture
def dwi_files(tmp_path: Path) -> tuple[str, str, list[str]]:
    """A small DWI (a bright cube in a dark background), a mask inside the cube,
    and the gradient options. The cube is large enough to survive the erosion of
    the automatic brain mask; the mask is smaller, to keep the fits fast."""
    bvals = np.array([0.0, 1000, 1000, 1000, 1000, 1000, 1000])
    bvecs = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]],
        dtype=float,
    )
    bvecs[1:] /= np.linalg.norm(bvecs[1:], axis=1, keepdims=True)

    rng = np.random.default_rng(0)
    data = rng.uniform(5.0, 15.0, (12, 12, 12, len(bvals))).astype(np.float32)
    data[2:10, 2:10, 2:10] = 1000.0 * np.exp(-bvals * 0.8e-3)

    cube = np.zeros(data.shape[:3], dtype=np.uint8)
    cube[3:9, 3:9, 3:9] = 1

    volume, mask = tmp_path / "dwi.nii.gz", tmp_path / "cube.nii.gz"
    save_volume(Volume(data, np.eye(4)), volume)
    save_volume(Volume(cube, np.eye(4)), mask)
    np.savetxt(tmp_path / "bvals", bvals[np.newaxis])
    np.savetxt(tmp_path / "bvecs", bvecs.T)
    gradients = ["--bvals", str(tmp_path / "bvals"), "--bvecs", str(tmp_path / "bvecs")]
    return str(volume), str(mask), gradients


@pytest.mark.parametrize("command", [*COMMANDS, *MULTI_SHELL_COMMANDS])
def test_every_command_has_the_shared_flags(command: str) -> None:
    result = _run(command, "--help")

    assert result.exit_code == 0
    for flag in ("--force", "--quiet", "--info", "--debug"):
        assert flag in result.output


@pytest.mark.parametrize("command", ["fod", "dti", "dki", "noddi"])
def test_fitting_commands_have_the_threads_flag(command: str) -> None:
    assert "--nt" in _run(command, "--help").output


def test_error_is_one_line_and_exit_code_1(tmp_path: Path) -> None:
    result = _run("mask", "missing.nii.gz", str(tmp_path / "m.nii"), *NO_GRADIENTS)

    assert result.exit_code == 1
    assert result.output.startswith("ERROR: ")
    assert "missing.nii.gz" in result.output
    assert len(result.output.strip().splitlines()) == 1


def test_debug_shows_the_traceback(tmp_path: Path) -> None:
    out = str(tmp_path / "m.nii")
    result = _run("mask", "missing.nii.gz", out, *NO_GRADIENTS, "--debug")

    assert result.exit_code == 1
    assert "ERROR: " in result.output
    assert "Traceback" in result.output


def _log_every_level() -> None:
    LarmorLogger.debug("debug-message")
    LarmorLogger.info("info-message")
    LarmorLogger.warning("warning-message")
    LarmorLogger.error("error-message")


# A throwaway app with one command that logs at every level, so each flag's effect is
# checked directly on the wrapper instead of on what a real command happens to log.
_levels_app = typer.Typer()
_levels_app.command()(larmor_command(_log_every_level))


def _shown_levels(*flags: str) -> set[str]:
    result = CliRunner().invoke(_levels_app, list(flags), env={"TERM": "dumb"})
    assert result.exit_code == 0, result.output
    levels = ("debug", "info", "warning", "error")
    return {level for level in levels if f"{level}-message" in result.output}


@pytest.mark.parametrize(
    ("flags", "shown"),
    [
        ([], {"warning", "error"}),
        (["--quiet"], {"error"}),
        (["--info"], {"info", "warning", "error"}),
        (["--debug"], {"debug", "info", "warning", "error"}),
        (["--quiet", "--info"], {"info", "warning", "error"}),  # most verbose wins
        (["--info", "--debug"], {"debug", "info", "warning", "error"}),
    ],
)
def test_verbosity_flags(flags: list[str], shown: set[str]) -> None:
    assert _shown_levels(*flags) == shown


def test_verbosity_does_not_leak_between_runs() -> None:
    _shown_levels("--debug")
    assert _shown_levels() == {"warning", "error"}


def test_mask_writes_the_mask(
    tmp_path: Path, dwi_files: tuple[str, str, list[str]]
) -> None:
    volume, _, gradients = dwi_files
    out = tmp_path / "mask.nii.gz"
    result = _run("mask", volume, str(out), *gradients)

    assert result.exit_code == 0, result.output
    written = _read(out)
    assert written.shape == (12, 12, 12)
    assert set(np.unique(written)) <= {0, 1} and written.any()


def test_existing_output_is_kept_without_force(
    tmp_path: Path, dwi_files: tuple[str, str, list[str]]
) -> None:
    volume, _, gradients = dwi_files
    out = tmp_path / "mask.nii.gz"
    out.write_bytes(b"old")
    result = _run("mask", volume, str(out), *gradients)

    assert result.exit_code == 1
    assert "exists" in result.output
    assert out.read_bytes() == b"old"


def test_existing_output_is_replaced_with_force(
    tmp_path: Path, dwi_files: tuple[str, str, list[str]]
) -> None:
    volume, _, gradients = dwi_files
    out = tmp_path / "mask.nii.gz"
    out.write_bytes(b"old")
    result = _run("mask", volume, str(out), *gradients, "--force")

    assert result.exit_code == 0, result.output
    assert out.read_bytes() != b"old"


def test_existing_data_file_of_a_mih_output_is_kept_without_force(
    tmp_path: Path, dwi_files: tuple[str, str, list[str]]
) -> None:
    # A .mih header keeps its data in a .dat, which would be overwritten too.
    volume, _, gradients = dwi_files
    data_file = tmp_path / "mask.dat"
    data_file.write_bytes(b"old")
    result = _run("mask", volume, str(tmp_path / "mask.mih"), *gradients)

    assert result.exit_code == 1
    assert "mask.dat" in result.output
    assert data_file.read_bytes() == b"old"


def test_outputs_keep_the_grid_of_the_input(
    tmp_path: Path, dwi_files: tuple[str, str, list[str]]
) -> None:
    # Nothing is reoriented on loading: an LPS input gives LPS outputs.
    volume, _, gradients = dwi_files
    lps = tmp_path / "lps.nii.gz"
    affine = np.diag([-2.0, -2.0, 2.0, 1.0])
    affine[:3, 3] = (30.0, 40.0, -5.0)
    save_volume(Volume(load_volume(volume).data, affine), lps)
    out = tmp_path / "mask.nii.gz"

    result = _run("mask", str(lps), str(out), *gradients)

    assert result.exit_code == 0, result.output
    assert load_volume(out).grid.affine.tolist() == affine.tolist()
    assert load_volume(out).orientation == ("L", "P", "S")


@pytest.mark.parametrize("command", ["dti", "fod"])
def test_mask_in_another_orientation_than_the_dwi_is_reoriented(
    tmp_path: Path, dwi_files: tuple[str, str, list[str]], command: str
) -> None:
    # The DWI is stored LPS and the mask RAS: the same voxels, other axes.
    volume, mask, gradients = dwi_files
    dwi = Volume(load_volume(volume).data, np.diag([-1.0, -1.0, 1.0, 1.0]))
    save_volume(dwi, tmp_path / "lps_dwi.nii.gz")
    cube = load_volume(mask)
    ras_mask = Volume(cube.data, np.diag([-1.0, -1.0, 1.0, 1.0]))
    ras_mask.to_closest_canonical()
    save_volume(ras_mask, tmp_path / "ras_mask.nii.gz")

    lps_dwi, ras = str(tmp_path / "lps_dwi.nii.gz"), str(tmp_path / "ras_mask.nii.gz")
    args = (
        [command, lps_dwi, str(tmp_path / "out"), *gradients, "--mask", ras]
        if command == "dti"
        else [command, lps_dwi, ras, str(tmp_path / "out"), *gradients]
        + ["--wm-rf", "1.7e-3", "0.2e-3", "0.2e-3"]
    )
    result = _run(*args)

    assert result.exit_code == 0, result.output
    assert "stored RAS, the volume LPS; reorienting" in result.output


# The inputs below do not exist: the commands must refuse because of the existing
# output before they try to read anything.


def test_dti_refuses_before_reading_inputs(tmp_path: Path) -> None:
    (tmp_path / "out_fa.nii.gz").write_bytes(b"old")
    result = _run("dti", "missing.nii.gz", str(tmp_path / "out"), *NO_GRADIENTS)

    assert result.exit_code == 1
    assert "out_fa.nii.gz" in result.output and "exists" in result.output


def test_fod_refuses_before_reading_inputs(tmp_path: Path) -> None:
    (tmp_path / "fod.nii.gz").write_bytes(b"old")
    result = _run("fod", "missing.nii.gz", "m", str(tmp_path / "fod"), *NO_GRADIENTS)

    assert result.exit_code == 1
    assert "fod.nii.gz" in result.output and "exists" in result.output


# fod writes a fractions file exactly when at least one isotropic response function
# (GM or CSF) is used, so only then may an existing one stop it.
@pytest.mark.parametrize(
    ("gm_rf", "csf_rf", "refuses"),
    [
        ("none", "none", False),
        ("none", "3.0e-3", True),
        ("0.7e-3", "none", True),
        ("0.7e-3", "3.0e-3", True),
        ("None", "NONE", False),  # "none" in any case
    ],
)
def test_fod_checks_the_fractions_file_only_when_it_writes_one(
    tmp_path: Path, gm_rf: str, csf_rf: str, refuses: bool
) -> None:
    (tmp_path / "fod_tissue_fraction.nii.gz").write_bytes(b"old")
    result = _run(
        "fod", "missing.nii.gz", "m", str(tmp_path / "fod"), *NO_GRADIENTS,
        "--gm-rf", gm_rf, "--csf-rf", csf_rf,
    )  # fmt: skip

    assert result.exit_code == 1
    # Refusing names the fractions file; otherwise the run gets as far as reading
    # its (missing) input.
    assert ("tissue_fraction.nii.gz exists" in result.output) == refuses
    assert ("missing.nii.gz" in result.output) != refuses


@pytest.mark.parametrize(
    "bad",
    [
        ["--gm-rf", "abc"],
        ["--csf-rf", ""],
        ["--wm-rf", "1.7e-3", "0.2e-3"],  # needs three eigenvalues
        ["--wm-rf", "1.7e-3,0.2e-3,0.2e-3"],  # space-separated, not commas
        ["--mk", "abc"],
    ],
)
def test_fod_rejects_malformed_response_options(tmp_path: Path, bad: list[str]) -> None:
    out = str(tmp_path / "fod")
    result = _run("fod", "missing.nii.gz", "m", out, *NO_GRADIENTS, *bad)

    assert result.exit_code == 2  # a usage error, reported before any work


def _gradient_run(
    command: str, volume: str, mask: str, out: Path, *options: str
) -> Result:
    inputs = [volume, mask] if command == "fod" else [volume]
    # The fixture's signal is isotropic: fod is given its white matter response.
    extra = {
        "dti": ["--mask", mask],
        "fod": ["--wm-rf", "1.7e-3", "0.2e-3", "0.2e-3", "--nt", "1"],
    }.get(command, [])
    return _run(command, *inputs, str(out), *options, *extra)


@pytest.mark.parametrize("command", COMMANDS)
@pytest.mark.parametrize("given", ["--bvals", "--bvecs"])
def test_one_gradient_file_alone_is_refused(
    tmp_path: Path, dwi_files: tuple[str, str, list[str]], command: str, given: str
) -> None:
    volume, mask, gradients = dwi_files
    option = gradients[gradients.index(given) : gradients.index(given) + 2]
    result = _gradient_run(command, volume, mask, tmp_path / "out", *option)

    assert result.exit_code == 1
    assert result.output.startswith("ERROR: ")
    assert "both the bvals and the bvecs" in result.output


@pytest.mark.parametrize("command", COMMANDS)
def test_nifti_without_gradient_files_is_refused(
    tmp_path: Path, dwi_files: tuple[str, str, list[str]], command: str
) -> None:
    # Larmor never looks for gradient files by name, even next to the volume.
    volume, mask, _ = dwi_files
    result = _gradient_run(command, volume, mask, tmp_path / "out")

    assert result.exit_code == 1
    assert "no gradient table: give the bvals and bvecs" in result.output


# The main direction of the tensor in `oblique_dwi`, in world coordinates.
MAIN_DIRECTION = np.array([1.0, 1.0, 0.0]) / np.sqrt(2)


@pytest.fixture
def oblique_dwi(
    tmp_path: Path, dwi_files: tuple[str, str, list[str]]
) -> tuple[str, str, list[str], Path]:
    """An anisotropic DWI on an LPS grid, its mask, FSL gradient options and the
    same directions as a world-frame bvecs file.

    On an LPS image (det > 0) the FSL bvecs are (R F) @ world with R F =
    diag(1, -1, 1), so the two files differ in y; the tensor's main direction,
    (1, 1, 0) in world, shows a lost or doubled y sign.
    """
    _, _, gradients = dwi_files
    bvals = np.loadtxt(gradients[1])
    world = np.loadtxt(gradients[3]).T
    rotation = np.array([[1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, np.sqrt(2)]])
    rotation /= np.sqrt(2)
    tensor = rotation @ np.diag([1.7e-3, 0.3e-3, 0.2e-3]) @ rotation.T
    signal = 1000.0 * np.exp(-bvals * np.einsum("ni,ij,nj->n", world, tensor, world))

    # An 8-voxel cube in a dark, noisy background, so that `lar mask` finds it.
    affine = np.diag([-1.0, -1.0, 1.0, 1.0])
    volume, mask = tmp_path / "lps.nii.gz", tmp_path / "lps_mask.nii.gz"
    rng = np.random.default_rng(0)
    data = rng.uniform(5.0, 15.0, (12, 12, 12, len(bvals))).astype(np.float32)
    data[2:10, 2:10, 2:10] = signal
    cube = np.zeros((12, 12, 12), np.uint8)
    cube[2:10, 2:10, 2:10] = 1
    save_volume(Volume(data, affine), volume)
    save_volume(Volume(cube, affine), mask)
    np.savetxt(tmp_path / "fsl_bvecs", np.diag([1.0, -1.0, 1.0]) @ world.T)
    np.savetxt(tmp_path / "world_bvecs", world.T)
    options = ["--bvals", gradients[1], "--bvecs", str(tmp_path / "fsl_bvecs")]
    return str(volume), str(mask), options, tmp_path / "world_bvecs"


def _main_direction(path: Path) -> np.ndarray:
    """The principal eigenvector a dti run wrote, at the centre, with its sign
    chosen to point along MAIN_DIRECTION (an eigenvector's sign is arbitrary)."""
    pe: np.ndarray = _read(path)[5, 5, 5]
    aligned: np.ndarray = pe * np.sign(pe @ MAIN_DIRECTION)
    return aligned


def _as_mif_with_gradients(tmp_path: Path, volume: str, gradients: list[str]) -> str:
    """The DWI saved as .mif with its gradient table in the header."""
    table = read_bvals_bvecs(gradients[1], gradients[3], ref_path=volume, frame="fsl")
    path = str(tmp_path / "dwi.mif")
    save_volume(load_volume(volume), path, gradients=table)
    return path


@pytest.mark.parametrize("command", COMMANDS)
def test_mif_with_gradients_needs_no_files(
    tmp_path: Path, oblique_dwi: tuple[str, str, list[str], Path], command: str
) -> None:
    volume, mask, gradients, _ = oblique_dwi
    mif = _as_mif_with_gradients(tmp_path, volume, gradients)
    from_files = _gradient_run(command, volume, mask, tmp_path / "a.nii.gz", *gradients)
    from_header = _gradient_run(command, mif, mask, tmp_path / "b.nii.gz")

    assert from_files.exit_code == 0, from_files.output
    assert from_header.exit_code == 0, from_header.output
    assert "override" not in from_header.output
    # The same table, from the header or from the files, gives the same result.
    outputs = sorted(tmp_path.glob("a*.nii.gz"))
    assert outputs
    for a in outputs:
        b = tmp_path / a.name.replace("a", "b", 1)
        np.testing.assert_array_equal(_read(a), _read(b))
    if command == "dti":
        direction = _main_direction(tmp_path / "b_pe.nii.gz")
        np.testing.assert_allclose(direction, MAIN_DIRECTION, atol=1e-4)


def test_files_override_the_mif_header_with_a_warning(
    tmp_path: Path, dwi_files: tuple[str, str, list[str]]
) -> None:
    volume, _, gradients = dwi_files
    mif = _as_mif_with_gradients(tmp_path, volume, gradients)
    result = _run("mask", mif, str(tmp_path / "m.nii.gz"), *gradients)

    assert result.exit_code == 0, result.output
    lines = result.output.strip().splitlines()
    assert len(lines) == 1
    assert lines[0].startswith("WARNING: ")
    assert "override" in lines[0]


def test_bvec_frame_world_gives_the_same_tensor_as_fsl(
    tmp_path: Path, oblique_dwi: tuple[str, str, list[str], Path]
) -> None:
    # The two files differ in y, and read in their own frames they must give the
    # same tensor.
    volume, mask, fsl_options, world_bvecs = oblique_dwi
    bvals = fsl_options[:2]
    runs = {
        "fsl": [*fsl_options, "--bvec-frame", "fsl"],
        "world": [*bvals, "--bvecs", str(world_bvecs), "--bvec-frame", "world"],
    }
    for frame, options in runs.items():
        out = str(tmp_path / f"out_{frame}")
        result = _run("dti", volume, out, *options, "--mask", mask)
        assert result.exit_code == 0, result.output
        direction = _main_direction(tmp_path / f"out_{frame}_pe.nii.gz")
        np.testing.assert_allclose(direction, MAIN_DIRECTION, atol=1e-4)


def test_dti_writes_every_metric(
    tmp_path: Path, dwi_files: tuple[str, str, list[str]]
) -> None:
    volume, mask, gradients = dwi_files
    result = _run("dti", volume, str(tmp_path / "out"), *gradients, "--mask", mask)

    assert result.exit_code == 0, result.output
    maps = {
        name: _read(tmp_path / f"out_{name}.nii.gz")
        for name in ("fa", "md", "ad", "rd", "pe", "npd")
    }
    assert not maps["npd"].any()
    assert maps["fa"].shape == (12, 12, 12) and maps["pe"].shape == (12, 12, 12, 3)
    # The cube diffuses isotropically at 0.8e-3 mm^2/s.
    np.testing.assert_allclose(maps["md"][3:9, 3:9, 3:9], 0.8e-3, atol=1e-6)
    np.testing.assert_allclose(maps["fa"][3:9, 3:9, 3:9], 0.0, atol=1e-3)


def test_dti_maps_equal_the_weighted_dti_model_fit(
    tmp_path: Path, dwi_files: tuple[str, str, list[str]]
) -> None:
    volume, mask, gradients = dwi_files
    out = str(tmp_path / "out")
    result = _run("dti", volume, out, *gradients, "--mask", mask, "--fit", "wls")
    assert result.exit_code == 0, result.output

    image, gtab = load_dwi(volume, bvals=gradients[1], bvecs=gradients[3], frame="fsl")
    fit = DTIModel(fit_method="wls").fit(image, gtab, load_volume(mask))
    for name in ("fa", "md", "ad", "rd", "pe"):
        np.testing.assert_array_equal(
            _read(tmp_path / f"out_{name}.nii.gz"), getattr(fit, name), err_msg=name
        )


@pytest.mark.parametrize(
    ("isotropic", "n_tissues"),
    [
        (["--gm-rf", "none", "--csf-rf", "none"], 0),  # single tissue: no fractions
        ([], 3),  # WM, GM and CSF by default
    ],
)
def test_fod_writes_the_fod_and_the_fractions(
    tmp_path: Path,
    dwi_files: tuple[str, str, list[str]],
    isotropic: list[str],
    n_tissues: int,
) -> None:
    volume, mask, gradients = dwi_files
    out = tmp_path / "out"
    result = _run(
        "fod", volume, mask, str(out), *gradients,
        "--wm-rf", "1.7e-3", "0.2e-3", "0.2e-3", "--nt", "1", *isotropic,
    )  # fmt: skip

    assert result.exit_code == 0, result.output
    # Six directions support lmax 2 only: 6 SH coefficients.
    fod = _read(tmp_path / "out.nii.gz")
    assert fod.shape == (12, 12, 12, 6) and np.isfinite(fod).all()
    fractions = tmp_path / "out_tissue_fraction.nii.gz"
    if n_tissues:
        assert _read(fractions).shape == (12, 12, 12, n_tissues)
    else:
        assert not fractions.exists()


def test_fod_csd_writes_the_fod_from_a_response_file(
    tmp_path: Path, dwi_files: tuple[str, str, list[str]]
) -> None:
    volume, mask, gradients = dwi_files
    response = tmp_path / "response.txt"
    response.write_text("450 -150 30\n")
    out = tmp_path / "out"
    result = _run(
        "fod", volume, mask, str(out), *gradients, "--method", "csd",
        "--response", str(response), "--lmax", "4", "--nt", "1",
    )  # fmt: skip

    assert result.exit_code == 0, result.output
    # CSD keeps lmax 4 (15 coefficients) on six directions and has no fractions.
    assert _read(tmp_path / "out.nii.gz").shape == (12, 12, 12, 15)
    assert not (tmp_path / "out_tissue_fraction.nii.gz").exists()


def test_fod_response_goes_with_csd_only(tmp_path: Path) -> None:
    response = tmp_path / "r.txt"
    response.write_text("450 -150 30\n")
    result = _run(
        "fod", "missing.nii.gz", "m", str(tmp_path / "fod"), *NO_GRADIENTS,
        "--response", str(response),
    )  # fmt: skip

    assert result.exit_code != 0
    assert "--response" in result.output


@pytest.mark.parametrize(
    "option",
    [
        ["--wm-rf", "1.7e-3", "0.2e-3", "0.2e-3"],
        ["--csf-rf", "none"],
        ["--fod-merge", "majority"],
    ],
)
def test_fod_csd_refuses_the_options_of_grl_and_mfod(
    tmp_path: Path, option: list[str]
) -> None:
    result = _run(
        "fod", "missing.nii.gz", "m", str(tmp_path / "fod"), *NO_GRADIENTS,
        "--method", "csd", *option,
    )  # fmt: skip

    assert result.exit_code != 0
    assert option[0] in result.output


@pytest.mark.parametrize("command", [*COMMANDS, *MULTI_SHELL_COMMANDS])
def test_missing_output_folder_is_refused_before_reading_inputs(
    tmp_path: Path, command: str
) -> None:
    inputs = ["missing.nii.gz", "m"] if command == "fod" else ["missing.nii.gz"]
    out = str(tmp_path / "no_such_folder" / "out")
    result = _run(command, *inputs, out, *NO_GRADIENTS)

    assert result.exit_code == 1
    assert "no_such_folder does not exist" in result.output


def _fod_run(
    volume: str, mask: str, out: Path, gradients: list[str], *options: str
) -> Result:
    result = _run(
        "fod", volume, mask, str(out), *gradients,
        "--wm-rf", "1.7e-3", "0.2e-3", "0.2e-3", "--nt", "1", *options,
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    return result


def test_fod_kurtosis_without_eigenvalues_sets_the_estimated_response(
    tmp_path: Path, oblique_dwi: tuple[str, str, list[str], Path]
) -> None:
    # The oblique fibre has FA > 0.7, so the response is estimated from it; on a
    # single shell its estimated kurtosis is 0, so --mk 0.5 must change the FOD.
    volume, mask, gradients, _ = oblique_dwi
    runs = {"estimated": [], "given": ["--mk", "0.5"]}
    for name, options in runs.items():
        result = _run(
            "fod", volume, mask, str(tmp_path / name), *gradients, "--nt", "1",
            "--info", *options,
        )  # fmt: skip
        assert result.exit_code == 0, result.output
        assert "Estimated WM eigenvalues" in result.output

    estimated = _read(tmp_path / "estimated.nii.gz")
    assert not np.allclose(_read(tmp_path / "given.nii.gz"), estimated)


def test_fod_kurtosis_defaults_to_zero_with_given_eigenvalues(
    tmp_path: Path, oblique_dwi: tuple[str, str, list[str], Path]
) -> None:
    volume, mask, gradients, _ = oblique_dwi
    _fod_run(volume, mask, tmp_path / "default", gradients)
    _fod_run(volume, mask, tmp_path / "zero", gradients, "--mk", "0")
    _fod_run(volume, mask, tmp_path / "half", gradients, "--mk", "0.5")

    default = _read(tmp_path / "default.nii.gz")
    np.testing.assert_array_equal(default, _read(tmp_path / "zero.nii.gz"))
    assert not np.allclose(default, _read(tmp_path / "half.nii.gz"))


@pytest.mark.parametrize("basis", ["descoteaux", "leemans"])
def test_fod_basis_matches_converting_the_default_output(
    tmp_path: Path, oblique_dwi: tuple[str, str, list[str], Path], basis: str
) -> None:
    volume, mask, gradients, _ = oblique_dwi
    _fod_run(volume, mask, tmp_path / "default", gradients)
    _fod_run(volume, mask, tmp_path / "direct", gradients, "--basis", basis)
    default, converted = tmp_path / "default.nii.gz", tmp_path / "converted.nii.gz"
    result = _run(
        "sh-convert", str(default), str(converted), "--from", "tournier", "--to", basis
    )
    assert result.exit_code == 0, result.output

    expected = _read(converted)
    # The oblique fibre has m != 0 terms, so the bases really differ here.
    assert not np.allclose(expected, _read(default))
    # Both files are float32: one conversion rounds before saving, the other after.
    np.testing.assert_allclose(
        _read(tmp_path / "direct.nii.gz"), expected, rtol=1e-6, atol=1e-7
    )


def test_fod_default_basis_is_tournier(
    tmp_path: Path, oblique_dwi: tuple[str, str, list[str], Path]
) -> None:
    volume, mask, gradients, _ = oblique_dwi
    default = _fod_run(volume, mask, tmp_path / "default", gradients, "--info")
    _fod_run(volume, mask, tmp_path / "tournier", gradients, "--basis", "tournier")

    np.testing.assert_array_equal(
        _read(tmp_path / "default.nii.gz"), _read(tmp_path / "tournier.nii.gz")
    )
    assert "(tournier basis)" in default.output


def test_fod_logs_the_basis_it_wrote(
    tmp_path: Path, oblique_dwi: tuple[str, str, list[str], Path]
) -> None:
    volume, mask, gradients, _ = oblique_dwi
    result = _fod_run(
        volume, mask, tmp_path / "out", gradients, "--basis", "leemans", "--info"
    )

    assert "(leemans basis)" in result.output


def test_fod_unknown_basis_is_a_usage_error_naming_the_choices(
    tmp_path: Path,
) -> None:
    out = str(tmp_path / "fod")
    result = _run("fod", "missing.nii.gz", "m", out, *NO_GRADIENTS, "--basis", "mrtrix")

    assert result.exit_code == 2
    for choice in ("tournier", "descoteaux", "leemans"):
        assert choice in result.output


def test_fod_method_choices_are_the_library_methods() -> None:
    # SDModel's methods, CSD (CSDModel) and MSMT-CSD (MSMTCSDModel).
    assert tuple(method.value for method in Method) == (
        *(method.value for method in SDMethod),
        "CSD",
        "MSMT-CSD",
    )


def test_fod_help_lists_the_methods() -> None:
    output = _run("fod", "--help").output
    (line,) = [line for line in output.splitlines() if "--method" in line]
    # Case-insensitive choices are shown lower-cased.
    for choice in ("grl", "mfod"):
        assert choice in line.lower()


@pytest.mark.parametrize("method", ["grl", "GRL", "mfod", "MFOD"])
def test_fod_method_is_accepted_in_any_case(tmp_path: Path, method: str) -> None:
    out = str(tmp_path / "fod")
    result = _run("fod", "missing.nii.gz", "m", out, *NO_GRADIENTS, "--method", method)

    # Accepted options get as far as reading the (missing) input.
    assert result.exit_code == 1
    assert "missing.nii.gz" in result.output


def test_fod_unknown_method_is_a_usage_error_naming_the_choices(
    tmp_path: Path,
) -> None:
    out = str(tmp_path / "fod")
    result = _run("fod", "missing.nii.gz", "m", out, *NO_GRADIENTS, "--method", "rl")

    assert result.exit_code == 2
    for choice in ("grl", "mfod"):
        assert choice in result.output.lower()


@pytest.mark.parametrize(
    ("options", "solver", "weight"),
    [
        ([], None, None),  # the method's own
        (["--solver", "drl", "--inner-shell-weight", "0.5"], DRL(), 0.5),
        (["--solver", "l2nnls", "--reg", "0.2"], L2NNLS(reg=0.2), None),
        (["--solver", "lasso", "--reg", "0.05"], Lasso(reg=0.05), None),
        (["--solver", "LASSO"], Lasso(), None),
        (["--method", "mfod"], None, None),  # mFOD's own, L2NNLS
    ],
)
def test_fod_passes_the_solver_and_the_weight_to_the_model(
    tmp_path: Path,
    dwi_files: tuple[str, str, list[str]],
    monkeypatch: pytest.MonkeyPatch,
    options: list[str],
    solver: DRL | L2NNLS | Lasso | None,
    weight: float | None,
) -> None:
    seen: dict[str, object] = {}
    real_model = larmor.sd.SDModel

    def spy(*args: object, **kwargs: object) -> larmor.sd.SDModel:
        seen.update(kwargs)
        return real_model(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(larmor.sd, "SDModel", spy)
    volume, mask, gradients = dwi_files
    _fod_run(volume, mask, tmp_path / "out", gradients, *options)

    assert seen["solver"] == solver and seen["inner_shell_weight"] == weight


@pytest.mark.parametrize("options", [[], ["--solver", "drl"]])
def test_fod_refuses_reg_with_the_drl_solver(
    tmp_path: Path, options: list[str]
) -> None:
    out = str(tmp_path / "fod")
    result = _run(
        "fod", "missing.nii.gz", "m", out, *NO_GRADIENTS, "--reg", "0.2", *options
    )  # fmt: skip

    # Refused as a usage error, before reading the (missing) input.
    assert result.exit_code == 2, result.output
    assert "--reg" in result.output and "missing.nii.gz" not in result.output


@pytest.mark.parametrize("weight", ["0", "-1"])
def test_fod_refuses_an_inner_shell_weight_outside_0_1(
    tmp_path: Path, dwi_files: tuple[str, str, list[str]], weight: str
) -> None:
    volume, mask, gradients = dwi_files
    result = _run(
        "fod", volume, mask, str(tmp_path / "out"), *gradients,
        "--wm-rf", "1.7e-3", "0.2e-3", "0.2e-3", "--inner-shell-weight", weight,
    )  # fmt: skip

    assert result.exit_code == 1
    assert result.output.startswith("ERROR: ")
    assert "inner_shell_weight must be in (0, 1]" in result.output
    assert "Traceback" not in result.output
    assert not (tmp_path / "out.nii.gz").exists()


DKI = Path(__file__).parents[1] / "data" / "dki"


@pytest.fixture
def dki_files(tmp_path: Path) -> tuple[str, str, list[str]]:
    """The real two-shell DWI block, a mask of eight voxels in it to keep the mFOD
    fits fast, and the gradient options."""
    affine = load_volume(DKI / "mask.nii.gz").affine
    cube = np.zeros((8, 8, 8), dtype=np.uint8)
    cube[3:5, 3:5, 3:5] = 1
    mask = tmp_path / "small_mask.nii.gz"
    save_volume(Volume(cube, affine), mask)
    gradients = ["--bvals", str(DKI / "bvals"), "--bvecs", str(DKI / "bvecs")]
    return str(DKI / "dwi.nii.gz"), str(mask), gradients


@pytest.mark.parametrize(
    ("method", "gm_rf", "noddi", "diffusivity"),
    [
        ("mfod", ["--gm-rf", "noddi"], True, None),
        ("mfod", ["--gm-rf", "NODDI"], True, None),
        ("mfod", [], True, None),
        ("grl", [], False, 0.7e-3),
        ("mfod", ["--gm-rf", "0.8e-3"], False, 0.8e-3),
    ],
    ids=["noddi", "noddi-any-case", "mfod-default", "grl-default", "mfod-isotropic"],
)
def test_fod_gm_rf_reaches_the_responses(
    tmp_path: Path,
    dki_files: tuple[str, str, list[str]],
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    gm_rf: list[str],
    noddi: bool,
    diffusivity: float | None,
) -> None:
    seen: dict[str, object] = {}
    real_build = larmor.sd.build_response_functions

    def spy(*args: object, **kwargs: object) -> list[larmor.sd.ResponseFunction]:
        seen.update(kwargs)
        return real_build(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(larmor.sd, "build_response_functions", spy)
    volume, mask, gradients = dki_files
    _fod_run(volume, mask, tmp_path / "out", gradients, "--method", method, *gm_rf)

    assert seen["gm_noddi"] is noddi
    assert seen["gm_diffusivity"] == diffusivity


@pytest.mark.parametrize("existing", ["fod1", "fod2"])
def test_fod_noddi_checks_both_fod_files_before_reading_inputs(
    tmp_path: Path, existing: str
) -> None:
    (tmp_path / f"out_{existing}.nii.gz").write_bytes(b"old")
    result = _run(
        "fod", "missing.nii.gz", "m", str(tmp_path / "out"), *NO_GRADIENTS,
        "--method", "mfod", "--gm-rf", "noddi", "--fod-merge", "none",
    )  # fmt: skip

    assert result.exit_code == 1
    assert f"out_{existing}.nii.gz exists" in result.output
    assert "missing.nii.gz" not in result.output


def test_fod_noddi_writes_two_fods_and_the_fractions(
    tmp_path: Path, dki_files: tuple[str, str, list[str]]
) -> None:
    volume, mask, gradients = dki_files
    _fod_run(
        volume,
        mask,
        tmp_path / "out",
        gradients,
        "--method",
        "mfod",
        "--gm-rf",
        "noddi",
        "--fod-merge",
        "none",
    )

    inside = load_volume(mask).data > 0
    for name in ("out_fod1", "out_fod2"):
        fod = _read(tmp_path / f"{name}.nii.gz")
        # lmax 6: the outer shell's 33 directions are too few for lmax 8.
        assert fod.shape == (8, 8, 8, 28) and np.isfinite(fod[inside]).all()
    # WM, NODDI GM and CSF.
    assert _read(tmp_path / "out_tissue_fraction.nii.gz").shape == (8, 8, 8, 3)
    assert not (tmp_path / "out.nii.gz").exists()


@pytest.mark.parametrize(
    "merge", [[], ["--fod-merge", "majority"]], ids=["ws", "majority"]
)
def test_fod_noddi_writes_one_fod_merged_from_the_two(
    tmp_path: Path, dki_files: tuple[str, str, list[str]], merge: list[str]
) -> None:
    volume, mask, gradients = dki_files
    mfod = ["--method", "mfod", "--gm-rf", "noddi"]
    _fod_run(volume, mask, tmp_path / "two", gradients, *mfod, "--fod-merge", "none")
    _fod_run(volume, mask, tmp_path / "one", gradients, *mfod, *merge)

    method: Literal["ws", "majority"] = "majority" if merge else "ws"
    fods = [_read(tmp_path / f"two_fod{i}.nii.gz") for i in (1, 2)]
    fractions = _read(tmp_path / "two_tissue_fraction.nii.gz")
    expected = larmor.sd.merge_fods(fods, fractions, method)
    np.testing.assert_allclose(_read(tmp_path / "one.nii.gz"), expected, atol=1e-6)
    assert (tmp_path / "one_tissue_fraction.nii.gz").exists()
    assert not (tmp_path / "one_fod1.nii.gz").exists()


def test_fod_msmt_csd_refuses_fod_merge(tmp_path: Path) -> None:
    result = _run(
        "fod", "missing.nii.gz", "m", str(tmp_path / "fod"), *NO_GRADIENTS,
        "--method", "msmt-csd", "--fod-merge", "majority",
    )  # fmt: skip

    assert result.exit_code != 0
    assert "--fod-merge" in result.output


def test_fod_merge_is_refused_for_a_fit_with_one_fod(
    tmp_path: Path, dki_files: tuple[str, str, list[str]]
) -> None:
    volume, mask, gradients = dki_files
    result = _run(
        "fod", volume, mask, str(tmp_path / "out"), *gradients,
        "--method", "mfod", "--gm-rf", "0.8e-3", "--fod-merge", "ws",
    )  # fmt: skip

    assert result.exit_code != 0
    assert "--fod-merge" in result.output


def test_fod_noddi_with_grl_is_a_clean_error(
    tmp_path: Path, dwi_files: tuple[str, str, list[str]]
) -> None:
    volume, mask, gradients = dwi_files
    result = _run(
        "fod", volume, mask, str(tmp_path / "out"), *gradients,
        "--wm-rf", "1.7e-3", "0.2e-3", "0.2e-3", "--nt", "1",
        "--method", "grl", "--gm-rf", "noddi",
    )  # fmt: skip

    assert result.exit_code == 1
    assert result.output.startswith("ERROR: ") and "GRL" in result.output
    assert "Traceback" not in result.output


def test_fod_invalid_reg_is_a_usage_error_naming_reg(tmp_path: Path) -> None:
    out = str(tmp_path / "fod")
    result = _run(
        "fod", "missing.nii.gz", "m", out, *NO_GRADIENTS,
        "--method", "mfod", "--reg", "-1",
    )  # fmt: skip

    assert result.exit_code == 2, result.output
    assert "--reg" in result.output and "missing.nii.gz" not in result.output


def test_version_option_prints_the_package_version() -> None:
    from larmor import __version__
    from larmor.main import app

    result = CliRunner().invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.output.strip() == f"larmor {__version__}"
