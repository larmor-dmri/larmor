from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from typer.testing import CliRunner, Result

from larmor.dwi import DKIModel
from larmor.io import load_dwi, load_volume, save_volume
from larmor.main import app

DKI_DIR = Path(__file__).parents[1] / "data" / "dki"
VOLUME, MASK = str(DKI_DIR / "dwi.nii.gz"), str(DKI_DIR / "mask.nii.gz")
GRADIENTS = ("--bvals", str(DKI_DIR / "bvals"), "--bvecs", str(DKI_DIR / "bvecs"))
# Gradient options for runs whose inputs do not exist.
NO_GRADIENTS = ("--bvals", "a", "--bvecs", "b")
METRICS = ("fa", "md", "ad", "rd", "pe", "npd", "mk", "ak", "rk", "ka")


def _run(*args: str | Path) -> Result:
    return CliRunner().invoke(app, ["dki", *map(str, args)], env={"TERM": "dumb"})


def _read(path: Path) -> npt.NDArray[np.floating]:
    return load_volume(path).as_float()


@pytest.mark.parametrize("options", [["--fit", "ols"], []], ids=["ols", "default"])
def test_dki_writes_every_metric(tmp_path: Path, options: list[str]) -> None:
    result = _run(VOLUME, tmp_path / "out", *GRADIENTS, "--mask", MASK, *options)

    assert result.exit_code == 0, result.output
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        f"out_{name}.nii.gz" for name in METRICS
    )
    mask = load_volume(MASK).data > 0
    for name in METRICS:
        value = _read(tmp_path / f"out_{name}.nii.gz")
        assert value.shape == ((8, 8, 8, 3) if name == "pe" else (8, 8, 8)), name
        assert np.isfinite(value).all(), name
        assert np.all(value[~mask] == 0), name
    # White matter of an adult brain: MK of order 1 inside the mask.
    assert 0.3 < float(np.median(_read(tmp_path / "out_mk.nii.gz")[mask])) < 3


def test_mask_is_computed_when_omitted(tmp_path: Path) -> None:
    result = _run(VOLUME, tmp_path / "out", *GRADIENTS, "--fit", "ols")

    assert result.exit_code == 0, result.output
    for name in METRICS:
        assert np.isfinite(_read(tmp_path / f"out_{name}.nii.gz")).all(), name
    assert np.any(_read(tmp_path / "out_mk.nii.gz") != 0)


@pytest.mark.parametrize(
    ("options", "fit_method", "constrained"),
    [
        (["--fit", "ols"], "ols", True),
        (["--fit", "wls"], "wls", True),
        ([], "wls", True),
        (["--unconstrained"], "wls", False),
        (["--fit", "ols", "--unconstrained"], "ols", False),
        (["--nt", "1"], "wls", True),
    ],
    ids=["ols", "wls", "default", "unconstrained", "ols unconstrained", "one thread"],
)
def test_maps_equal_the_dki_model_fit(
    tmp_path: Path, options: list[str], fit_method: str, constrained: bool
) -> None:
    result = _run(VOLUME, tmp_path / "out", *GRADIENTS, "--mask", MASK, *options)
    assert result.exit_code == 0, result.output

    image, gtab = load_dwi(VOLUME, bvals=GRADIENTS[1], bvecs=GRADIENTS[3], frame="fsl")
    model = DKIModel(fit_method=fit_method, constrained=constrained)
    fit = model.fit(image, gtab, load_volume(MASK))
    for name in METRICS:
        np.testing.assert_array_equal(
            _read(tmp_path / f"out_{name}.nii.gz"), getattr(fit, name), err_msg=name
        )


def test_voxels_outside_the_mask_are_not_fitted(tmp_path: Path) -> None:
    # The block's own mask covers every voxel: half of it is used instead.
    image = load_volume(VOLUME)
    half = np.zeros((8, 8, 8), dtype=np.uint8)
    half[:4] = 1
    save_volume(image.with_data(half), tmp_path / "half.nii.gz")
    result = _run(
        VOLUME, tmp_path / "out", *GRADIENTS, "--mask", tmp_path / "half.nii.gz"
    )

    assert result.exit_code == 0, result.output
    for name in METRICS:
        value = _read(tmp_path / f"out_{name}.nii.gz")
        assert np.all(value[4:] == 0), name
        # The non-positive-definite flag is 0 in every fitted voxel of the block.
        assert name == "npd" or np.all(value[:4] != 0), name


@pytest.mark.parametrize("name", METRICS)
def test_existing_output_is_kept_without_force(tmp_path: Path, name: str) -> None:
    # The inputs do not exist: the refusal comes before anything is read.
    existing = tmp_path / f"out_{name}.nii.gz"
    existing.write_bytes(b"old")
    result = _run("missing.nii.gz", tmp_path / "out", *NO_GRADIENTS)

    assert result.exit_code == 1
    assert existing.name in result.output and "exists" in result.output
    assert existing.read_bytes() == b"old"


def test_existing_outputs_are_replaced_with_force(tmp_path: Path) -> None:
    for name in METRICS:
        (tmp_path / f"out_{name}.nii.gz").write_bytes(b"old")
    result = _run(VOLUME, tmp_path / "out", *GRADIENTS, "--mask", MASK, "--force")

    assert result.exit_code == 0, result.output
    for name in METRICS:
        assert (tmp_path / f"out_{name}.nii.gz").read_bytes() != b"old", name


def test_single_shell_input_is_refused_with_one_error(tmp_path: Path) -> None:
    image, _ = load_dwi(VOLUME, bvals=GRADIENTS[1], bvecs=GRADIENTS[3], frame="fsl")
    bvals = np.loadtxt(GRADIENTS[1])
    keep = (bvals == 0) | (bvals == 1000)
    save_volume(image.with_data(image.data[..., keep]), tmp_path / "shell.nii.gz")
    np.savetxt(tmp_path / "bvals", bvals[keep][np.newaxis])
    np.savetxt(tmp_path / "bvecs", np.loadtxt(GRADIENTS[3])[:, keep])

    result = _run(
        tmp_path / "shell.nii.gz", tmp_path / "out",
        "--bvals", tmp_path / "bvals", "--bvecs", tmp_path / "bvecs", "--mask", MASK,
    )  # fmt: skip

    assert result.exit_code == 1
    errors = [line for line in result.output.splitlines() if line.startswith("ERROR:")]
    assert len(errors) == 1, result.output
    assert "at least two non-zero b-value shells" in errors[0]
    assert not list(tmp_path.glob("out_*"))


def test_unknown_fit_method_is_a_usage_error(tmp_path: Path) -> None:
    result = _run(VOLUME, tmp_path / "out", *GRADIENTS, "--fit", "nls")

    assert result.exit_code == 2
    assert not list(tmp_path.glob("out_*"))
