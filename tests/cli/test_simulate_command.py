from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner, Result

from larmor.io import load_volume
from larmor.main import app
from larmor.simulation import (
    example_phantom,
    read_phantom,
    simulation_files,
    write_phantom,
)
from larmor.types import GradientTable


def _run(*args: str | Path) -> Result:
    return CliRunner().invoke(app, ["simulate", *map(str, args)], env={"TERM": "dumb"})


def test_the_example_is_simulated_by_name_and_every_output_is_written(
    tmp_path: Path,
) -> None:
    out = tmp_path / "out"
    result = _run("crossing", out, "--snr", "30", "--seed", "1")
    assert result.exit_code == 0, result.output
    loaded = example_phantom()
    assert loaded.protocol is not None
    files = simulation_files(loaded.phantom, loaded.protocol, out, noise=True)
    assert all(p.is_file() for p in files.all_paths())
    assert load_volume(files.dwi[0]).data.shape == (40, 40, 40, 66)
    assert len(np.loadtxt(files.bvals)) == 66


def test_a_phantom_file_is_read_from_its_path(tmp_path: Path) -> None:
    loaded = example_phantom()
    path = tmp_path / "mine.phantom.json"
    write_phantom(loaded.phantom, path, protocol=loaded.protocol)
    result = _run(path, tmp_path / "out", "--seed", "2")
    assert result.exit_code == 0, result.output
    assert (tmp_path / "out" / "dwi.nii.gz").is_file()
    assert not (tmp_path / "out" / "dwi_noise_free.nii.gz").exists()


def test_the_same_seed_gives_the_same_images_and_another_seed_does_not(
    tmp_path: Path,
) -> None:
    for name, seed in (("a", "1"), ("b", "1"), ("c", "2")):
        assert (
            _run(
                "crossing", tmp_path / name, "--snr", "20", "--seed", str(seed)
            ).exit_code
            == 0
        )
    first, same, other = (load_volume(tmp_path / n / "dwi.nii.gz").data for n in "abc")
    np.testing.assert_array_equal(first, same)
    assert not np.array_equal(first, other)


def test_an_existing_output_is_refused_before_any_work_without_force(
    tmp_path: Path,
) -> None:
    out = tmp_path / "out"
    assert _run("crossing", out, "--seed", "1").exit_code == 0
    before = (out / "dwi.nii.gz").read_bytes()
    result = _run("crossing", out, "--seed", "1", "--snr", "10")
    assert result.exit_code == 1
    assert "exists" in result.output and "--force" in result.output
    assert (out / "dwi.nii.gz").read_bytes() == before
    assert _run("crossing", out, "--seed", "1", "--snr", "10", "--force").exit_code == 0
    assert (out / "dwi.nii.gz").read_bytes() != before


def test_a_file_where_the_folder_should_be_is_an_error(tmp_path: Path) -> None:
    (tmp_path / "out").write_text("x")
    result = _run("crossing", tmp_path / "out")
    assert result.exit_code == 1
    assert "is not a folder" in result.output


def test_shells_replace_the_gradients_of_the_file(tmp_path: Path) -> None:
    out = tmp_path / "out"
    result = _run(
        "crossing", out, "--shell", "500:6", "--shell", "2500:10", "--num-b0", "3"
    )
    assert result.exit_code == 0, result.output
    bvals = np.loadtxt(out / "dwi.bval")
    assert bvals.tolist() == [0.0] * 3 + [500.0] * 6 + [2500.0] * 10


def test_new_gradients_keep_the_b0_threshold_of_the_file(tmp_path: Path) -> None:
    loaded = example_phantom()
    assert loaded.protocol is not None
    table = loaded.protocol.gradients
    narrow = GradientTable(table.bvals, table.bvecs, b0_threshold=5.0)
    protocol = replace(loaded.protocol, gradients=narrow)
    path = tmp_path / "mine.phantom.json"
    write_phantom(loaded.phantom, path, protocol=protocol)
    assert _run(path, tmp_path / "out", "--shell", "1000:6").exit_code == 0
    written = read_phantom(tmp_path / "out" / "phantom.json").protocol
    assert written is not None
    assert written.gradients.b0_threshold == 5.0


def test_bvals_and_bvecs_files_are_read_in_the_given_frame(tmp_path: Path) -> None:
    bvals = np.array([0.0, 1000.0, 1000.0])
    world = np.array([[0.0, 0, 0], [0, 1, 0], [0.6, 0, 0.8]])
    np.savetxt(tmp_path / "w.bval", bvals[None])
    np.savetxt(tmp_path / "w.bvec", world.T)
    # The example's grid is RAS with positive voxels, so FSL negates x.
    fsl = world.copy()
    fsl[:, 0] *= -1
    np.savetxt(tmp_path / "f.bvec", fsl.T)
    first = _run("crossing", tmp_path / "w", "--bvals", tmp_path / "w.bval",
                 "--bvecs", tmp_path / "w.bvec", "--bvec-frame", "world")  # fmt: skip
    second = _run("crossing", tmp_path / "f", "--bvals", tmp_path / "w.bval",
                  "--bvecs", tmp_path / "f.bvec")  # fmt: skip
    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    np.testing.assert_array_equal(
        load_volume(tmp_path / "w" / "dwi.nii.gz").data,
        load_volume(tmp_path / "f" / "dwi.nii.gz").data,
    )
    assert load_volume(tmp_path / "w" / "dwi.nii.gz").data.shape[3] == 3


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--shell", "1000"], "not a shell"),
        (["--shell", "abc:3"], "not a shell"),
        (["--shell", "1000:0"], "at least one direction"),
        (["--shell", "-5:8"], "b > 0"),
        (["--shell", "1000:8", "--num-b0", "-1"], "num_b0"),
        (["--bvals", "x.bval"], "go together"),
        (["--snr", "0"], "snr must be positive"),
        (["--snr", "20", "--noise-reference", "nope"], "no tissue 'nope'"),
        (["--nt", "0"], "--nt"),
        (["--num-b0", "3"], "needs --shell"),
        (["--bvec-frame", "world"], "needs --bvals"),
        (["--shell", "1000:8", "--bvals", "a", "--bvecs", "b"], "either --shell"),
    ],
)
def test_invalid_options_give_one_clear_error(
    tmp_path: Path, args: list[str], message: str
) -> None:
    result = _run("crossing", tmp_path / "out", *args)
    assert result.exit_code != 0
    assert message in result.output
    assert not (tmp_path / "out").exists()


def test_a_bvecs_file_of_the_wrong_length_names_the_files(tmp_path: Path) -> None:
    np.savetxt(tmp_path / "a.bval", np.array([[0.0, 1000, 1000]]))
    np.savetxt(tmp_path / "a.bvec", np.eye(3)[:, :2])
    result = _run(
        "crossing", tmp_path / "out", "--bvals", tmp_path / "a.bval", "--bvecs",
        tmp_path / "a.bvec",
    )  # fmt: skip
    assert result.exit_code == 1
    assert "a.bvec" in result.output


def test_an_unknown_phantom_names_the_examples(tmp_path: Path) -> None:
    result = _run("nowhere.phantom.json", tmp_path / "out")
    assert result.exit_code == 1
    assert "nowhere.phantom.json" in result.output and "crossing" in result.output


def test_a_phantom_file_without_a_scan_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "bare.phantom.json"
    write_phantom(example_phantom().phantom, path)
    result = _run(path, tmp_path / "out")
    assert result.exit_code == 1
    assert "has no scan" in result.output


def test_force_replaces_the_files_of_the_earlier_run(tmp_path: Path) -> None:
    out = tmp_path / "out"
    assert _run("crossing", out, "--snr", "20", "--seed", "1").exit_code == 0
    assert (out / "dwi_noise_free.nii.gz").exists()
    assert _run("crossing", out, "--seed", "1", "--force").exit_code == 0
    assert not (out / "dwi_noise_free.nii.gz").exists()


def test_a_folder_of_other_files_needs_force(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / "notes.txt").write_text("mine")
    result = _run("crossing", out)
    assert result.exit_code == 1
    assert "not empty" in result.output
    assert not (out / "dwi.nii.gz").exists()
    assert _run("crossing", out, "--force").exit_code == 0
    assert (out / "notes.txt").exists()


def test_without_a_seed_the_one_drawn_is_in_the_info_file(tmp_path: Path) -> None:
    import json

    assert _run("crossing", tmp_path / "out", "--snr", "20").exit_code == 0
    seed = json.loads((tmp_path / "out" / "simulation.json").read_text())["seed"]
    assert isinstance(seed, int)
    assert (
        _run(
            "crossing", tmp_path / "again", "--snr", "20", "--seed", str(seed)
        ).exit_code
        == 0
    )
    np.testing.assert_array_equal(
        load_volume(tmp_path / "out" / "dwi.nii.gz").data,
        load_volume(tmp_path / "again" / "dwi.nii.gz").data,
    )


def test_the_bundle_names_of_the_example_reach_the_viewer(tmp_path: Path) -> None:
    from larmor.viewer import Viewer

    out = tmp_path / "crossing-sim"
    assert _run("crossing", out, "--seed", "1").exit_code == 0
    for as_folder in (True, False):
        viewer = Viewer()
        if as_folder:
            viewer.open_simulation(out)
        else:
            viewer.load_phantom(out / "phantom.json")
        state = viewer.phantom
        assert state is not None
        names = [viewer.scene.item(b.item_id).name for b in state.bundles]
        assert names == ["Left-right", "Front-back"]
