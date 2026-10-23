import json
import logging
import math
import os
from pathlib import Path

import numpy as np
import pytest
import typer
from typer.testing import CliRunner, Result

from larmor.io import load_tractogram, load_volume, save_volume
from larmor.main import app
from larmor.sd import convert_sh_basis
from larmor.sd.sh import get_sh_indices
from larmor.tractography import EndStatus, Tracker, seeds_per_voxel
from larmor.types import Tractogram, Volume

DATA = Path(__file__).parent.parent / "data" / "tracking"
FOD = DATA / "fod.nii.gz"
MASK = DATA / "mask.nii.gz"
# Four voxels, so per-voxel seed counts are easy to check.
SEED_VOXELS = DATA / "seeds.nii.gz"
SEED = 3
SELECT = 30
# Close to the FOD's auto cutoff (0.082); given, so most runs skip measuring it.
CUTOFF = "0.08"


pytestmark = pytest.mark.usefixtures("keep_log_level")


def _run(*args: str | Path) -> Result:
    return CliRunner().invoke(app, ["track", *map(str, args)], env={"TERM": "dumb"})


def _track(out: Path, *args: str | Path) -> Result:
    """A small run seeded in the mask, or from the source given in args."""
    sources = ("--seed-image", "--seed-random-per-voxel", "--seed-file")
    defaults: list[str | Path] = []
    if not any(a in sources for a in args):
        defaults += ["--seed-image", MASK, "--select", str(SELECT)]
    if "--cutoff" not in args:
        defaults += ["--cutoff", CUTOFF]
    result = _run(FOD, out, "--mask", MASK, *defaults, *args)
    assert result.exit_code == 0, result.output
    return result


def _assert_same(a: Tractogram, b: Tractogram) -> None:
    np.testing.assert_array_equal(a.offsets, b.offsets)
    np.testing.assert_array_equal(a.points, b.points)
    np.testing.assert_array_equal(
        a.streamline_data["end_status"], b.streamline_data["end_status"]
    )


def _record(path: Path) -> dict[str, object]:
    record: dict[str, object] = json.loads(path.read_text(encoding="utf-8"))
    return record


def _parameters(path: Path) -> dict[str, object]:
    parameters = _record(path)["parameters"]
    assert isinstance(parameters, dict)
    return parameters


# Usage errors: inputs do not exist, so a refusal must come before any reading,
# and an existing output given with --force must not be touched.

MISSING = "missing.nii.gz"
ONE_SOURCE = ("--seed-image", MISSING, "--select", "5")


@pytest.mark.parametrize(
    ("args", "named"),
    [
        ([], ["--seed-image", "--seed-random-per-voxel", "--seed-file"]),
        (
            ["--seed-image", MISSING, "--seed-file", "s.txt", "--select", "5"],
            ["--seed-image", "--seed-random-per-voxel", "--seed-file"],
        ),
        (
            ["--seed-random-per-voxel", MISSING, "2", "--seed-file", "s.txt"],
            ["--seed-image", "--seed-random-per-voxel", "--seed-file"],
        ),
        (
            [
                "--seed-image",
                MISSING,
                "--seed-random-per-voxel",
                MISSING,
                "2",
                "--seed-file",
                "s.txt",
                "--select",
                "5",
            ],
            ["--seed-image", "--seed-random-per-voxel", "--seed-file"],
        ),
        (["--seed-image", MISSING], ["--select"]),
        ([*ONE_SOURCE[:2], "--select", "0"], ["--select"]),
        (["--seed-file", "s.txt", "--select", "-1"], ["--select"]),
        ([*ONE_SOURCE, "--cutoff", "abc"], ["--cutoff"]),
        ([*ONE_SOURCE, "--cutoff", "-1"], ["--cutoff"]),
        ([*ONE_SOURCE, "--cutoff", "inf"], ["--cutoff"]),
        ([*ONE_SOURCE, "--cutoff", "nan"], ["--cutoff"]),
        ([*ONE_SOURCE, "--nt", "0"], ["--nt"]),
        ([*ONE_SOURCE, "--nt", "-2"], ["--nt"]),
        ([*ONE_SOURCE, "--nt", str((os.cpu_count() or 1) + 1)], ["--nt"]),
        ([*ONE_SOURCE, "--seed", "-1"], ["--seed"]),
        ([*ONE_SOURCE, "--seed", str(2**64)], ["--seed"]),
        (["--seed-random-per-voxel", MISSING, "0"], ["--seed-random-per-voxel"]),
        (["--seed-random-per-voxel", MISSING, "-3"], ["--seed-random-per-voxel"]),
        ([*ONE_SOURCE, "--algorithm", "ifod2"], ["--algorithm"]),
    ],
)
def test_usage_error_names_the_option_and_touches_no_output(
    tmp_path: Path, args: list[str], named: list[str]
) -> None:
    out = tmp_path / "out.trk"
    out.write_bytes(b"old")
    json_path, seeds_path = tmp_path / "run.json", tmp_path / "seeds.txt"
    result = _run(
        tmp_path / MISSING,
        out,
        *args,
        "--json",
        json_path,
        "--output-seeds",
        seeds_path,
        "--force",
    )

    assert result.exit_code == 2, result.output
    for option in named:
        assert option in result.output
    assert out.read_bytes() == b"old"
    assert not json_path.exists()
    assert not seeds_path.exists()


@pytest.mark.parametrize("name", ["out.vtk", "out.txt", "out.tck.gz", "out"])
def test_output_that_is_not_tck_or_trk_is_a_usage_error(
    tmp_path: Path, name: str
) -> None:
    out = tmp_path / name
    result = _run(tmp_path / MISSING, out, *ONE_SOURCE)

    assert result.exit_code == 2, result.output
    assert "OUTPUT" in result.output
    assert ".tck" in result.output
    assert not out.exists()


def test_unknown_basis_is_a_usage_error_naming_the_choices(tmp_path: Path) -> None:
    result = _run(tmp_path / MISSING, tmp_path / "out.tck", *ONE_SOURCE, "--basis", "x")

    assert result.exit_code == 2
    for choice in ("tournier", "descoteaux", "leemans"):
        assert choice in result.output


@pytest.mark.parametrize("which", ["output", "json", "seeds"])
def test_existing_output_is_refused_before_reading(tmp_path: Path, which: str) -> None:
    paths = {
        "output": tmp_path / "out.tck",
        "json": tmp_path / "run.json",
        "seeds": tmp_path / "seeds.txt",
    }
    paths[which].write_bytes(b"old")
    result = _run(
        tmp_path / MISSING,
        paths["output"],
        *ONE_SOURCE,
        "--json",
        paths["json"],
        "--output-seeds",
        paths["seeds"],
    )

    assert result.exit_code == 1
    assert result.output.startswith("ERROR: ")
    assert "exists" in result.output
    assert str(paths[which]) in result.output
    assert paths[which].read_bytes() == b"old"
    assert [p for p in paths.values() if p.exists()] == [paths[which]]


def test_force_overwrites_an_existing_output(tmp_path: Path) -> None:
    out, reference = tmp_path / "out.trk", tmp_path / "reference.trk"
    _track(reference, "--seed", str(SEED), "--nt", "1")
    out.write_bytes(b"old")
    _track(out, "--seed", str(SEED), "--nt", "1", "--force")

    assert out.read_bytes() == reference.read_bytes()


@pytest.mark.parametrize("which", ["output", "json", "seeds"])
def test_missing_output_folder_is_an_error_before_reading(
    tmp_path: Path, which: str
) -> None:
    paths = {
        "output": tmp_path / "out.tck",
        "json": tmp_path / "run.json",
        "seeds": tmp_path / "seeds.txt",
    }
    paths[which] = tmp_path / "nowhere" / paths[which].name
    result = _run(
        tmp_path / MISSING,
        paths["output"],
        *ONE_SOURCE,
        "--json",
        paths["json"],
        "--output-seeds",
        paths["seeds"],
    )

    assert result.exit_code == 1
    assert result.output.startswith("ERROR: ")
    assert "nowhere" in result.output
    assert not any(p.exists() for p in paths.values())


# Runs


@pytest.mark.parametrize("algorithm", ["det", "prob"])
def test_both_algorithms_write_the_selected_streamlines(
    tmp_path: Path, algorithm: str
) -> None:
    out = tmp_path / "out.trk"
    _track(out, "--algorithm", algorithm, "--seed", str(SEED))

    tractogram = load_tractogram(out)
    assert tractogram.num_streamlines == SELECT
    assert tractogram.streamline_data["end_status"].shape == (SELECT, 2)


def test_trk_keeps_the_end_status_and_tck_does_not(tmp_path: Path) -> None:
    trk, tck = tmp_path / "out.trk", tmp_path / "out.tck"
    _track(trk, "--seed", str(SEED))
    _track(tck, "--seed", str(SEED))

    with_status, without = load_tractogram(trk), load_tractogram(tck)
    assert "end_status" in with_status.streamline_data
    assert without.streamline_data == {}
    np.testing.assert_array_equal(with_status.offsets, without.offsets)
    # TRK stores voxel-mm, so float32 points of ~15 mm move by an ulp or two.
    np.testing.assert_allclose(with_status.points, without.points, rtol=0, atol=1e-5)


@pytest.mark.parametrize("algorithm", ["det", "prob"])
@pytest.mark.parametrize("name", ["out.trk", "out.tck"])
def test_same_seed_gives_identical_files_whatever_the_threads(
    tmp_path: Path, algorithm: str, name: str
) -> None:
    one, two = tmp_path / f"1_{name}", tmp_path / f"2_{name}"
    common = ("--algorithm", algorithm, "--seed", str(SEED))
    _track(one, *common, "--nt", "1")
    _track(two, *common, "--nt", "2")

    assert one.read_bytes() == two.read_bytes()


@pytest.mark.parametrize("algorithm", ["det", "prob"])
def test_different_seeds_give_different_files(tmp_path: Path, algorithm: str) -> None:
    a, b = tmp_path / "a.tck", tmp_path / "b.tck"
    _track(a, "--algorithm", algorithm, "--seed", "1")
    _track(b, "--algorithm", algorithm, "--seed", "2")

    assert a.read_bytes() != b.read_bytes()


def test_largest_seed_is_accepted(tmp_path: Path) -> None:
    out, json_path = tmp_path / "out.tck", tmp_path / "run.json"
    _track(out, "--seed", str(2**64 - 1), "--json", json_path)

    assert _parameters(json_path)["seed"] == 2**64 - 1


def test_seed_is_drawn_when_not_given_and_recorded(tmp_path: Path) -> None:
    a, b = tmp_path / "a.tck", tmp_path / "b.tck"
    _track(a, "--algorithm", "prob", "--json", tmp_path / "a.json")
    _track(b, "--algorithm", "prob", "--json", tmp_path / "b.json")

    seed_a, seed_b = (_parameters(tmp_path / f"{n}.json")["seed"] for n in "ab")
    assert isinstance(seed_a, int)
    assert 0 <= seed_a < 2**64
    assert seed_a != seed_b
    assert a.read_bytes() != b.read_bytes()


# Run record


# det's default angle keeps the radius of curvature at 0.65 voxel at 0.1 voxel steps.
DET_ANGLE = 2 * math.degrees(math.asin(0.1 / 1.3))


@pytest.mark.parametrize(("algorithm", "angle"), [("det", DET_ANGLE), ("prob", 15.0)])
def test_json_record_holds_the_resolved_parameters(
    tmp_path: Path, algorithm: str, angle: float
) -> None:
    out, json_path = tmp_path / "out.trk", tmp_path / "run.json"
    _track(
        out,
        "--algorithm",
        algorithm,
        "--seed",
        str(SEED),
        "--nt",
        "2",
        "--json",
        json_path,
    )

    record = _record(json_path)
    parameters = _parameters(json_path)
    assert parameters["algorithm"] == algorithm
    assert parameters["max_angle"] == pytest.approx(angle)
    assert parameters["seed"] == SEED
    assert parameters["num_threads"] == 2
    assert parameters["num_streamlines_target"] == SELECT
    assert parameters["cutoff"] == float(CUTOFF)
    assert parameters["cutoff_mode"] == "given"
    assert parameters["bidirectional"] is True
    assert parameters["mask"] is True
    assert record["num_streamlines"] == SELECT
    assert record["path"] == str(out)
    if algorithm == "prob":
        sampling = record["sampling"]
        assert isinstance(sampling, dict)
        assert sampling["sampled_steps"] > 0
    else:
        assert record["sampling"] is None


def _command_from_json(path: Path) -> list[str]:
    """The arguments of a run, all read from its JSON record."""
    record = _record(path)
    p, inputs = _parameters(path), record["inputs"]
    assert isinstance(inputs, dict)
    assert isinstance(p["cutoff"], float)
    args = [
        inputs["fod"],
        "--algorithm",
        p["algorithm"],
        "--step-size",
        p["step_size"],
        "--angle",
        p["max_angle"],
        "--cutoff",
        p["cutoff"],
        "--min-length",
        p["min_length"],
        "--max-length",
        p["max_length"],
        "--basis",
        p["basis"],
        "--seed",
        p["seed"],
    ]
    if not p["bidirectional"]:
        args.append("--unidirectional")
    if p["num_streamlines_target"] is not None:
        args += ["--select", p["num_streamlines_target"]]
    if inputs["mask"] is not None:
        args += ["--mask", inputs["mask"]]
    if inputs["seed_image"] is not None:
        args += ["--seed-image", inputs["seed_image"]]
    if inputs["seed_random_per_voxel"] is not None:
        args += ["--seed-random-per-voxel", *inputs["seed_random_per_voxel"]]
    if inputs["seed_file"] is not None:
        args += ["--seed-file", inputs["seed_file"]]
    return list(map(str, args))


# Each run draws its seed, and the second command holds nothing but the record.
REPEATED_RUNS: dict[str, list[str | Path]] = {
    "image": [
        "--seed-image",
        MASK,
        "--select",
        str(SELECT),
        "--unidirectional",
        "--angle",
        "40",
        "--cutoff",
        "auto",
    ],
    "per_voxel": ["--seed-random-per-voxel", SEED_VOXELS, "4", "--cutoff", CUTOFF],
}


@pytest.mark.parametrize("algorithm", ["det", "prob"])
@pytest.mark.parametrize("source", list(REPEATED_RUNS))
def test_command_from_the_json_alone_repeats_the_run(
    tmp_path: Path, algorithm: str, source: str
) -> None:
    first, again, json_path = (
        tmp_path / "a.trk",
        tmp_path / "b.trk",
        tmp_path / "r.json",
    )
    args = ["--algorithm", algorithm, *REPEATED_RUNS[source], "--json", json_path]
    result = _run(FOD, first, "--mask", MASK, *args)
    assert result.exit_code == 0, result.output

    command = _command_from_json(json_path)
    result = _run(command[0], again, *command[1:])
    assert result.exit_code == 0, result.output

    assert again.read_bytes() == first.read_bytes()


@pytest.mark.parametrize("source", ["image", "per_voxel", "file"])
@pytest.mark.parametrize("with_mask", [True, False])
def test_json_records_the_inputs_as_absolute_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str, with_mask: bool
) -> None:
    seeds_path = tmp_path / "seeds.txt"
    _track(tmp_path / "seeds.tck", "--seed", str(SEED), "--output-seeds", seeds_path)
    # Relative paths, so the record must resolve them.
    monkeypatch.chdir(tmp_path)

    def rel(path: Path) -> str:
        try:
            return os.path.relpath(path, tmp_path)
        except ValueError:  # on Windows, a path on another drive has no relative form
            return str(path)

    sources = {
        "image": ["--seed-image", rel(SEED_VOXELS), "--select", "5"],
        "per_voxel": ["--seed-random-per-voxel", rel(SEED_VOXELS), "2"],
        "file": ["--seed-file", "seeds.txt"],
    }
    mask = ["--mask", rel(MASK)] if with_mask else []
    result = _run(
        rel(FOD),
        "out.tck",
        *mask,
        *sources[source],
        "--cutoff",
        CUTOFF,
        "--json",
        "run.json",
    )
    assert result.exit_code == 0, result.output

    seed_voxels = str(SEED_VOXELS.resolve())
    assert _record(tmp_path / "run.json")["inputs"] == {
        "fod": str(FOD.resolve()),
        "mask": str(MASK.resolve()) if with_mask else None,
        "seed_image": seed_voxels if source == "image" else None,
        "seed_random_per_voxel": [seed_voxels, 2] if source == "per_voxel" else None,
        "seed_file": str(seeds_path.resolve()) if source == "file" else None,
    }


@pytest.mark.parametrize("algorithm", ["det", "prob"])
def test_seed_file_with_the_same_seed_reproduces_the_streamlines(
    tmp_path: Path, algorithm: str
) -> None:
    first, again = tmp_path / "first.trk", tmp_path / "again.trk"
    seeds_path, json_path = tmp_path / "seeds.txt", tmp_path / "run.json"
    common = ("--algorithm", algorithm, "--seed", str(SEED))
    _track(first, *common, "--output-seeds", seeds_path)
    _track(again, *common, "--seed-file", seeds_path, "--json", json_path)

    # The first run rejects seeds, so the files only match if each streamline's
    # random numbers follow its seed, not its draw index.
    assert _record(json_path)["num_seeds"] == SELECT
    _assert_same(load_tractogram(first), load_tractogram(again))


def test_seed_file_with_another_seed_gives_other_prob_streamlines(
    tmp_path: Path,
) -> None:
    first, again = tmp_path / "first.trk", tmp_path / "again.trk"
    seeds_path = tmp_path / "seeds.txt"
    _track(first, "--algorithm", "prob", "--seed", "1", "--output-seeds", seeds_path)
    _track(again, "--algorithm", "prob", "--seed", "2", "--seed-file", seeds_path)

    a, b = load_tractogram(first), load_tractogram(again)
    assert a.points.shape != b.points.shape or not np.array_equal(a.points, b.points)


# Seeds per voxel


def _per_voxel(out: Path, n: int, *args: str | Path) -> Result:
    return _track(out, "--seed-random-per-voxel", SEED_VOXELS, str(n), *args)


@pytest.mark.parametrize("algorithm", ["det", "prob"])
def test_seeds_per_voxel_follow_the_run_seed(tmp_path: Path, algorithm: str) -> None:
    a, b, c = tmp_path / "a.tck", tmp_path / "b.tck", tmp_path / "c.tck"
    seeds = {p: tmp_path / f"{p.stem}.txt" for p in (a, b, c)}
    _per_voxel(
        a, 5, "--algorithm", algorithm, "--seed", "1", "--output-seeds", seeds[a]
    )
    _per_voxel(
        b, 5, "--algorithm", algorithm, "--seed", "1", "--output-seeds", seeds[b]
    )
    _per_voxel(
        c, 5, "--algorithm", algorithm, "--seed", "2", "--output-seeds", seeds[c]
    )

    assert a.read_bytes() == b.read_bytes()
    assert seeds[a].read_bytes() == seeds[b].read_bytes()
    assert seeds[a].read_bytes() != seeds[c].read_bytes()


@pytest.mark.parametrize("n", [1, 3])
def test_every_seed_per_voxel_is_tracked(tmp_path: Path, n: int) -> None:
    json_path = tmp_path / "run.json"
    _per_voxel(tmp_path / "out.tck", n, "--seed", str(SEED), "--json", json_path)

    voxels = int(np.count_nonzero(load_volume(SEED_VOXELS).data))
    record = _record(json_path)
    assert record["num_seeds"] == n * voxels
    assert _parameters(json_path)["num_streamlines_target"] is None


def test_seeds_per_voxel_come_from_their_own_generator_of_the_run_seed(
    tmp_path: Path,
) -> None:
    out, expected = tmp_path / "out.trk", tmp_path / "expected.trk"
    _per_voxel(out, 4, "--algorithm", "prob", "--seed", str(SEED), "--nt", "1")

    seeds = seeds_per_voxel(load_volume(SEED_VOXELS), 4, SEED)
    Tracker("prob", cutoff=float(CUTOFF), progress=False).track(
        load_volume(FOD),
        expected,
        seeds,
        mask=load_volume(MASK),
        seed=SEED,
        n_jobs=1,
    )
    assert out.read_bytes() == expected.read_bytes()


# Basis


def test_fod_in_another_basis_tracks_as_in_tournier(tmp_path: Path) -> None:
    fod = load_volume(FOD)
    descoteaux = tmp_path / "descoteaux.nii.gz"
    coeffs = convert_sh_basis(fod.data.astype(np.float64), "tournier", "descoteaux")
    save_volume(Volume(coeffs.astype(np.float32), fod.affine), descoteaux)
    expected, out, misread = (tmp_path / f"{n}.tck" for n in ("ref", "out", "bad"))
    _track(expected, "--seed", str(SEED))
    for path, args in ((out, ["--basis", "descoteaux"]), (misread, [])):
        result = _run(
            descoteaux,
            path,
            "--mask",
            MASK,
            "--seed-image",
            MASK,
            "--select",
            str(SELECT),
            "--cutoff",
            CUTOFF,
            "--seed",
            str(SEED),
            *args,
        )
        assert result.exit_code == 0, result.output

    a, b = load_tractogram(expected), load_tractogram(out)
    np.testing.assert_array_equal(a.offsets, b.offsets)
    # The float32 FOD is rounded differently in each basis.
    np.testing.assert_allclose(a.points, b.points, rtol=0, atol=1e-3)
    c = load_tractogram(misread)
    assert a.points.shape != c.points.shape or not np.allclose(a.points, c.points)


@pytest.mark.parametrize("asymmetric", [False, True])
def test_asymmetric_fod_is_tracked_with_one_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, asymmetric: bool
) -> None:
    fod = load_volume(FOD)
    ls, ms = get_sh_indices(8)
    full = np.zeros((*fod.data.shape[:3], 81), dtype=np.float32)
    full[..., ls * ls + ls + ms] = fod.data
    source = tmp_path / "fod.nii.gz"
    save_volume(Volume(full if asymmetric else fod.data, fod.affine), source)
    out = tmp_path / "out.tck"

    result = _run(
        source,
        out,
        "--mask",
        MASK,
        "--seed-image",
        MASK,
        "--select",
        str(SELECT),
        "--cutoff",
        CUTOFF,
        "--seed",
        str(SEED),
    )

    assert result.exit_code == 0, result.output
    assert load_tractogram(out).num_streamlines == SELECT
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    message = (
        f"{source} has 81 coefficients: read as an asymmetric FOD (all degrees, "
        "lmax 8)."
    )
    assert warnings == ([message] if asymmetric else [])


# Direction


@pytest.mark.parametrize("algorithm", ["det", "prob"])
def test_unidirectional_streamlines_end_at_their_seed_with_status_none(
    tmp_path: Path, algorithm: str
) -> None:
    both, one = tmp_path / "both.trk", tmp_path / "one.trk"
    _track(both, "--algorithm", algorithm, "--seed", str(SEED))
    _track(one, "--algorithm", algorithm, "--seed", str(SEED), "--unidirectional")

    status = load_tractogram(one).streamline_data["end_status"]
    assert (status[:, 0] == EndStatus.NONE).all()
    assert (status[:, 1] != EndStatus.NONE).all()
    assert (load_tractogram(both).streamline_data["end_status"] != EndStatus.NONE).all()


# Messages


@pytest.mark.parametrize("algorithm", ["det", "prob"])
def test_info_logs_the_parameters_and_a_summary(tmp_path: Path, algorithm: str) -> None:
    out, json_path = tmp_path / "out.trk", tmp_path / "run.json"
    result = _track(
        out,
        "--algorithm",
        algorithm,
        "--seed",
        str(SEED),
        "--json",
        json_path,
        "--info",
    )

    info = [line for line in result.output.splitlines() if line.startswith("INFO: ")]
    for name, value in _parameters(json_path).items():
        assert f"INFO: {name}: {value}" in info
    record = _record(json_path)
    sampling = record["sampling"]
    if algorithm == "prob":
        assert isinstance(sampling, dict)
        for name, value in sampling.items():
            assert f"INFO: {name}: {value}" in info
    else:
        assert not any("mean_samples_per_step" in line for line in info)
    summary = [line for line in info if line.startswith("INFO: Saved ")]
    assert len(summary) == 1
    assert f"Saved {SELECT} streamlines to {out}" in summary[0]
    assert f"from {record['num_seeds']} seeds" in summary[0]


def test_without_info_nothing_is_logged_at_info_level(tmp_path: Path) -> None:
    result = _track(tmp_path / "out.tck", "--algorithm", "prob", "--seed", str(SEED))

    assert "INFO" not in result.output


def test_quiet_shows_no_progress_bar(tmp_path: Path) -> None:
    result = _track(tmp_path / "out.trk", "--seed", str(SEED), "--quiet")

    assert result.output == ""


def test_help_lists_the_options_without_a_seed_direction() -> None:
    result = CliRunner().invoke(app, ["track", "--help"], env={"TERM": "dumb"})
    command = typer.main.get_command(app)
    assert isinstance(command, typer.core.TyperGroup)
    # Read from the command, as rich cuts long option names at 80 columns.
    options = {o for p in command.commands["track"].params for o in p.opts}

    assert result.exit_code == 0
    assert options >= {
        "--algorithm",
        "--seed-image",
        "--seed-random-per-voxel",
        "--seed-file",
        "--select",
        "--mask",
        "--unidirectional",
        "--step-size",
        "--angle",
        "--min-length",
        "--max-length",
        "--cutoff",
        "--basis",
        "--seed",
        "--json",
        "--output-seeds",
        "--force",
        "--nt",
        "--info",
    }
    assert "--seed-direction" not in options
    assert "--seed-direction" not in result.output
    assert "--seed-image" in result.output
    assert "prob" in result.output
