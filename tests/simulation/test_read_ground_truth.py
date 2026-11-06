import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from larmor.io import load_volume
from larmor.simulation import (
    BundleGeometry,
    GroundTruth,
    SimulatedDWI,
    example_phantom,
    read_ground_truth,
    save_simulation,
    simulate,
)


@pytest.fixture(scope="module")
def run() -> tuple[SimulatedDWI, GroundTruth]:
    example = example_phantom("crossing")
    assert example.protocol is not None
    return simulate(
        example.phantom, example.protocol, seed=1, bundle_names=["LR", "AP"]
    )


@pytest.fixture(scope="module")
def folder(
    run: tuple[SimulatedDWI, GroundTruth], tmp_path_factory: pytest.TempPathFactory
) -> Path:
    path = tmp_path_factory.mktemp("sim") / "out"
    save_simulation(*run, path)
    return path


def test_the_ground_truth_reads_back_as_simulate_returned_it(
    run: tuple[SimulatedDWI, GroundTruth], folder: Path
) -> None:
    _, truth = run
    read = read_ground_truth(folder)
    assert read.partition is None
    assert read.tissue_names == truth.tissue_names
    assert read.bundle_names == truth.bundle_names == ("LR", "AP")
    # The saved phantom gives the same streamlines again: same tissues and seeds.
    assert [t.name for t in read.phantom.tissues] == [
        t.name for t in truth.phantom.tissues
    ]
    assert len(read.phantom.bundles) == len(truth.phantom.bundles)
    for read_bundle, bundle in zip(
        read.phantom.bundles, truth.phantom.bundles, strict=True
    ):
        assert isinstance(read_bundle.source, BundleGeometry)
        assert isinstance(bundle.source, BundleGeometry)
        assert bundle.source.seed is not None
        assert read_bundle.source.seed == bundle.source.seed
    np.testing.assert_allclose(read.grid.affine, truth.grid.affine)
    assert read.grid.shape == truth.grid.shape
    # The maps are stored as float32, fractions of at most 1.
    for name in ("tissue_fractions", "background", "bundle_fractions"):
        np.testing.assert_allclose(
            getattr(read, name), getattr(truth, name), atol=1e-7, err_msg=name
        )
    np.testing.assert_allclose(read.peaks, truth.peaks, atol=1e-7)
    assert len(read.streamlines) == len(truth.streamlines)
    for got, expected in zip(read.streamlines, truth.streamlines, strict=True):
        np.testing.assert_allclose(got.points, expected.points, atol=1e-4)
        np.testing.assert_array_equal(got.offsets, expected.offsets)


def test_the_peaks_of_simulate_are_the_saved_peaks(
    run: tuple[SimulatedDWI, GroundTruth], folder: Path
) -> None:
    _, truth = run
    saved = load_volume(folder / "peaks.nii.gz").data
    assert np.isnan(truth.peaks).any()
    np.testing.assert_array_equal(np.isnan(saved), np.isnan(truth.peaks))
    np.testing.assert_allclose(saved, truth.peaks, atol=1e-7)


def test_a_folder_saved_with_a_connectome_reads_and_loses_it_when_saved_over(
    run: tuple[SimulatedDWI, GroundTruth], folder: Path, tmp_path: Path
) -> None:
    old = tmp_path / "old"
    shutil.copytree(folder, old)
    info = json.loads((old / "simulation.json").read_text())
    names = {
        "connectome_parcellation": "connectome_nodes.nii.gz",
        "connectome_lookup_table": "connectome_nodes.txt",
        "connectome": "connectome.csv",
        "connectome_proportion": "connectome_proportion.csv",
    }
    info["files"].update(names)
    info["connectome"] = {"nodes": ["LR start", "LR end"]}
    (old / "simulation.json").write_text(json.dumps(info))
    for name in names.values():
        (old / name).write_text("old")

    assert len(read_ground_truth(old).streamlines) == len(run[1].streamlines)

    save_simulation(*run, old)
    assert not [name for name in names.values() if (old / name).exists()]
    assert "connectome" not in json.loads((old / "simulation.json").read_text())


@pytest.mark.parametrize("broken", ["missing", "format"])
def test_a_folder_that_is_not_a_simulation_is_refused(
    folder: Path, tmp_path: Path, broken: str
) -> None:
    copy = tmp_path / "copy"
    shutil.copytree(folder, copy)
    info_path = copy / "simulation.json"
    if broken == "missing":
        info_path.unlink()
        match = "holds no simulation.json"
    else:
        info = json.loads(info_path.read_text())
        info["format"] = "something-else"
        info_path.write_text(json.dumps(info))
        match = "not a Larmor simulation file"
    with pytest.raises(ValueError, match=match):
        read_ground_truth(copy)
