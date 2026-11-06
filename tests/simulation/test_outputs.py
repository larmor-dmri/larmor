import json
import math
import shutil
from dataclasses import replace
from pathlib import Path
from typing import cast

import nibabel as nib
import numpy as np
import pytest

from larmor.io import (
    load_tractogram,
    load_volume,
    read_bvals_bvecs,
)
from larmor.simulation import (
    EXAMPLE_PHANTOMS,
    Bundle,
    BundleGeometry,
    CompartmentModel,
    GroundTruth,
    Phantom,
    Protocol,
    PulseSequence,
    SimulatedDWI,
    Tissue,
    TissueMap,
    create_gradient_table,
    csf_tissue,
    example_phantom,
    partition_voxels,
    read_phantom,
    save_simulation,
    simulate,
    simulation_files,
    standard_model_tissue,
)
from larmor.simulation.ground_truth import _voxel_peaks, build_ground_truth
from larmor.types import Grid, Tractogram, Volume

ANGLE = math.radians(25)
AFFINE = np.array(
    [
        [2 * math.cos(ANGLE), -2 * math.sin(ANGLE), 0, -9.0],
        [2 * math.sin(ANGLE), 2 * math.cos(ANGLE), 0, 4.0],
        [0, 0, 2.5, -6.0],
        [0, 0, 0, 1],
    ]
)
GRID = Grid((14, 14, 8), AFFINE)
PROTOCOL = Protocol(
    create_gradient_table([(1000, 10)], num_b0=2, seed=0), 15.0, 30.0, 80.0, 3000.0
)
X = np.array([1.0, 0, 0])


def _load(path: Path) -> np.ndarray:
    return np.asarray(cast(nib.Nifti1Image, nib.load(path)).dataobj)


def crossing(angle_deg: float = 90.0) -> Phantom:
    centre = (AFFINE @ np.array([7.0, 7.0, 3.0, 1.0]))[:3]
    a = math.radians(angle_deg)
    bundles = []
    for index, direction in enumerate(
        (np.array([1.0, 0, 0]), np.array([math.cos(a), math.sin(a), 0]))
    ):
        line = np.array([centre - 12 * direction, centre + 12 * direction])
        geometry = BundleGeometry(line, num_streamlines=60, radius=2.0, seed=index)
        bundles.append(Bundle(geometry))
    return Phantom(GRID, bundles=tuple(bundles))


@pytest.fixture(scope="module")
def run() -> tuple[SimulatedDWI, GroundTruth]:
    return simulate(crossing(), PROTOCOL, snr=25, seed=3)


@pytest.fixture(scope="module")
def simulated(run: tuple[SimulatedDWI, GroundTruth]) -> SimulatedDWI:
    return run[0]


@pytest.fixture(scope="module")
def truth(run: tuple[SimulatedDWI, GroundTruth]) -> GroundTruth:
    return run[1]


@pytest.fixture(scope="module")
def saved(
    simulated: SimulatedDWI,
    truth: GroundTruth,
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    folder = tmp_path_factory.mktemp("sim") / "out"
    save_simulation(simulated, truth, folder)
    return folder


def test_the_images_and_gradients_read_back(
    simulated: SimulatedDWI, saved: Path
) -> None:
    image = cast(nib.Nifti1Image, nib.load(saved / "dwi.nii.gz"))
    np.testing.assert_array_equal(
        np.asarray(image.dataobj), simulated.echoes[0].data.astype(np.float32)
    )
    np.testing.assert_allclose(image.affine, AFFINE)
    clean = cast(nib.Nifti1Image, nib.load(saved / "dwi_noise_free.nii.gz"))
    np.testing.assert_array_equal(
        np.asarray(clean.dataobj), simulated.noise_free[0].data
    )
    np.testing.assert_allclose(
        np.loadtxt(saved / "dwi.bval"), simulated.gradients.bvals
    )


def test_the_bvecs_are_the_world_directions_in_the_fsl_frame(
    simulated: SimulatedDWI, saved: Path
) -> None:
    # FSL: components along the image axes (unit columns of the affine), x flipped
    # when the affine has a positive determinant.
    rotation = AFFINE[:3, :3] / np.linalg.norm(AFFINE[:3, :3], axis=0)
    expected = simulated.gradients.bvecs @ rotation
    if np.linalg.det(AFFINE[:3, :3]) > 0:
        expected[:, 0] *= -1
    written = np.loadtxt(saved / "dwi.bvec")
    assert written.shape == (3, len(simulated.gradients.bvals))
    np.testing.assert_allclose(written.T, expected, atol=1e-7)
    table = read_bvals_bvecs(
        saved / "dwi.bval", saved / "dwi.bvec", ref_path=saved / "dwi.nii.gz"
    )
    np.testing.assert_allclose(table.bvecs, simulated.gradients.bvecs, atol=1e-12)


def test_the_fractions_masks_and_bundles_are_the_partition(
    truth: GroundTruth, saved: Path
) -> None:
    partition = truth.partition
    assert partition is not None
    np.testing.assert_allclose(
        _load(saved / "tissue_fractions.nii.gz"),
        partition.tissue_fractions,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        _load(saved / "bundle_fractions.nii.gz"),
        partition.bundle_fractions,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        _load(saved / "background_fraction.nii.gz"),
        partition.background,
        atol=1e-6,
    )
    wm = _load(saved / "mask_WM.nii.gz")
    np.testing.assert_array_equal(wm, partition.tissue_fractions[..., 0] >= 0.5)
    assert wm.any()
    assert not _load(saved / "mask_GM.nii.gz").any()
    for index, streamlines in enumerate(partition.streamlines, 1):
        read = load_tractogram(saved / f"bundle_{index}.tck")
        np.testing.assert_allclose(read.points, streamlines.points, atol=1e-4)
        np.testing.assert_array_equal(read.offsets, streamlines.offsets)


def test_a_crossing_of_two_bundles_at_90_degrees_has_two_peaks_90_degrees_apart(
    saved: Path,
) -> None:
    peaks = _load(saved / "peaks.nii.gz")
    voxel = peaks[7, 7, 3]
    first, second = voxel[:3], voxel[3:6]
    assert np.isfinite(voxel[:6]).all()
    cosine = abs(first @ second) / np.linalg.norm(first) / np.linalg.norm(second)
    assert math.degrees(math.acos(min(cosine, 1.0))) == pytest.approx(90.0, abs=3.0)
    # Far from the crossing one bundle remains: a single peak along x, and no second.
    single = peaks[2, 2, 3]
    assert np.isfinite(single).sum() in (0, 3)


def test_two_bundles_crossing_at_15_degrees_stay_two_peaks() -> None:
    # A straight bundle's fibres all point one way, so any merging at the crossing
    # would join the two bundles, not one bundle's own spread.
    phantom = crossing(15.0)
    truth = build_ground_truth(phantom, partition_voxels(phantom, seed=3))
    voxel = truth.peaks[7, 7, 3]
    found = int(np.isfinite(voxel).sum()) // 3
    assert found == 2, f"the two bundles gave {found} peak(s) at the crossing"
    first, second = voxel[:3], voxel[3:6]
    cosine = abs(first @ second) / np.linalg.norm(first) / np.linalg.norm(second)
    assert math.degrees(math.acos(min(cosine, 1.0))) == pytest.approx(15.0, abs=3.0)


def test_peaks_have_the_amplitude_of_the_fibres_share_of_the_voxel(
    truth: GroundTruth, saved: Path
) -> None:
    peaks = _load(saved / "peaks.nii.gz")
    amplitudes = np.nansum(
        np.linalg.norm(peaks.reshape(*peaks.shape[:3], -1, 3), axis=-1), axis=-1
    )
    occupied = truth.tissue_fractions.sum(axis=-1)
    inside = occupied > 0.2
    # Peaks below 10% of the largest are dropped, so the sum cannot exceed the share.
    assert (amplitudes[inside] <= occupied[inside] + 1e-5).all()
    assert (amplitudes[inside] > 0.8 * occupied[inside]).all()


def test_the_info_file_names_the_protocol_noise_seed_and_files(
    simulated: SimulatedDWI, saved: Path
) -> None:
    info = json.loads((saved / "simulation.json").read_text())
    assert info["seed"] == 3
    assert info["snr"] == 25
    assert info["sigma"] == pytest.approx(simulated.sigma)
    assert info["protocol"]["te_ms"] == [80.0]
    assert info["files"]["dwi"] == ["dwi.nii.gz"]
    assert info["files"]["bundles"] == ["bundle_1.tck", "bundle_2.tck"]
    for relative in [
        info["files"]["bvals"],
        info["files"]["peaks"],
        info["files"]["phantom"],
    ]:
        assert (saved / relative).is_file()


def test_the_files_returned_are_the_files_written(
    simulated: SimulatedDWI, truth: GroundTruth, saved: Path
) -> None:
    assert simulated.protocol is not None
    planned = simulation_files(truth.phantom, simulated.protocol, saved, noise=True)
    on_disk = {p for p in saved.iterdir()}
    assert set(planned.all_paths()) == on_disk


def test_running_the_saved_phantom_file_again_gives_the_same_images(
    simulated: SimulatedDWI, saved: Path, tmp_path: Path
) -> None:
    moved = tmp_path / "moved"
    shutil.copytree(saved, moved)
    loaded = read_phantom(moved / "phantom.json")
    assert loaded.protocol is not None
    again, _ = simulate(loaded.phantom, loaded.protocol, snr=25, seed=3)
    np.testing.assert_array_equal(again.echoes[0].data, simulated.echoes[0].data)


def test_a_phantom_without_geometry_seeds_is_rerun_exactly(tmp_path: Path) -> None:
    centre = (AFFINE @ np.array([7.0, 7.0, 3.0, 1.0]))[:3]
    line = np.array([centre - 10 * X, centre + 10 * X])
    phantom = Phantom(
        GRID, bundles=(Bundle(BundleGeometry(line, num_streamlines=40, radius=2.0)),)
    )
    first, truth = simulate(phantom, PROTOCOL, seed=11)
    save_simulation(first, truth, tmp_path / "out")
    loaded = read_phantom(tmp_path / "out" / "phantom.json")
    assert loaded.protocol is not None
    second, _ = simulate(loaded.phantom, loaded.protocol, seed=99)
    np.testing.assert_array_equal(second.echoes[0].data, first.echoes[0].data)


def test_tractogram_bundles_and_tissue_maps_are_copied_into_the_folder(
    tmp_path: Path,
) -> None:
    centre = (AFFINE @ np.array([7.0, 7.0, 3.0, 1.0]))[:3]
    steps = np.linspace(-10, 10, 40)[:, None]
    points = centre + steps * X
    tractogram = Tractogram(
        np.concatenate([points + [0, off, 0] for off in (-1.0, 0.0, 1.0)]),
        np.array([0, 40, 80, 120]),
    )
    gm = np.zeros(GRID.shape)
    gm[:3] = 0.5
    phantom = Phantom(
        GRID,
        bundles=(Bundle(tractogram),),
        tissue_maps=(TissueMap("GM", Volume(gm, AFFINE)),),
    )
    first, truth = simulate(phantom, PROTOCOL, seed=1)
    save_simulation(first, truth, tmp_path / "out")
    assert (tmp_path / "out" / "tissue_map_GM.nii.gz").is_file()
    moved = tmp_path / "elsewhere"
    shutil.move(tmp_path / "out", moved)
    loaded = read_phantom(moved / "phantom.json")
    second, _ = simulate(loaded.phantom, loaded.protocol or PROTOCOL, seed=1)
    np.testing.assert_allclose(second.echoes[0].data, first.echoes[0].data, rtol=1e-4)


def test_each_echo_has_its_own_file_and_shares_the_gradients(tmp_path: Path) -> None:
    protocol = Protocol(PROTOCOL.gradients, 15.0, 30.0, [60.0, 90.0], 3000.0)
    result, truth = simulate(crossing(), protocol, seed=0)
    files = save_simulation(result, truth, tmp_path / "multi")
    assert [p.name for p in files.dwi] == ["dwi_te60ms.nii.gz", "dwi_te90ms.nii.gz"]
    assert files.noise_free == ()
    for path, echo in zip(files.dwi, result.echoes, strict=True):
        np.testing.assert_array_equal(load_volume(path).data, echo.data)


def test_a_result_that_did_not_come_from_simulate_is_refused(
    simulated: SimulatedDWI, truth: GroundTruth, tmp_path: Path
) -> None:
    bare = SimulatedDWI(
        simulated.echoes,
        simulated.noise_free,
        simulated.gradients,
        simulated.echo_times_ms,
        simulated.sigma,
    )
    with pytest.raises(ValueError, match="no protocol"):
        save_simulation(bare, truth, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_tissue_names_that_give_one_file_name_are_refused(
    simulated: SimulatedDWI, truth: GroundTruth
) -> None:
    assert simulated.protocol is not None
    first, second, third = truth.phantom.tissues
    phantom = replace(
        truth.phantom,
        tissues=(replace(first, name="a b"), replace(second, name="a_b"), third),
        bundles=(),
    )
    with pytest.raises(ValueError, match="same file names"):
        simulation_files(phantom, simulated.protocol, "x", noise=False)


def test_the_example_phantom_has_a_scan_and_crosses_two_bundles() -> None:
    assert EXAMPLE_PHANTOMS == ("crossing",)
    example = example_phantom()
    assert example.protocol is not None
    assert example.protocol.gradients.bvals.shape == (66,)
    assert len(example.phantom.bundles) == 2
    assert example.phantom.grid.shape == (40, 40, 40)
    assert {t.name for t in example.phantom.tissues} >= {"WM"}
    with pytest.raises(ValueError, match="no example phantom 'nope'"):
        example_phantom("nope")


def test_without_a_seed_the_drawn_seed_reruns_the_images_exactly(
    tmp_path: Path,
) -> None:
    centre = (AFFINE @ np.array([7.0, 7.0, 3.0, 1.0]))[:3]
    line = np.array([centre - 10 * X, centre + 10 * X])
    phantom = Phantom(
        GRID, bundles=(Bundle(BundleGeometry(line, num_streamlines=40, radius=2.0)),)
    )
    first, truth = simulate(phantom, PROTOCOL, snr=20)
    save_simulation(first, truth, tmp_path / "out")
    info = json.loads((tmp_path / "out" / "simulation.json").read_text())
    assert info["seed"] == first.seed is not None
    loaded = read_phantom(tmp_path / "out" / "phantom.json")
    assert loaded.protocol is not None
    second, _ = simulate(loaded.phantom, loaded.protocol, snr=20, seed=info["seed"])
    np.testing.assert_array_equal(second.echoes[0].data, first.echoes[0].data)


def test_a_new_simulation_removes_the_files_of_the_one_before(
    simulated: SimulatedDWI, truth: GroundTruth, tmp_path: Path
) -> None:
    folder = tmp_path / "out"
    save_simulation(simulated, truth, folder)
    (folder / "notes.txt").write_text("mine")
    assert (folder / "dwi_noise_free.nii.gz").exists()
    save_simulation(*simulate(crossing(), PROTOCOL, seed=3), folder)
    assert not (folder / "dwi_noise_free.nii.gz").exists()
    assert (folder / "notes.txt").read_text() == "mine"


def test_a_folder_of_other_files_is_refused_unless_forced(
    simulated: SimulatedDWI, truth: GroundTruth, tmp_path: Path
) -> None:
    folder = tmp_path / "out"
    folder.mkdir()
    (folder / "notes.txt").write_text("mine")
    with pytest.raises(FileExistsError, match="not empty"):
        save_simulation(simulated, truth, folder)
    assert not (folder / "dwi.nii.gz").exists()
    save_simulation(simulated, truth, folder, force=True)
    assert (folder / "dwi.nii.gz").exists() and (folder / "notes.txt").exists()


def test_compartment_fractions_are_the_tissue_fraction_times_the_share(
    truth: GroundTruth, saved: Path
) -> None:
    maps = _load(saved / "compartment_fractions.nii.gz")
    shares = [c.fraction for t in truth.phantom.tissues for c in t.compartments]
    assert maps.shape[3] == len(shares)
    tissues = truth.tissue_fractions
    np.testing.assert_allclose(maps[..., 0], tissues[..., 0] * shares[0], atol=1e-6)
    np.testing.assert_allclose(maps.sum(axis=-1), tissues.sum(axis=-1), atol=1e-5)
    info = json.loads((saved / "simulation.json").read_text())
    assert [c["model"] for c in info["compartments"][:2]] == [
        "watson_stick",
        "zeppelin",
    ]


def test_the_free_water_fraction_counts_free_water_in_tissues_and_background(
    tmp_path: Path,
) -> None:
    # The background is half free water and half a plain ball, which is not
    # free water.
    (water,) = csf_tissue().compartments
    ball = replace(water, model=CompartmentModel.BALL, fraction=0.5)
    mixed = Tissue("BG", (replace(water, fraction=0.5), ball))
    tissues = (standard_model_tissue(free_water_fraction=0.1), mixed)
    phantom = replace(crossing(), tissues=tissues, background="BG")
    simulated, truth = simulate(phantom, PROTOCOL, seed=3)
    save_simulation(simulated, truth, tmp_path / "out")
    found = _load(tmp_path / "out" / "free_water_fraction.nii.gz")
    wm, bg = np.moveaxis(truth.tissue_fractions, -1, 0)
    expected = 0.1 * wm + 0.5 * (bg + truth.background)
    np.testing.assert_allclose(found, expected, atol=1e-6)
    assert found.max() == pytest.approx(0.5, abs=1e-6)


def _free_water_phantom() -> Phantom:
    # WM with 10% free water, and a CSF background: free water has a far longer
    # T2 and a higher proton density than the stick and the zeppelin.
    tissues = (standard_model_tissue(free_water_fraction=0.1), csf_tissue())
    return replace(crossing(), tissues=tissues, background="CSF")


@pytest.mark.parametrize("sequence", list(PulseSequence))
def test_signal_fractions_weight_each_compartment_by_its_relaxation(
    tmp_path: Path, sequence: PulseSequence
) -> None:
    protocol = Protocol(
        PROTOCOL.gradients, 15.0, 30.0, [60.0, 120.0], 3000.0, sequence=sequence
    )
    simulated, truth = simulate(_free_water_phantom(), protocol, seed=0)
    files = save_simulation(simulated, truth, tmp_path / "out")
    assert [p.name for p in files.signal_fractions] == [
        "signal_fractions_te60ms.nii.gz",
        "signal_fractions_te120ms.nii.gz",
    ]
    volume = _load(files.compartment_fractions).astype(np.float64)
    compartments = [c for t in truth.phantom.tissues for c in t.compartments]
    free = [c.model == CompartmentModel.FREE_WATER for c in compartments]
    seen = []
    for te, path, water in zip(
        (60.0, 120.0),
        files.signal_fractions,
        files.free_water_signal_fraction,
        strict=True,
    ):
        weights = np.array(
            [
                c.proton_density
                * (1 - math.exp(-3000.0 / c.t1_ms))
                * math.exp(
                    -te
                    / (
                        c.t2star_ms
                        if sequence == PulseSequence.GRADIENT_ECHO
                        else c.t2_ms
                    )
                )
                for c in compartments
            ]
        )
        weighted = volume * weights
        total = weighted.sum(axis=-1, keepdims=True)
        expected = np.divide(
            weighted, total, out=np.zeros_like(weighted), where=total > 0
        )
        found = _load(path)
        np.testing.assert_allclose(found, expected, atol=1e-6)
        np.testing.assert_allclose(_load(water), expected[..., free].sum(-1), atol=1e-6)
        seen.append(found)
    # In a voxel of WM only, free water fills 0.1 of the volume but gives more of
    # the signal, and more at the longer echo time.
    wm_only = volume[..., 0] > compartments[0].fraction - 1e-6
    assert wm_only.any()
    water_index = free.index(True)
    assert np.all(seen[0][wm_only, water_index] > 0.15)
    assert np.all(seen[1][wm_only, water_index] > seen[0][wm_only, water_index])


def test_opposite_fibres_are_one_peak_with_the_summed_amplitude() -> None:
    directions = np.array([[1.0, 0, 0], [-1.0, 0, 0], [0, 1.0, 0]])
    peaks = _voxel_peaks(directions, np.array([0.4, 0.3, 0.3]))
    assert [round(a, 6) for _, a in peaks] == [0.7, 0.3]
    assert abs(peaks[0][0] @ X) == pytest.approx(1.0)


def test_the_background_fraction_is_in_the_mask_of_the_background_tissue(
    tmp_path: Path,
) -> None:
    phantom = replace(crossing(), background="CSF")
    result, truth = simulate(phantom, PROTOCOL, seed=1)
    save_simulation(result, truth, tmp_path / "out")
    csf = _load(tmp_path / "out" / "mask_CSF.nii.gz")
    expected = (truth.background + truth.tissue_fractions[..., 2]) >= 0.5
    np.testing.assert_array_equal(csf, expected)
    assert csf.sum() > 0.5 * csf.size


def test_a_peak_points_along_the_bundle_in_world_coordinates(tmp_path: Path) -> None:
    direction = np.array([math.cos(math.radians(35)), math.sin(math.radians(35)), 0])
    centre = (AFFINE @ np.array([7.0, 7.0, 3.0, 1.0]))[:3]
    line = np.array([centre - 14 * direction, centre + 14 * direction])
    phantom = Phantom(
        GRID,
        bundles=(Bundle(BundleGeometry(line, num_streamlines=60, radius=1.0, seed=1)),),
    )
    save_simulation(*simulate(phantom, PROTOCOL, seed=1), tmp_path / "out")
    peak = _load(tmp_path / "out" / "peaks.nii.gz")[7, 7, 3, :3]
    assert abs(peak @ direction) / np.linalg.norm(peak) == pytest.approx(1.0, abs=1e-3)


def test_the_bundle_mask_holds_a_crossing_that_neither_tissue_mask_fills(
    tmp_path: Path,
) -> None:
    example = example_phantom()
    assert example.protocol is not None
    result, truth = simulate(example.phantom, example.protocol, seed=1)
    save_simulation(result, truth, tmp_path / "out")
    wm, wm2, both = (
        _load(tmp_path / "out" / f"mask_{n}.nii.gz")
        for n in ("WM", "WM-sparse", "bundles")
    )
    filled = truth.bundle_fractions.sum(axis=-1) >= 0.5
    hole = filled & ~(wm | wm2).astype(bool)
    assert hole.any()
    assert both[hole].all()


@pytest.mark.parametrize("names", [["only one"], ["a", "b", "c"]])
def test_bundle_names_must_name_every_bundle(names: list[str]) -> None:
    with pytest.raises(ValueError, match="must name 2 bundles"):
        simulate(crossing(), PROTOCOL, seed=1, bundle_names=names)


def test_a_bundle_of_csf_gives_no_peaks_and_the_bundle_mask_skips_the_background(
    tmp_path: Path,
) -> None:
    centre = (AFFINE @ np.array([7.0, 7.0, 3.0, 1.0]))[:3]
    line = np.array([centre - 12 * X, centre + 12 * X])
    bundle = Bundle(BundleGeometry(line, num_streamlines=60, seed=0), tissue="CSF")
    phantom = Phantom(GRID, bundles=(bundle,), background="CSF")
    result, truth = simulate(phantom, PROTOCOL, seed=1)
    save_simulation(result, truth, tmp_path / "out")
    assert np.isnan(_load(tmp_path / "out" / "peaks.nii.gz")).all()
    mask = _load(tmp_path / "out" / "mask_bundles.nii.gz").astype(bool)
    np.testing.assert_array_equal(mask, truth.bundle_fractions.sum(axis=-1) >= 0.5)


def test_echo_times_that_round_alike_get_files_of_their_own() -> None:
    protocol = Protocol(PROTOCOL.gradients, 15.0, 30.0, [80.0, 80.0000001], 3000.0)
    names = [
        p.name for p in simulation_files(crossing(), protocol, "x", noise=False).dwi
    ]
    assert len(set(names)) == 2


def test_saving_reports_its_progress_to_the_end(
    simulated: SimulatedDWI, truth: GroundTruth, tmp_path: Path
) -> None:
    seen: list[tuple[int, int, str]] = []
    save_simulation(
        simulated, truth, tmp_path / "out", progress=lambda *a: seen.append(a)
    )
    assert [done for done, _, _ in seen] == list(range(1, seen[-1][1] + 1))
    assert seen[-1][0] == seen[-1][1] and seen[-1][2] == "save"


def test_the_ground_truth_checks_that_bundle_names_name_every_bundle() -> None:
    phantom = crossing()
    partition = partition_voxels(phantom, seed=1)
    with pytest.raises(ValueError, match="must name 2 bundles"):
        build_ground_truth(phantom, partition, bundle_names=["only one"])


def _small_bundle_phantom(tissue_name: str) -> tuple[Phantom, Protocol]:
    grid = Grid((8, 8, 4), np.eye(4))
    gradients = create_gradient_table([(1000.0, 6)], seed=1)
    geometry = BundleGeometry(
        np.array([[1.0, 4.0, 2.0], [6.0, 4.0, 2.0]]), num_streamlines=20, seed=1
    )
    tissue = standard_model_tissue(name=tissue_name)
    bundle = Bundle(geometry, tissue=tissue_name)
    return Phantom(grid, tissues=(tissue,), bundles=(bundle,)), Protocol(
        gradients, 15.0, 30.0, 80.0, 3000.0
    )


def test_a_tissue_whose_mask_would_take_the_bundles_mask_file_is_refused(
    tmp_path: Path,
) -> None:
    phantom, protocol = _small_bundle_phantom("bundles")
    with pytest.raises(ValueError, match="rename the tissue"):
        simulation_files(phantom, protocol, tmp_path, noise=False)


def test_a_numpy_seed_and_snr_are_saved_as_plain_numbers(tmp_path: Path) -> None:
    phantom, protocol = _small_bundle_phantom("WM")
    simulated, truth = simulate(
        phantom,
        protocol,
        snr=np.float32(20),
        seed=np.int64(3),  # type: ignore[arg-type]
    )
    save_simulation(simulated, truth, tmp_path)
    info = json.loads((tmp_path / "simulation.json").read_text())
    assert info["seed"] == 3 and info["snr"] == 20
