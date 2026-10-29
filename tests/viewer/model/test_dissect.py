"""The Dissect workspace's model: the fit, the tracking, the steps and the save."""

import json
import queue
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner, Result

from larmor.io import load_tractogram, load_volume
from larmor.main import app
from larmor.types import Tractogram, Volume
from larmor.viewer import Basis, Category, OptionName, Viewer
from larmor.viewer.dissect_jobs import start_tracking
from larmor.viewer.jobs import JobRunner
from larmor.viewer.model.dissect import (
    Dissection,
    DissectStep,
    DwiFiles,
    FodMerge,
    FodMethod,
    FodSettings,
    Seeding,
    StartFrom,
    TrackedRun,
    TrackSettings,
    dissect_status,
    fit_fods,
    plan_save,
    track,
    write_regions,
    write_save,
)

DATA = Path(__file__).parents[2] / "data"
DKI, TRACKING, CONNECTOME = DATA / "dki", DATA / "tracking", DATA / "connectome"


def _lar(*args: str | Path) -> Result:
    result = CliRunner().invoke(app, [str(a) for a in args], env={"TERM": "dumb"})
    assert result.exit_code == 0, result.output
    return result


def _same(a: Tractogram, b: Tractogram) -> None:
    np.testing.assert_array_equal(a.offsets, b.offsets)
    np.testing.assert_allclose(a.points, b.points, atol=1e-5)


# The data is too small to estimate the three MSMT-CSD responses, so they are given.
MSMT = DATA / "msmt"
MSMT_SETTINGS = FodSettings(
    FodMethod.MSMT_CSD,
    lmax=4,
    wm_response=MSMT / "wm.txt",
    gm_response=MSMT / "gm.txt",
    csf_response=MSMT / "csf.txt",
)
MSMT_ARGS = ("--wm-response", MSMT / "wm.txt", "--gm-response", MSMT / "gm.txt",
             "--csf-response", MSMT / "csf.txt")  # fmt: skip


@pytest.mark.parametrize(
    ("settings", "args"),
    [
        (FodSettings(lmax=4), ()),
        (FodSettings(FodMethod.CSD, lmax=4), ()),
        (MSMT_SETTINGS, MSMT_ARGS),
    ],
    ids=["GRL", "CSD", "MSMT-CSD"],
)
def test_the_fit_gives_the_fods_of_lar_fod(
    tmp_path: Path, settings: FodSettings, args: tuple[str | Path, ...]
) -> None:
    dwi = DwiFiles(DKI / "dwi.nii.gz", DKI / "bvals", DKI / "bvecs")
    mask = load_volume(DKI / "mask.nii.gz")
    stages: list[str] = []
    fit = fit_fods(dwi, mask, settings, lambda d, t, stage: stages.append(stage))
    _lar("fod", dwi.path, DKI / "mask.nii.gz", tmp_path / "out", "--bvals",
         DKI / "bvals", "--bvecs", DKI / "bvecs", "--method",
         settings.method.value, "--lmax", "4", *args)  # fmt: skip
    expected = load_volume(tmp_path / "out.nii.gz").data
    np.testing.assert_allclose(fit.fods[0].data, expected, rtol=1e-5, atol=1e-6)
    fractions = tmp_path / "out_tissue_fraction.nii.gz"
    assert (fit.fractions is None) == (not fractions.exists())
    if fit.fractions is not None:
        np.testing.assert_allclose(
            fit.fractions.data, load_volume(fractions).data, rtol=1e-5, atol=1e-6
        )
    assert "Fitting voxels" in stages


@pytest.mark.parametrize("merge", list(FodMerge))
def test_an_mfod_fit_merges_its_two_fods_as_lar_fod_does(
    tmp_path: Path, merge: FodMerge
) -> None:
    mask = load_volume(DKI / "mask.nii.gz")
    dwi = DwiFiles(DKI / "dwi.nii.gz", DKI / "bvals", DKI / "bvecs")
    settings = FodSettings(FodMethod.MFOD, lmax=4, fod_merge=merge)

    fit = fit_fods(dwi, mask, settings, lambda *_: None)
    _lar("fod", dwi.path, DKI / "mask.nii.gz", tmp_path / "out", "--bvals",
         DKI / "bvals", "--bvecs", DKI / "bvecs", "--method", "mfod", "--lmax",
         "4", "--fod-merge", merge.value)  # fmt: skip

    names = ["out_fod1", "out_fod2"] if merge == FodMerge.NONE else ["out"]
    assert len(fit.fods) == len(names)
    for fod, name in zip(fit.fods, names, strict=True):
        expected = load_volume(tmp_path / f"{name}.nii.gz").data
        np.testing.assert_allclose(fod.data, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    ("settings", "refused"),
    [
        (FodSettings(FodMethod.CSD, csf_diffusivity=None), "CSF diffusivity"),
        (FodSettings(FodMethod.MSMT_CSD, inner_shell_weight=0.5), "inner-shell"),
        (FodSettings(FodMethod.GRL, response=Path("r.txt")), "CSD response"),
        (FodSettings(FodMethod.GRL, fod_merge=FodMerge.MAJORITY), "NODDI"),
    ],
    ids=["csd-csf", "msmt-inner-shell", "grl-csd-response", "grl-merge"],
)
def test_a_fit_refuses_the_choices_its_method_does_not_take(
    settings: FodSettings, refused: str
) -> None:
    dwi = DwiFiles(DKI / "dwi.nii.gz", DKI / "bvals", DKI / "bvecs")
    mask = load_volume(DKI / "mask.nii.gz")
    with pytest.raises(ValueError, match=refused):
        fit_fods(dwi, mask, settings, lambda *_: None)


def test_the_tracking_gives_the_tractogram_of_lar_track(tmp_path: Path) -> None:
    fod, mask = (
        load_volume(TRACKING / "fod.nii.gz"),
        load_volume(TRACKING / "mask.nii.gz"),
    )
    settings = TrackSettings(streamlines=30, cutoff=0.08, seed=3)
    run = track(fod, Basis.TOURNIER, mask, mask, settings, lambda *_: None)
    _lar("track", TRACKING / "fod.nii.gz", tmp_path / "t.tck", "--seed-image",
         TRACKING / "mask.nii.gz", "--mask", TRACKING / "mask.nii.gz", "--select",
         "30", "--cutoff", "0.08", "--seed", "3")  # fmt: skip
    assert run.tractogram is not None
    _same(run.tractogram, load_tractogram(tmp_path / "t.tck"))


def test_a_tractogram_start_skips_the_fods_and_track_steps() -> None:
    state = Dissection()
    state.begin(StartFrom.TRACTOGRAM)
    rows = {s.step: s for s in dissect_status(state, Viewer().scene)}
    assert rows[DissectStep.FODS].skipped and rows[DissectStep.TRACK].skipped
    assert not rows[DissectStep.REGIONS].skipped and rows[DissectStep.START].ready


# The run of the save tests: 40 streamlines from the whole mask.
WHOLE_MASK = TrackSettings(streamlines=40, cutoff=0.08, seed=5)


def _tracked(
    viewer: Viewer,
    state: Dissection,
    settings: TrackSettings = WHOLE_MASK,
) -> tuple[Tractogram, Volume]:
    """Tracks the test FOD added without a file, as the Track step does."""
    fod = load_volume(TRACKING / "fod.nii.gz")
    fod_id = viewer.add(fod, Category.FODS, name="fod").id
    mask = load_volume(TRACKING / "mask.nii.gz")
    mask_id = viewer.add(mask, Category.OVERLAYS, name="mask").id
    run = track(fod, Basis.TOURNIER, mask, mask, settings, lambda *_: None)
    assert run.tractogram is not None
    tract_id = viewer.add(run.tractogram, Category.TRACTOGRAMS, name="wb").id
    state.begin(StartFrom.FOD)
    state.fod, state.tractogram = fod_id, tract_id
    state.runs[tract_id] = TrackedRun(
        run, fod_id, mask_id, mask_id, settings.seeding, settings.seeds_per_voxel
    )
    state.adopt(tract_id)
    return run.tractogram, mask


def test_a_save_reopens_as_the_same_bundle_and_reruns_with_lar_track(
    tmp_path: Path,
) -> None:
    viewer, state = Viewer(), Dissection()
    whole, _ = _tracked(viewer, state)
    tract = state.tractogram
    assert tract is not None
    x, y, z = (float(v) for v in whole.points[0])  # an end of the first streamline
    viewer.add_sphere((x, y, z), 2.0, apply=False)
    region = viewer.scene.in_category(Category.REGIONS)[0].id
    viewer.apply_region(tract, region)
    viewer.set(tract, OptionName.ENDS_ONLY, True)
    kept = viewer.kept_streamlines(tract)
    assert 0 < kept < whole.num_streamlines

    unsaved = state.unsaved(viewer.scene)
    assert [viewer.scene.item(i).name for i in unsaved] == ["fod", "mask", "wb"]
    plan = plan_save(state, viewer.scene, tmp_path, "cst", [tract])
    write_save(viewer, state, plan)
    assert all(p.is_file() for p in plan.paths)
    assert load_tractogram(plan.bundle).num_streamlines == kept

    other = Viewer()
    other_id = other.load(plan.items[tract], Category.TRACTOGRAMS).id
    report = other.open_regions(plan.regions, other_id)
    assert report.problems == ()
    assert other.kept_streamlines(other_id) == kept

    assert plan.run is not None
    record = json.loads(plan.run.read_text(encoding="utf-8"))
    inputs, parameters = record["inputs"], record["parameters"]
    _lar("track", inputs["fod"], tmp_path / "again.tck", "--seed-image",
         inputs["seed_image"], "--mask", inputs["mask"], "--select",
         parameters["num_streamlines_target"], "--cutoff", parameters["cutoff"],
         "--seed", parameters["seed"])  # fmt: skip
    _same(load_tractogram(tmp_path / "again.tck"), whole)


def test_a_per_voxel_run_reruns_from_its_record_with_lar_track(
    tmp_path: Path,
) -> None:
    viewer, state = Viewer(), Dissection()
    settings = TrackSettings(
        cutoff=0.08, seed=7, seeding=Seeding.PER_VOXEL, seeds_per_voxel=1
    )
    whole, _ = _tracked(viewer, state, settings)
    plan = plan_save(state, viewer.scene, tmp_path, "cst", [])
    write_save(viewer, state, plan)
    assert plan.run is not None
    record = json.loads(plan.run.read_text(encoding="utf-8"))
    inputs, parameters = record["inputs"], record["parameters"]
    seeds, per_voxel = inputs["seed_random_per_voxel"]
    _lar("track", inputs["fod"], tmp_path / "again.tck", "--seed-random-per-voxel",
         seeds, per_voxel, "--mask", inputs["mask"], "--cutoff",
         parameters["cutoff"], "--seed", parameters["seed"])  # fmt: skip
    _same(load_tractogram(tmp_path / "again.tck"), whole)


def test_a_saved_fod_reloads_as_the_same_fod_in_its_basis(tmp_path: Path) -> None:
    viewer = Viewer()
    fod = load_volume(TRACKING / "fod.nii.gz")
    item = viewer.add(fod, Category.FODS, name="fod")
    viewer.set(item.id, OptionName.BASIS, Basis.DESCOTEAUX)
    viewer.save_item(item.id, tmp_path / "fod.nii.gz")
    viewer.save_session(tmp_path / "s.lar.json")
    other = Viewer()
    other.load_session(tmp_path / "s.lar.json")
    (again,) = other.scene.in_category(Category.FODS)
    assert again.options.basis == Basis.DESCOTEAUX  # type: ignore[attr-defined]
    assert isinstance(again.data, Volume)
    np.testing.assert_array_equal(again.data.data, fod.data)


def test_a_save_refuses_a_missing_tractogram_or_name(tmp_path: Path) -> None:
    viewer, state = Viewer(), Dissection()
    with pytest.raises(ValueError, match="no tractogram"):
        plan_save(state, viewer.scene, tmp_path, "cst", [])
    _tracked(viewer, state)
    with pytest.raises(ValueError, match="name"):
        plan_save(state, viewer.scene, tmp_path, " ", [])


def test_a_regions_file_keeps_a_label_region_and_its_parcellation(
    tmp_path: Path,
) -> None:
    viewer = Viewer()
    tract = viewer.load(CONNECTOME / "tracks.tck", Category.TRACTOGRAMS).id
    parcellation = viewer.load(CONNECTOME / "parcellation.nii.gz", Category.OVERLAYS)
    viewer.add_label_region(parcellation.id, labels_a=[1], apply=True)
    kept = viewer.kept_streamlines(tract)
    write_regions(viewer.scene, tract, tmp_path / "r.regions.json")

    other = Viewer()
    other_id = other.load(CONNECTOME / "tracks.tck", Category.TRACTOGRAMS).id
    report = other.open_regions(tmp_path / "r.regions.json", other_id)
    assert report.problems == ()
    assert other.kept_streamlines(other_id) == kept


def _track_loaded(viewer: Viewer, state: Dissection) -> int:
    """Tracks the test FOD loaded from its file, through the job runner."""
    fod = viewer.load(TRACKING / "fod.nii.gz", Category.FODS).id
    viewer.set(fod, OptionName.BASIS, Basis.TOURNIER)
    mask = viewer.load(TRACKING / "mask.nii.gz", Category.OVERLAYS).id
    posted: queue.Queue[Callable[[], None]] = queue.Queue()
    jobs = JobRunner(posted.put)
    settings = TrackSettings(streamlines=20, cutoff=0.08, seed=5)
    state.begin(StartFrom.FOD)
    start_tracking(viewer, jobs, state, fod, mask, mask, settings)
    while jobs.busy:
        posted.get(timeout=10.0)()
    return fod


def test_a_save_after_removing_the_fod_names_its_file_in_the_run(
    tmp_path: Path,
) -> None:
    viewer, state = Viewer(), Dissection()
    fod = _track_loaded(viewer, state)
    viewer.remove(fod)
    plan = plan_save(state, viewer.scene, tmp_path, "cst", [])
    write_save(viewer, state, plan)
    assert plan.run is not None and plan.notes == ()
    record = json.loads(plan.run.read_text(encoding="utf-8"))
    assert record["inputs"]["fod"] == str((TRACKING / "fod.nii.gz").resolve())


def test_a_save_after_removing_an_unsaved_fod_leaves_out_the_run(
    tmp_path: Path,
) -> None:
    viewer, state = Viewer(), Dissection()
    _tracked(viewer, state)
    assert state.fod is not None
    viewer.remove(state.fod)
    plan = plan_save(state, viewer.scene, tmp_path, "cst", [])
    write_save(viewer, state, plan)
    assert plan.run is None and len(plan.notes) == 1
    assert plan.bundle.is_file() and plan.regions.is_file()


def test_items_of_the_same_name_are_saved_to_different_files(tmp_path: Path) -> None:
    viewer, state = Viewer(), Dissection()
    _tracked(viewer, state)
    mask = load_volume(TRACKING / "mask.nii.gz")
    a = viewer.add(mask, Category.OVERLAYS, name="a b").id
    b = viewer.add(mask, Category.OVERLAYS, name="a_b").id
    plan = plan_save(state, viewer.scene, tmp_path, "cst", [a, b])
    assert len(set(plan.paths)) == len(plan.paths)
    assert plan.items[b].name == "cst_a_b_2.nii.gz"
