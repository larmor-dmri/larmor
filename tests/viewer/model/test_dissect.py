"""The Dissect workspace's model: the fit, the tracking, the steps and the save."""

import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner, Result

from larmor.io import load_tractogram, load_volume
from larmor.main import app
from larmor.types import Tractogram, Volume
from larmor.viewer import Category, OptionName, Viewer
from larmor.viewer.model.dissect import (
    Dissection,
    DissectStep,
    DwiFiles,
    FodSettings,
    StartFrom,
    TrackedRun,
    TrackSettings,
    dissect_status,
    fit_fods,
    plan_save,
    track,
    write_save,
)
from larmor.viewer.model.options import Basis

DATA = Path(__file__).parents[2] / "data"
DKI, TRACKING = DATA / "dki", DATA / "tracking"


def _lar(*args: str | Path) -> Result:
    result = CliRunner().invoke(app, [str(a) for a in args], env={"TERM": "dumb"})
    assert result.exit_code == 0, result.output
    return result


def _same(a: Tractogram, b: Tractogram) -> None:
    np.testing.assert_array_equal(a.offsets, b.offsets)
    np.testing.assert_allclose(a.points, b.points, atol=1e-5)


def test_the_fit_gives_the_fods_of_lar_fod(tmp_path: Path) -> None:
    dwi = DwiFiles(DKI / "dwi.nii.gz", DKI / "bvals", DKI / "bvecs")
    mask = load_volume(DKI / "mask.nii.gz")
    stages: list[str] = []
    fit = fit_fods(dwi, mask, FodSettings(), lambda d, t, stage: stages.append(stage))
    _lar("fod", dwi.path, DKI / "mask.nii.gz", tmp_path / "out", "--bvals",
         DKI / "bvals", "--bvecs", DKI / "bvecs")  # fmt: skip
    expected = load_volume(tmp_path / "out.nii.gz").data
    np.testing.assert_allclose(fit.fods[0].data, expected, rtol=1e-5, atol=1e-6)
    assert fit.fractions is not None
    assert "Fitting voxels" in stages


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


def _tracked(viewer: Viewer, state: Dissection) -> tuple[Tractogram, Volume]:
    """Tracks the test FOD added without a file, as the Track step does."""
    fod = load_volume(TRACKING / "fod.nii.gz")
    fod_id = viewer.add(fod, Category.FODS, name="fod").id
    mask = load_volume(TRACKING / "mask.nii.gz")
    mask_id = viewer.add(mask, Category.OVERLAYS, name="mask").id
    settings = TrackSettings(streamlines=40, cutoff=0.08, seed=5)
    run = track(fod, Basis.TOURNIER, mask, mask, settings, lambda *_: None)
    assert run.tractogram is not None
    tract_id = viewer.add(run.tractogram, Category.TRACTOGRAMS, name="wb").id
    state.begin(StartFrom.FOD)
    state.fod, state.tractogram = fod_id, tract_id
    state.runs[tract_id] = TrackedRun(
        run, fod_id, mask_id, mask_id, settings.seeding, 1
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
