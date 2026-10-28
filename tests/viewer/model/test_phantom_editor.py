"""The phantom editor through the Viewer's actions, without a window."""

import subprocess
import sys
import threading
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

import larmor.simulation as sim
from larmor.io import load_volume, save_tractogram
from larmor.types import Grid, Tractogram, Volume
from larmor.viewer import (
    BundleField,
    BvecFrame,
    Category,
    Colouring,
    OptionName,
    ProxyMap,
    ScanField,
    TissuePreset,
    Viewer,
)
from larmor.viewer.jobs import SAVING, JobCancelledError, JobRunner, Report
from larmor.viewer.model.phantom import build_tissue, run_simulation
from larmor.viewer.model.shortcuts import Command, run
from larmor.viewer.phantom_jobs import start_run

LINE_A = [(-12.0, 0.0, 0.0), (0.0, 2.0, 0.0), (12.0, 0.0, 0.0)]
LINE_B = [(0.0, -12.0, 0.0), (0.0, 0.0, 2.0), (0.0, 12.0, 0.0)]


def _draw(viewer: Viewer, points: list[tuple[float, float, float]]) -> int:
    viewer.start_bundle()
    for point in points:
        viewer.add_bundle_point(point)
    return viewer.finish_bundle().id


@pytest.fixture
def viewer() -> Viewer:
    """A small phantom with two drawn bundles and a short scan."""
    viewer = Viewer()
    viewer.new_phantom((30.0, 30.0, 30.0), 2.0)
    viewer.set_scan(ScanField.SHELLS, [[1000, 6]])
    for points in (LINE_A, LINE_B):
        _draw(viewer, points)
        state = viewer.phantom
        assert state is not None
        viewer.set_bundle(state.bundles[-1].item_id, BundleField.NUM_STREAMLINES, 200)
    return viewer


def test_opening_the_viewer_does_not_load_the_simulation() -> None:
    code = (
        "import sys; from larmor.viewer import Viewer; Viewer(); "
        "print('larmor.simulation' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "False"


def test_a_saved_phantom_loads_back_the_same(viewer: Viewer, tmp_path: Path) -> None:
    viewer.set_tissue("WM", TissuePreset.STANDARD_MODEL, {"intra_t2_ms": 70.0})
    viewer.set_scan(ScanField.TE_MS, [70.0, 90.0])
    viewer.save_phantom(tmp_path / "scene.phantom.json")
    other = Viewer()
    other.load_phantom(tmp_path / "scene.phantom.json")
    saved, loaded = viewer.phantom, other.phantom
    assert saved is not None and loaded is not None
    assert loaded.grid.matches(saved.grid)
    assert loaded.tissues == saved.tissues
    assert loaded.scan == saved.scan
    assert [b.geometry for b in loaded.bundles] == [b.geometry for b in saved.bundles]
    names = [other.scene.item(b.item_id).name for b in loaded.bundles]
    assert names == ["Bundle 1", "Bundle 2"]


def test_the_new_compartments_build_tissues_and_only_unoriented_fill_the_background(
    viewer: Viewer,
) -> None:
    viewer.set_tissue("WM", TissuePreset.NODDI, {"kappa": 2.0})
    state = viewer.phantom
    assert state is not None
    built = build_tissue(state.tissue("WM"))
    assert built.compartments[0].model == sim.CompartmentModel.WATSON_STICK
    assert built.compartments[0].kappa == 2.0

    relaxation = {"t1_ms": 900.0, "t2_ms": 80.0, "t2star_ms": 50.0}
    viewer.set_compartments(
        "Still",
        [
            {"kind": "dot", "fraction": 0.3, "diffusivity": 0.0, **relaxation},
            {"kind": "free_water", "fraction": 0.7, "diffusivity": 3e-3, **relaxation},
        ],
    )
    viewer.set_background_tissue("Still")
    viewer.set_compartments(
        "Axons",
        [
            {
                "kind": "cylinder",
                "fraction": 1.0,
                "diffusivity": 2e-3,
                "radius_um": 2.0,
                **relaxation,
            }
        ],
    )
    with pytest.raises(ValueError, match="fibre direction"):
        viewer.set_background_tissue("Axons")
    state = viewer.phantom
    assert state is not None and state.background == "Still"


def test_editing_a_bundle_keeps_its_item_and_options(viewer: Viewer) -> None:
    state = viewer.phantom
    assert state is not None
    item_id = state.bundles[0].item_id
    viewer.set(item_id, OptionName.SUBSET, 50.0)
    before = viewer.scene.item(item_id).data
    viewer.edit_bundle(item_id)
    viewer.add_bundle_point((14.0, 6.0, 0.0))
    assert viewer.finish_bundle().id == item_id
    item = viewer.scene.item(item_id)
    assert item.data is not before
    assert item.options.subset == 50.0  # type: ignore[attr-defined]
    assert [i.id for i in viewer.scene.in_category(Category.TRACTOGRAMS)] == [
        b.item_id for b in state.bundles
    ]


def test_point_edits_within_one_frame_draw_the_bundle_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    viewer = Viewer()
    frame: list[Callable[[], None]] = []
    viewer.set_scheduler(frame.append)
    viewer.new_phantom((30.0, 30.0, 30.0), 2.0)
    viewer.start_bundle()
    built = []
    original = sim.create_bundle

    def counting(*args: object, **kwargs: object) -> Tractogram:
        built.append(1)
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(sim, "create_bundle", counting)
    for point in LINE_A:
        viewer.add_bundle_point(point)
    while frame:
        frame.pop()()
    assert len(built) == 1
    viewer.add_bundle_point((14.0, 2.0, 0.0))
    viewer.add_bundle_point((13.0, 3.0, 0.0))
    while frame:
        frame.pop()()
    assert len(built) == 2


def test_loading_n_bundles_builds_each_once(
    viewer: Viewer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    viewer.show_proxy(ProxyMap.FILL)
    path = tmp_path / "scene.phantom.json"
    viewer.save_phantom(path)
    built = []
    original = sim.create_bundle

    def counting(*args: object, **kwargs: object) -> Tractogram:
        built.append(1)
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(sim, "create_bundle", counting)
    Viewer().load_phantom(path)
    assert len(built) == 2


def test_enter_outside_draw_mode_does_nothing(viewer: Viewer) -> None:
    before = viewer.phantom
    assert not run(viewer, Command.FINISH_BUNDLE)
    assert viewer.phantom is before


def test_a_finished_bundle_is_an_ordinary_tractogram(viewer: Viewer) -> None:
    state = viewer.phantom
    assert state is not None
    item = viewer.scene.item(state.bundles[0].item_id)
    assert item.category == Category.TRACTOGRAMS
    viewer.set(item.id, OptionName.COLOURING, Colouring.SINGLE)
    assert item.options.colouring == Colouring.SINGLE  # type: ignore[attr-defined]


def test_the_expected_b0_is_the_noise_free_b0_of_a_run(viewer: Viewer) -> None:
    viewer.set_background_tissue("CSF")
    expected = viewer.show_proxy(ProxyMap.EXPECTED_B0)
    assert expected is not None
    inputs = viewer.prepare_run("unused")
    result = sim.simulate(inputs.phantom, inputs.protocol)
    b0 = result.noise_free[0].data[..., 0]
    assert np.allclose(expected.data.data, b0, rtol=1e-5, atol=1e-7)  # type: ignore[attr-defined]


def test_tissue_and_fill_edits_update_the_proxy_maps(viewer: Viewer) -> None:
    expected = viewer.show_proxy(ProxyMap.EXPECTED_B0)
    fill = viewer.show_proxy(ProxyMap.FILL)
    assert expected is not None and fill is not None
    b0 = expected.data.data.max()  # type: ignore[attr-defined]
    viewer.set_tissue("WM", TissuePreset.STANDARD_MODEL, {"proton_density": 0.35})
    assert expected.data.data.max() == pytest.approx(b0 / 2, rel=1e-5)  # type: ignore[attr-defined]
    total = fill.data.data[..., 0].sum()  # type: ignore[attr-defined]
    state = viewer.phantom
    assert state is not None
    viewer.set_bundle(state.bundles[0].item_id, BundleField.STREAMLINE_AREA, 1e-3)
    assert fill.data.data[..., 0].sum() < total  # type: ignore[attr-defined]


def test_a_session_restores_the_editor(viewer: Viewer, tmp_path: Path) -> None:
    viewer.show_proxy(ProxyMap.FILL)
    assert viewer.save_session(tmp_path / "s.lar.json") == []
    other = Viewer()
    report = other.load_session(tmp_path / "s.lar.json")
    state = other.phantom
    assert report.problems == () and state is not None
    assert len(state.bundles) == 2
    assert state.proxy(ProxyMap.FILL) is not None


def test_a_run_opens_its_outputs_with_files_and_a_session_keeps_them(
    viewer: Viewer, tmp_path: Path
) -> None:
    items = viewer.run_phantom(tmp_path / "sim")
    categories = {i.category for i in items}
    assert categories == {Category.VOLUMES, Category.OVERLAYS, Category.PEAKS}
    assert all(i.source is not None and i.source.is_file() for i in viewer.scene.items)
    viewer.save_session(tmp_path / "s.lar.json")
    other = Viewer()
    other.load_session(tmp_path / "s.lar.json")
    sources = sorted(str(i.source) for i in other.scene.items)
    assert sources == sorted(str(i.source) for i in viewer.scene.items)


def test_a_run_refuses_to_overwrite(viewer: Viewer, tmp_path: Path) -> None:
    viewer.run_phantom(tmp_path / "sim")
    with pytest.raises(FileExistsError):
        viewer.prepare_run(tmp_path / "sim")


def test_cancel_stops_the_run_and_opens_nothing(
    viewer: Viewer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    posted: list[Callable[[], None]] = []
    jobs = JobRunner(posted.append)
    started, cancelled, ended = threading.Event(), threading.Event(), threading.Event()

    def slow(inputs: object, folder: Path, report: Report, force: bool) -> object:
        started.set()
        cancelled.wait(10)
        try:
            report(1, 2, "signal")  # raises once cancelled
        finally:
            ended.set()
        return run_simulation(inputs, folder, report)  # type: ignore[arg-type]

    monkeypatch.setattr("larmor.viewer.phantom_jobs.run_simulation", slow)
    opened: list[list[Path]] = []
    start_run(viewer, jobs, tmp_path / "sim", lambda paths, _: opened.append(paths))
    assert started.wait(10)
    jobs.cancel()
    cancelled.set()
    assert ended.wait(10)
    for function in posted:
        function()
    # Only that nothing opens: a cancel before saving is the next test's.
    assert opened == []


def test_a_cancel_before_saving_writes_nothing(viewer: Viewer, tmp_path: Path) -> None:
    inputs = viewer.prepare_run(tmp_path / "sim")

    def report(done: int, total: int, stage: str) -> None:
        if stage == SAVING:
            raise JobCancelledError

    with pytest.raises(JobCancelledError):
        run_simulation(inputs, tmp_path / "sim", report)
    assert not (tmp_path / "sim").exists() or not any((tmp_path / "sim").iterdir())


def test_a_run_that_is_saving_finishes_despite_a_cancel(
    viewer: Viewer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    posted: list[Callable[[], None]] = []
    jobs = JobRunner(posted.append)
    saving, go = threading.Event(), threading.Event()

    def slow(inputs: object, folder: Path, report: Report, force: bool) -> object:
        def held(done: int, total: int, stage: str) -> None:
            report(done, total, stage)
            if stage == SAVING:
                saving.set()
                go.wait(10)

        return run_simulation(inputs, folder, held)  # type: ignore[arg-type]

    monkeypatch.setattr("larmor.viewer.phantom_jobs.run_simulation", slow)
    opened: list[Path] = []
    start_run(viewer, jobs, tmp_path / "sim", lambda paths, _: opened.extend(paths))
    assert saving.wait(30)
    jobs.cancel()  # refused: the files are being written
    assert jobs.saving
    go.set()
    for _ in range(3000):
        while posted:
            posted.pop(0)()
        if not jobs.busy:
            break
        threading.Event().wait(0.01)
    assert (tmp_path / "sim" / "dwi.nii.gz") in opened
    state = viewer.phantom
    assert state is not None and state.output == tmp_path / "sim"


def test_an_empty_phantom_is_refused() -> None:
    viewer = Viewer()
    viewer.new_phantom()
    [problem] = viewer.check_phantom()
    assert "at least one bundle or tissue map" in problem


def test_a_broken_protocol_is_refused(viewer: Viewer) -> None:
    viewer.set_scan(ScanField.BIG_DELTA_MS, 90.0)  # Delta + delta past TE
    [problem] = viewer.check_phantom()
    assert "The scan is not valid" in problem and "echo time" in problem


def test_a_map_on_another_grid_is_refused(viewer: Viewer) -> None:
    other = Volume(np.zeros((4, 4, 4), np.float32), np.eye(4))
    overlay = viewer.add(other, Category.OVERLAYS)
    with pytest.raises(ValueError, match="phantom's grid"):
        viewer.set_tissue_map("GM", overlay.id)


def test_a_run_does_not_give_its_folder_to_another_phantom(
    viewer: Viewer, tmp_path: Path
) -> None:
    inputs = viewer.prepare_run(tmp_path / "sim")
    outputs = run_simulation(inputs, tmp_path / "sim", lambda *_: None)
    viewer.new_phantom()
    viewer.finish_run(outputs)
    state = viewer.phantom
    assert state is not None and state.output is None


def test_a_point_outside_the_grid_is_refused(viewer: Viewer) -> None:
    viewer.start_bundle()
    with pytest.raises(ValueError, match="outside the phantom's grid"):
        viewer.add_bundle_point((100.0, 0.0, 0.0))
    state = viewer.phantom
    assert state is not None and state.draft is not None
    assert state.draft.points == ()


def test_a_session_keeps_the_grid_box_hidden(viewer: Viewer, tmp_path: Path) -> None:
    viewer.show_grid_box(False)
    viewer.save_session(tmp_path / "s.lar.json")
    other = Viewer()
    other.load_session(tmp_path / "s.lar.json")
    state = other.phantom
    assert state is not None and not state.show_box


def test_a_simulation_folder_opens_with_its_phantom(tmp_path: Path) -> None:
    phantom = sim.example_phantom("crossing")
    simulated = sim.simulate(
        phantom.phantom,
        phantom.protocol,  # type: ignore[arg-type]
        seed=1,
    )
    sim.save_simulation(simulated, tmp_path / "sim")
    viewer = Viewer()
    items = viewer.open_simulation(tmp_path / "sim")
    categories = [i.category for i in items]
    assert categories.count(Category.TRACTOGRAMS) == 2
    assert {Category.VOLUMES, Category.OVERLAYS, Category.PEAKS} <= set(categories)
    assert all(i.source is not None for i in items)
    names = {i.name for i in items}
    assert "mask_bundles.nii.gz" in names
    state = viewer.phantom
    assert state is not None and len(state.bundles) == 2


def test_a_run_of_a_phantom_file_gives_the_images_of_lar_simulate(
    tmp_path: Path,
) -> None:
    grid = Grid((10, 10, 10), np.diag([2.0, 2.0, 2.0, 1.0]))
    line = sim.BundleGeometry(np.array([[1.0, 9, 9], [17.0, 9, 9]]), num_streamlines=50)
    table = sim.create_gradient_table([(1000, 7), (2000, 5)], num_b0=2, seed=5)
    protocol = sim.Protocol(table, 15.0, 30.0, 80.0, 3000.0)
    path = tmp_path / "p.phantom.json"
    sim.write_phantom(
        sim.Phantom(grid, bundles=(sim.Bundle(line),)), path, protocol=protocol
    )
    viewer = Viewer()
    viewer.load_phantom(path)
    viewer.set_scan(ScanField.SNR, 20.0)
    viewer.run_phantom(tmp_path / "viewer")
    read = sim.read_phantom(path)
    assert read.protocol is not None
    expected = sim.simulate(read.phantom, read.protocol, snr=20.0, seed=0)
    images = sim.read_phantom(tmp_path / "viewer" / "phantom.json")
    assert images.protocol is not None
    np.testing.assert_array_equal(images.protocol.gradients.bvecs, table.bvecs)
    run = load_volume(tmp_path / "viewer" / "dwi.nii.gz")
    np.testing.assert_allclose(run.data, expected.echoes[0].data, atol=1e-6)


def test_editing_the_shells_replaces_the_table_of_a_phantom_file(
    viewer: Viewer, tmp_path: Path
) -> None:
    viewer.save_phantom(tmp_path / "p.phantom.json")
    other = Viewer()
    other.load_phantom(tmp_path / "p.phantom.json")
    assert other.phantom is not None and other.phantom.scan.table is None
    path = tmp_path / "file.phantom.json"
    phantom = sim.read_phantom(tmp_path / "p.phantom.json")
    sim.write_phantom(phantom.phantom, path, protocol=phantom.protocol)
    other.load_phantom(path)
    assert other.phantom is not None and other.phantom.scan.table is not None
    other.set_scan(ScanField.SHELLS, [[2000, 6]])
    assert other.phantom.scan.table is None


def test_bvecs_files_are_in_the_fsl_frame_unless_told(
    viewer: Viewer, tmp_path: Path
) -> None:
    (tmp_path / "b.bval").write_text("0 1000\n")
    (tmp_path / "b.bvec").write_text("0 1\n0 0\n0 0\n")
    viewer.set_scan(
        ScanField.GRADIENT_FILES, [tmp_path / "b.bval", tmp_path / "b.bvec"]
    )
    fsl = viewer.prepare_run(tmp_path / "a").protocol.gradients.bvecs[1]
    viewer.set_scan(ScanField.BVEC_FRAME, BvecFrame.WORLD)
    world = viewer.prepare_run(tmp_path / "a").protocol.gradients.bvecs[1]
    assert world.tolist() == [1.0, 0.0, 0.0]
    assert fsl.tolist() == [-1.0, 0.0, 0.0]


def test_noise_needs_a_reference_tissue_of_the_phantom(viewer: Viewer) -> None:
    viewer.set_tissue("Axons", TissuePreset.STANDARD_MODEL)
    state = viewer.phantom
    assert state is not None
    for bundle in state.bundles:
        viewer.set_bundle(bundle.item_id, BundleField.TISSUE, "Axons")
    viewer.remove_tissue("WM")
    viewer.set_scan(ScanField.SNR, 30.0)
    [problem] = viewer.check_phantom()
    assert "noise reference" in problem
    viewer.set_scan(ScanField.NOISE_REFERENCE, "Axons")
    assert viewer.check_phantom() == []


def test_a_phantom_file_keeps_the_grid_box_hidden(
    viewer: Viewer, tmp_path: Path
) -> None:
    viewer.show_grid_box(False)
    viewer.save_phantom(tmp_path / "p.phantom.json")
    other = Viewer()
    other.load_phantom(tmp_path / "p.phantom.json")
    assert other.phantom is not None and not other.phantom.show_box


def test_a_registered_tractogram_is_saved_where_it_is_shown(tmp_path: Path) -> None:
    points = np.array([[x, 0.0, 0.0] for x in np.linspace(-8, 8, 20)], np.float32)
    save_tractogram(Tractogram(points, np.array([0, 20])), tmp_path / "t.tck")
    (tmp_path / "shift.txt").write_text("1 0 0 0\n0 1 0 6\n0 0 1 0\n0 0 0 1\n")
    viewer = Viewer()
    viewer.new_phantom((30.0, 30.0, 30.0), 2.0)
    item = viewer.load(
        tmp_path / "t.tck",
        Category.TRACTOGRAMS,
        transforms=[str(tmp_path / "shift.txt")],
        format="mrtrix",
    )
    viewer.use_as_bundle(item.id, "WM")
    viewer.save_phantom(tmp_path / "p.phantom.json")
    other = Viewer()
    other.load_phantom(tmp_path / "p.phantom.json")
    assert other.phantom is not None
    shown = other.scene.item(other.phantom.bundles[0].item_id).data
    assert isinstance(shown, Tractogram) and isinstance(item.data, Tractogram)
    np.testing.assert_allclose(shown.points, item.data.points, atol=1e-5)


def test_the_proxy_maps_follow_a_bundle_being_edited(viewer: Viewer) -> None:
    state = viewer.phantom
    assert state is not None
    first = state.bundles[0].item_id
    proxy = viewer.show_proxy(ProxyMap.FILL)
    assert proxy is not None
    before = np.asarray(viewer.scene.item(proxy.id).data.data)[..., 1].copy()
    viewer.edit_bundle(first)
    viewer.set_bundle_points([(-12.0, 8.0, 0.0), (12.0, 8.0, 0.0)])
    after = np.asarray(viewer.scene.item(proxy.id).data.data)[..., 1]
    assert not np.array_equal(before, after)


def test_a_folder_of_other_files_is_refused_before_the_run(
    viewer: Viewer, tmp_path: Path
) -> None:
    (tmp_path / "notes.txt").write_text("mine")
    with pytest.raises(FileExistsError, match="other files"):
        viewer.prepare_run(tmp_path)
    viewer.run_phantom(tmp_path, overwrite=True)
    assert (tmp_path / "notes.txt").read_text() == "mine"
    assert (tmp_path / "dwi.nii.gz").is_file()


def test_a_point_repeated_by_a_click_or_a_stroke_is_dropped(viewer: Viewer) -> None:
    viewer.start_bundle()
    viewer.add_bundle_point((-10.0, 0.0, 0.0))
    viewer.add_bundle_point((-10.0, 0.0, 0.0))
    viewer.add_bundle_stroke([(-10.0, 0.0, 0.0), (0.0, 2.0, 0.0), (10.0, 0.0, 0.0)])
    item = viewer.finish_bundle()
    state = viewer.phantom
    assert state is not None
    geometry = state.bundle(item.id).geometry
    assert geometry is not None and geometry.control_points[0] == (-10.0, 0.0, 0.0)
    assert len(set(geometry.control_points)) == len(geometry.control_points)


def test_cancelling_an_edit_draws_the_bundle_as_it_was(viewer: Viewer) -> None:
    state = viewer.phantom
    assert state is not None
    second = state.bundles[1].item_id
    data = viewer.scene.item(second).data
    assert isinstance(data, Tractogram)
    before = data.points.copy()
    viewer.edit_bundle(second)
    viewer.set_bundle(None, BundleField.RADIUS, 4.0)
    viewer.cancel_bundle()
    after = viewer.scene.item(second).data
    assert isinstance(after, Tractogram)
    np.testing.assert_array_equal(after.points, before)
    assert set(after.streamline_data["bundle"].tolist()) == {1}


def test_a_run_does_not_open_its_outputs_over_another_phantom(
    viewer: Viewer, tmp_path: Path
) -> None:
    posted: list[Callable[[], None]] = []
    jobs = JobRunner(posted.append)
    opened: list[Path] = []
    start_run(viewer, jobs, tmp_path / "sim", lambda paths, _: opened.extend(paths))
    viewer.new_phantom()
    for _ in range(3000):
        while posted:
            posted.pop(0)()
        if not jobs.busy:
            break
        threading.Event().wait(0.01)
    assert (tmp_path / "sim" / "dwi.nii.gz").is_file()
    assert opened == []


def test_removing_the_background_tissue_names_the_background(viewer: Viewer) -> None:
    viewer.set_background_tissue("CSF")
    with pytest.raises(ValueError, match="is the background"):
        viewer.remove_tissue("CSF")


def test_a_bundle_being_drawn_stops_a_run(viewer: Viewer) -> None:
    state = viewer.phantom
    assert state is not None
    viewer.edit_bundle(state.bundles[0].item_id)
    [problem] = viewer.check_phantom()
    assert "finish or cancel" in problem


def test_a_grid_too_large_is_refused(viewer: Viewer) -> None:
    with pytest.raises(ValueError, match="too large"):
        viewer.set_phantom_grid((1200.0, 1200.0, 1200.0), 1.0)
    with pytest.raises(ValueError, match="too large"):
        viewer.new_phantom((1200.0, 1200.0, 1200.0), 1.0)


def test_a_radius_of_0_is_refused_and_a_tube_too_thin_stops_a_run(
    viewer: Viewer,
) -> None:
    state = viewer.phantom
    assert state is not None
    bundle = state.bundles[0].item_id
    with pytest.raises(ValueError, match="positive"):
        viewer.set_bundle(bundle, BundleField.RADIUS, 0.0)
    viewer.set_bundle(bundle, BundleField.RADIUS, 0.05)
    [problem] = viewer.check_phantom()
    assert "too thin" in problem
