"""Where the steps to a first simulation stand, and that Run agrees with a run."""

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest

from larmor.viewer import ItemId, TissuePreset
from larmor.viewer.model.phantom import (
    BundleSettings,
    Draft,
    GeometrySettings,
    ItemSource,
    PhantomState,
    PhantomStep,
    ScanSettings,
    TissueSettings,
    box_grid,
    default_tissues,
    problems,
    step_status,
)

type Items = dict[ItemId, ItemSource]


def _new() -> PhantomState:
    return PhantomState(box_grid((60.0, 60.0, 60.0), 2.0), default_tissues())


def _drawn() -> PhantomState:
    return replace(_new(), bundles=(BundleSettings(1, geometry=GeometrySettings()),))


def _by_step(
    state: PhantomState, items: Items | None = None
) -> dict[PhantomStep, tuple[bool, str]]:
    return {s.step: (s.ready, s.text) for s in step_status(state, items or {})}


def test_a_new_phantom_needs_only_a_bundle() -> None:
    assert _by_step(_new()) == {
        PhantomStep.GRID: (True, "30 x 30 x 30 voxels of 2 mm"),
        PhantomStep.BUNDLE: (False, "No bundle yet"),
        PhantomStep.TISSUE: (True, "3 tissues to choose from"),
        PhantomStep.SCAN: (True, "b = 1000, 32 directions"),
        PhantomStep.RUN: (False, "Needs a bundle"),
    }


def test_a_bundle_drawn_and_finished_readies_the_run() -> None:
    drawing = replace(_new(), draft=Draft(((0.0, 0.0, 0.0), (5.0, 0.0, 0.0))))
    status = _by_step(drawing)
    assert status[PhantomStep.BUNDLE] == (False, "Drawing: 2 points")
    assert status[PhantomStep.TISSUE] == (True, "WM (standard model)")
    finished = replace(_new(), bundles=(BundleSettings(1, "GM", GeometrySettings()),))
    status = _by_step(finished)
    assert status[PhantomStep.BUNDLE] == (True, "1 bundle")
    assert status[PhantomStep.TISSUE] == (True, "GM (SANDI)")
    assert status[PhantomStep.RUN] == (True, "Ready")


def _invalid_tissue(_: Path) -> tuple[PhantomState, Items]:
    wm = TissueSettings("WM", TissuePreset.CUSTOM)  # no compartments
    tissues = (wm, *default_tissues()[1:])
    return replace(_drawn(), tissues=tissues), {}


def _missing_tissue(_: Path) -> tuple[PhantomState, Items]:
    return replace(_new(), bundles=(BundleSettings(1, "Gone", GeometrySettings()),)), {}


def _only_tissue_maps(_: Path) -> tuple[PhantomState, Items]:
    state = replace(_new(), tissue_maps=(("GM", 5),))
    return state, {5: ItemSource("gm.nii.gz", True, True)}


def _missing_gradient_file(tmp_path: Path) -> tuple[PhantomState, Items]:
    files = (tmp_path / "dwi.bval", tmp_path / "dwi.bvec")
    return replace(_drawn(), scan=ScanSettings(gradient_files=files)), {}


def _malformed_gradient_file(tmp_path: Path) -> tuple[PhantomState, Items]:
    bvals, bvecs = tmp_path / "dwi.bval", tmp_path / "dwi.bvec"
    bvals.write_text("0 1000 b\n")
    bvecs.write_text("0 1 0\n0 0 1\n0 0 0\n")
    return replace(_drawn(), scan=ScanSettings(gradient_files=(bvals, bvecs))), {}


def _unsaved_loaded_bundle(_: Path) -> tuple[PhantomState, Items]:
    state = replace(_new(), bundles=(BundleSettings(3),))  # no geometry: loaded
    return state, {3: ItemSource("cst.tck", True, False)}


def _unsaved_tissue_map(_: Path) -> tuple[PhantomState, Items]:
    state = replace(_new(), tissue_maps=(("GM", 5),))
    return state, {5: ItemSource("gm.nii.gz", True, False)}


def _map_of_a_missing_tissue(_: Path) -> tuple[PhantomState, Items]:
    state = replace(_new(), tissue_maps=(("Gone", 5),))
    return state, {5: ItemSource("gone.nii.gz", True, True)}


def _missing_background(_: Path) -> tuple[PhantomState, Items]:
    return replace(_drawn(), background="Gone"), {}


@pytest.mark.parametrize(
    ("make", "step", "line", "run_line"),
    [
        (
            _invalid_tissue,
            PhantomStep.TISSUE,
            "WM is not valid",
            "Needs a valid tissue",
        ),
        (_missing_tissue, PhantomStep.TISSUE, "Gone is gone", "Needs a valid tissue"),
        (
            _only_tissue_maps,
            PhantomStep.BUNDLE,
            "No bundle; tissue maps fill it",
            "Ready",
        ),
        (
            _missing_gradient_file,
            PhantomStep.SCAN,
            "A gradient file is missing",
            "Needs a valid scan",
        ),
        (
            _malformed_gradient_file,
            PhantomStep.SCAN,
            "The gradient files are not valid",
            "Needs a valid scan",
        ),
        (
            _unsaved_loaded_bundle,
            PhantomStep.BUNDLE,
            "cst.tck has no file",
            "Needs a bundle",
        ),
        (
            _unsaved_tissue_map,
            PhantomStep.TISSUE,
            "The map of GM has no file",
            "Needs a valid tissue",
        ),
        (
            _map_of_a_missing_tissue,
            PhantomStep.TISSUE,
            "Gone is gone",
            "Needs a valid tissue",
        ),
        (
            _missing_background,
            PhantomStep.TISSUE,
            "Gone is gone",
            "Needs a valid tissue",
        ),
    ],
)
def test_each_state_shows_its_step_and_run_agrees_with_a_run(
    make: Callable[[Path], tuple[PhantomState, Items]],
    step: PhantomStep,
    line: str,
    run_line: str,
    tmp_path: Path,
) -> None:
    state, items = make(tmp_path)
    status = _by_step(state, items)
    ready = run_line == "Ready"
    assert status[step] == (ready, line)
    assert status[PhantomStep.RUN] == (ready, run_line)
    assert ready == (problems(state, items) == [])


def test_timings_that_do_not_fit_hold_the_run() -> None:
    scan = ScanSettings(shells=((1000.0, 30), (2000.0, 60)), te_ms=(40.0,))
    state = replace(_drawn(), scan=scan)
    status = _by_step(state)
    assert status[PhantomStep.SCAN] == (False, "The timings do not fit")
    assert status[PhantomStep.RUN] == (False, "Needs a valid scan")
    fixed = replace(state, scan=replace(scan, te_ms=(80.0,)))
    assert _by_step(fixed)[PhantomStep.SCAN] == (True, "b = 1000, 2000; 90 directions")
