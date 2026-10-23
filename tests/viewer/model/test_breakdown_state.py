"""The Signal breakdown through the Viewer's actions, without a window."""

import numpy as np
import pytest

import larmor.simulation as sim
from larmor.types import Volume
from larmor.viewer import BundleField, Category, ScanField, TissuePreset, Viewer
from larmor.viewer.model.breakdown import (
    NO_VOXEL,
    card_rows,
    heading,
    title,
    voxel_at,
)
from larmor.viewer.model.events import BreakdownChanged, Event
from larmor.viewer.model.shortcuts import Command, run

LINE_A = [(-12.0, 0.0, 0.0), (0.0, 2.0, 0.0), (12.0, 0.0, 0.0)]
LINE_B = [(0.0, -12.0, 0.0), (0.0, 0.0, 2.0), (0.0, 12.0, 0.0)]
CROSSING = (0.0, 1.0, 0.0)
ALONG_A = (-8.0, 0.0, 0.0)


@pytest.fixture
def viewer() -> Viewer:
    """Two crossing bundles over a GM background, the breakdown shown."""
    viewer = Viewer()
    viewer.new_phantom((30.0, 30.0, 30.0), 2.0)
    viewer.set_scan(ScanField.SHELLS, [[1000, 6], [3000, 6]])
    viewer.set_background_tissue("GM")
    for points in (LINE_A, LINE_B):
        viewer.start_bundle()
        for point in points:
            viewer.add_bundle_point(point)
        item = viewer.finish_bundle()
        viewer.set_bundle(item.id, BundleField.NUM_STREAMLINES, 200)
    viewer.show_breakdown()
    return viewer


def events(viewer: Viewer) -> list[Event]:
    found: list[Event] = []
    viewer.scene.subscribe(
        lambda e: found.append(e) if isinstance(e, BreakdownChanged) else None
    )
    return found


def test_the_breakdown_follows_the_pointer_until_pinned(viewer: Viewer) -> None:
    state = viewer.breakdown
    assert state is not None and state.problem == NO_VOXEL
    viewer.hover_breakdown(CROSSING)
    grid = viewer.phantom.grid  # type: ignore[union-attr]
    assert viewer.breakdown.voxel == voxel_at(grid, CROSSING)  # type: ignore[union-attr]
    viewer.pin_breakdown(ALONG_A)
    viewer.hover_breakdown(CROSSING)
    assert viewer.breakdown.voxel == voxel_at(grid, ALONG_A)  # type: ignore[union-attr]
    assert viewer.breakdown.pinned  # type: ignore[union-attr]
    assert run(viewer, Command.LEAVE_DRAW_MODE)  # Escape unpins
    viewer.hover_breakdown(CROSSING)
    assert viewer.breakdown.voxel == voxel_at(grid, CROSSING)  # type: ignore[union-attr]


def test_the_breakdown_is_the_signal_a_run_simulates(viewer: Viewer) -> None:
    viewer.pin_breakdown(CROSSING)
    viewer.set_breakdown_volume(9)
    state = viewer.breakdown
    assert state is not None and state.breakdown is not None
    inputs = viewer._phantom.run_inputs()
    result = sim.simulate(inputs.phantom, inputs.protocol, seed=0)
    expected = result.noise_free[0].data[state.breakdown.voxel][9]
    assert state.breakdown.signal == pytest.approx(expected, rel=1e-6)
    assert len(state.breakdown.bundles) == 2


def test_one_setting_change_gives_one_update(viewer: Viewer) -> None:
    viewer.pin_breakdown(ALONG_A)
    found = events(viewer)
    viewer.set_scan(ScanField.TE_MS, [100.0])
    assert len(found) == 1
    state = viewer.breakdown
    assert state is not None and state.breakdown is not None
    assert state.breakdown.te_ms == 100.0
    viewer.set_tissue("WM", TissuePreset.STANDARD_MODEL, {"intra_t2_ms": 60.0})
    assert len(found) == 2


def test_choosing_a_volume_shows_its_b_value(
    viewer: Viewer,
) -> None:
    viewer.pin_breakdown(ALONG_A)
    viewer.set_scan(ScanField.TR_MS, 4000.0)
    gradients = viewer.breakdown_gradients()
    assert gradients is not None
    volume = int(np.flatnonzero(gradients.bvals > 2000)[0])
    viewer.set_breakdown_volume(volume)
    state = viewer.breakdown
    assert state is not None and state.breakdown is not None
    assert state.breakdown.b_value == pytest.approx(3000)
    with pytest.raises(ValueError, match="volume"):
        viewer.set_breakdown_volume(len(gradients.bvals))


def test_closing_stops_following(viewer: Viewer) -> None:
    viewer.show_breakdown(False)
    found = events(viewer)
    viewer.hover_breakdown(CROSSING)
    viewer.set_scan(ScanField.TE_MS, [100.0])
    assert viewer.breakdown is None
    assert found == []


def test_the_card_lists_the_tissues_and_under_them_each_bundle(
    viewer: Viewer,
) -> None:
    viewer.pin_breakdown(CROSSING)
    state = viewer.breakdown
    assert state is not None and state.breakdown is not None
    voxel = "({}, {}, {})".format(*state.breakdown.voxel)
    assert heading(state) == f"Voxel {voxel}"
    assert title(state) == f"Signal breakdown - voxel {voxel}"
    viewer.pin_breakdown(None)
    assert title(viewer.breakdown) == f"Signal breakdown - voxel {voxel}"
    rows = card_rows(state.breakdown)
    assert [(r.name, r.bundle) for r in rows[:3]] == [
        ("WM", False),
        ("Bundle 1", True),
        ("Bundle 2", True),
    ]
    assert rows[0].angle == ""
    assert all(r.angle.endswith(" deg") for r in rows[1:3])
    assert all(r.share != "0%" for r in rows)  # CSF fills none of it
    assert "CSF" not in [r.name for r in rows]
    viewer.set_breakdown_volume(0)  # b = 0: no gradient
    state = viewer.breakdown
    assert state is not None and state.breakdown is not None
    assert all(r.angle == "" for r in card_rows(state.breakdown))


def test_a_new_tissue_map_refills_the_voxel(viewer: Viewer) -> None:
    grid = viewer.phantom.grid  # type: ignore[union-attr]
    viewer.set_background_tissue("CSF")
    viewer.pin_breakdown((10.0, 10.0, 10.0))  # far from the bundles
    for fraction in (0.2, 0.8):
        item = viewer.add(
            Volume(np.full(grid.shape, fraction, np.float32), grid.affine),
            Category.OVERLAYS,
        )
        viewer.set_tissue_map("GM", item.id)
        state = viewer.breakdown
        assert state is not None and state.breakdown is not None
        assert dict(state.breakdown.tissue_fractions)["GM"] == pytest.approx(fraction)
