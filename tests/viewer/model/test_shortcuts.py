import subprocess
import sys

import numpy as np
import pytest

from larmor.types import Volume
from larmor.viewer import CameraPreset, Category, OptionName, Tool, Viewer, ViewKind
from larmor.viewer.model.camera import Camera2D, Camera3D, screen_axes
from larmor.viewer.model.item import VIEW_KINDS
from larmor.viewer.model.shortcuts import SHORTCUTS, ZOOM_STEP, Command, run
from larmor.viewer.model.view_state import ViewState, Workspace

VIEWS_2D: list[ViewKind] = [ViewKind.AXIAL, ViewKind.CORONAL, ViewKind.SAGITTAL]
AXIS: dict[ViewKind, int] = {
    ViewKind.SAGITTAL: 0,
    ViewKind.CORONAL: 1,
    ViewKind.AXIAL: 2,
}
# The documented keys of each command.
KEYS: dict[Command, tuple[str, ...]] = {
    Command.MAXIMISE_AXIAL: ("A",),
    Command.MAXIMISE_CORONAL: ("C",),
    Command.MAXIMISE_SAGITTAL: ("S",),
    Command.MAXIMISE_3D: ("3",),
    Command.NEXT_SLICE: ("Up",),
    Command.PREVIOUS_SLICE: ("Down",),
    Command.NEXT_VOLUME: ("Right",),
    Command.PREVIOUS_VOLUME: ("Left",),
    Command.ZOOM_IN: ("Ctrl++", "Ctrl+="),
    Command.ZOOM_OUT: ("Ctrl+-",),
    Command.RESET_VIEW: ("R",),
    Command.INFORMATION_OVERLAY: ("I",),
    Command.CROSSHAIR: ("X",),
    Command.CAMERA_AXIAL: ("Shift+A",),
    Command.CAMERA_CORONAL: ("Shift+C",),
    Command.CAMERA_SAGITTAL: ("Shift+S",),
    Command.CAMERA_RESET: ("Shift+R",),
    Command.OPEN: ("Ctrl+O",),
    Command.OPEN_SESSION: ("Ctrl+Shift+O",),
    Command.SAVE_SESSION: ("Ctrl+S",),
    Command.WORKSPACE_VIEW: ("Ctrl+1",),
    Command.WORKSPACE_SIMULATE: ("Ctrl+2",),
    Command.TOOL_CURSOR: ("V",),
    Command.TOOL_SPHERE: ("O",),
    Command.TOOL_BOX: ("B",),
    Command.TOOL_ZOOM_BOX: ("Z",),
    Command.DELETE_ITEM: ("Delete", "Ctrl+Backspace"),
    Command.NEXT_ROLE: ("Tab",),
    Command.EXPORT: ("Ctrl+Shift+S",),
    Command.NEW_BUNDLE: ("N",),
    Command.FINISH_BUNDLE: ("Return", "Enter"),
    Command.REMOVE_POINT: ("Backspace",),
    Command.LEAVE_DRAW_MODE: ("Escape",),
    Command.RUN_PHANTOM: ("Ctrl+R",),
    Command.SIGNAL_BREAKDOWN: ("Ctrl+6",),
    Command.SETTINGS: ("Ctrl+,",),
    Command.HELP: ("F1",),
    Command.HIDE_PANELS: ("Ctrl+0",),
    Command.QUIT: ("Ctrl+Q",),
}
MAXIMISE: dict[Command, ViewKind] = {
    Command.MAXIMISE_AXIAL: ViewKind.AXIAL,
    Command.MAXIMISE_CORONAL: ViewKind.CORONAL,
    Command.MAXIMISE_SAGITTAL: ViewKind.SAGITTAL,
    Command.MAXIMISE_3D: ViewKind.THREE_D,
}
CAMERA_PRESETS: dict[Command, CameraPreset | None] = {
    Command.CAMERA_AXIAL: CameraPreset.AXIAL,
    Command.CAMERA_CORONAL: CameraPreset.CORONAL,
    Command.CAMERA_SAGITTAL: CameraPreset.SAGITTAL,
    Command.CAMERA_RESET: None,
}


def _viewer(selected: ViewKind = ViewKind.AXIAL) -> Viewer:
    viewer = Viewer()
    viewer.add(
        Volume(np.zeros((8, 8, 8, 3), np.float32), np.diag([2.0, 2.0, 2.0, 1.0])),
        Category.VOLUMES,
    )
    viewer.select_view(selected)
    return viewer


def _volume_index(viewer: Viewer) -> object:
    return getattr(viewer.scene.items[0].options, OptionName.VOLUME_INDEX)


def test_every_command_has_one_entry_keyed_by_it() -> None:
    assert list(SHORTCUTS) == list(Command)
    assert all(s.command == c for c, s in SHORTCUTS.items())
    assert all(s.description and s.views for s in SHORTCUTS.values())


@pytest.mark.parametrize("command", list(Command))
def test_each_command_has_its_keys(command: Command) -> None:
    assert SHORTCUTS[command].keys == KEYS[command]


def test_no_two_shortcuts_share_a_key() -> None:
    keys = [k.upper() for s in SHORTCUTS.values() for k in s.keys]
    assert len(keys) == len(set(keys))


@pytest.mark.parametrize("selected", VIEW_KINDS)
@pytest.mark.parametrize("command", list(MAXIMISE))
def test_a_view_key_toggles_that_view_as_its_button_does(
    command: Command, selected: ViewKind
) -> None:
    by_key, by_button = _viewer(selected), _viewer(selected)
    seen = []
    for _ in range(2):
        assert run(by_key, command)
        by_button.toggle_maximised(MAXIMISE[command])
        assert by_key.scene.view == by_button.scene.view
        seen.append(by_key.scene.view.maximised)
    assert seen == [MAXIMISE[command], None]


@pytest.mark.parametrize("command", list(MAXIMISE))
def test_a_view_key_switches_from_another_enlarged_view(command: Command) -> None:
    viewer = _viewer()
    other = next(v for v in VIEW_KINDS if v != MAXIMISE[command])
    viewer.toggle_maximised(other)
    run(viewer, command)
    assert viewer.scene.view.maximised == MAXIMISE[command]


WINDOW_COMMANDS = [
    Command.OPEN,
    Command.OPEN_SESSION,
    Command.SAVE_SESSION,
    Command.EXPORT,
    Command.RUN_PHANTOM,
    Command.SIGNAL_BREAKDOWN,
    Command.SETTINGS,
    Command.HELP,
    Command.HIDE_PANELS,
    Command.QUIT,
]


@pytest.mark.parametrize("selected", VIEW_KINDS)
@pytest.mark.parametrize("command", CAMERA_PRESETS)
def test_the_3d_camera_keys_move_the_3d_camera_from_any_view(
    command: Command, selected: ViewKind
) -> None:
    # They name the view they act on, so the selected view does not matter.
    assert SHORTCUTS[command].views == VIEW_KINDS
    by_key, by_call = _viewer(selected), _viewer(selected)
    for viewer in (by_key, by_call):
        viewer.zoom(ViewKind.THREE_D, 2.0)
        viewer.rotate_camera(30.0, 20.0)
    before = by_key.scene.view.cameras.three_d
    assert run(by_key, command)
    if command == Command.CAMERA_RESET:
        by_call.reset_camera(ViewKind.THREE_D)
    else:
        preset = CAMERA_PRESETS[command]
        assert preset is not None
        by_call.set_camera(ViewKind.THREE_D, preset)
    assert by_key.scene.view.cameras.three_d == by_call.scene.view.cameras.three_d
    assert by_key.scene.view.cameras.three_d != before


def test_nothing_is_inactive_now() -> None:
    assert [c for c, s in SHORTCUTS.items() if not s.active] == []


def test_only_the_window_commands_are_done_by_the_window() -> None:
    assert [c for c, s in SHORTCUTS.items() if s.window] == WINDOW_COMMANDS


@pytest.mark.parametrize("selected", VIEW_KINDS)
@pytest.mark.parametrize("command", WINDOW_COMMANDS)
def test_a_window_shortcut_is_active_but_not_run_on_the_viewer(
    command: Command, selected: ViewKind
) -> None:
    shortcut = SHORTCUTS[command]
    assert shortcut.active
    assert shortcut.views == VIEW_KINDS
    viewer = _viewer(selected)
    before = viewer.scene.view
    assert not run(viewer, command)
    assert viewer.scene.view == before


@pytest.mark.parametrize(
    ("command", "step"), [(Command.NEXT_SLICE, 1), (Command.PREVIOUS_SLICE, -1)]
)
@pytest.mark.parametrize("view", VIEWS_2D)
def test_up_and_down_step_the_slice_of_the_selected_view(
    view: ViewKind, command: Command, step: int
) -> None:
    viewer = _viewer(view)
    before = np.array(viewer.scene.view.cursor)
    assert run(viewer, command)
    expected = before.copy()
    expected[AXIS[view]] += 2.0 * step  # one 2 mm voxel
    assert viewer.scene.view.cursor == tuple(expected)


@pytest.mark.parametrize("command", [Command.NEXT_SLICE, Command.PREVIOUS_SLICE])
def test_a_2d_command_does_nothing_with_the_3d_view_selected(command: Command) -> None:
    viewer = _viewer(ViewKind.THREE_D)
    before = viewer.scene.view
    assert not run(viewer, command)
    assert viewer.scene.view == before


@pytest.mark.parametrize("selected", VIEW_KINDS)
def test_left_and_right_step_the_volume_in_any_view(selected: ViewKind) -> None:
    viewer = _viewer(selected)
    assert run(viewer, Command.NEXT_VOLUME)
    assert run(viewer, Command.NEXT_VOLUME)
    assert _volume_index(viewer) == 2
    assert run(viewer, Command.PREVIOUS_VOLUME)
    assert _volume_index(viewer) == 1


@pytest.mark.parametrize(
    ("command", "zoom"),
    [(Command.ZOOM_IN, ZOOM_STEP), (Command.ZOOM_OUT, 1 / ZOOM_STEP)],
)
@pytest.mark.parametrize("view", VIEWS_2D)
def test_zoom_keys_zoom_the_selected_view_about_the_cursor(
    view: ViewKind, command: Command, zoom: float
) -> None:
    viewer = _viewer(view)
    # Off the view centre, so a zoom about the centre would move it on the screen.
    viewer.pan(view, 5.0, -3.0)
    right, up = (np.array(a) for a in screen_axes(view, viewer.scene.view.convention))

    def on_screen() -> np.ndarray:
        offset = np.array(viewer.scene.view.cursor) - viewer.view_centre(view)
        camera = viewer.scene.view.cameras.of(view)
        assert isinstance(camera, Camera2D)
        return np.array([offset @ right, offset @ up]) * camera.zoom

    before = on_screen()
    assert run(viewer, command)
    camera = viewer.scene.view.cameras.of(view)
    assert isinstance(camera, Camera2D)
    assert camera.zoom == pytest.approx(zoom)
    assert np.allclose(on_screen(), before)
    others = [v for v in VIEWS_2D if v != view]
    assert all(viewer.scene.view.cameras.of(v) == Camera2D() for v in others)


@pytest.mark.parametrize(
    ("command", "zoom"),
    [(Command.ZOOM_IN, ZOOM_STEP), (Command.ZOOM_OUT, 1 / ZOOM_STEP)],
)
def test_zoom_keys_zoom_the_3d_camera_about_its_centre(
    command: Command, zoom: float
) -> None:
    viewer = _viewer(ViewKind.THREE_D)
    before = viewer.scene.view
    assert run(viewer, command)
    assert viewer.scene.view.cameras.three_d == Camera3D(zoom=zoom)
    assert viewer.scene.view.cameras.axial == before.cameras.axial
    assert viewer.scene.view.cursor == before.cursor


@pytest.mark.parametrize("selected", VIEW_KINDS)
def test_r_resets_only_the_selected_view(selected: ViewKind) -> None:
    viewer = _viewer(selected)
    for view in VIEWS_2D:
        viewer.zoom(view, 2.0)
        viewer.pan(view, 5.0, -3.0)
    viewer.zoom(ViewKind.THREE_D, 2.0)
    viewer.rotate_camera(30.0, 20.0)
    before = viewer.scene.view.cameras
    assert run(viewer, Command.RESET_VIEW)
    for view in VIEW_KINDS:
        camera = viewer.scene.view.cameras.of(view)
        if view == selected:
            assert camera == (Camera3D() if view == ViewKind.THREE_D else Camera2D())
        else:
            assert camera == before.of(view)


@pytest.mark.parametrize("selected", VIEW_KINDS)
def test_the_overlay_key_toggles_the_information_overlay(selected: ViewKind) -> None:
    viewer = _viewer(selected)
    run(viewer, Command.INFORMATION_OVERLAY)
    assert not viewer.scene.view.overlay_visible
    run(viewer, Command.INFORMATION_OVERLAY)
    assert viewer.scene.view.overlay_visible


@pytest.mark.parametrize("command", ["zoom_in", "bogus", None])
def test_run_refuses_what_is_not_a_command(command: object) -> None:
    viewer = _viewer()
    before = viewer.scene.view
    with pytest.raises(TypeError, match="Expected a Command"):
        run(viewer, command)  # type: ignore[arg-type]  # wrong on purpose
    assert viewer.scene.view == before


def test_the_shortcut_registry_loads_no_qt_or_gpu_library() -> None:
    heavy = ("PySide6", "pygfx", "wgpu", "rendercanvas")
    code = (
        "import sys, larmor.viewer.model.shortcuts\n"
        f"print(','.join(m for m in {heavy!r} if m in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ""


def _view(viewer: Viewer) -> ViewState:
    return viewer.scene.view


def test_the_workspace_keys_switch_the_workspace() -> None:
    viewer = _viewer()
    assert run(viewer, Command.WORKSPACE_SIMULATE)
    assert _view(viewer).workspace == Workspace.SIMULATE
    assert run(viewer, Command.WORKSPACE_VIEW)
    assert _view(viewer).workspace == Workspace.VIEW


def test_the_region_tool_keys_work_only_in_the_view_workspace() -> None:
    viewer = _viewer()
    assert run(viewer, Command.TOOL_SPHERE)
    assert _view(viewer).tool == Tool.SPHERE
    viewer.set_workspace(Workspace.SIMULATE)
    assert _view(viewer).tool == Tool.CURSOR  # Simulate offers no spheres
    assert not run(viewer, Command.TOOL_BOX)
    assert _view(viewer).tool == Tool.CURSOR
