"""Lists the keyboard shortcuts and mouse controls, and runs a shortcut."""

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from larmor.viewer.model.camera import CameraPreset
from larmor.viewer.model.item import VIEW_KINDS, ViewKind
from larmor.viewer.model.phantom import drawable
from larmor.viewer.model.view_state import Tool, Workspace
from larmor.viewer.model.viewer import Viewer

# The three 2D views.
VIEWS_2D: tuple[ViewKind, ...] = (ViewKind.AXIAL, ViewKind.CORONAL, ViewKind.SAGITTAL)
# The factor one zoom-in shortcut zooms by.
ZOOM_STEP = 1.25


class Command(StrEnum):
    """What a shortcut does; the menus find their shortcut by it."""

    MAXIMISE_AXIAL = "maximise_axial"
    MAXIMISE_CORONAL = "maximise_coronal"
    MAXIMISE_SAGITTAL = "maximise_sagittal"
    MAXIMISE_3D = "maximise_3d"
    NEXT_SLICE = "next_slice"
    PREVIOUS_SLICE = "previous_slice"
    NEXT_VOLUME = "next_volume"
    PREVIOUS_VOLUME = "previous_volume"
    ZOOM_IN = "zoom_in"
    ZOOM_OUT = "zoom_out"
    RESET_VIEW = "reset_view"
    INFORMATION_OVERLAY = "information_overlay"
    CROSSHAIR = "crosshair"
    CAMERA_AXIAL = "camera_axial"
    CAMERA_CORONAL = "camera_coronal"
    CAMERA_SAGITTAL = "camera_sagittal"
    CAMERA_RESET = "camera_reset"
    OPEN = "open"
    OPEN_SESSION = "open_session"
    SAVE_SESSION = "save_session"
    WORKSPACE_VIEW = "workspace_view"
    WORKSPACE_SIMULATE = "workspace_simulate"
    TOOL_CURSOR = "tool_cursor"
    TOOL_SPHERE = "tool_sphere"
    TOOL_BOX = "tool_box"
    TOOL_ZOOM_BOX = "tool_zoom_box"
    EXPORT = "export"
    NEW_BUNDLE = "new_bundle"
    FINISH_BUNDLE = "finish_bundle"
    REMOVE_POINT = "remove_point"
    LEAVE_DRAW_MODE = "leave_draw_mode"
    RUN_PHANTOM = "run_phantom"
    SIGNAL_BREAKDOWN = "signal_breakdown"
    SETTINGS = "settings"
    HELP = "help"
    HIDE_PANELS = "hide_panels"
    QUIT = "quit"


@dataclass(frozen=True)
class Shortcut:
    """One shortcut: its keys, what it does, and the views it works in."""

    command: Command
    # Qt's portable key text, such as "A", "Shift+A", "Up" or "Ctrl++"; Ctrl is Cmd
    # on macOS. The first is the one menus show.
    keys: tuple[str, ...]
    description: str
    views: tuple[ViewKind, ...]  # the selected view must be one of these
    # Called with the viewer and the selected view; None until its feature exists,
    # or when the window carries it out (`window`).
    action: Callable[[Viewer, ViewKind], None] | None
    window: bool = False  # done by the window, such as opening a file dialog
    # Enabled only while this holds, so the key is free for others otherwise; None
    # for always.
    enabled_when: Callable[[Viewer], bool] | None = None

    @property
    def active(self) -> bool:
        """Tells whether the shortcut does something yet."""
        return self.action is not None or self.window


@dataclass(frozen=True)
class MouseControl:
    """One way of using the mouse or trackpad in the views, for the Help list."""

    gesture: str  # Ctrl is Cmd on macOS, as for the keys
    description: str
    views: tuple[ViewKind, ...]


def _maximise(view: ViewKind) -> Callable[[Viewer, ViewKind], None]:
    """Returns an action that maximises a view, or shows all four again."""
    return lambda viewer, _: viewer.toggle_maximised(view)


def _zoom(factor: float) -> Callable[[Viewer, ViewKind], None]:
    """Returns an action that zooms the selected view by factor."""

    def zoom(viewer: Viewer, view: ViewKind) -> None:
        """Zooms the view, a 2D one about the cursor."""
        if view == ViewKind.THREE_D:
            viewer.zoom(view, factor)  # the 3D view zooms about its centre only
        else:
            # About the cursor, so the crosshair stays where it is on the screen.
            viewer.zoom(view, factor, about=viewer.scene.view.cursor)

    return zoom


def _preset(preset: CameraPreset) -> Callable[[Viewer, ViewKind], None]:
    """Returns an action that moves the 3D camera to a preset."""
    return lambda viewer, _: viewer.set_camera(ViewKind.THREE_D, preset)


def _window(command: Command, keys: tuple[str, ...], text: str) -> Shortcut:
    """Returns a shortcut the window carries out, such as opening a sheet."""
    return Shortcut(command, keys, text, VIEW_KINDS, None, window=True)


def _workspace(workspace: Workspace) -> Callable[[Viewer, ViewKind], None]:
    """Returns an action that shows a workspace."""
    return lambda viewer, _: viewer.set_workspace(workspace)


def _tool(tool: Tool) -> Callable[[Viewer, ViewKind], None]:
    """Returns an action that chooses a pointer tool."""
    return lambda viewer, _: viewer.set_tool(tool)


def _in_view(viewer: Viewer) -> bool:
    """Tells whether the View workspace shows, whose tools the region keys choose."""
    return viewer.scene.view.workspace == Workspace.VIEW


def _new_bundle(viewer: Viewer, view: ViewKind) -> None:
    """Starts drawing a bundle with the Draw bundle tool."""
    viewer.start_bundle()
    viewer.set_tool(Tool.DRAW_BUNDLE)


def _drawing(viewer: Viewer) -> bool:
    """Tells whether a bundle is being drawn."""
    return viewer.drawing


def _finish_bundle(viewer: Viewer, view: ViewKind) -> None:
    """Finishes the bundle being drawn, once it has the points it needs."""
    state = viewer.phantom
    if state is not None and state.draft is not None and drawable(state.draft.points):
        viewer.finish_bundle()


def _while_drawing(
    action: Callable[[Viewer], None],
) -> Callable[[Viewer, ViewKind], None]:
    """Returns an action that runs action only while a bundle is drawn."""

    def run(viewer: Viewer, view: ViewKind) -> None:
        """Runs the action if a bundle is being drawn."""
        if viewer.drawing:
            action(viewer)

    return run


def _escapable(viewer: Viewer) -> bool:
    """Tells whether a bundle is drawn or the Signal breakdown has a pinned voxel."""
    state = viewer.breakdown
    return viewer.drawing or (state is not None and state.pinned)


def _escape(viewer: Viewer, view: ViewKind) -> None:
    """Stops drawing; else unpins the Signal breakdown's voxel."""
    if viewer.drawing:
        viewer.cancel_bundle()
    elif _escapable(viewer):
        viewer.pin_breakdown(None)


def _enlarge(name: str) -> str:
    """Returns the description of a maximise shortcut."""
    return f"Fill the window with the {name} view, or show all four views"


# Every shortcut, by its command.
SHORTCUTS: dict[Command, Shortcut] = {
    s.command: s
    for s in (
        Shortcut(
            Command.MAXIMISE_AXIAL,
            ("A",),
            _enlarge("axial"),
            VIEW_KINDS,
            _maximise(ViewKind.AXIAL),
        ),
        Shortcut(
            Command.MAXIMISE_CORONAL,
            ("C",),
            _enlarge("coronal"),
            VIEW_KINDS,
            _maximise(ViewKind.CORONAL),
        ),
        Shortcut(
            Command.MAXIMISE_SAGITTAL,
            ("S",),
            _enlarge("sagittal"),
            VIEW_KINDS,
            _maximise(ViewKind.SAGITTAL),
        ),
        Shortcut(
            Command.MAXIMISE_3D,
            ("3",),
            _enlarge("3D"),
            VIEW_KINDS,
            _maximise(ViewKind.THREE_D),
        ),
        Shortcut(
            Command.NEXT_SLICE,
            ("Up",),
            "Next slice",
            VIEWS_2D,
            lambda viewer, view: viewer.step_slice(view, 1),
        ),
        Shortcut(
            Command.PREVIOUS_SLICE,
            ("Down",),
            "Previous slice",
            VIEWS_2D,
            lambda viewer, view: viewer.step_slice(view, -1),
        ),
        Shortcut(
            Command.NEXT_VOLUME,
            ("Right",),
            "Next volume of the selected 4D image, or of the main volume",
            VIEW_KINDS,
            lambda viewer, _: viewer.step_volume(1),
        ),
        Shortcut(
            Command.PREVIOUS_VOLUME,
            ("Left",),
            "Previous volume of the selected 4D image, or of the main volume",
            VIEW_KINDS,
            lambda viewer, _: viewer.step_volume(-1),
        ),
        Shortcut(
            Command.ZOOM_IN,
            ("Ctrl++", "Ctrl+="),
            "Zoom in",
            VIEW_KINDS,
            _zoom(ZOOM_STEP),
        ),
        Shortcut(
            Command.ZOOM_OUT,
            ("Ctrl+-",),
            "Zoom out",
            VIEW_KINDS,
            _zoom(1 / ZOOM_STEP),
        ),
        Shortcut(
            Command.RESET_VIEW,
            ("R",),
            "Show the whole view again (zoom 1, centred)",
            VIEW_KINDS,
            lambda viewer, view: viewer.reset_camera(view),
        ),
        Shortcut(
            Command.INFORMATION_OVERLAY,
            ("I",),
            "Show or hide the text, labels and colour bars over the views",
            VIEW_KINDS,
            lambda viewer, _: viewer.toggle_overlay(),
        ),
        Shortcut(
            Command.CROSSHAIR,
            ("X",),
            "Show or hide the crosshair",
            VIEW_KINDS,
            lambda viewer, _: viewer.toggle_crosshair(),
        ),
        Shortcut(
            Command.CAMERA_AXIAL,
            ("Shift+A",),
            "Look at the 3D view from above",
            VIEW_KINDS,  # it names its view, so it works from any selected one
            _preset(CameraPreset.AXIAL),
        ),
        Shortcut(
            Command.CAMERA_CORONAL,
            ("Shift+C",),
            "Look at the 3D view from the front",
            VIEW_KINDS,  # it names its view, so it works from any selected one
            _preset(CameraPreset.CORONAL),
        ),
        Shortcut(
            Command.CAMERA_SAGITTAL,
            ("Shift+S",),
            "Look at the 3D view from the side",
            VIEW_KINDS,  # it names its view, so it works from any selected one
            _preset(CameraPreset.SAGITTAL),
        ),
        Shortcut(
            Command.CAMERA_RESET,
            ("Shift+R",),
            "Reset the 3D camera",
            VIEW_KINDS,  # it names its view, so it works from any selected one
            lambda viewer, _: viewer.reset_camera(ViewKind.THREE_D),
        ),
        Shortcut(
            Command.OPEN, ("Ctrl+O",), "Open files", VIEW_KINDS, None, window=True
        ),
        Shortcut(
            Command.OPEN_SESSION,
            ("Ctrl+Shift+O",),
            "Open a saved session, replacing what is shown",
            VIEW_KINDS,
            None,
            window=True,
        ),
        Shortcut(
            Command.SAVE_SESSION,
            ("Ctrl+S",),
            "Save the items and the views as a session",
            VIEW_KINDS,
            None,
            window=True,
        ),
        Shortcut(
            Command.WORKSPACE_VIEW,
            ("Ctrl+1",),
            "Show the View workspace: the scene tree and the inspector",
            VIEW_KINDS,
            _workspace(Workspace.VIEW),
        ),
        Shortcut(
            Command.WORKSPACE_SIMULATE,
            ("Ctrl+2",),
            "Show the Simulate workspace: the steps of a phantom",
            VIEW_KINDS,
            _workspace(Workspace.SIMULATE),
        ),
        Shortcut(
            Command.TOOL_CURSOR,
            ("V",),
            "Choose the Cursor tool: clicks and drags move the crosshair",
            VIEW_KINDS,
            _tool(Tool.CURSOR),
        ),
        Shortcut(
            Command.TOOL_SPHERE,
            ("O",),
            "Choose the Sphere ROI tool (View workspace)",
            VIEW_KINDS,
            _tool(Tool.SPHERE),
            enabled_when=_in_view,
        ),
        Shortcut(
            Command.TOOL_BOX,
            ("B",),
            "Choose the Box ROI tool (View workspace)",
            VIEW_KINDS,
            _tool(Tool.BOX),
            enabled_when=_in_view,
        ),
        Shortcut(
            Command.TOOL_ZOOM_BOX,
            ("Z",),
            "Choose the Zoom box tool (View workspace)",
            VIEW_KINDS,
            _tool(Tool.ZOOM_BOX),
            enabled_when=_in_view,
        ),
        _window(
            Command.EXPORT,
            ("Ctrl+Shift+S",),
            "Export the views as images or a video",
        ),
        Shortcut(
            Command.NEW_BUNDLE,
            ("N",),
            "Draw a new bundle: clicks in the 2D views add its points",
            VIEW_KINDS,
            _new_bundle,
        ),
        Shortcut(
            Command.FINISH_BUNDLE,
            ("Return", "Enter"),
            "Finish the bundle being drawn (only while drawing)",
            VIEW_KINDS,
            _finish_bundle,
            enabled_when=_drawing,
        ),
        Shortcut(
            Command.REMOVE_POINT,
            ("Backspace",),
            "Remove the last point of the bundle being drawn (only while drawing)",
            VIEW_KINDS,
            _while_drawing(lambda viewer: viewer.remove_bundle_point()),
            enabled_when=_drawing,
        ),
        Shortcut(
            Command.LEAVE_DRAW_MODE,
            ("Escape",),
            "Stop drawing: drop the new bundle, or keep an edited one as it was; "
            "else unpin the voxel of the Signal breakdown",
            VIEW_KINDS,
            _escape,
            enabled_when=_escapable,
        ),
        Shortcut(
            Command.RUN_PHANTOM,
            ("Ctrl+R",),
            "Simulate the phantom and open the outputs",
            VIEW_KINDS,
            None,
            window=True,
        ),
        _window(
            Command.SIGNAL_BREAKDOWN,
            ("Ctrl+6",),
            "Show or hide the Signal breakdown of a voxel of the phantom",
        ),
        _window(Command.SETTINGS, ("Ctrl+,",), "Open the settings"),
        _window(Command.HELP, ("F1",), "Open the help"),
        _window(
            Command.HIDE_PANELS,
            ("Ctrl+0",),
            "Hide the panels either side of the views, or show them again",
        ),
        Shortcut(Command.QUIT, ("Ctrl+Q",), "Quit", VIEW_KINDS, None, window=True),
    )
}

# The mouse and trackpad controls the Help panel lists.
MOUSE_CONTROLS: tuple[MouseControl, ...] = (
    MouseControl("Click", "Select the view and move the crosshair there", VIEWS_2D),
    MouseControl("Drag", "Move the crosshair", VIEWS_2D),
    MouseControl("Scroll", "Next or previous slice", VIEWS_2D),
    MouseControl("Ctrl + scroll, or pinch", "Zoom about the pointer", VIEWS_2D),
    MouseControl("Ctrl + drag", "Pan", VIEWS_2D),
    MouseControl(
        "Click, while drawing a bundle",
        "Add a point to the bundle and move the crosshair there",
        VIEWS_2D,
    ),
    MouseControl("Drag", "Turn the scene", (ViewKind.THREE_D,)),
    MouseControl("Scroll, or pinch", "Zoom", (ViewKind.THREE_D,)),
)


def views_text(views: tuple[ViewKind, ...]) -> str:
    """Returns the views a shortcut or mouse control works in, as Help shows them."""
    if set(views) == set(VIEW_KINDS):
        return "any"
    if set(views) == set(VIEWS_2D):
        return "2D views"
    return ", ".join("3D" if v == ViewKind.THREE_D else v.capitalize() for v in views)


def run(viewer: Viewer, command: Command) -> bool:
    """Runs a shortcut on the selected view; False if it is inactive or not for it.

    A shortcut the window carries out (`window`) is not run here.

    Args:
        viewer: The viewer to run it on.
        command: The shortcut.

    Returns:
        Whether the shortcut ran.

    Raises:
        TypeError: If `command` is not a `Command`.
    """
    if not isinstance(command, Command):
        raise TypeError(
            f"Expected a Command, such as Command.ZOOM_IN; got {command!r}."
        )
    shortcut = SHORTCUTS[command]
    view = viewer.scene.view.selected_view
    if shortcut.action is None or view not in shortcut.views:
        return False
    if shortcut.enabled_when is not None and not shortcut.enabled_when(viewer):
        return False
    shortcut.action(viewer, view)
    return True
