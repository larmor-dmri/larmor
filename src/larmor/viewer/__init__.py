"""BrainViewer, Larmor's viewer, and the functions that open it from Python."""

import os
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from larmor.types import Surface, Tractogram, Volume
from larmor.viewer.loading import DEFAULT, Default
from larmor.viewer.mcp_server import PortInUseError
from larmor.viewer.model.camera import CameraPreset, Convention
from larmor.viewer.model.item import Category, Item, ItemId, ViewKind
from larmor.viewer.model.matlab_import import LoadAs, MatlabPreset
from larmor.viewer.model.options import (
    Basis,
    Colormap,
    Colouring,
    GlyphScaling,
    OptionError,
    OptionName,
    RegionRole,
    TractStyle,
)
from larmor.viewer.model.phantom import (
    BundleField,
    EchoKind,
    ProxyMap,
    ScanField,
    TissuePreset,
    simulation_outputs,
)
from larmor.viewer.model.readout import Readout
from larmor.viewer.model.registry import MAX_STREAMLINES, item_type
from larmor.viewer.model.session import SessionError, SessionReport, read_session
from larmor.viewer.model.viewer import Viewer
from larmor.viewer.system import ViewerUnavailableError

if TYPE_CHECKING:
    # mypy and the IDE see the real import, so a misspelled name is still an error.
    from larmor.viewer.render.fod_inspector import FodInspector
else:

    def __getattr__(name: str) -> object:
        """Imports the FOD inspector on first use, as it needs pygfx."""
        if name == "FodInspector":
            from larmor.viewer.render.fod_inspector import FodInspector

            globals()[name] = FodInspector
            return FodInspector
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    def __dir__() -> list[str]:
        """Lists the lazy name too, for tab completion in IPython and Jupyter."""
        return sorted({*globals(), *__all__})


APP_NAME = "BrainViewer"
# Windows taskbar grouping, the Wayland desktop-file name and the macOS bundle id.
APP_ID = "larmor.brainviewer"

__all__ = [
    "MAX_STREAMLINES",
    "Basis",
    "BundleField",
    "CameraPreset",
    "Category",
    "Colormap",
    "Colouring",
    "Convention",
    "EchoKind",
    "FodInspector",
    "GlyphScaling",
    "Item",
    "ItemId",
    "LoadAs",
    "MatlabPreset",
    "OptionError",
    "OptionName",
    "PortInUseError",
    "ProxyMap",
    "Readout",
    "RegionRole",
    "ScanField",
    "SessionError",
    "SessionReport",
    "TissuePreset",
    "TractStyle",
    "ViewKind",
    "Viewer",
    "ViewerUnavailableError",
    "view",
    "wait",
]

type Source = str | os.PathLike[str] | Volume | Tractogram | Surface
type Sources = Source | Sequence[Source]


def view(
    viewer: Viewer | None = None,
    *,
    volumes: Sources | None = None,
    overlays: Sources | None = None,
    tracts: Sources | None = None,
    fods: Sources | None = None,
    peaks: Sources | None = None,
    surfaces: Sources | None = None,
    max_streamlines: int | None | Default = DEFAULT,
    session: str | os.PathLike[str] | None = None,
    phantom: str | os.PathLike[str] | None = None,
    simulation: str | os.PathLike[str] | None = None,
    block: bool = True,
    mcp: bool = False,
    mcp_port: int | None = None,
) -> Viewer:
    """Opens the viewer's window on files or objects, given by category.

    Items appear in the order given. Files are read in the background and appear
    as each one is read; a file that cannot be read shows a message. Objects are
    never copied: those given before any file of their category are added at
    once, the others after the files before them. Calling `view` again with the
    same viewer brings its window back and adds to it. A session replaces what
    the viewer shows; the files and objects given with it are added after its
    items.

    Args:
        viewer: The viewer to show; a new one if None.
        volumes: Paths or `Volume` objects; one or a list, like every category.
        overlays: Masks, maps or parcellations, as paths or `Volume` objects.
        tracts: Paths or `Tractogram` objects.
        fods: Paths or 4D `Volume` objects of SH coefficients, in the tournier
            basis unless their basis option says otherwise.
        peaks: Paths or 4D `Volume` objects as `lar peaks` writes them, with 3
            volumes per peak (its direction scaled by its amplitude).
        surfaces: GIfTI (.gii) or FreeSurfer files (such as lh.white), or
            `Surface` objects. A FreeSurfer surface's curvature (lh.curv in its
            folder) and cortex label (../label/lh.cortex.label) are read with it
            when they exist.
        max_streamlines: A tractogram file given here with more streamlines is
            reduced to a random subset of this size when read; the same file
            always gives the same subset. None keeps all. If not given, the read
            limit of the Settings panel (1,000,000 unless changed), which files
            opened later from the window use too. Objects are never reduced.
        session: A session file saved from the viewer (File > Save Session or
            `Viewer.save_session`). Its files are read in the background, then
            its options and views are restored. Missing files are reported in a
            message box, and the rest still opens.
        phantom: A phantom file to open in the Phantom panel, its bundles drawn
            in the background.
        simulation: A simulation folder (from `lar simulate` or the Phantom
            panel's Run), whose phantom opens in the Phantom panel and whose DWI,
            tissue masks and ground-truth peaks open as items.
        block: Wait until the window is closed. With False, return at once: in
            IPython or Jupyter, run `%gui qt` first so the window stays live; in
            a script, call `wait` to let the files load.
        mcp: Serve the viewer's actions to agents over MCP, at
            http://127.0.0.1:<port>/mcp; the URL and the token that requests
            need are printed. Tool calls run only while Qt's events run, that is
            while the window blocks or with `%gui qt`.
        mcp_port: The port for mcp, which it implies; the port in the Settings
            window (8765 unless changed) if None. If the port is taken, a
            message is printed and the window opens without the server.

    Returns:
        The viewer, whose actions change what the window shows.

    Raises:
        ViewerUnavailableError: If the viewer cannot start on this machine.
        ValueError: If an object is not of its category's type, an FOD's
            number of coefficients fits no SH layout, a peaks object does not
            hold 3 volumes per peak, max_streamlines is negative, mcp_port is
            not between 1024 and 65535, simulation is not a simulation
            folder, or both phantom and simulation are given.
        SessionError: If session is not a session file this version reads.
        FileNotFoundError: If the phantom file does not exist.
        OSError: If the session file cannot be read.
    """
    from larmor.viewer import system

    system.check_viewer_can_start()  # before Qt is imported
    session_path = _check_arguments(
        max_streamlines, mcp_port, session, phantom, simulation
    )
    viewer = viewer if viewer is not None else Viewer()

    given = {
        Category.VOLUMES: volumes,
        Category.OVERLAYS: overlays,
        Category.TRACTOGRAMS: tracts,
        Category.FODS: fods,
        Category.PEAKS: peaks,
        Category.SURFACES: surfaces,
    }
    now, queued = _split_sources(given, session_path)
    for category, data in now:
        viewer.add(data, category)

    from larmor.viewer.app import show

    # DEFAULT asks for the port in the Settings panel, None for no server.
    port = mcp_port if mcp_port is not None else (DEFAULT if mcp else None)
    show(
        viewer,
        queued,
        max_streamlines,
        block,
        mcp_port=port,
        session=session_path,
        phantom=None if phantom is None else Path(phantom),
        simulation=None if simulation is None else Path(simulation),
    )
    return viewer


def wait(viewer: Viewer, timeout: float | None = None) -> None:
    """Waits until the files given to `view` or opened in the window are loaded.

    Needed after `view(..., block=False)` in a script, where nothing else runs
    Qt's events; in IPython or Jupyter with `%gui qt`, files load on their own.

    Args:
        viewer: The viewer whose window loads the files.
        timeout: The longest wait in seconds; None waits as long as it takes.

    Raises:
        TimeoutError: If the files are still loading after timeout seconds.
    """
    from larmor.viewer.app import wait as wait_for_window

    wait_for_window(viewer, timeout)


def _check_arguments(
    max_streamlines: int | None | Default,
    mcp_port: int | None,
    session: str | os.PathLike[str] | None,
    phantom: str | os.PathLike[str] | None,
    simulation: str | os.PathLike[str] | None,
) -> Path | None:
    """Returns the session's path, after raising for `view`'s wrong arguments."""
    if isinstance(max_streamlines, int) and max_streamlines < 0:
        raise ValueError(f"max_streamlines must be 0 or more, got {max_streamlines}.")
    if mcp_port is not None and not 1024 <= mcp_port <= 65535:
        raise ValueError(f"mcp_port must be between 1024 and 65535, got {mcp_port}.")
    session_path = None if session is None else Path(session)
    if session_path is not None:
        read_session(session_path)
    if phantom is not None and simulation is not None:
        raise ValueError(
            "Give a phantom file or a simulation folder, not both: each opens a "
            "phantom in the editor."
        )
    if phantom is not None and not Path(phantom).is_file():
        raise FileNotFoundError(f"{phantom}: no such phantom file.")
    if simulation is not None:
        simulation_outputs(Path(simulation))
    return session_path


def _split_sources(
    given: dict[Category, Sources | None], session_path: Path | None
) -> tuple[list[tuple[Category, object]], list[tuple[Category, list[Path | object]]]]:
    """Splits the sources into objects to add now and the rest to queue, in order.

    Objects before any file of their category are added at once; with a session,
    every source is queued, as the session replaces the scene first.

    Returns:
        The (category, object) pairs to add now, and the sources of each
        category to queue.

    Raises:
        ValueError: If an object is not of its category's type.
    """
    now: list[tuple[Category, object]] = []
    queued: list[tuple[Category, list[Path | object]]] = []
    for category, sources in given.items():
        later: list[Path | object] = []
        for source in _as_list(sources):
            if isinstance(source, (str, os.PathLike)):
                later.append(Path(source))
                continue
            item_type(category).data(source)
            if later or session_path is not None:
                later.append(source)
            else:
                now.append((category, source))
        queued.append((category, later))
    return now, queued


def _as_list(sources: Sources | None) -> list[Source]:
    """Returns one source or several as a list."""
    if sources is None:
        return []
    if isinstance(sources, (str, os.PathLike, Volume, Tractogram, Surface)):
        return [sources]
    return list(sources)
