"""Starting the viewer's Qt application and opening its windows."""

import os
import sys
import time
from importlib.resources import files
from pathlib import Path

import shiboken6
from PySide6.QtCore import QEvent, QEventLoop, QObject, Qt, QTimer
from PySide6.QtGui import QFileOpenEvent, QGuiApplication, QIcon
from PySide6.QtWidgets import QApplication

from larmor._log import LarmorLogger
from larmor.viewer import APP_ID, APP_NAME
from larmor.viewer.loading import Default, sort_files
from larmor.viewer.mcp_server import PortInUseError
from larmor.viewer.model.item import Category
from larmor.viewer.model.settings import (
    SettingChanged,
    SettingName,
    Settings,
    Theme,
)
from larmor.viewer.model.viewer import Viewer
from larmor.viewer.qt.loading import ErrorBox
from larmor.viewer.qt.main_window import MainWindow
from larmor.viewer.qt.style import apply_style
from larmor.viewer.render.gpu import select_gpu

ICON = files("larmor.viewer") / "resources" / "icon.svg"

# The open window of each viewer; this keeps it alive after view() returns.
_WINDOWS: dict[Viewer, MainWindow] = {}
# The user's settings, read from their file when the application starts.
_SETTINGS: Settings | None = None

COLOR_SCHEMES = {
    Theme.LIGHT: Qt.ColorScheme.Light,
    Theme.DARK: Qt.ColorScheme.Dark,
}


def _set_windows_app_id() -> None:
    """Gives the process its own app ID, so Windows shows our icon in the taskbar."""
    if sys.platform == "win32":  # mypy skips this branch on other platforms
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)


def _give_streams() -> None:
    """Gives a process without a console (pythonw) streams, so that writes work."""
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w")  # noqa: SIM115 - open for the process
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w")  # noqa: SIM115


class _FileOpener(QObject):
    """Opens the files the system asks the app to open, from Finder on macOS."""

    def __init__(self) -> None:
        super().__init__()
        self.pending: list[Path] = []  # files waiting for a window
        self._flushing = False

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        """Gathers the files of a file-open event, and opens them on the next event."""
        if event.type() != QEvent.Type.FileOpen:
            return False
        assert isinstance(event, QFileOpenEvent)
        self.pending.append(Path(event.file()))
        # A selection comes as one event per file, in a row: they open as one batch.
        if not self._flushing:
            self._flushing = True
            QTimer.singleShot(0, self._flush)
        return True

    def _flush(self) -> None:
        """Opens the files gathered so far, once a window is open to take them."""
        self._flushing = False
        windows = [w for w in _WINDOWS.values() if shiboken6.isValid(w)]
        if windows and self.pending:
            self.open_in(windows[-1])

    def open_in(self, w: MainWindow, has_session: bool = False) -> None:
        """Opens the pending files in a window; has_session refuses another session."""
        paths, self.pending = self.pending, []
        given = sort_files(paths)
        if len(given.sessions) > 1:
            names = ", ".join(str(s) for s in given.sessions)
            _report(w, f"Only one session can be opened at a time, got {names}.")
        elif given.sessions and has_session:
            _report(
                w, f"A session is already being opened; {given.sessions[0]} is not."
            )
        elif given.sessions:
            w.open_session(given.sessions[0])
        w.loader.open(given.volumes, Category.VOLUMES)
        w.loader.open(given.tracts, Category.TRACTOGRAMS)


def _report(w: MainWindow, message: str) -> None:
    """Logs why files cannot be opened, and shows it with the window's load errors."""
    LarmorLogger.error(message)
    box = w.findChild(ErrorBox)
    if box is not None:
        box.add(message)


_FILE_OPENER: _FileOpener | None = None


def application() -> QApplication:
    """Returns the running QApplication, creating it with the viewer's name and icon."""
    global _FILE_OPENER
    _give_streams()
    _set_windows_app_id()
    app = QApplication.instance()
    if app is None:
        if sys.platform == "darwin":
            from larmor.viewer import macos

            macos.prepare(APP_NAME)  # before Qt reads the name
        app = QApplication([sys.argv[0]])
    assert isinstance(app, QApplication)
    # Only macOS sends files to open as events; elsewhere they come as arguments.
    if _FILE_OPENER is None and sys.platform == "darwin":
        _FILE_OPENER = _FileOpener()
        app.installEventFilter(_FILE_OPENER)

    app.setApplicationName(APP_NAME)
    app.setDesktopFileName(APP_ID)
    app.setWindowIcon(QIcon(str(ICON)))
    values = settings().values
    apply_style(values.theme, values.accent)
    return app


def settings() -> Settings:
    """Returns the user's settings; the first call reads them, sets theme and GPU."""
    global _SETTINGS
    if _SETTINGS is None:
        _SETTINGS = Settings.load()
        # Only before anything is drawn: wgpu keeps the GPU it starts with.
        select_gpu(_SETTINGS.values.gpu)
        _apply_theme(_SETTINGS.values.theme)
        _SETTINGS.subscribe(_on_setting)
    return _SETTINGS


def _on_setting(event: SettingChanged) -> None:
    """Restyles the application when the theme or the accent colour changes."""
    if event.name == SettingName.THEME:
        assert isinstance(event.new, Theme)
        _apply_theme(event.new)
    if event.name in (SettingName.THEME, SettingName.ACCENT):
        assert _SETTINGS is not None
        apply_style(_SETTINGS.values.theme, _SETTINGS.values.accent)


def _apply_theme(theme: Theme) -> None:
    """Follows the system's light or dark colours, or sets one of them, for Qt."""
    hints = QGuiApplication.styleHints()
    if theme == Theme.SYSTEM:
        hints.unsetColorScheme()
    else:
        hints.setColorScheme(COLOR_SCHEMES[theme])


def window(viewer: Viewer) -> MainWindow | None:
    """Returns the viewer's open window, or None if it has none."""
    w = _WINDOWS.get(viewer)
    return w if w is not None and shiboken6.isValid(w) else None


def show(
    viewer: Viewer,
    sources: list[tuple[Category, list[Path | object]]],
    max_streamlines: int | None | Default,
    block: bool,
    mcp_port: int | None | Default = None,
    session: Path | None = None,
    phantom: Path | None = None,
    simulation: Path | None = None,
) -> None:
    """Shows the viewer's window, opens a session and queues sources on it.

    Waits until the window is closed if block is set. With mcp_port, the window
    also serves agents on that port, or on the one in the settings for DEFAULT.
    A taken port prints a message, and the window opens without the server.
    """
    application()
    w = window(viewer)
    if w is None:
        w = MainWindow(viewer, settings=settings())
        w.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        _WINDOWS[viewer] = w
        w.destroyed.connect(lambda *_: _WINDOWS.pop(viewer, None))
    if session is not None:
        w.open_session(session, max_streamlines)
    if phantom is not None:
        w.open_phantom(phantom)
    if simulation is not None:
        w.open_simulation(simulation)
    if _FILE_OPENER is not None and _FILE_OPENER.pending:
        _FILE_OPENER.open_in(w, has_session=session is not None)
    for category, queued in sources:
        w.loader.open(queued, category, max_streamlines)
    if mcp_port is not None and not w.mcp.running:
        try:
            w.start_mcp(None if isinstance(mcp_port, Default) else mcp_port)
        except PortInUseError as error:
            sys.stderr.write(f"{error}\n")
    w.show()
    w.raise_()
    w.activateWindow()
    if block:
        loop = QEventLoop()
        w.destroyed.connect(loop.quit)
        loop.exec()


def wait(viewer: Viewer, timeout: float | None) -> None:
    """Runs Qt's events until the viewer's window has loaded its files."""
    app = application()
    deadline = None if timeout is None else time.monotonic() + timeout
    while (w := window(viewer)) is not None and (w.loader.busy or w.jobs.busy):
        if deadline is not None and time.monotonic() > deadline:
            raise TimeoutError(f"The files were not loaded within {timeout} s.")
        app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 50)
        time.sleep(0.01)
