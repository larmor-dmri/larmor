"""The main window: menus, the views, and the panels docked around them."""

import sys
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import QObject, QPoint, QSignalBlocker, Qt, QTimer
from PySide6.QtGui import QAction, QCloseEvent, QShowEvent
from PySide6.QtWidgets import (
    QFileDialog,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QWidget,
)

from larmor.types import Tractogram
from larmor.viewer import APP_NAME
from larmor.viewer.jobs import JobBusyError, JobRunner
from larmor.viewer.loading import DEFAULT, Default, Loader
from larmor.viewer.mcp_server import (
    Background,
    Done,
    McpServer,
    PortInUseError,
)
from larmor.viewer.model.breakdown import title
from larmor.viewer.model.item import Category, Item, ItemId
from larmor.viewer.model.matlab_import import (
    LoadAs,
    MatlabPreset,
    plan_import,
    values_spec,
)
from larmor.viewer.model.options import OptionGroup, OptionName
from larmor.viewer.model.registry import REGISTRY, SUBSETS
from larmor.viewer.model.session import SUFFIX, SessionReport, read_session
from larmor.viewer.model.settings import (
    AppSettings,
    PhantomLayout,
    SettingChanged,
    SettingName,
    Settings,
)
from larmor.viewer.model.shortcuts import Command
from larmor.viewer.model.viewer import SAVABLE, Viewer
from larmor.viewer.phantom_jobs import (
    start_open,
    start_open_simulation,
    start_run,
)
from larmor.viewer.qt.docks import (
    LAYOUT_FILE,
    PANELS,
    Docks,
    ItemPanel,
    Panel,
    Side,
)
from larmor.viewer.qt.fod_inspector_panel import FodInspectorPanel
from larmor.viewer.qt.help_panel import HelpPanel
from larmor.viewer.qt.labels_panel import LabelsButton, LabelsPanel
from larmor.viewer.qt.layers import LayersPanel
from larmor.viewer.qt.loading import (
    JobBar,
    LoadingBar,
    MainThread,
    create_jobs,
    create_loader,
)
from larmor.viewer.qt.matlab_dialog import MatlabChoice, MatlabImportDialog
from larmor.viewer.qt.options_panel import OptionsPanel
from larmor.viewer.qt.phantom_panel import PhantomPanel
from larmor.viewer.qt.phantom_steps import PhantomEditor
from larmor.viewer.qt.regions_editor import RegionsEditor
from larmor.viewer.qt.registration_dialog import RegistrationDialog
from larmor.viewer.qt.screenshot_panel import ScreenshotPanel
from larmor.viewer.qt.settings_panel import McpSwitch, SettingsPanel
from larmor.viewer.qt.shortcuts import ShortcutsDialog, shortcut_actions
from larmor.viewer.qt.views import ViewGrid
from larmor.viewer.render.scene_renderer import SceneRenderer

if TYPE_CHECKING:
    # matplotlib, which the panel draws with, loads when the panel first opens.
    from larmor.viewer.qt.breakdown_panel import BreakdownPanel


class MainWindow(QMainWindow):
    """The viewer's window."""

    def __init__(
        self,
        viewer: Viewer | None = None,
        parent: QWidget | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Creates the main window; settings default to the built-in ones, unsaved.

        With saved settings, the panels' layout is saved next to them on close and
        restored when the window first shows.
        """
        super().__init__(parent)
        self.setWindowTitle(APP_NAME)
        self.resize(1400, 900)

        self.viewer = viewer if viewer is not None else Viewer()
        self.views = ViewGrid(self.viewer, self)

        self.renderer = SceneRenderer(
            self.viewer, {kind: panel.view for kind, panel in self.views.panels.items()}
        )
        # Not `self.renderer` in the lambda: it would keep the window alive.
        renderer = self.renderer
        self.destroyed.connect(lambda *_: renderer.close())

        self.layers = LayersPanel(self.viewer, self)
        self.options = self._create_options()
        self._create_loader_and_jobs()
        self.settings = settings if settings is not None else Settings()
        self._create_mcp()
        self._follow_settings()
        self.shortcuts = shortcut_actions(self, self.viewer)
        self._create_docks()
        self._build_menus()
        self.layout_path = (
            self.settings.path.parent / LAYOUT_FILE
            if self.settings.path is not None
            else None
        )
        self._layout_restored = False  # restored when the window first shows

    def _create_options(self) -> OptionsPanel:
        """Creates the Options panel, with the widgets some option groups end with."""
        # An FOD's options end with the button that inspects it, an overlay's with
        # the one that chooses its labels in 3D, a tractogram's regions with
        # their list.
        return OptionsPanel(
            self.viewer,
            self,
            extras={
                Category.FODS: (OptionGroup.GLYPHS, _inspect_button),
                Category.OVERLAYS: (
                    OptionGroup.LABELS,
                    partial(_labels_button, self.viewer),
                ),
                Category.TRACTOGRAMS: (
                    OptionGroup.REGIONS,
                    partial(RegionsEditor, self.viewer),
                ),
            },
        )

    def _create_loader_and_jobs(self) -> None:
        """Creates the loader, the job runner, their bars and the frame scheduler.

        It also connects the Layers panel's drop and Save As, the views' refused
        messages to the status bar, and the views' context menu.
        """
        self.loading_bar = LoadingBar(self)
        self.statusBar().addPermanentWidget(self.loading_bar)
        self.loader = create_loader(self.viewer, self, self.loading_bar)
        loader = self.loader  # not `self` in the lambda, as for the renderer
        self.destroyed.connect(lambda *_: loader.stop())
        self.layers.files_dropped.connect(self.loader.open)
        self.layers.save_requested.connect(self.ask_save_item)
        # Long jobs, such as a simulation, with their own bar and Cancel.
        self.job_bar = JobBar(self)
        self.statusBar().addPermanentWidget(self.job_bar)
        self.jobs = create_jobs(self, self.job_bar)
        jobs = self.jobs  # not `self` in the lambda, as for the renderer
        self.destroyed.connect(lambda *_: jobs.stop())
        status_bar = self.statusBar()
        for view_input in self.views.inputs.values():
            view_input.refused.connect(
                lambda text: status_bar.showMessage(text, NOTICE_MS)
            )
            view_input.menu_requested.connect(self._popup_view_menu)
        # The phantom's rebuilds wait for the next frame, so that several point
        # edits in one frame draw the bundle once. The viewer outlives the window.
        self.frames = FrameScheduler(self)
        self.viewer.set_scheduler(self.frames.schedule)
        viewer_ = self.viewer
        self.destroyed.connect(lambda *_: viewer_.set_scheduler(None))

    def _create_mcp(self) -> None:
        """Creates the MCP server, which serves the actions to agents once started.

        Its tool calls run on the main thread.
        """
        background: dict[str, Background] = {
            "load": self.loader.load,
            # Not a method: what the server holds must not keep the window alive.
            "load_session": partial(_agent_session, self.loader),
            "load_matlab": partial(_agent_matlab, self.loader),
            "run_phantom": partial(_agent_run, self.viewer, self.jobs, self.loader),
            "load_phantom": partial(_agent_phantom, self.viewer, self.jobs),
            "open_simulation": partial(
                _agent_simulation, self.viewer, self.jobs, self.loader
            ),
        }
        self.mcp = McpServer(self.viewer, MainThread(self).post, background=background)
        mcp = self.mcp  # not `self` in the lambda, as for the renderer
        self.destroyed.connect(lambda *_: mcp.stop())

    def _follow_settings(self) -> None:
        """Applies the settings now and on every change, until the window goes.

        Also moves a running MCP server to a new port, and shows the renderer's
        notices, such as tubes drawn as lines, in the status bar.
        """
        # Not `self` in the callbacks, as for the renderer.
        apply = partial(
            _apply_setting,
            self.settings,
            self.viewer,
            self.renderer,
            self.loader,
        )
        for name in APPLIED:
            apply(name)
        # The settings outlive the window: stop listening when the window goes.
        unsubscribe = self.settings.subscribe(
            lambda event: apply(event.name) if event.name in APPLIED else None
        )
        self.destroyed.connect(lambda *_: unsubscribe())
        mcp, values, status = self.mcp, self.settings, self.statusBar()

        def move_mcp(event: SettingChanged) -> None:
            if event.name == SettingName.MCP_PORT and mcp.running:
                try:
                    _start_mcp(mcp, values.values.mcp_port)
                except PortInUseError as error:
                    status.showMessage(str(error), NOTICE_MS)

        unsubscribe_port = self.settings.subscribe(move_mcp)
        self.destroyed.connect(lambda *_: unsubscribe_port())
        status = self.statusBar()
        stop_notices = self.renderer.subscribe_notices(
            lambda text: status.showMessage(text, NOTICE_MS)
        )
        self.destroyed.connect(lambda *_: stop_notices())

    def _create_docks(self) -> None:
        """Creates the docks, whose panels are made on first use."""
        # Not `self` in the makers, as for the renderer.
        viewer, views, values = self.viewer, self.views, self.settings
        layers, options, mcp = self.layers, self.options, self.mcp
        switch = McpSwitch(mcp, lambda: _start_mcp(mcp, values.values.mcp_port))
        self.docks = Docks(
            self,
            self.views,
            {
                Panel.LAYERS: lambda: layers,
                Panel.OPTIONS: lambda: options,
                Panel.LABELS: lambda: LabelsPanel(viewer),
                Panel.SCREENSHOTS: lambda: ScreenshotPanel(viewer, views, values),
                Panel.PHANTOM: lambda: _phantom_editor(viewer, values),
                Panel.INSPECT_FOD: lambda: FodInspectorPanel(viewer),
                # `docks` is bound below, before any panel is made.
                Panel.SIGNAL_BREAKDOWN: lambda: _breakdown_panel(viewer, docks),
                Panel.SETTINGS: lambda: SettingsPanel(values, None, switch),
                Panel.HELP: lambda: HelpPanel(),
            },
            self.shortcuts,
        )
        # A double click on a layer shows its options; a single click does too
        # while no right panel shows. Bound to the docks, not the window, so the
        # connections keep no reference to the window and it can be freed.
        docks = self.docks
        layers.name_double_clicked.connect(partial(docks.open, Panel.OPTIONS))
        layers.name_clicked.connect(partial(_options_if_right_empty, docks))

    def _build_menus(self) -> None:
        """Builds the menu bar."""
        bar = self.menuBar()
        self._build_file_menu(bar.addMenu("&File"))
        self._build_view_menu(bar.addMenu("&View"))
        settings = bar.addMenu("&Settings")
        preferences = self.shortcuts[Command.SETTINGS]
        # Kept in this menu: macOS would move a preferences item into the app menu,
        # titled "Python" when run from Python, and hide this menu as empty.
        preferences.setMenuRole(QAction.MenuRole.NoRole)
        settings.addAction(preferences)
        self._build_help_menu(bar.addMenu("&Help"))

    def _build_file_menu(self, file: QMenu) -> None:
        """Fills the File menu."""
        quit_action = self.shortcuts[Command.QUIT]
        quit_action.setText("&Quit")
        quit_action.setMenuRole(QAction.MenuRole.QuitRole)
        quit_action.triggered.connect(self.close)

        open_action = self.shortcuts[Command.OPEN]
        open_action.setText("&Open...")
        open_action.triggered.connect(self._open_files)

        open_registered = QAction("Open Through a &Registration...", self)
        open_registered.triggered.connect(self.open_through_registration)

        open_session = self.shortcuts[Command.OPEN_SESSION]
        open_session.setText("Open Sess&ion...")
        open_session.triggered.connect(self._ask_open_session)

        save_session = self.shortcuts[Command.SAVE_SESSION]
        save_session.setText("Sa&ve Session...")
        save_session.triggered.connect(self._ask_save_session)

        self.save_item_action = QAction("Save &As...", self)
        self.save_item_action.setToolTip("Save the item selected in Layers to a file")
        self.save_item_action.triggered.connect(self._save_selected)

        screenshot = self.shortcuts[Command.SCREENSHOT]

        file.addAction(open_action)
        file.addAction(open_registered)
        file.addAction(self.save_item_action)
        file.addSeparator()
        file.addAction(open_session)
        file.addAction(save_session)
        file.addSeparator()
        file.addAction(screenshot)
        file.addSeparator()
        self._add_phantom_items(file)
        file.addSeparator()
        file.addAction(quit_action)
        file.aboutToShow.connect(self._show_file_menu)
        self._show_file_menu()

    def _add_phantom_items(self, file: QMenu) -> None:
        """Adds the phantom's and the simulation's items to the File menu."""
        new_phantom = QAction("&New Phantom", self)
        new_phantom.triggered.connect(self.new_phantom)
        open_phantom = QAction("Open P&hantom...", self)
        open_phantom.triggered.connect(self.ask_open_phantom)
        self.save_phantom_action = QAction("Save Phan&tom...", self)
        self.save_phantom_action.triggered.connect(self.ask_save_phantom)
        open_simulation = QAction("Open Si&mulation...", self)
        open_simulation.triggered.connect(self._ask_open_simulation)
        run = self.shortcuts[Command.RUN_PHANTOM]
        run.setText("&Run Phantom")
        run.triggered.connect(self.run_phantom)
        for action in (new_phantom, open_phantom, self.save_phantom_action, run):
            file.addAction(action)
        file.addAction(open_simulation)

    def _build_view_menu(self, view: QMenu) -> None:
        """Fills the View menu.

        Its ticks are read from the settings and the view state when it opens.
        """
        overlay = self.shortcuts[Command.INFORMATION_OVERLAY]
        overlay.setText("&Information overlay")
        overlay.setCheckable(True)
        view.addAction(overlay)
        self.crosshair = self.shortcuts[Command.CROSSHAIR]
        self.crosshair.setText("C&rosshair")
        self.crosshair.setCheckable(True)
        view.addAction(self.crosshair)
        reset = self.shortcuts[Command.RESET_VIEW]
        reset.setText("Reset &view")
        view.addAction(reset)
        panels = view.addMenu("&Panels")
        for info in PANELS.values():
            panels.addAction(self.shortcuts[info.command])
        panels.addSeparator()
        panels.addAction(self.shortcuts[Command.HIDE_PANELS])
        reset_layout = QAction("Reset &Layout", self)
        reset_layout.setToolTip("Put the panels back where they were at first start")
        reset_layout.triggered.connect(self.reset_layout)
        panels.addAction(reset_layout)
        self.grid_box_action = QAction("Phantom &grid box", self)
        self.grid_box_action.setCheckable(True)
        self.grid_box_action.toggled.connect(self._toggle_grid_box)
        view.addAction(self.grid_box_action)
        self.guided_action = QAction("G&uided phantom editor", self)
        self.guided_action.setToolTip(
            "The phantom editor's steps to a first simulation, one at a time"
        )
        self.guided_action.setCheckable(True)
        self.guided_action.toggled.connect(self._toggle_guided)
        view.addAction(self.guided_action)
        view.aboutToShow.connect(self._show_view_menu)
        self._show_view_menu()
        camera = view.addMenu("3D &camera")
        presets = {
            Command.CAMERA_AXIAL: "&Axial",
            Command.CAMERA_CORONAL: "&Coronal",
            Command.CAMERA_SAGITTAL: "&Sagittal",
            Command.CAMERA_RESET: "&Reset",
        }
        for command, name in presets.items():
            self.shortcuts[command].setText(name)
            camera.addAction(self.shortcuts[command])

    def _build_help_menu(self, help_menu: QMenu) -> None:
        """Fills the Help menu."""
        help_menu.addAction(self.shortcuts[Command.HELP])
        shortcuts = QAction("&Shortcuts", self)
        shortcuts.triggered.connect(self.show_shortcuts)
        help_menu.addAction(shortcuts)
        about = QAction("&About", self)
        about.setMenuRole(QAction.MenuRole.AboutRole)
        about.triggered.connect(self._about)
        help_menu.addAction(about)

    def _open_files(self) -> None:
        """Asks for files and a category, whose filter is the category choice.

        MATLAB files open the import dialog, which asks what to load from them.
        """
        filters: dict[str, Category | None] = {
            f"{kind.label} ({kind.patterns()})": category
            for category, kind in REGISTRY.items()
            if kind.from_files
        }
        filters[MATLAB_FILTER] = None
        current = _category_of(self.viewer, self.viewer.scene.view.selected)
        chosen = next(
            (f for f, c in filters.items() if c == current), next(iter(filters))
        )

        paths, chosen = QFileDialog.getOpenFileNames(
            self, "Open", "", ";;".join(filters), chosen
        )
        matlab = [p for p in paths if p.lower().endswith(".mat")]
        others = [p for p in paths if p not in matlab]
        category = filters.get(chosen, current)
        if others:
            self.loader.open(others, category if category is not None else current)
        for path in matlab:
            self.import_matlab(path)

    def open_through_registration(self) -> RegistrationDialog:
        """Asks for files and a registration, then loads them in the background."""
        current = _category_of(self.viewer, self.viewer.scene.view.selected)
        dialog = RegistrationDialog(current, self)
        dialog.accepted.connect(lambda: self._open_registered(dialog))
        dialog.open()
        return dialog

    def _open_registered(self, dialog: RegistrationDialog) -> None:
        """Loads the files and registration chosen in the dialog."""
        choice = dialog.choice()
        self.loader.open(
            list(choice.paths), choice.category, registration=choice.registration
        )

    def import_matlab(self, path: str | Path) -> None:
        """Asks what to load from a MATLAB file, then loads it in the background."""
        try:
            dialog = MatlabImportDialog(path, self.viewer, self)
        except (OSError, ValueError) as e:
            self._message("Cannot open file", f"Cannot open {path}: {e}")
            return
        if dialog.exec() == MatlabImportDialog.DialogCode.Accepted:
            self.load_matlab(dialog.choice())

    def load_matlab(self, choice: MatlabChoice) -> None:
        """Loads what the import dialog chose; problems are shown in a message box."""
        try:
            planned, values = plan_import(
                choice.path,
                choice.preset,
                choice.reference,
                choice.variables,
                choice.tractogram,
            )
        except (OSError, ValueError) as e:
            self._message("Cannot import file", str(e))
            return

        def done(items: list[Item], problems: list[str]) -> None:
            for variable in values:
                if choice.tractogram is None:
                    break
                try:
                    spec = values_spec(choice.path, variable)
                    self.viewer.set(choice.tractogram, OptionName.VALUES_FILE, spec)
                except ValueError as e:
                    problems.append(f"{variable}: {e}")
            if problems:
                self._message("Imported with problems", "\n\n".join(problems))

        self.loader.load_planned(choice.path, planned, done)

    def _ask_open_session(self) -> None:
        """Asks for a session file and opens it."""
        path, _ = QFileDialog.getOpenFileName(self, "Open Session", "", _SESSION_FILTER)
        if path:
            self.open_session(path)

    def open_session(
        self, path: str | Path, max_streamlines: int | None | Default = DEFAULT
    ) -> None:
        """Replaces the scene with a session's; its files load in the background.

        A file that is not a session, or that cannot be read, is reported in a
        message box, as are the session's missing files and refused options once
        it has opened. The read limit defaults to the Settings panel's.
        """
        try:
            self.loader.open_session(path, self._session_opened, max_streamlines)
        except (OSError, ValueError) as e:
            self._message("Cannot open session", str(e))

    def _session_opened(self, report: SessionReport) -> None:
        """Shows the problems of a session that has opened, if any."""
        if report.problems:
            text = "The session opened, except:\n\n" + "\n\n".join(report.problems)
            self._message("Session opened with problems", text)

    def _ask_save_session(self) -> None:
        """Asks for a file, and whether to replace it, then saves the session to it."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Session", f"session{SUFFIX}", _SESSION_FILTER
        )
        if not path:
            return  # cancelled
        final = _session_path(path)
        # The dialog asked about the name as typed; the suffix may name another file.
        if final != Path(path) and final.exists():
            answer = QMessageBox.question(
                self,
                "Replace session",
                f"{final.name} already exists. Replace it?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        self.save_session(final)

    def save_session(self, path: str | Path) -> None:
        """Saves the session; a message box names the items it leaves out."""
        path = _session_path(path)
        try:
            skipped = self.viewer.save_session(path)
        except (OSError, ValueError) as e:
            self._message("Cannot save session", f"Cannot save {path}: {e}")
            return
        self.statusBar().showMessage(f"Session saved to {path}", NOTICE_MS)
        if skipped:
            names = "\n".join(f"- {name}" for name in skipped)
            self._message(
                "Items left out of the session",
                f"These items were added from Python without a file, so the "
                f"session cannot reopen them:\n\n{names}",
            )

    def _show_file_menu(self) -> None:
        """Enables the File menu's items that apply to the scene now."""
        phantom = self.viewer.phantom is not None
        self.save_phantom_action.setEnabled(phantom)
        self.shortcuts[Command.RUN_PHANTOM].setEnabled(True)
        selected = self.viewer.scene.view.selected
        self.save_item_action.setEnabled(
            isinstance(selected, int)
            and self.viewer.scene.item(selected).category in SAVABLE
        )

    def _save_selected(self) -> None:
        """Asks for a file and saves the selected item to it."""
        self.ask_save_item()

    def ask_save_item(self, item_id: ItemId | None = None) -> None:
        """Asks for a file and saves an item to it; the selected item if None."""
        target = self.viewer.scene.view.selected if item_id is None else item_id
        if not isinstance(target, int):
            return
        item = self.viewer.scene.item(target)
        kind = REGISTRY[item.category]
        # A subset's file holds more streamlines: it is never suggested.
        subset = isinstance(item.data, Tractogram) and item.data in SUBSETS
        if item.source is not None and item.part is None and not subset:
            suggested = str(item.source)  # not a MATLAB file
        else:
            name = Path(item.name).name
            for extension in kind.extensions:
                if name.lower().endswith(extension):
                    name = name[: -len(extension)]
                    break
            folder = item.source.parent if item.source is not None else Path()
            ending = "_subset" if subset else ""
            suffix = ".tck" if item.category == Category.TRACTOGRAMS else ".nii.gz"
            suggested = str(folder / f"{name}{ending}{suffix}")
        path, _ = QFileDialog.getSaveFileName(
            self,
            f"Save {item.name}",
            suggested,
            f"{kind.label} ({kind.patterns()});;All files (*)",
        )
        if path:
            self.save_item(target, path)

    def save_item(self, item_id: ItemId, path: str | Path) -> None:
        """Saves an item to a file; a message box tells why it cannot."""
        try:
            item = self.viewer.save_item(item_id, path)
        except (OSError, ValueError) as e:
            self._message("Cannot save item", f"Cannot save {path}: {e}")
            return
        self.statusBar().showMessage(f"{item.name} saved to {item.source}", NOTICE_MS)

    # The phantom editor

    def show_phantom(self) -> PhantomPanel:
        """Shows the Phantom panel."""
        return self._show(Panel.PHANTOM, PhantomEditor).panel

    def show_breakdown(self) -> BreakdownPanel:
        """Shows the Signal breakdown panel."""
        from larmor.viewer.qt.breakdown_panel import BreakdownPanel

        return self._show(Panel.SIGNAL_BREAKDOWN, BreakdownPanel)

    def show_breakdown_at(self, position: tuple[float, float, float]) -> None:
        """Shows the Signal breakdown panel, pinned to the voxel at a position."""
        self.show_breakdown()
        try:
            self.viewer.pin_breakdown(position)
        except ValueError as e:
            self.statusBar().showMessage(str(e), NOTICE_MS)

    def view_menu(self, position: tuple[float, float, float]) -> QMenu:
        """Returns the context menu of a voxel of the phantom in a 2D view."""
        menu = QMenu(self)
        menu.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        here = menu.addAction("Signal breakdown here")
        here.triggered.connect(lambda: self.show_breakdown_at(position))
        return menu

    def _popup_view_menu(
        self, position: tuple[float, float, float], point: QPoint
    ) -> None:
        """Shows the context menu of a voxel at a point on the screen."""
        self.view_menu(position).popup(point)

    def new_phantom(self) -> None:
        """Starts a phantom on the views' grid, or the default box, and shows it."""
        grid_of = self.viewer.scene.view.main_volume
        self.viewer.new_phantom(grid_of=grid_of)
        self.show_phantom()

    def ask_open_phantom(self) -> None:
        """Asks for a phantom file and opens it."""
        path, _ = QFileDialog.getOpenFileName(self, "Open Phantom", "", _PHANTOM_FILTER)
        if path:
            self.open_phantom(path)

    def open_phantom(self, path: str | Path) -> None:
        """Opens a phantom file in the editor, its bundles built in the background."""
        try:
            start_open(self.viewer, self.jobs, path, on_done=self.show_phantom)
        except JobBusyError as e:
            self._message("Cannot open phantom", str(e))

    def _ask_open_simulation(self) -> None:
        """Asks for a simulation folder and opens it."""
        folder = QFileDialog.getExistingDirectory(self, "Open Simulation")
        if folder:
            self.open_simulation(folder)

    def open_simulation(self, folder: str | Path) -> None:
        """Opens a simulation folder: its phantom in the editor, its files as items."""
        try:
            start_open_simulation(
                self.viewer,
                self.jobs,
                folder,
                self.loader.open,
                on_done=lambda files: self.show_phantom(),
            )
        except (JobBusyError, OSError, ValueError) as e:
            self._message("Cannot open simulation", str(e))

    def ask_save_phantom(self) -> None:
        """Asks for a file and saves the phantom to it."""
        if self.viewer.phantom is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Phantom", "scene.phantom.json", _PHANTOM_FILTER
        )
        if path:
            try:
                self.viewer.save_phantom(path)
            except (OSError, ValueError) as e:
                self._message("Cannot save phantom", str(e))
                return
            self.statusBar().showMessage(f"Phantom saved to {path}", NOTICE_MS)

    def cancel_job(self) -> None:
        """Cancels the running job, such as a simulation."""
        self.jobs.cancel()

    def run_phantom(self) -> None:
        """Runs the phantom in the background into its folder, asked for if unset.

        Asks before replacing the outputs of an earlier run; a phantom that
        cannot run is explained in a message box.
        """
        state = self.viewer.phantom
        if state is None:
            self._message("No phantom", "Start or open a phantom first.")
            return
        problems = self.viewer.check_phantom()
        if problems:
            self._message("The phantom cannot run", "\n".join(problems))
            return
        folder = state.output
        if folder is None:
            chosen = QFileDialog.getExistingDirectory(self, "Output folder")
            if not chosen:
                return
            folder = Path(chosen)
            self.viewer.set_run_folder(folder)
        try:
            self._start_run(folder, overwrite=False)
        except FileExistsError as e:
            answer = QMessageBox.question(
                self,
                "Replace outputs",
                f"{e}\n\nReplace them?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer == QMessageBox.StandardButton.Yes:
                self._start_run(folder, overwrite=True)

    def _start_run(self, folder: Path, *, overwrite: bool) -> None:
        """Starts running the phantom; a message box tells why it cannot."""
        try:
            start_run(
                self.viewer,
                self.jobs,
                folder,
                self.loader.open,
                overwrite=overwrite,
                on_done=lambda outputs: self.statusBar().showMessage(
                    f"Simulation saved to {outputs.files.folder}", NOTICE_MS
                ),
            )
        except (JobBusyError, ValueError) as e:
            self._message("Cannot run the phantom", str(e))

    def _message(self, title: str, text: str) -> None:
        """Shows a warning without blocking the window."""
        box = QMessageBox(
            QMessageBox.Icon.Warning, title, text, QMessageBox.StandardButton.Ok, self
        )
        box.setWindowModality(Qt.WindowModality.NonModal)
        box.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        box.show()

    def reset_layout(self) -> None:
        """Puts the panels back as at first start, and forgets the saved layout."""
        self.docks.reset()
        if self.layout_path is not None:
            self.layout_path.unlink(missing_ok=True)

    def _toggle_grid_box(self, shown: bool) -> None:
        """Shows or hides the phantom's grid box."""
        if self.viewer.phantom is not None:
            self.viewer.show_grid_box(shown)

    def _toggle_guided(self, guided: bool) -> None:
        """Switches the phantom editor between its guided steps and all its sections."""
        layout = PhantomLayout.GUIDED if guided else PhantomLayout.ALL
        self.settings.set(SettingName.PHANTOM_LAYOUT, layout)

    def _show_view_menu(self) -> None:
        """Updates the View menu's ticks from the settings and the view state."""
        phantom = self.viewer.phantom
        with QSignalBlocker(self.guided_action):
            guided = self.settings.values.phantom_layout == PhantomLayout.GUIDED
            self.guided_action.setChecked(guided)
        self.grid_box_action.setEnabled(phantom is not None)
        with QSignalBlocker(self.grid_box_action):
            self.grid_box_action.setChecked(phantom is not None and phantom.show_box)
        state = self.viewer.scene.view
        self.shortcuts[Command.INFORMATION_OVERLAY].setChecked(state.overlay_visible)
        self.crosshair.setChecked(state.crosshair_visible)

    def show_settings(self) -> SettingsPanel:
        """Shows the Settings panel."""
        return self._show(Panel.SETTINGS, SettingsPanel)

    def show_screenshot(self) -> ScreenshotPanel:
        """Shows the Screenshots panel."""
        return self._show(Panel.SCREENSHOTS, ScreenshotPanel)

    def inspect_fod(self, item_id: ItemId | None = None) -> FodInspectorPanel:
        """Shows the Inspect FOD panel, on this FOD if given.

        Returns:
            FodInspectorPanel: The panel.

        Raises:
            ValueError: If the item is not an FOD.
        """
        self._check_category(item_id, Category.FODS)
        return self._show_item(Panel.INSPECT_FOD, FodInspectorPanel, item_id)

    def choose_labels(self, item_id: ItemId | None = None) -> LabelsPanel:
        """Shows the Labels panel, on this overlay if given.

        Returns:
            LabelsPanel: The panel.

        Raises:
            ValueError: If the item is not an overlay.
        """
        self._check_category(item_id, Category.OVERLAYS)
        return self._show_item(Panel.LABELS, LabelsPanel, item_id)

    def _check_category(self, item_id: ItemId | None, category: Category) -> None:
        """Raises ValueError if the item, when given, is not in the category."""
        if item_id is None:
            return
        item = self.viewer.scene.item(item_id)
        if item.category != category:
            raise ValueError(
                f"{item.name} is in {item.category.value}, not {category.value}."
            )

    def _show[T: QWidget](self, panel: Panel, kind: type[T]) -> T:
        """Opens a panel and returns its widget."""
        self.docks.open(panel)
        widget = self.docks.widget(panel)
        assert isinstance(widget, kind)
        return widget

    def _show_item[T: ItemPanel](
        self, panel: Panel, kind: type[T], item_id: ItemId | None
    ) -> T:
        """Opens an item panel, on the item if given, and returns its widget."""
        widget = self._show(panel, kind)
        if item_id is not None:
            widget.show_item(item_id)
        return widget

    def showEvent(self, event: QShowEvent) -> None:  # noqa: N802
        """Restores the panels' layout the first time the window shows."""
        super().showEvent(event)
        if not self._layout_restored:
            self._layout_restored = True
            if self.layout_path is not None:
                self.docks.restore(self.layout_path)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        """Saves the panels' layout, stops the MCP server and the item panels."""
        if self.layout_path is not None:
            self.docks.save(self.layout_path)
        self.mcp.stop()
        # The viewer outlives the window: the panels' items stop following it.
        for panel in self.findChildren(ItemPanel):
            panel.drop()
        super().closeEvent(event)

    def start_mcp(self, port: int | None = None) -> None:
        """Serves the actions to agents on port, the setting's if None.

        Prints the URL and the token to the terminal.

        Raises:
            PortInUseError: If another program listens on the port.
        """
        _start_mcp(
            self.mcp, port if port is not None else self.settings.values.mcp_port
        )

    def show_shortcuts(self) -> None:
        """Opens the list of shortcuts."""
        ShortcutsDialog(self).exec()

    def _about(self) -> None:
        """Shows the About box with the version."""
        from larmor import __version__

        QMessageBox.about(self, "About", f"<b>{APP_NAME}</b><br>Larmor {__version__}")


def _start_mcp(mcp: McpServer, port: int) -> None:
    """Starts the MCP server on port and prints its URL and token to the terminal."""
    mcp.start(port)
    sys.stderr.write(f"Serving agents (MCP) at {mcp.url()}\nToken: {mcp.token}\n")


# How long a notice stays in the status bar.
NOTICE_MS = 10_000
_SESSION_FILTER = f"Sessions (*{SUFFIX} *.json);;All files (*)"
_PHANTOM_FILTER = "Phantom files (*.phantom.json *.json);;All files (*)"


class FrameScheduler(QObject):
    """Runs the functions scheduled within one frame once, at the next frame."""

    FRAME_MS = 16  # one frame at 60 frames a second

    def __init__(self, parent: QObject) -> None:
        """Creates the scheduler, with nothing waiting."""
        super().__init__(parent)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(self.FRAME_MS)
        self._timer.timeout.connect(self._run)
        self._waiting: list[Callable[[], None]] = []

    def schedule(self, function: Callable[[], None]) -> None:
        """Runs function at the next frame."""
        self._waiting.append(function)
        if not self._timer.isActive():
            self._timer.start()

    def _run(self) -> None:
        """Runs the functions waiting, in the order they were scheduled."""
        waiting, self._waiting = self._waiting, []
        for function in waiting:
            function()


MATLAB_FILTER = "MATLAB files (*.mat)"

# The settings each window applies to its viewer, renderer and loader; the theme and
# the GPU are the application's (app.py).
APPLIED = (
    SettingName.CONVENTION,
    SettingName.BACKGROUND,
    SettingName.CROSSHAIR,
    SettingName.INTERPOLATE,
    SettingName.READ_LIMIT,
    SettingName.DRAW_CAP,
    SettingName.TUBE_LIMIT,
    SettingName.POINTS_PER_STREAMLINE,
    SettingName.SLAB_THICKNESS,
    SettingName.SLAB_MEMORY,
    SettingName.LOWER_RESOLUTION,
)


def _apply_setting(
    settings: Settings,
    viewer: Viewer,
    renderer: SceneRenderer,
    loader: Loader,
    name: SettingName,
) -> None:
    """Makes one setting take effect on a window's viewer, renderer and loader."""
    values = settings.values
    if name == SettingName.READ_LIMIT:
        loader.max_streamlines = values.max_streamlines
    elif name in _RENDERER_SETTINGS:
        _apply_to_renderer(values, renderer, name)
    else:
        _apply_to_viewer(values, viewer, name)


# The settings that take effect on the renderer.
_RENDERER_SETTINGS = (
    SettingName.BACKGROUND,
    SettingName.CROSSHAIR,
    SettingName.DRAW_CAP,
    SettingName.SLAB_MEMORY,
    SettingName.TUBE_LIMIT,
    SettingName.LOWER_RESOLUTION,
)


def _apply_to_renderer(
    values: AppSettings, renderer: SceneRenderer, name: SettingName
) -> None:
    """Makes a setting of `_RENDERER_SETTINGS` take effect on the renderer."""
    if name in (SettingName.BACKGROUND, SettingName.CROSSHAIR):
        for view in renderer.views.values():
            view.set_colours(values.background, values.crosshair)
    elif name in (
        SettingName.DRAW_CAP,
        SettingName.SLAB_MEMORY,
        SettingName.TUBE_LIMIT,
    ):
        renderer.set_tract_limits(values.draw_cap, values.slab_bytes, values.tube_limit)
    elif name == SettingName.LOWER_RESOLUTION:
        for view in renderer.views.values():
            view.set_lower_resolution(values.lower_resolution)


def _apply_to_viewer(values: AppSettings, viewer: Viewer, name: SettingName) -> None:
    """Makes a setting take effect on the viewer; the others are ignored."""
    if name == SettingName.CONVENTION:
        viewer.set_convention(values.convention)
    elif name == SettingName.INTERPOLATE:
        for category in (Category.VOLUMES, Category.OVERLAYS):
            viewer.set_default(category, OptionName.INTERPOLATE, values.interpolate)
    elif name == SettingName.POINTS_PER_STREAMLINE:
        viewer.set_default(
            Category.TRACTOGRAMS, OptionName.POINTS, values.points_per_streamline
        )
    elif name == SettingName.SLAB_THICKNESS:
        viewer.set_default(
            Category.TRACTOGRAMS, OptionName.SLAB_THICKNESS, values.slab_thickness
        )


def _inspect_button(item_id: ItemId, parent: QWidget) -> QWidget:
    """Creates an FOD's Inspect FOD button, which shows it in the Inspect FOD panel."""
    button = QPushButton("Inspect FOD", parent)
    button.setToolTip("Show this FOD at the crosshair, large, in the Inspect FOD panel")

    def clicked() -> None:
        # The window found when clicked: holding it would keep it alive.
        window = button.window()
        if isinstance(window, MainWindow):
            window.inspect_fod(item_id)

    button.clicked.connect(clicked)
    return button


def _labels_button(viewer: Viewer, item_id: ItemId, parent: QWidget) -> QWidget:
    """Creates an overlay's Choose labels button, which shows it in the Labels panel."""
    button = LabelsButton(viewer, item_id, parent)

    def clicked() -> None:
        # The window found when clicked: holding it would keep it alive.
        window = button.window()
        if isinstance(window, MainWindow):
            window.choose_labels(item_id)

    button.clicked.connect(clicked)
    return button


def _main_window(widget: QWidget) -> MainWindow | None:
    """Returns the main window a widget is in, also from a floating panel."""
    # Through the parents: a floating panel's window() is its own container.
    found = widget.parentWidget()
    while found is not None and not isinstance(found, MainWindow):
        found = found.parentWidget()
    return found


def _phantom_editor(viewer: Viewer, settings: Settings) -> PhantomEditor:
    """Creates the Phantom editor, whose file and run buttons the window carries out."""
    editor = PhantomEditor(viewer, settings)
    panel = editor.panel

    def call(name: str, *args: object) -> None:
        # Found when clicked: holding the window would keep it alive.
        found = _main_window(editor)
        if found is not None:
            getattr(found, name)(*args)

    panel.new_requested.connect(lambda: call("new_phantom"))
    panel.open_requested.connect(lambda: call("ask_open_phantom"))
    panel.save_requested.connect(lambda: call("ask_save_phantom"))
    panel.run_requested.connect(lambda: call("run_phantom"))
    panel.cancel_requested.connect(lambda: call("cancel_job"))
    panel.save_bundle_requested.connect(lambda item_id: call("ask_save_item", item_id))
    return editor


def _breakdown_panel(viewer: Viewer, docks: Docks) -> BreakdownPanel:
    """Creates the Signal breakdown panel, whose dock's title names the voxel shown."""
    from larmor.viewer.qt.breakdown_panel import BreakdownPanel

    panel = BreakdownPanel(viewer)
    panel.title_changed.connect(partial(docks.set_title, Panel.SIGNAL_BREAKDOWN))
    docks.set_title(Panel.SIGNAL_BREAKDOWN, title(viewer.breakdown))
    return panel


def _options_if_right_empty(docks: Docks) -> None:
    """Shows the Options panel if no panel of the right stripe shows."""
    if not docks.side_open(Side.RIGHT):
        docks.open(Panel.OPTIONS)


def _category_of(viewer: Viewer, target: ItemId | Category | None) -> Category:
    """Returns the category a Layers selection stands for; Volumes if none."""
    if target is None:
        return Category.VOLUMES
    if isinstance(target, Category):
        return target
    return viewer.scene.item(target).category


def _session_path(path: str | Path) -> Path:
    """Returns the file a session is saved to, with the suffix if it has none."""
    path = Path(path)
    return path if path.name.endswith(".json") else path.with_name(path.name + SUFFIX)


def _agent_matlab(
    loader: Loader,
    path: str | Path,
    preset: MatlabPreset,
    reference: str | Path | None = None,
    variables: dict[str, LoadAs] | None = None,
    tractogram_id: ItemId | None = None,
    max_streamlines: int | None | Default = DEFAULT,
    *,
    then: Done,
) -> None:
    """Imports a MATLAB file for an agent, read in the background as files are."""
    source = Path(path).absolute()
    ref = None if reference is None else Path(reference).absolute()
    planned, values = plan_import(source, preset, ref, variables, tractogram_id)

    def done(items: list[Item], problems: list[str]) -> None:
        for variable in values:
            assert tractogram_id is not None
            try:
                spec = values_spec(source, variable)
                loader.viewer.set(tractogram_id, OptionName.VALUES_FILE, spec)
            except ValueError as e:
                problems.append(f"{variable}: {e}")
        then(items, " ".join(problems) if problems else None)

    loader.load_planned(source, planned, done, max_streamlines)


def _agent_run(
    viewer: Viewer,
    jobs: JobRunner,
    loader: Loader,
    folder: str | Path,
    *,
    overwrite: bool = False,
    then: Done,
) -> None:
    """Runs the phantom for an agent in the background; told the files once opened."""
    start_run(
        viewer,
        jobs,
        folder,
        loader.open,
        overwrite=overwrite,
        on_done=lambda outputs: then(str(outputs.files.folder), None),
        on_failed=lambda message: then(None, message),
    )


def _agent_phantom(
    viewer: Viewer, jobs: JobRunner, path: str | Path, *, then: Done
) -> None:
    """Opens a phantom file for an agent, its bundles built in the background."""
    start_open(
        viewer,
        jobs,
        path,
        on_done=lambda: then(None, None),
        on_failed=lambda message: then(None, message),
    )


def _agent_simulation(
    viewer: Viewer, jobs: JobRunner, loader: Loader, folder: str | Path, *, then: Done
) -> None:
    """Opens a simulation folder for an agent in the background."""
    start_open_simulation(
        viewer,
        jobs,
        folder,
        loader.open,
        on_done=lambda files: then(str(files.folder), None),
        on_failed=lambda message: then(None, message),
    )


def _agent_session(
    loader: Loader,
    path: str | Path,
    max_streamlines: int | None | Default = DEFAULT,
    *,
    then: Done,
) -> None:
    """Opens a session for an agent, once the files being read are dropped.

    A file that is not a session is refused first, and drops nothing.
    """
    read_session(Path(path))
    loader.cancel()  # so none of them lands in the session's scene
    limit = loader.max_streamlines if max_streamlines is DEFAULT else max_streamlines
    assert not isinstance(limit, Default)
    then(loader.viewer.load_session(path, max_streamlines=limit), None)
