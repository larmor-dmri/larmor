import logging
import socket
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pygfx as gfx
import pytest
import wgpu
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QFormLayout, QLabel, QWidget
from pytestqt.qtbot import QtBot
from viewer_helpers import free_port

from larmor.types import Tractogram, Volume
from larmor.viewer import Category, Convention, ItemId, OptionName, TractStyle, Viewer
from larmor.viewer import app as viewer_app
from larmor.viewer.mcp_server import McpServer
from larmor.viewer.model import settings as settings_module
from larmor.viewer.model.settings import (
    AppSettings,
    SettingName,
    Settings,
    Theme,
    setting_fields,
)
from larmor.viewer.qt import settings_panel
from larmor.viewer.qt.field import ChoiceField, ColourField, SliderField
from larmor.viewer.qt.main_window import MainWindow
from larmor.viewer.qt.settings_panel import (
    SCENE_SETTINGS,
    SHEET_SETTINGS,
    McpSection,
    McpSwitch,
    SettingsPanel,
    SettingsSheet,
)
from larmor.viewer.render import gpu
from larmor.viewer.render.tractogram_adapter import (
    SLAB_BUDGET,
    TractogramAdapter,
)
from larmor.viewer.render.view import INTERACTIVE_SCALE

GPUS = ["GPU A (Vulkan)", "GPU B (Vulkan)"]


@pytest.fixture(autouse=True)
def _gpus(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings_panel, "gpu_names", lambda: GPUS)


@pytest.fixture(autouse=True)
def _tract_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    # Windows set these for every tractogram; give them back after each test.
    monkeypatch.setattr(TractogramAdapter, "draw_cap", TractogramAdapter.draw_cap)
    monkeypatch.setattr(TractogramAdapter, "tube_limit", TractogramAdapter.tube_limit)
    monkeypatch.setattr(SLAB_BUDGET, "limit", SLAB_BUDGET.limit)


@pytest.fixture
def path(tmp_path: Path) -> Path:
    return tmp_path / "viewer.toml"


def _form(dialog: SettingsPanel) -> list[tuple[str, QWidget]]:
    # Each row with a field, by its label; error and note rows have an empty one.
    form = dialog.findChild(QFormLayout)
    assert form is not None
    rows: list[tuple[str, QWidget]] = []
    for row in range(form.rowCount()):
        label = form.itemAt(row, QFormLayout.ItemRole.LabelRole)
        field = form.itemAt(row, QFormLayout.ItemRole.FieldRole)
        if field is None:  # a section heading
            continue
        widget = field.widget()
        assert widget is not None
        text = "" if label is None else label.widget()
        rows.append((text.text() if isinstance(text, QLabel) else "", widget))
    return rows


def _rows(dialog: SettingsPanel) -> dict[str, QWidget]:
    return {text: widget for text, widget in _form(dialog) if text}


def _field(dialog: SettingsPanel, name: SettingName) -> QWidget:
    return _rows(dialog)[setting_fields()[name][1].label]


def _error(dialog: SettingsPanel, field: QWidget) -> QLabel:
    # The message sits in the row under its field.
    widgets = [widget for _, widget in _form(dialog)]
    label = widgets[widgets.index(field) + 1]
    assert isinstance(label, QLabel)
    return label


# The main window


def _interpolation(window: MainWindow) -> list[object]:
    for category in (Category.VOLUMES, Category.OVERLAYS):
        window.viewer.add(Volume(np.zeros((2, 2, 2), np.float32), np.eye(4)), category)
    return [
        getattr(i.options, OptionName.INTERPOLATE) for i in window.viewer.scene.items
    ]


def _backgrounds(window: MainWindow) -> set[str]:
    views = [p.view for p in window.views.panels.values()]
    return {v.background.material.color_top_left.hex for v in views}


def _lower_resolution(window: MainWindow) -> set[bool]:
    return {p.view.lower_resolution for p in window.views.panels.values()}


def _crosshairs(window: MainWindow) -> set[str]:
    views = [p.view for p in window.views.panels.values()]
    return {
        o.material.color.hex
        for v in views
        for o in v.guides.children
        if isinstance(o, gfx.Line)
    }


def _streamlines(count: int) -> Tractogram:
    points = np.arange(count * 6, dtype=np.float32).reshape(-1, 3)
    return Tractogram(points, np.arange(count + 1) * 2)


def _new_tractogram(name: OptionName) -> Callable[[MainWindow], object]:
    def option(window: MainWindow) -> object:
        item = window.viewer.add(_streamlines(1), Category.TRACTOGRAMS)
        return getattr(item.options, name)

    return option


def _adapter(window: MainWindow, item: ItemId) -> TractogramAdapter:
    adapter = window.renderer.adapters[item]
    assert isinstance(adapter, TractogramAdapter)
    return adapter


APPLIED: list[tuple[SettingName, object, Callable[[MainWindow], object], object]] = [
    (
        SettingName.CONVENTION,
        Convention.LAS,
        lambda w: w.viewer.scene.view.convention,
        Convention.LAS,
    ),
    (SettingName.READ_LIMIT, 0, lambda w: w.loader.max_streamlines, None),
    (SettingName.READ_LIMIT, 5000, lambda w: w.loader.max_streamlines, 5000),
    (SettingName.INTERPOLATE, False, _interpolation, [False, False]),
    (SettingName.BACKGROUND, "#ff0000", _backgrounds, {"#ff0000"}),
    (SettingName.CROSSHAIR, "#0000ff", _crosshairs, {"#0000ff"}),
    (SettingName.LOWER_RESOLUTION, True, _lower_resolution, {True}),
    (SettingName.DRAW_CAP, 2000, lambda w: TractogramAdapter.draw_cap, 2000),
    (SettingName.TUBE_LIMIT, 3000, lambda w: TractogramAdapter.tube_limit, 3000),
    (SettingName.SLAB_MEMORY, 1.5, lambda w: SLAB_BUDGET.limit, 3 * 1024**3 // 2),
    (
        SettingName.POINTS_PER_STREAMLINE,
        7,
        _new_tractogram(OptionName.POINTS),
        7,
    ),
    (
        SettingName.SLAB_THICKNESS,
        2.5,
        _new_tractogram(OptionName.SLAB_THICKNESS),
        2.5,
    ),
]


@pytest.mark.parametrize("when", ["at start", "on change"])
@pytest.mark.parametrize(
    ("name", "value", "shown", "expected"),
    APPLIED,
    ids=[f"{name}={value}" for name, value, _, _ in APPLIED],
)
def test_the_window_applies_a_setting_at_once(
    when: str,
    name: SettingName,
    value: object,
    shown: Callable[[MainWindow], object],
    expected: object,
    qtbot: QtBot,
) -> None:
    changes: dict[str, Any] = {name: value}
    if when == "at start":
        settings = Settings(replace(AppSettings(), **changes))
        window = MainWindow(settings=settings)
    else:
        settings = Settings()
        window = MainWindow(settings=settings)
        settings.set(name, value)
    qtbot.addWidget(window)
    assert shown(window) == expected


def test_lower_resolution_off_mid_interaction_returns_the_views_to_full(
    qtbot: QtBot,
) -> None:
    settings = Settings(replace(AppSettings(), lower_resolution=True))
    window = MainWindow(settings=settings)
    qtbot.addWidget(window)
    views = [p.view for p in window.views.panels.values()]
    for view in views:
        view.start_interaction()
        view.request_draw()
    assert {v.scale for v in views} == {INTERACTIVE_SCALE}
    settings.set(SettingName.LOWER_RESOLUTION, False)
    assert {v.scale for v in views} == {1.0}


@pytest.mark.parametrize("windows", [1, 2])
def test_a_new_draw_cap_reshapes_the_tractograms_of_every_open_window(
    windows: int, qtbot: QtBot
) -> None:
    settings = Settings()
    # qtbot holds its widgets weakly: keep each window alive.
    open_windows = [MainWindow(settings=settings) for _ in range(windows)]
    adapters = []
    for window in open_windows:
        qtbot.addWidget(window)
        item = window.viewer.add(_streamlines(1500), Category.TRACTOGRAMS)
        adapters.append(_adapter(window, item.id))
    settings.set(SettingName.DRAW_CAP, 1000)
    qtbot.waitUntil(
        lambda: [len(a.drawn.offsets) - 1 for a in adapters] == [1000] * windows
    )


def test_tubes_drawn_as_lines_are_noticed_in_the_status_bar(qtbot: QtBot) -> None:
    settings = Settings()
    window = MainWindow(settings=settings)
    qtbot.addWidget(window)
    settings.set(SettingName.TUBE_LIMIT, 1000)
    item = window.viewer.add(_streamlines(1500), Category.TRACTOGRAMS)
    window.viewer.set(item.id, OptionName.STYLE, TractStyle.TUBES)
    assert "1,500 streamlines" in window.statusBar().currentMessage()


def test_more_slab_memory_sorts_a_slab_that_did_not_fit(
    monkeypatch: pytest.MonkeyPatch, qtbot: QtBot
) -> None:
    settings = Settings()
    window = MainWindow(settings=settings)
    qtbot.addWidget(window)
    monkeypatch.setattr(SLAB_BUDGET, "limit", 1)
    adapter = _adapter(
        window, window.viewer.add(_streamlines(5), Category.TRACTOGRAMS).id
    )
    assert adapter.sorted_copies == {}
    settings.set(SettingName.SLAB_MEMORY, 1.0)
    qtbot.waitUntil(lambda: set(adapter.sorted_copies) == {0, 1, 2})


@pytest.mark.parametrize(
    ("name", "option", "value"),
    [
        (SettingName.POINTS_PER_STREAMLINE, OptionName.POINTS, 7),
        (SettingName.SLAB_THICKNESS, OptionName.SLAB_THICKNESS, 2.5),
    ],
)
def test_a_tractogram_default_leaves_the_open_tractograms_as_they_are(
    name: SettingName, option: OptionName, value: object, qtbot: QtBot
) -> None:
    settings = Settings()
    window = MainWindow(settings=settings)
    qtbot.addWidget(window)
    item = window.viewer.add(_streamlines(1), Category.TRACTOGRAMS)
    before = getattr(item.options, option)
    settings.set(name, value)
    assert getattr(window.viewer.scene.item(item.id).options, option) == before


def test_ctrl_comma_and_the_gear_open_one_settings_sheet_on_the_window_settings(
    qtbot: QtBot,
) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)
    window.activateWindow()
    # waitActive can return before the first window of a run is active.
    qtbot.waitUntil(lambda: QApplication.activeWindow() is window)
    QTest.keyClick(window, Qt.Key.Key_Comma, Qt.KeyboardModifier.ControlModifier)
    [sheet] = window.findChildren(SettingsSheet)
    assert sheet.isVisible() and sheet.isWindow()
    assert sheet.panel.settings is window.settings
    sheet.close()
    window.header.settings_button.click()
    assert window.findChildren(SettingsSheet) == [sheet] and sheet.isVisible()


def test_the_sheet_and_the_scene_tab_share_the_settings_between_them(
    qtbot: QtBot,
) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    sheet = window.show_settings()
    # The GPU shows on a computer with two of them, as in these tests.
    assert set(sheet.panel._fields) == set(SHEET_SETTINGS)
    assert set(window.inspector.scene_tab.settings_form._fields) == SCENE_SETTINGS
    assert not SCENE_SETTINGS & set(sheet.panel._fields)


# The Settings panel


@pytest.mark.parametrize("gpus", [GPUS[:1], GPUS], ids=["one GPU", "two GPUs"])
def test_the_settings_panel_has_a_field_per_setting(
    gpus: list[str], monkeypatch: pytest.MonkeyPatch, qtbot: QtBot
) -> None:
    monkeypatch.setattr(settings_panel, "gpu_names", lambda: gpus)
    dialog = SettingsPanel(Settings())
    qtbot.addWidget(dialog)
    rows = _rows(dialog)
    shown = [n for n in SettingName if n != SettingName.GPU or len(gpus) > 1]
    assert list(rows) == [setting_fields()[n][1].label for n in shown]
    for name in (SettingName.BACKGROUND, SettingName.CROSSHAIR):
        assert isinstance(_field(dialog, name), ColourField)
    if len(gpus) > 1:
        choice = _field(dialog, SettingName.GPU)
        assert isinstance(choice, ChoiceField)
        items = [choice.itemText(i) for i in range(choice.count())]
        assert items == ["Default", *gpus]


@pytest.mark.parametrize("gpus", [GPUS[:1], GPUS], ids=["one GPU", "two GPUs"])
def test_lower_resolution_is_the_first_setting_under_graphics(
    gpus: list[str], monkeypatch: pytest.MonkeyPatch, qtbot: QtBot
) -> None:
    monkeypatch.setattr(settings_panel, "gpu_names", lambda: gpus)
    dialog = SettingsPanel(Settings())
    qtbot.addWidget(dialog)
    form = dialog.findChild(QFormLayout)
    assert form is not None
    texts = []
    for row in range(form.rowCount()):
        item = form.itemAt(row, QFormLayout.ItemRole.SpanningRole)
        if item is None:
            item = form.itemAt(row, QFormLayout.ItemRole.LabelRole)
        widget = None if item is None else item.widget()
        if isinstance(widget, QLabel) and widget.text():
            texts.append(widget.text())
    heading = texts.index("Graphics")
    assert texts[heading + 1] == setting_fields()[SettingName.LOWER_RESOLUTION][1].label


@pytest.mark.parametrize(
    ("name", "start", "chosen", "saved"),
    [
        (SettingName.CONVENTION, AppSettings(), "LAS", Convention.LAS),
        (SettingName.GPU, AppSettings(gpu=GPUS[1]), "Default", ""),
    ],
)
def test_a_choice_in_the_settings_panel_is_set_and_saved(
    name: SettingName,
    start: AppSettings,
    chosen: str,
    saved: object,
    path: Path,
    qtbot: QtBot,
) -> None:
    settings = Settings(start, path)
    dialog = SettingsPanel(settings)
    qtbot.addWidget(dialog)
    field = _field(dialog, name)
    assert isinstance(field, ChoiceField)
    index = field.findText(chosen)
    field.setCurrentIndex(index)
    field.activated.emit(index)
    assert getattr(settings.values, name) == saved
    assert getattr(Settings.load(path).values, name) == saved


def test_a_bad_value_shows_an_error_under_its_field(path: Path, qtbot: QtBot) -> None:
    settings = Settings(path=path)
    dialog = SettingsPanel(settings)
    qtbot.addWidget(dialog)
    field = _field(dialog, SettingName.POINTS_PER_STREAMLINE)
    assert isinstance(field, SliderField)
    field.text.setText("900")
    field.text.editingFinished.emit()
    error = _error(dialog, field)
    assert error.isVisibleTo(dialog)
    assert "between 2 and 200" in error.text()
    assert settings.values == AppSettings()
    assert not path.exists()


@pytest.fixture
def mcp(tmp_path: Path) -> Iterator[McpServer]:
    server = McpServer(Viewer(), lambda function: None, token_file=tmp_path / "token")
    yield server
    server.stop()


def _mcp_section(port: int, server: McpServer, qtbot: QtBot) -> McpSection:
    settings = Settings(AppSettings(mcp_port=port))
    switch = McpSwitch(server, lambda: server.start(settings.values.mcp_port))
    dialog = SettingsPanel(settings, None, switch)
    qtbot.addWidget(dialog)
    assert dialog.mcp is not None
    return dialog.mcp


def test_the_serve_agents_switch_starts_and_stops_the_server_at_the_port(
    mcp: McpServer, qtbot: QtBot
) -> None:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    section = _mcp_section(port, mcp, qtbot)
    section.serve.setChecked(True)
    assert mcp.running
    assert mcp.port == port
    assert section.url.text() == f"http://127.0.0.1:{port}/mcp"
    assert section.token.text() == mcp.token
    section.serve.setChecked(False)
    assert not mcp.running


def test_the_serve_agents_switch_shows_a_taken_port_and_stays_off(
    mcp: McpServer, qtbot: QtBot
) -> None:
    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen()
        port = taken.getsockname()[1]
        section = _mcp_section(port, mcp, qtbot)
        section.serve.setChecked(True)
    assert not mcp.running
    assert not section.serve.isChecked()
    assert not section.error.isHidden()
    assert f"port {port}" in section.error.text()


def _serves(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=5):
            return True
    except OSError:
        return False


def test_a_running_server_moves_to_a_free_port_and_stays_if_the_new_one_is_taken(
    qtbot: QtBot, tmp_path: Path
) -> None:
    settings = Settings(AppSettings(mcp_port=free_port()))
    window = MainWindow(settings=settings)
    qtbot.addWidget(window)
    window.mcp.token_file = tmp_path / "token"
    window.start_mcp()
    try:
        section = window.show_settings().panel.mcp
        assert section is not None
        free = free_port()
        settings.set(SettingName.MCP_PORT, free)
        assert window.mcp.port == free
        assert _serves(free)
        with socket.socket() as taken:
            taken.bind(("127.0.0.1", 0))
            taken.listen()
            port = taken.getsockname()[1]
            settings.set(SettingName.MCP_PORT, port)
        assert window.mcp.running
        assert window.mcp.port == free
        assert _serves(free)
        assert f"port {port}" in window.statusBar().currentMessage()
        assert section.serve.isChecked()
        assert not section.error.isHidden()
        assert f"port {port}" in section.error.text()
    finally:
        window.mcp.stop()


# The application: saved settings, theme and GPU


@dataclass
class _Adapter:
    summary: str


@pytest.mark.parametrize(
    ("name", "chosen"),
    [("GPU B (Vulkan)", 1), ("GPU C (Vulkan)", None)],
    ids=["present", "missing"],
)
def test_the_gpu_is_selected_by_name_or_left_to_wgpu_if_missing(
    name: str,
    chosen: int | None,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    adapters = [_Adapter(n) for n in GPUS]
    selected: list[object] = []
    monkeypatch.setattr(wgpu.gpu, "enumerate_adapters_sync", lambda: adapters)
    monkeypatch.setattr(gfx.renderers.wgpu, "select_adapter", selected.append)
    with caplog.at_level(logging.WARNING, logger="larmor"):
        found = gpu.select_gpu(name)
    assert found == (chosen is not None)
    assert selected == ([] if chosen is None else [adapters[chosen]])
    assert ("not available" in caplog.text) == (chosen is None)


def test_identical_gpus_are_numbered_and_each_can_be_selected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapters = [_Adapter("A"), _Adapter("A"), _Adapter("B")]
    selected: list[object] = []
    monkeypatch.setattr(wgpu.gpu, "enumerate_adapters_sync", lambda: adapters)
    monkeypatch.setattr(gfx.renderers.wgpu, "select_adapter", selected.append)
    assert gpu.gpu_names() == ["A (GPU 1)", "A (GPU 2)", "B"]
    assert gpu.select_gpu("A (GPU 2)")
    # Equal dataclasses compare equal, so check identity.
    assert len(selected) == 1
    assert selected[0] is adapters[1]


class _Hints:
    def __init__(self) -> None:
        self.schemes: list[Qt.ColorScheme | None] = []

    def setColorScheme(self, scheme: Qt.ColorScheme) -> None:  # noqa: N802
        self.schemes.append(scheme)

    def unsetColorScheme(self) -> None:  # noqa: N802
        self.schemes.append(None)  # None: the system's


@pytest.fixture
def started(
    path: Path, monkeypatch: pytest.MonkeyPatch
) -> Callable[[], tuple[Settings, _Hints, list[str]]]:
    hints = _Hints()
    gpus: list[str] = []

    class Application:
        @staticmethod
        def styleHints() -> _Hints:  # noqa: N802
            return hints

    monkeypatch.setattr(settings_module, "settings_path", lambda: path)
    monkeypatch.setattr(viewer_app, "_SETTINGS", None)
    monkeypatch.setattr(viewer_app, "QGuiApplication", Application)
    monkeypatch.setattr(viewer_app, "select_gpu", gpus.append)

    def start() -> tuple[Settings, _Hints, list[str]]:
        return viewer_app.settings(), hints, gpus

    return start


def test_the_saved_settings_theme_and_gpu_are_restored_at_start(
    path: Path, started: Callable[[], tuple[Settings, _Hints, list[str]]]
) -> None:
    saved = Settings(path=path)
    saved.set(SettingName.THEME, Theme.DARK)
    saved.set(SettingName.GPU, GPUS[1])
    saved.set(SettingName.READ_LIMIT, 0)
    settings, hints, gpus = started()
    assert settings.values == saved.values
    assert hints.schemes == [Qt.ColorScheme.Dark]
    assert gpus == [GPUS[1]]
    assert viewer_app.settings() is settings


@pytest.mark.parametrize(
    ("theme", "scheme"),
    [
        (Theme.LIGHT, Qt.ColorScheme.Light),
        (Theme.DARK, Qt.ColorScheme.Dark),
        (Theme.SYSTEM, None),
    ],
)
def test_the_theme_follows_the_setting_or_the_system_at_once(
    theme: Theme,
    scheme: Qt.ColorScheme | None,
    started: Callable[[], tuple[Settings, _Hints, list[str]]],
) -> None:
    settings, hints, _ = started()
    settings.set(SettingName.THEME, Theme.LIGHT if theme != Theme.LIGHT else Theme.DARK)
    hints.schemes.clear()
    settings.set(SettingName.THEME, theme)
    assert hints.schemes == [scheme]


def test_every_settings_row_has_its_help_on_its_label_and_field(
    mcp: McpServer, qtbot: QtBot
) -> None:
    dialog = SettingsPanel(Settings(), None, McpSwitch(mcp, lambda: None))
    qtbot.addWidget(dialog)
    form = dialog.findChild(QFormLayout)
    assert form is not None
    helps = {info.label: info.help for _, info in setting_fields().values()}
    helps |= {
        "Serve agents": settings_panel.SERVE_HELP,
        "URL": settings_panel.URL_HELP,
        "Token": settings_panel.TOKEN_HELP,
    }
    rows = _rows(dialog)
    assert set(rows) == set(helps)
    for text, field in rows.items():
        label = form.labelForField(field)
        assert isinstance(label, QLabel)
        assert helps[text]
        assert (label.toolTip(), field.toolTip()) == (helps[text], helps[text])


def test_the_mcp_url_shows_whole_in_the_sheet(mcp: McpServer, qtbot: QtBot) -> None:
    sheet = SettingsSheet(Settings(), None, McpSwitch(mcp, lambda: None))
    qtbot.addWidget(sheet)
    sheet.show()
    qtbot.waitExposed(sheet)
    assert sheet.panel.mcp is not None
    url = sheet.panel.mcp.url
    assert url.fontMetrics().horizontalAdvance(url.text()) < url.contentsRect().width()
