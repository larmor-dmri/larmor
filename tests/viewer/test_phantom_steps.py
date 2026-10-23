"""The phantom editor's guided steps and All settings, and the layout remembered."""

from pathlib import Path

from pytestqt.qtbot import QtBot

from larmor.viewer import Category, Viewer
from larmor.viewer.model.phantom import PhantomStep
from larmor.viewer.model.settings import PhantomLayout, SettingName, Settings
from larmor.viewer.qt.docks import Panel
from larmor.viewer.qt.help_panel import GUIDES
from larmor.viewer.qt.main_window import MainWindow, _main_window
from larmor.viewer.qt.phantom_steps import Extra, PhantomEditor, default_scan_text


def _shown(editor: PhantomEditor) -> list[str]:
    return [n for n, s in editor.panel.sections.items() if s.isVisibleTo(editor)]


def _window(qtbot: QtBot, settings: Settings) -> MainWindow:
    window = MainWindow(settings=settings)
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)
    return window


def test_a_new_user_reaches_a_run_by_following_next(
    qtbot: QtBot, tmp_path: Path
) -> None:
    window = _window(qtbot, Settings())
    window.new_phantom()
    editor = window.docks.widget(Panel.PHANTOM)
    assert isinstance(editor, PhantomEditor)
    panel, viewer = editor.panel, window.viewer
    assert editor.page == PhantomStep.GRID and _shown(editor) == ["Grid"]
    for box in panel.size_boxes:
        box.setValue(16.0)  # a small grid, for a quick run
    panel.apply_box.click()
    assert editor.rows[PhantomStep.GRID].status.text().startswith("8 x 8 x 8")

    editor.next_button.click()
    assert _shown(editor) == ["Bundles"]
    assert editor.rows[PhantomStep.RUN].status.text() == "Needs a bundle"
    panel.draw_button.click()
    for point in [(-6.0, 0.0, 0.0), (0.0, 2.0, 0.0), (6.0, 0.0, 0.0)]:
        viewer.add_bundle_point(point)
    panel.finish_button.click()
    assert editor.rows[PhantomStep.BUNDLE].status.text() == "1 bundle"

    editor.next_button.click()
    assert _shown(editor) == ["Tissues"]
    panel.parameter_boxes["proton_density"].setValue(0.8)  # the tissue of WM

    editor.next_button.click()
    assert _shown(editor) == ["Scan"]
    panel.noise.click()

    editor.next_button.click()
    assert _shown(editor) == ["Run"] and not editor.next_button.isVisible()
    assert editor.rows[PhantomStep.RUN].status.text() == "Ready"
    viewer.set_run_folder(tmp_path)  # in place of the folder dialog
    panel.run_button.click()
    qtbot.waitUntil(
        lambda: bool(viewer.scene.in_category(Category.VOLUMES)), timeout=60_000
    )
    state = viewer.phantom
    assert state is not None and state.scan.snr is not None
    assert state.tissue("WM").value("proton_density") == 0.8


def test_any_step_opens_from_the_list_and_back_returns_from_an_extra_page(
    qtbot: QtBot,
) -> None:
    viewer = Viewer()
    viewer.new_phantom()
    editor = PhantomEditor(viewer, Settings())
    qtbot.addWidget(editor)
    editor.rows[PhantomStep.SCAN].clicked.emit()
    assert editor.page == PhantomStep.SCAN and _shown(editor) == ["Scan"]
    editor.extras[Extra.PROXY_MAPS].click()
    assert _shown(editor) == ["Proxy maps"]
    editor.back_button.click()
    assert editor.page == PhantomStep.SCAN


def test_the_layout_chosen_is_remembered_and_all_settings_shows_every_section(
    qtbot: QtBot, tmp_path: Path
) -> None:
    path = tmp_path / "viewer.toml"
    window = _window(qtbot, Settings.load(path))
    window.new_phantom()
    editor = window.docks.widget(Panel.PHANTOM)
    assert isinstance(editor, PhantomEditor)
    assert editor.guided and editor.rail.isVisibleTo(editor)
    editor.all_button.click()
    assert Settings.load(path).values.phantom_layout == PhantomLayout.ALL
    assert len(_shown(editor)) == len(editor.panel.sections)
    assert not editor.rail.isVisibleTo(editor)
    assert editor.chips_row.isVisibleTo(editor)
    window._show_view_menu()  # as the View menu opens
    assert not window.guided_action.isChecked()
    window.guided_action.trigger()
    assert Settings.load(path).values.phantom_layout == PhantomLayout.GUIDED
    assert editor.guided_button.isChecked() and editor.rail.isVisibleTo(editor)


def test_a_step_chip_scrolls_all_settings_to_its_section(qtbot: QtBot) -> None:
    viewer = Viewer()
    viewer.new_phantom()
    settings = Settings()
    settings.set(SettingName.PHANTOM_LAYOUT, PhantomLayout.ALL)
    editor = PhantomEditor(viewer, settings)
    qtbot.addWidget(editor)
    editor.resize(420, 400)  # one column, taller than the window
    editor.show()
    qtbot.waitExposed(editor)
    editor.chips[PhantomStep.SCAN].click()
    scan = editor.panel.sections["Scan"]
    top = scan.mapTo(editor.page_area.viewport(), scan.rect().topLeft()).y()
    assert editor.page_area.verticalScrollBar().value() > 0
    assert abs(top) <= 1


def test_the_guide_quotes_the_default_scan_as_the_scan_page_does() -> None:
    guide = GUIDES / "15-phantoms.md"
    text = " ".join(guide.read_text(encoding="utf-8").split())
    sentence = default_scan_text().removeprefix("The ").removesuffix(".")
    assert sentence in text


def test_the_floating_editor_finds_the_main_window(qtbot: QtBot) -> None:
    window = _window(qtbot, Settings())
    window.new_phantom()
    assert window.docks.docks[Panel.PHANTOM].isInFloatingContainer()
    editor = window.docks.widget(Panel.PHANTOM)
    assert editor.window() is not window
    assert _main_window(editor) is window
