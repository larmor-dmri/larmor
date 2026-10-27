"""The Simulate workspace: the steps, the page of one step, and its links."""

from pathlib import Path

from pytestqt.qtbot import QtBot

from larmor.viewer import Category, Viewer
from larmor.viewer.model.phantom import PhantomStep
from larmor.viewer.qt.docks import Panel
from larmor.viewer.qt.help_panel import GUIDES
from larmor.viewer.qt.main_window import MainWindow
from larmor.viewer.qt.phantom_steps import (
    StepNavigator,
    StepPage,
    StepsPanel,
    default_scan_text,
)


def _shown(page: StepPage) -> list[str]:
    return [n for n, s in page.panel.sections.items() if s.isVisibleTo(page)]


def _window(qtbot: QtBot) -> MainWindow:
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)
    return window


def test_a_new_user_reaches_a_run_by_following_next(
    qtbot: QtBot, tmp_path: Path
) -> None:
    window = _window(qtbot)
    window.steps_panel.new_button.click()
    page, rows, viewer = window.step_page, window.steps_panel.rows, window.viewer
    panel = page.panel
    assert window.steps.page == PhantomStep.GRID and _shown(page) == ["Grid"]
    for box in panel.size_boxes:
        box.setValue(16.0)  # a small grid, for a quick run
    panel.apply_box.click()
    assert rows[PhantomStep.GRID].status.text().startswith("8 x 8 x 8")

    page.next_button.click()
    assert _shown(page) == ["Bundles"]
    assert rows[PhantomStep.RUN].status.text() == "Needs a bundle"
    panel.draw_button.click()
    for point in [(-6.0, 0.0, 0.0), (0.0, 2.0, 0.0), (6.0, 0.0, 0.0)]:
        viewer.add_bundle_point(point)
    panel.finish_button.click()
    assert rows[PhantomStep.BUNDLE].status.text() == "1 bundle"

    page.next_button.click()
    assert _shown(page) == ["Tissues"]
    panel.parameter_boxes["proton_density"].setValue(0.8)  # the tissue of WM

    page.next_button.click()
    assert _shown(page) == ["Scan"]
    panel.noise.click()

    page.next_button.click()
    assert _shown(page) == ["Run"] and not page.next_button.isVisible()
    assert rows[PhantomStep.RUN].status.text() == "Ready"
    viewer.set_run_folder(tmp_path)  # in place of the folder dialog
    window.header.run_button.click()
    assert not window.header.run_button.isEnabled()  # while it runs
    assert panel.stop_button.isVisible()
    qtbot.waitUntil(
        lambda: bool(viewer.scene.in_category(Category.VOLUMES)), timeout=60_000
    )
    qtbot.waitUntil(lambda: not panel.stop_button.isVisible())
    state = viewer.phantom
    assert state is not None and state.scan.snr is not None
    assert state.tissue("WM").value("proton_density") == 0.8


def test_any_step_opens_from_the_list_and_back_returns_from_an_extra_page(
    qtbot: QtBot,
) -> None:
    viewer = Viewer()
    viewer.new_phantom()
    steps = StepNavigator(viewer)
    rail, page = StepsPanel(viewer, steps), StepPage(viewer, steps)
    for widget in (rail, page):
        qtbot.addWidget(widget)
    rail.rows[PhantomStep.SCAN].clicked.emit()
    assert steps.page == PhantomStep.SCAN and _shown(page) == ["Scan"]
    rail.links["maps"].click()
    assert _shown(page) == ["Grid"]
    assert page.panel.background.isVisibleTo(page)  # under More, unfolded
    page.back_button.click()
    assert steps.page == PhantomStep.SCAN


def test_a_new_phantom_starts_again_at_the_first_step(qtbot: QtBot) -> None:
    viewer = Viewer()
    viewer.new_phantom()
    steps = StepNavigator(viewer)
    steps.go_to(PhantomStep.SCAN)
    viewer.new_phantom()
    assert steps.page == PhantomStep.GRID


def test_the_links_open_the_signal_breakdown_and_all_settings(qtbot: QtBot) -> None:
    window = _window(qtbot)
    window.new_phantom()
    window.steps_panel.links["breakdown"].click()
    assert window.docks.is_open(Panel.SIGNAL_BREAKDOWN)
    window.steps_panel.links["all"].click()
    panel = window.show_all_settings()
    assert panel.window().isVisible() and panel.window() is not window


def test_without_a_phantom_the_steps_say_how_to_start(qtbot: QtBot) -> None:
    viewer = Viewer()
    steps = StepNavigator(viewer)
    rail = StepsPanel(viewer, steps)
    qtbot.addWidget(rail)
    assert rail.hint.isVisibleTo(rail)
    assert not rail.rows[PhantomStep.GRID].isVisibleTo(rail)
    assert not rail.save_button.isEnabled()
    viewer.new_phantom()
    assert not rail.hint.isVisibleTo(rail)
    assert rail.rows[PhantomStep.GRID].isVisibleTo(rail)


def test_the_guide_quotes_the_default_scan_as_the_scan_page_does() -> None:
    guide = GUIDES / "15-phantoms.md"
    text = " ".join(guide.read_text(encoding="utf-8").split())
    sentence = default_scan_text().removeprefix("The ").removesuffix(".")
    assert sentence in text
