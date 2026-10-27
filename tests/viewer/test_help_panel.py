import re
from pathlib import Path

import pytest
from PySide6.QtCore import QObject, Qt, QUrl, Slot
from PySide6.QtGui import QColor, QDesktopServices, QImage, QTextCursor, QTextDocument
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QMenu
from pytestqt.qtbot import QtBot

from larmor.viewer.qt import help_panel
from larmor.viewer.qt.help_panel import (
    MAX_IMAGE_WIDTH,
    HelpPanel,
    HelpWindow,
    guides,
)
from larmor.viewer.qt.main_window import MainWindow

SHIPPED = Path(str(help_panel.GUIDES))
SHIPPED_TITLES = [
    "Getting started",
    "The 2D views",
    "Volumes and overlays",
    "Parcellations",
    "Tractograms",
    "The 3D view",
    "FOD glyphs",
    "Peaks",
    "Settings",
    "Exporting images and videos",
    "Sessions",
    "Driving the viewer from an agent",
    "Surfaces",
    "MATLAB files",
    "Phantoms",
]


@pytest.fixture
def folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    # The window reads the package's guides; point it at a scratch folder.
    (tmp_path / "images").mkdir()
    (tmp_path / "01-first.md").write_text(
        "# First\n\nSee [the second](02-second.md) or [the web](https://example.org).\n",
        encoding="utf-8",
    )
    (tmp_path / "02-second.md").write_text(
        "# Second\n\nSecond guide text.\n", encoding="utf-8"
    )
    monkeypatch.setattr(help_panel, "GUIDES", tmp_path)
    return tmp_path


@pytest.fixture
def window(folder: Path, qtbot: QtBot) -> HelpPanel:
    window = HelpPanel()
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)
    return window


def _click_link(window: HelpPanel, href: str) -> None:
    document = window.browser.document()
    block = document.begin()
    while block.isValid():
        it = block.begin()
        while not it.atEnd():
            fragment = it.fragment()
            if fragment.charFormat().anchorHref() == href:
                cursor = QTextCursor(document)
                cursor.setPosition(fragment.position() + fragment.length() // 2)
                point = window.browser.cursorRect(cursor).center()
                QTest.mouseClick(
                    window.browser.viewport(), Qt.MouseButton.LeftButton, pos=point
                )
                return
            it += 1
        block = block.next()
    raise AssertionError(f"no link to {href}")


def test_shipped_guides_are_listed_in_order_with_their_titles() -> None:
    found = guides()
    assert [g.title for g in found] == SHIPPED_TITLES
    assert [g.path.name[:2] for g in found] == [f"{i:02}" for i in range(1, 16)]


def test_a_guide_without_a_heading_is_titled_from_its_file_name(
    tmp_path: Path,
) -> None:
    (tmp_path / "03-loading-many-files.md").write_text("No heading.\n")
    assert [g.title for g in guides(tmp_path)] == ["Loading many files"]


def test_a_new_guide_in_the_folder_is_listed_in_its_place(tmp_path: Path) -> None:
    for name in ("01-a.md", "03-c.md"):
        (tmp_path / name).write_text(f"# {name}\n")
    (tmp_path / "notes.md").write_text("# Not a guide\n")
    (tmp_path / "4-d.md").write_text("# One digit: would sort after 03\n")
    (tmp_path / "02-b.md").write_text("# 02-b.md\n")
    assert [g.title for g in guides(tmp_path)] == ["01-a.md", "02-b.md", "03-c.md"]


@pytest.mark.parametrize("path", sorted(SHIPPED.glob("*.md")), ids=lambda p: p.name)
def test_every_image_and_guide_a_shipped_guide_links_to_exists(path: Path) -> None:
    targets = re.findall(r"\]\(([^)\s]+)\)", path.read_text(encoding="utf-8"))
    local = [t for t in targets if not re.match(r"[a-z]+:", t)]
    assert all(t.startswith("images/") or t.endswith(".md") for t in local)
    missing = [t for t in local if not (SHIPPED / t.split("#")[0]).is_file()]
    assert missing == []


def test_every_shipped_guide_renders_with_its_images(qtbot: QtBot) -> None:
    panel = HelpPanel()
    qtbot.addWidget(panel)
    for row, guide in enumerate(panel.guides):
        panel.list.setCurrentRow(row)
        document = panel.browser.document()
        assert guide.title in panel.browser.toPlainText()
        assert "](" not in panel.browser.toPlainText()  # Markdown, not raw text
        images = []
        block = document.begin()
        while block.isValid():
            it = block.begin()
            while not it.atEnd():
                char_format = it.fragment().charFormat()
                if char_format.isImageFormat():
                    images.append(char_format.toImageFormat().name())
                it += 1
            block = block.next()
        written = guide.path.read_text(encoding="utf-8").count("![")
        assert len(images) == written
        for name in images:
            loaded = panel.browser.loadResource(
                QTextDocument.ResourceType.ImageResource.value, QUrl(name)
            )
            image = loaded if isinstance(loaded, QImage) else QImage.fromData(loaded)
            assert not image.isNull(), f"{guide.path.name}: {name}"


def test_selecting_a_guide_shows_its_text(window: HelpPanel) -> None:
    assert "See the second" in window.browser.toPlainText()
    window.list.setCurrentRow(1)
    assert "Second guide text." in window.browser.toPlainText()


def test_a_link_to_another_guide_opens_it_and_selects_it(window: HelpPanel) -> None:
    _click_link(window, "02-second.md")
    assert "Second guide text." in window.browser.toPlainText()
    assert "# Second" not in window.browser.toPlainText()  # rendered, not raw text
    assert window.list.currentRow() == 1


class _UrlCatcher(QObject):
    def __init__(self) -> None:
        super().__init__()
        self.urls: list[QUrl] = []

    @Slot(QUrl)
    def open_url(self, url: QUrl) -> None:
        self.urls.append(url)


def test_a_web_link_opens_in_the_system_browser_not_the_window(
    window: HelpPanel,
) -> None:
    # The browser calls QDesktopServices from C++, so catch it with a URL handler.
    catcher = _UrlCatcher()
    # PySide takes the slot's name as a str, though its stubs say bytes.
    QDesktopServices.setUrlHandler("https", catcher, "open_url")  # type: ignore[arg-type]
    try:
        _click_link(window, "https://example.org")
    finally:
        QDesktopServices.unsetUrlHandler("https")
    assert catcher.urls == [QUrl("https://example.org")]
    assert "See the second" in window.browser.toPlainText()
    assert window.list.currentRow() == 0


def test_a_wide_image_is_shrunk_to_the_maximum_width(
    folder: Path, window: HelpPanel
) -> None:
    image = QImage(3 * MAX_IMAGE_WIDTH, 300, QImage.Format.Format_RGB32)
    image.fill(QColor("red"))
    assert image.save(str(folder / "images" / "wide.png"))
    loaded = window.browser.loadResource(
        QTextDocument.ResourceType.ImageResource.value, QUrl("images/wide.png")
    )
    assert isinstance(loaded, QImage)
    assert loaded.width() / loaded.devicePixelRatio() == pytest.approx(MAX_IMAGE_WIDTH)


def test_the_help_action_opens_one_help_window(qtbot: QtBot) -> None:
    main = MainWindow()
    qtbot.addWidget(main)
    main.show()
    qtbot.waitExposed(main)
    help_menu = next(a.menu() for a in main.menuBar().actions() if a.text() == "&Help")
    assert isinstance(help_menu, QMenu)
    action = next(a for a in help_menu.actions() if a.text() == "&Help")
    action.trigger()
    [window] = main.findChildren(HelpWindow)
    assert window.isVisible() and window.isWindow()
    window.close()
    action.trigger()
    assert main.findChildren(HelpWindow) == [window] and window.isVisible()
