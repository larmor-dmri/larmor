"""The Help panel, which shows the guides shipped with the package."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, override

from PySide6.QtCore import QByteArray, QSignalBlocker, Qt, QUrl
from PySide6.QtGui import QImage, QTextDocument
from PySide6.QtWidgets import (
    QListWidget,
    QSplitter,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from larmor.viewer.qt.style import RESOURCES

GUIDES = RESOURCES / "guides"
# A guide is a Markdown file whose name starts with its place in the list.
GUIDE_NAME = re.compile(r"\d{2}-.+\.md")  # two digits, so the names sort in order
MAX_IMAGE_WIDTH = 260  # wider images, such as the whole window, fit the panel


@dataclass(frozen=True)
class Guide:
    """Holds a guide's title and its Markdown file."""

    title: str
    path: Path


def guides(folder: Path | None = None) -> list[Guide]:
    """Returns the guides in folder (the package's by default), in file name order."""
    folder = folder if folder is not None else Path(str(GUIDES))
    paths = sorted(p for p in folder.iterdir() if GUIDE_NAME.fullmatch(p.name))
    return [Guide(_title(p), p) for p in paths]


def _title(path: Path) -> str:
    """Returns the guide's first `# ` heading, else its file name without the number."""
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return path.stem.split("-", 1)[1].replace("-", " ").capitalize()


class GuideBrowser(QTextBrowser):
    """Shows a guide, with its images no wider than MAX_IMAGE_WIDTH."""

    @override
    def loadResource(self, type: int, name: QUrl | str) -> Any:  # noqa: N802
        """Loads a resource, shrinking a wide image to fit the panel."""
        resource = super().loadResource(type, name)
        if type != QTextDocument.ResourceType.ImageResource.value:
            return resource
        image = QImage.fromData(QByteArray(resource)) if resource else QImage()
        if image.isNull() or image.width() <= MAX_IMAGE_WIDTH:
            return resource
        # Drawn at the screen's resolution, so it stays sharp on a HiDPI screen.
        ratio = self.devicePixelRatioF()
        image = image.scaledToWidth(
            round(MAX_IMAGE_WIDTH * ratio), Qt.TransformationMode.SmoothTransformation
        )
        image.setDevicePixelRatio(ratio)
        return image


class HelpPanel(QWidget):
    """Lists the guides at the top and shows the selected one below."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Lists the guides and shows the first one."""
        super().__init__(parent)
        self.guides = guides()

        self.list = QListWidget(self)
        self.list.setObjectName("guides")
        self.list.addItems([g.title for g in self.guides])
        self.browser = GuideBrowser(self)
        self.browser.setObjectName("guide")
        # Links to other guides open here; web links in the system's browser.
        self.browser.setOpenExternalLinks(True)
        self.browser.setSearchPaths([str(GUIDES)])
        self.list.currentRowChanged.connect(self._show)
        self.browser.sourceChanged.connect(self._follow)

        splitter = QSplitter(Qt.Orientation.Vertical, self)
        splitter.addWidget(self.list)
        splitter.addWidget(self.browser)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([160, 600])
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)
        if self.guides:
            self.list.setCurrentRow(0)

    def _show(self, row: int) -> None:
        """Shows the guide of a row of the list."""
        if 0 <= row < len(self.guides):
            url = QUrl.fromLocalFile(str(self.guides[row].path))
            self.browser.setSource(url, QTextDocument.ResourceType.MarkdownResource)

    def _follow(self, url: QUrl) -> None:
        """Selects the guide a link opened, without loading it again."""
        path = Path(url.toLocalFile()).resolve()
        for row, guide in enumerate(self.guides):
            if guide.path.resolve() == path:
                with QSignalBlocker(self.list):
                    self.list.setCurrentRow(row)
