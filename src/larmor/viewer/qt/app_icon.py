"""The viewer's icon as image files, drawn from its SVG without a window."""

from importlib.resources import files

from PySide6.QtCore import QBuffer, QByteArray, QIODevice
from PySide6.QtGui import QImage, QImageWriter, QPainter
from PySide6.QtSvg import QSvgRenderer

ICON_SVG = files("larmor.viewer") / "resources" / "icon.svg"


def render_icon(size: int, image_format: str = "PNG") -> bytes:
    """Draws the icon, size pixels square, in one of Qt's image formats.

    Returns:
        bytes: The image file's contents.

    Raises:
        OSError: If Qt cannot write the format.
    """
    image = QImage(size, size, QImage.Format.Format_ARGB32)
    image.fill(0)
    painter = QPainter(image)
    QSvgRenderer(str(ICON_SVG)).render(painter)
    painter.end()
    data = QByteArray()
    buffer = QBuffer(data)
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    if not QImageWriter(buffer, QByteArray(image_format.encode())).write(image):
        raise OSError(f"Qt cannot write the icon as {image_format}.")
    return bytes(data.data())
