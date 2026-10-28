"""The settings forms, and the Settings sheet that holds the application's."""

from collections.abc import Callable, Collection

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from larmor.viewer.mcp_server import McpServer, PortInUseError
from larmor.viewer.model.options import OptionError
from larmor.viewer.model.settings import (
    SettingChanged,
    SettingName,
    Settings,
    setting_fields,
)
from larmor.viewer.qt.field import (
    ChoiceField,
    OptionField,
    SliderField,
    add_field_row,
    field_for,
)
from larmor.viewer.qt.style import form_layout
from larmor.viewer.render.gpu import gpu_names

# A heading above the first setting of each group; the group runs to the next.
SECTIONS: dict[SettingName, str] = {
    SettingName.THEME: "Appearance",
    SettingName.CONVENTION: "Views",
    SettingName.INTERPOLATE: "Images",
    SettingName.READ_LIMIT: "Tractograms",
    SettingName.LOWER_RESOLUTION: "Graphics",
    SettingName.MCP_PORT: "Agents (MCP)",
}
# The settings the inspector's Scene tab shows instead of the Settings sheet.
SCENE_SETTINGS = frozenset(
    {
        SettingName.CONVENTION,
        SettingName.BACKGROUND,
        SettingName.CROSSHAIR,
        SettingName.INTERPOLATE,
    }
)
# The settings the Settings sheet shows.
SHEET_SETTINGS = tuple(n for n in SettingName if n not in SCENE_SETTINGS)
SHEET_SIZE = (520, 640)  # px
DEFAULT_GPU = "Default"  # the GPU choice that leaves it to wgpu, saved as ""
# The explanations of the MCP rows, which are not settings.
SERVE_HELP = "Lets agents on this computer drive the viewer through MCP."
URL_HELP = "The address an agent connects to."
TOKEN_HELP = "The key an agent must send; a new token refuses agents with the old one."
# The longest URL, for the least width of the URL and token fields.
URL_SAMPLE = "http://127.0.0.1:65535/mcp"


class SettingsPanel(QWidget):
    """Edits settings; each change applies at once and is saved."""

    def __init__(
        self,
        settings: Settings,
        parent: QWidget | None = None,
        mcp: McpSwitch | None = None,
        names: Collection[SettingName] | None = None,
    ) -> None:
        """Builds a field for each of the settings named, every one if None.

        The MCP server's switch is added below its port if given.
        """
        super().__init__(parent)
        self.settings = settings
        self.names = frozenset(SettingName if names is None else names)
        self._fields: dict[SettingName, OptionField] = {}
        self._errors: dict[SettingName, QLabel] = {}
        self.mcp: McpSection | None = None
        form = self._build_form(mcp)

        scroll = QScrollArea(self)
        scroll.setObjectName("settings")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(form)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        for name in self._fields:
            self._show(name)
        # The settings outlive the panel: stop listening when the panel goes.
        unsubscribe = settings.subscribe(self._on_change)
        self.destroyed.connect(lambda *_: unsubscribe())

    def _build_form(self, mcp: McpSwitch | None) -> QWidget:
        """Builds the form of the settings, with the MCP server's rows if given.

        Returns:
            QWidget: The form.
        """
        form = QWidget()
        form.setObjectName("settingsForm")
        layout = form_layout(form)
        gpus = gpu_names()
        current, shown = "", ""  # the group of the setting, and the last heading
        for name, (f, info) in setting_fields().items():
            current = SECTIONS.get(name, current)
            if name not in self.names:
                continue
            if name == SettingName.GPU and len(gpus) < 2:
                continue  # nothing to choose on a machine with one GPU
            if current != shown:
                shown = current
                section = QLabel(current)
                section.setObjectName("formSection")
                layout.addRow(section)
            field: OptionField
            if name == SettingName.GPU:
                field = ChoiceField((DEFAULT_GPU, *gpus), form)
            else:
                field = field_for(f, info, form)
            error = QLabel(form)
            error.setObjectName("fieldError")
            error.setWordWrap(True)
            error.hide()
            field.edited.connect(lambda value, name=name: self._edit(name, value))
            add_field_row(layout, info.label, info.help, field, form)
            layout.addRow(error)  # a hidden row takes no room
            self._fields[name] = field
            self._errors[name] = error
            if name == SettingName.GPU:
                layout.addRow("", QLabel("Applies when the viewer starts again."))
            if name == SettingName.MCP_PORT:
                assert isinstance(field, SliderField)
                field.slider.hide()  # a port is typed; a drag would try every one
                row = field.layout()
                assert isinstance(row, QHBoxLayout)
                row.addStretch(1)  # the text where the other fields start
                if mcp is not None:
                    self.mcp = McpSection(mcp, self.settings, layout, form)
        return form

    def _on_change(self, event: SettingChanged) -> None:
        """Shows a setting's new value, and the MCP server after a new port."""
        if event.name in self._fields:
            self._show(event.name)
        if event.name == SettingName.MCP_PORT and self.mcp is not None:
            self.mcp.moved()

    def _show(self, name: SettingName) -> None:
        """Shows a setting's value in its field and hides its error."""
        value = getattr(self.settings.values, name)
        if name == SettingName.GPU and not value:
            value = DEFAULT_GPU
        self._fields[name].show_value(value)
        self._errors[name].hide()  # the message was about text now replaced

    def _edit(self, name: SettingName, value: object) -> None:
        """Sets a setting from its field; shows the error if it is refused."""
        if name == SettingName.GPU and value == DEFAULT_GPU:
            value = ""
        try:
            self.settings.set(name, value)
        except OptionError as error:
            self._errors[name].setText(str(error))
            self._errors[name].show()
            return
        # Setting the current value sends no change, so tidy the field here.
        self._show(name)


class SettingsSheet(QDialog):
    """Holds the application's settings in a window beside the main one.

    The settings of how the views show the scene are in the inspector's Scene
    tab instead.
    """

    def __init__(
        self,
        settings: Settings,
        parent: QWidget | None = None,
        mcp: McpSwitch | None = None,
    ) -> None:
        """Builds the form of the sheet's settings, with the MCP switch if given."""
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setModal(False)
        self.panel = SettingsPanel(settings, self, mcp, SHEET_SETTINGS)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, self)
        buttons.rejected.connect(self.close)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 8)
        layout.addWidget(self.panel, 1)
        layout.addWidget(buttons)
        self.resize(*SHEET_SIZE)


class McpSwitch:
    """Holds a window's MCP server and the function that starts it."""

    def __init__(self, server: McpServer, start: Callable[[], None]) -> None:
        """Holds server, and start, which starts it at the settings' port.

        start raises PortInUseError if the port is taken.
        """
        self.server = server
        self.start = start


class McpSection:
    """Shows the MCP server's switch, URL and token as rows of the Settings form."""

    def __init__(
        self, switch: McpSwitch, settings: Settings, layout: QFormLayout, form: QWidget
    ) -> None:
        """Adds the rows to layout, below the port."""
        self.switch = switch
        self.settings = settings
        self.serve = QCheckBox(form)
        self.url = QLineEdit(form)
        self.token = QLineEdit(form)
        for line in (self.url, self.token):
            line.setReadOnly(True)
        self.new_token = QPushButton("New token", form)
        self.new_token.clicked.connect(self._renew)
        self.error = QLabel(form)
        self.error.setObjectName("fieldError")
        self.error.setWordWrap(True)
        self.error.hide()
        add_field_row(layout, "Serve agents", SERVE_HELP, self.serve, form)
        url = _with_buttons(form, self.url, _copy_button(form, self.url))
        copy = _copy_button(form, self.token)
        token = _with_buttons(form, self.token, copy, self.new_token)
        # As wide as the URL, so the form puts them below their labels rather than
        # cut the URL short.
        for line in (self.url, self.token):
            line.setMinimumWidth(line.fontMetrics().horizontalAdvance(URL_SAMPLE) + 16)
        add_field_row(layout, "URL", URL_HELP, url, form)
        add_field_row(layout, "Token", TOKEN_HELP, token, form)
        layout.addRow(self.error)
        self.serve.toggled.connect(self._toggle)
        self.refresh()

    def refresh(self) -> None:
        """Shows whether the server runs, its URL and the token."""
        server = self.switch.server
        self.serve.blockSignals(True)
        self.serve.setChecked(server.running)
        self.serve.blockSignals(False)
        port = server.port if server.running else self.settings.values.mcp_port
        self.url.setText(server.url(port))
        self.token.setText(server.token)
        for line in (self.url, self.token):
            line.setCursorPosition(0)  # the start shows if it is cut short

    def moved(self) -> None:
        """Shows the server after a new port, and why it stayed if it could not move."""
        self.error.hide()
        self.refresh()
        # The window moves a running server before this runs, keeping it if it fails.
        server, port = self.switch.server, self.settings.values.mcp_port
        if server.running and server.port != port:
            self.error.setText(
                f"Cannot serve agents on port {port}: another program may be using "
                f"it. Still serving on port {server.port}."
            )
            self.error.show()

    def _toggle(self, on: bool) -> None:
        """Starts or stops the MCP server; shows why it cannot start."""
        self.error.hide()
        if on:
            try:
                self.switch.start()
            except PortInUseError as error:
                self.error.setText(str(error))
                self.error.show()
        else:
            self.switch.server.stop()
        self.refresh()

    def _renew(self) -> None:
        """Gives the MCP server a new token."""
        self.switch.server.renew_token()
        self.refresh()


def _copy_button(parent: QWidget, line: QLineEdit) -> QPushButton:
    """Creates a button that copies line's text to the clipboard."""
    button = QPushButton("Copy", parent)
    button.clicked.connect(lambda: QGuiApplication.clipboard().setText(line.text()))
    return button


def _with_buttons(parent: QWidget, line: QLineEdit, *buttons: QPushButton) -> QWidget:
    """Returns a line of text with buttons after it, as one row's field."""
    row = QWidget(parent)
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(line, 1)
    for button in buttons:
        layout.addWidget(button)
    return row
