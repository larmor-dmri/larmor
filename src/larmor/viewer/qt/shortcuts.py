"""Binds the shortcut registry to Qt actions, and lists the shortcuts in a dialog."""

import sys

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtGui import QAction, QKeyEvent, QKeySequence
from PySide6.QtWidgets import (
    QAbstractItemView,
    QAbstractSpinBox,
    QApplication,
    QDialog,
    QDialogButtonBox,
    QHeaderView,
    QLineEdit,
    QPlainTextEdit,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from larmor.viewer.model.shortcuts import (
    MOUSE_CONTROLS,
    SHORTCUTS,
    Command,
    run,
    views_text,
)
from larmor.viewer.model.viewer import Viewer

# Widgets that take typed text: while one has the focus, it keeps the keys it types
# with, and the shortcuts do not fire.
TEXT_WIDGETS = (QLineEdit, QAbstractSpinBox, QTextEdit, QPlainTextEdit)
COMMAND_MODIFIERS = (
    Qt.KeyboardModifier.ControlModifier
    | Qt.KeyboardModifier.MetaModifier
    | Qt.KeyboardModifier.AltModifier
)


def shortcut_actions(window: QWidget, viewer: Viewer) -> dict[Command, QAction]:
    """Creates a QAction per registry entry, whose keys work in all of the window.

    An entry whose feature does not exist yet gets a disabled action.

    Returns:
        The actions by command.
    """
    actions: dict[Command, QAction] = {}
    for command, shortcut in SHORTCUTS.items():
        action = QAction(shortcut.description, window)
        action.setShortcuts([QKeySequence(k) for k in shortcut.keys])
        action.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
        action.setEnabled(shortcut.active)
        if not shortcut.window:  # the window connects its own
            # Not run(viewer, command) bound now: the command must be this entry's.
            action.triggered.connect(lambda *_, c=command: run(viewer, c))
        window.addAction(action)
        actions[command] = action
    window.installEventFilter(KeyGuard(window))
    # The keys that work only at times, such as Enter while drawing, are enabled
    # only then, so that otherwise they reach the widgets.
    conditional = {
        c: s.enabled_when for c, s in SHORTCUTS.items() if s.enabled_when is not None
    }

    def enable(event: object = None) -> None:
        for command, when in conditional.items():
            actions[command].setEnabled(when(viewer))

    enable()
    unsubscribe = viewer.scene.subscribe(enable)
    window.destroyed.connect(lambda *_: unsubscribe())
    return actions


class KeyGuard(QObject):
    """Leaves the keys a text field types with to it, not to the shortcuts."""

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        """Accepts a ShortcutOverride for a key the focused text field types with."""
        if event.type() == QEvent.Type.ShortcutOverride:
            assert isinstance(event, QKeyEvent)
            if _widget_keeps(QApplication.focusWidget(), event):
                event.accept()
                return True
        return False


def _widget_keeps(widget: QWidget | None, event: QKeyEvent) -> bool:
    """Returns whether a text field has the focus and the key has no Ctrl or Cmd."""
    if widget is None or event.modifiers() & COMMAND_MODIFIERS:
        return False
    return isinstance(widget, TEXT_WIDGETS)


class ShortcutsDialog(QDialog):
    """Lists every shortcut of the registry, and the mouse controls of the views."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Builds the list from the registry."""
        super().__init__(parent)
        self.setWindowTitle("Shortcuts")
        self.table = QTableWidget(len(SHORTCUTS), 3, self)
        self.table.setHorizontalHeaderLabels(["Keys", "Action", "Selected view"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        for row, shortcut in enumerate(SHORTCUTS.values()):
            keys = ", ".join(
                QKeySequence(k).toString(QKeySequence.SequenceFormat.NativeText)
                for k in shortcut.keys
            )
            description = shortcut.description
            if not shortcut.active:
                description += " (not available yet)"
            cells = (keys, description, views_text(shortcut.views))
            for column, text in enumerate(cells):
                self.table.setItem(row, column, QTableWidgetItem(text))
        _fit_columns(self.table)

        self.mouse_table = QTableWidget(len(MOUSE_CONTROLS), 3, self)
        self.mouse_table.setHorizontalHeaderLabels(["Mouse", "Action", "Views"])
        self.mouse_table.verticalHeader().setVisible(False)
        self.mouse_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.mouse_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        for row, control in enumerate(MOUSE_CONTROLS):
            gesture = control.gesture
            if sys.platform == "darwin":
                gesture = gesture.replace("Ctrl", "Cmd")
            cells = (gesture, control.description, views_text(control.views))
            for column, text in enumerate(cells):
                self.mouse_table.setItem(row, column, QTableWidgetItem(text))
        _fit_columns(self.mouse_table)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, self)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(self.table, 3)
        layout.addWidget(self.mouse_table, 1)
        layout.addWidget(buttons)
        self.resize(640, 600)


def _fit_columns(table: QTableWidget) -> None:
    """Fits a table's first and last columns to their text; the middle one stretches."""
    header = table.horizontalHeader()
    header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
    header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
    header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
