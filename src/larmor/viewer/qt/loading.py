"""Connects the loader and the job runner to the main thread, the bars and errors."""

from collections.abc import Callable

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QWidget,
)

from larmor.viewer.jobs import CANCELLING, SAVING, JobRunner, JobStatus
from larmor.viewer.loading import Loader, LoadStatus
from larmor.viewer.model.viewer import Viewer

_STEPS = 1000  # the progress bar's resolution


class MainThread(QObject):
    """Runs functions on the thread it lives on (the main thread), from any thread."""

    _call = Signal(object)

    def __init__(self, parent: QObject) -> None:
        """Lives on the thread that creates it, as a child of parent."""
        super().__init__(parent)
        self._call.connect(self._run, Qt.ConnectionType.QueuedConnection)

    def post(self, function: Callable[[], None]) -> None:
        """Queues function to run on the main thread, from any thread."""
        try:
            self._call.emit(function)
        except RuntimeError:  # the window is gone; nothing is left to update
            pass

    def _run(self, function: Callable[[], None]) -> None:
        """Runs a function posted from another thread."""
        function()


class LoadingBar(QWidget):
    """Shows the file being read, a progress bar and Cancel; hidden when idle."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Creates the bar, hidden."""
        super().__init__(parent)
        self.label = QLabel(self)
        self.progress = QProgressBar(self)
        self.progress.setRange(0, _STEPS)
        self.progress.setMaximumWidth(200)
        self.progress.setTextVisible(False)
        self.cancel = QPushButton("Cancel", self)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        for widget in (self.label, self.progress, self.cancel):
            layout.addWidget(widget)
        self.hide()

    def show_status(self, status: LoadStatus | None) -> None:
        """Shows what is being read, or hides the bar when nothing is."""
        if status is None:
            self.hide()
            return
        count = f" ({status.number} of {status.total})" if status.total > 1 else ""
        self.label.setText(f"Loading {status.name}{count}")
        self.progress.setValue(round(status.fraction * _STEPS))
        self.show()


class JobBar(LoadingBar):
    """Shows the running job, its stage, a progress bar and Cancel; hidden when idle."""

    def show_job(self, status: JobStatus | None) -> None:
        """Shows how far the job is, or hides the bar when none runs."""
        if status is None:
            self.hide()
            return
        saving = status.stage == SAVING
        stage = f": {status.stage}" if status.stage else ""
        if saving:
            stage = ": Saving, cannot be cancelled"
        self.label.setText(f"{status.name}{stage}")
        self.cancel.setEnabled(not saving and status.stage != CANCELLING)
        if status.total > 0:
            self.progress.setRange(0, _STEPS)
            self.progress.setValue(round(status.fraction * _STEPS))
        else:
            self.progress.setRange(0, 0)  # busy, without a known end
        self.show()


class ErrorBox(QMessageBox):
    """Collects errors, such as files that could not be opened, in one box."""

    def __init__(self, parent: QWidget, title: str = "Cannot open file") -> None:
        """Creates the box, hidden."""
        super().__init__(parent)
        self.setIcon(QMessageBox.Icon.Warning)
        self.setWindowTitle(title)
        self.setStandardButtons(QMessageBox.StandardButton.Ok)
        self.setWindowModality(Qt.WindowModality.NonModal)

    def add(self, message: str) -> None:
        """Shows message, below those not yet dismissed."""
        if self.isVisible():
            self.setText(f"{self.text()}\n\n{message}")
        else:
            self.setText(message)
            self.show()


def create_loader(viewer: Viewer, window: QWidget, bar: LoadingBar) -> Loader:
    """Creates a loader that reports to bar and to an error box owned by window."""
    errors = ErrorBox(window)
    loader = Loader(
        viewer,
        MainThread(window).post,
        on_status=bar.show_status,
        on_error=errors.add,
    )
    bar.cancel.clicked.connect(loader.cancel)
    return loader


def create_jobs(window: QWidget, bar: JobBar) -> JobRunner:
    """Creates a job runner that reports to bar and to an error box owned by window."""
    errors = ErrorBox(window, "Job failed")
    jobs = JobRunner(
        MainThread(window).post, on_status=bar.show_job, on_error=errors.add
    )
    bar.cancel.clicked.connect(jobs.cancel)
    return jobs
