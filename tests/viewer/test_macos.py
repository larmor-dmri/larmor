import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin" or importlib.util.find_spec("PySide6") is None,
    reason="AppKit's argument handling, through Qt",
)

# Prints each file Qt is asked to open while its event loop runs for half a second.
PROBE = """
import sys
from PySide6.QtCore import QEvent, QObject, QTimer
from PySide6.QtWidgets import QApplication

if sys.argv[1] == "viewer":
    from larmor.viewer.app import application

    app = application()
else:
    app = QApplication(sys.argv[:1])


class Spy(QObject):
    def eventFilter(self, watched, event):
        if event.type() == QEvent.Type.FileOpen:
            print(event.file())
        return False


spy = Spy()
app.installEventFilter(spy)
QTimer.singleShot(500, app.quit)
app.exec()
"""


def _opened(python: Path, probe: Path, mode: str) -> list[str]:
    # The real Cocoa platform: offscreen Qt never sees AppKit's open events.
    env = {k: v for k, v in os.environ.items() if k != "QT_QPA_PLATFORM"}
    result = subprocess.run(
        [str(python), str(probe), mode, "larmor.main", "bv"],
        capture_output=True,
        text=True,
        env=env,
        cwd=probe.parent,
        timeout=30,
        check=True,
    )
    return result.stdout.split()


def test_command_line_words_are_not_opened_as_files(tmp_path: Path) -> None:
    probe = tmp_path / "probe.py"
    probe.write_text(PROBE)
    # Whether AppKit opens the words depends on the path Python runs from: a venv's
    # python, linked straight to the interpreter, gets none; its python3 does.
    here = Path(sys.executable)
    pythons = [p for p in (here, here.with_name("python3")) if p.exists()]
    python = next((p for p in pythons if _opened(p, probe, "bare")), None)
    if python is None:
        pytest.skip("AppKit sends no open events here")
    assert _opened(python, probe, "viewer") == []
