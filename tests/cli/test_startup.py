import subprocess
import sys

import pytest

import larmor

HEAVY = ("scipy", "h5py", "skimage", "PySide6", "pygfx", "wgpu")


def test_starting_the_cli_does_not_load_numerical_libraries() -> None:
    # Every Tab completion starts Larmor, so this keeps completion fast. A fresh
    # interpreter is needed: this test process has imported everything already.
    code = (
        "import sys, larmor.main\n"
        f"print(','.join(m for m in {HEAVY!r} if m in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ""


@pytest.mark.parametrize("name", larmor.__all__)
def test_every_public_name_imports(name: str) -> None:
    assert getattr(larmor, name) is not None


def test_an_unknown_name_is_an_attribute_error() -> None:
    with pytest.raises(AttributeError, match="no_such_name"):
        _ = larmor.no_such_name  # type: ignore[attr-defined]


@pytest.mark.parametrize("name", larmor.__all__)
def test_every_public_name_is_listed_by_dir(name: str) -> None:
    # IPython and Jupyter complete `larmor.<Tab>` from dir().
    assert name in dir(larmor)
