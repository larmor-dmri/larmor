import subprocess
import sys

import pytest

from larmor.viewer import Category
from larmor.viewer.model.options import ItemOptions
from larmor.viewer.model.registry import item_type


@pytest.mark.parametrize("category", list(Category))
def test_every_category_has_an_entry_keyed_by_it_with_item_options(
    category: Category,
) -> None:
    entry = item_type(category)
    assert entry.category == category
    assert issubclass(entry.options, ItemOptions)


def test_the_model_loads_no_qt_or_gpu_library() -> None:
    # A fresh interpreter: this test process may have imported them already.
    heavy = ("PySide6", "pygfx", "wgpu", "rendercanvas")
    code = (
        "import sys, larmor.viewer.model.viewer\n"
        f"print(','.join(m for m in {heavy!r} if m in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ""
