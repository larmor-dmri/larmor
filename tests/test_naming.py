import ast
import inspect
from pathlib import Path

import pytest

import larmor.io

# iter_ reads a file piece by piece; has_ and is_ ask a yes/no question about a file.
IO_PREFIXES = ("load_", "save_", "read_", "write_", "open_", "iter_", "has_", "is_")
IO_EXCEPTIONS = {"file_kind", "output_files"}

ROOT = Path(larmor.io.__file__).parents[1]


def _read_all(init: Path) -> list[str] | None:
    # Read from the source so no package (and no Qt) is imported.
    for node in ast.parse(init.read_text()).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets
        ):
            names: list[str] = ast.literal_eval(node.value)
            return names
    return None


PACKAGES = {
    ".".join(init.parent.relative_to(ROOT.parent).parts): names
    for init in sorted(ROOT.rglob("__init__.py"))
    if (names := _read_all(init)) is not None
}


def test_io_functions_are_named_after_their_action() -> None:
    functions = [
        n for n in larmor.io.__all__ if inspect.isfunction(getattr(larmor.io, n))
    ]
    offenders = [
        n for n in functions if not n.startswith(IO_PREFIXES) and n not in IO_EXCEPTIONS
    ]
    assert offenders == []


def test_format_functions_only_read_or_write() -> None:
    names = PACKAGES["larmor.io.formats"]
    assert [n for n in names if not n.startswith(("read_", "write_"))] == []


def test_packages_are_discovered() -> None:
    assert {"larmor", "larmor.io.formats", "larmor.viewer"} <= set(PACKAGES)


@pytest.mark.parametrize("package", sorted(PACKAGES))
def test_no_public_name_starts_with_get(package: str) -> None:
    assert [n for n in PACKAGES[package] if n.startswith("get_")] == []
