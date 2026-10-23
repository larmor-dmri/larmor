import subprocess
import sys
from pathlib import Path

import pytest

import larmor

COMMITTED = Path(larmor.__file__).parent / "_core.pyi"
STUBGEN = ["-m", "nanobind.stubgen", "-q", "-m", "larmor._core", "-o"]


def test_committed_stub_matches_the_compiled_module(tmp_path: Path) -> None:
    # mypy trusts the stub, so a stale one would silently give wrong types.
    pytest.importorskip("nanobind")
    fresh = tmp_path / "_core.pyi"
    subprocess.run([sys.executable, *STUBGEN, str(fresh)], check=True)
    regenerate = " ".join(["uv run python", *STUBGEN, "src/larmor/_core.pyi"])
    # Line by line: stubgen writes the platform's line endings.
    fresh_lines = fresh.read_text(encoding="utf-8").splitlines()
    committed_lines = COMMITTED.read_text(encoding="utf-8").splitlines()
    assert fresh_lines == committed_lines, (
        f"_core.pyi is out of date; regenerate it with: {regenerate}"
    )
