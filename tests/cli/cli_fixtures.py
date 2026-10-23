"""Fixtures shared by the command and connectome tests."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from larmor._log import LarmorLogger

CONNECTOME = Path(__file__).parents[1] / "data" / "connectome"


@pytest.fixture
def keep_log_level() -> Iterator[None]:
    # A command sets the package logger's level and leaves it, which would hide
    # warnings from the tests that run after --quiet.
    level = LarmorLogger.level
    yield
    LarmorLogger.setLevel(level)


@pytest.fixture(scope="module")
def connectome_ref() -> dict[str, np.ndarray]:
    return dict(np.load(CONNECTOME / "ref.npz"))
