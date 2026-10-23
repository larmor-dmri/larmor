import importlib
from itertools import combinations
from types import ModuleType

import pytest

import larmor
import larmor.io
import larmor.sd
import larmor.types

# Label is a type but not an everyday name, so the top level leaves it out.
TOP_LEVEL_TYPES = {"Grid", "Volume", "Tractogram", "Surface", "GradientTable"}
TYPES = TOP_LEVEL_TYPES | {"Label"}

SUBPACKAGES = [
    "larmor.types",
    "larmor.io",
    "larmor.io.formats",
    "larmor.surfaces",
    "larmor.dwi",
    "larmor.data",
    "larmor.sd",
    "larmor.tractography",
    "larmor.registration",
    "larmor.simulation",
]


def _module(name: str) -> ModuleType:
    return importlib.import_module(name)


def test_types_exports_exactly_the_core_types() -> None:
    assert set(larmor.types.__all__) == TYPES


@pytest.mark.parametrize("name", sorted(TOP_LEVEL_TYPES))
def test_top_level_type_is_the_types_object(name: str) -> None:
    assert getattr(larmor, name) is getattr(larmor.types, name)


@pytest.mark.parametrize("name", sorted(TYPES))
def test_io_does_not_export_core_types(name: str) -> None:
    assert name not in larmor.io.__all__
    assert not hasattr(larmor.io, name)


def test_importing_a_type_from_io_fails() -> None:
    with pytest.raises(ImportError):
        # The import is meant to fail: io no longer exports the types.
        from larmor.io import Volume  # type: ignore[attr-defined]  # noqa: F401


def test_top_level_exports_everyday_io() -> None:
    assert {"load_surface", "save_surface", "load_dwi"} <= set(larmor.__all__)


def test_sd_exports_peaks_and_sh_conversion() -> None:
    names = {"Peaks", "find_peaks", "convert_sh_basis", "sh_layout"}
    assert names <= set(larmor.sd.__all__)


@pytest.mark.parametrize("package", ["larmor", *SUBPACKAGES])
def test_every_exported_name_is_importable(package: str) -> None:
    module = _module(package)
    missing = [name for name in module.__all__ if not hasattr(module, name)]
    assert missing == []


@pytest.mark.parametrize(("first", "second"), list(combinations(SUBPACKAGES, 2)))
def test_no_name_is_exported_from_two_subpackages(first: str, second: str) -> None:
    shared = set(_module(first).__all__) & set(_module(second).__all__)
    assert shared == set()
