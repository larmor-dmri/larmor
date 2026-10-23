"""Diffusion MRI processing, with the names most scripts need at the top level."""

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING

from larmor.io import (
    load_dwi,
    load_surface,
    load_tractogram,
    load_volume,
    read_bvals_bvecs,
    save_surface,
    save_tractogram,
    save_volume,
    write_bvals_bvecs,
)
from larmor.types import GradientTable, Grid, Surface, Tractogram, Volume

try:
    #: The version of Larmor installed, such as "1.0.0".
    __version__ = version("larmor-dmri")
except PackageNotFoundError:  # a source tree that is not installed
    __version__ = "unknown"

# The models imported on first use, with the package each comes from.
_LAZY = {
    "DTIModel": "larmor.dwi",
    "DTIFit": "larmor.dwi",
    "create_brain_mask": "larmor.dwi",
    "SDModel": "larmor.sd",
    "SDFit": "larmor.sd",
    "AnisotropicRF": "larmor.sd",
    "IsotropicRF": "larmor.sd",
}


if TYPE_CHECKING:
    # Type checkers see the real imports, so a misspelled name is still an error.
    from larmor.dwi import DTIFit, DTIModel, create_brain_mask
    from larmor.sd import AnisotropicRF, IsotropicRF, SDFit, SDModel
else:

    def __getattr__(name: str) -> object:
        """Imports a model on first use, so that the CLI starts fast."""
        if name in _LAZY:
            value = getattr(import_module(_LAZY[name]), name)
            globals()[name] = value
            return value
        raise AttributeError(f"module 'larmor' has no attribute {name!r}")

    def __dir__() -> list[str]:
        """Lists the names imported on first use too, for tab completion."""
        return sorted({*globals(), *__all__})


__all__ = [
    "AnisotropicRF",
    "DTIFit",
    "DTIModel",
    "GradientTable",
    "Grid",
    "IsotropicRF",
    "SDFit",
    "SDModel",
    "Surface",
    "Tractogram",
    "Volume",
    "__version__",
    "create_brain_mask",
    "load_dwi",
    "load_surface",
    "load_tractogram",
    "load_volume",
    "read_bvals_bvecs",
    "save_surface",
    "save_tractogram",
    "save_volume",
    "write_bvals_bvecs",
]
