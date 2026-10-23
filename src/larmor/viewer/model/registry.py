"""Describes how each category of item is read and which options it has."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Final
from weakref import WeakKeyDictionary

import numpy as np

from larmor.io import (
    TRACTOGRAM_EXTENSIONS,
    VOLUME_EXTENSIONS,
    Progress,
    load_tractogram,
    load_tractogram_subset,
    load_volume,
)
from larmor.types import Surface, Tractogram, Volume
from larmor.viewer.model.item import Category
from larmor.viewer.model.options import (
    FodOptions,
    ItemOptions,
    OverlayOptions,
    PeaksOptions,
    SurfaceOptions,
    TractogramOptions,
    VolumeOptions,
)
from larmor.viewer.model.surface import SurfaceData, read_surface

#: Tractograms with more streamlines are reduced to a random subset when read.
MAX_STREAMLINES: Final = 1_000_000
# The seed of the random subset, so the same file always gives the same subset.
SUBSET_SEED: Final = 0

# Reads a file, given its path, a progress callback and the streamline limit.
type Loader = Callable[[Path, Progress | None, int | None], object]


@dataclass(frozen=True)
class ItemType:
    """How one category of item is loaded, and which options it has."""

    category: Category
    label: str  # the name in the Layers tree, such as "Volumes"
    load: Loader
    data_type: type  # the type `load` returns and objects from Python must have
    options: type[ItemOptions]
    extensions: tuple[str, ...]  # the extensions `load` reads, such as ".nii.gz"
    # The name starts of files without an extension, such as FreeSurfer's "lh.".
    prefixes: tuple[str, ...] = ()
    # An object given from Python that data_type wraps, such as a Surface.
    wraps: type | None = None

    def reads(self, path: Path) -> bool:
        """Tells whether `load` reads this file, from its name only."""
        name = path.name.lower()
        return name.endswith(self.extensions) or name.startswith(self.prefixes)

    def patterns(self) -> str:
        """Returns the file dialog's patterns for the category, such as "*.gii lh.*"."""
        return " ".join(
            [*("*" + e for e in self.extensions), *(p + "*" for p in self.prefixes)]
        )

    def data(self, obj: object) -> object:
        """Returns the item's data for an object read or given from Python.

        Returns:
            The data.

        Raises:
            ValueError: If the object is not of the category's type.
        """
        if self.wraps is not None and isinstance(obj, self.wraps):
            return self.data_type(obj)
        if not isinstance(obj, self.data_type):
            given = self.wraps or self.data_type
            raise ValueError(
                f"{self.label} hold {given.__name__} objects, not {type(obj).__name__}."
            )
        return obj


def _read_volume(path: Path, progress: Progress | None, limit: int | None) -> object:
    """Reads a volume; the streamline limit is not used."""
    return load_volume(path, on_progress=progress)


@dataclass(frozen=True, eq=False)
class Subset:
    """Records which of its file's streamlines a tractogram read as a subset kept."""

    indices: np.ndarray  # (S,) int64, ascending, into the file's streamlines
    total: int  # the number of streamlines in the file


# The subset each tractogram read as a random subset of its file holds.
SUBSETS: WeakKeyDictionary[Tractogram, Subset] = WeakKeyDictionary()


def _read_tractogram(
    path: Path, progress: Progress | None, limit: int | None
) -> object:
    """Reads a tractogram, or a random subset of limit streamlines of it."""
    if limit is None:
        return load_tractogram(path, on_progress=progress)
    tractogram, indices, total = load_tractogram_subset(
        path, limit, seed=SUBSET_SEED, on_progress=progress
    )
    if len(indices) < total:
        SUBSETS[tractogram] = Subset(indices, total)
    return tractogram


def _read_surface(path: Path, progress: Progress | None, limit: int | None) -> object:
    """Reads a surface; the progress and the streamline limit are not used."""
    return read_surface(path)


_VOLUME_EXTENSIONS = tuple(e for exts in VOLUME_EXTENSIONS.values() for e in exts)
_TRACTOGRAM_EXTENSIONS = tuple(
    e for exts in TRACTOGRAM_EXTENSIONS.values() for e in exts
)

# The item type of each category.
REGISTRY: dict[Category, ItemType] = {
    t.category: t
    for t in (
        ItemType(
            Category.VOLUMES,
            "Volumes",
            _read_volume,
            Volume,
            VolumeOptions,
            _VOLUME_EXTENSIONS,
        ),
        ItemType(
            Category.OVERLAYS,
            "Overlays",
            _read_volume,
            Volume,
            OverlayOptions,
            _VOLUME_EXTENSIONS,
        ),
        ItemType(
            Category.TRACTOGRAMS,
            "Tractograms",
            _read_tractogram,
            Tractogram,
            TractogramOptions,
            _TRACTOGRAM_EXTENSIONS,
        ),
        # A 4D image whose last axis holds the SH coefficients.
        ItemType(
            Category.FODS,
            "FODs",
            _read_volume,
            Volume,
            FodOptions,
            _VOLUME_EXTENSIONS,
        ),
        ItemType(
            Category.PEAKS,
            "Peaks",
            _read_volume,
            Volume,
            PeaksOptions,
            _VOLUME_EXTENSIONS,
        ),
        # FreeSurfer surfaces, such as lh.white, are known by their hemisphere prefix.
        ItemType(
            Category.SURFACES,
            "Surfaces",
            _read_surface,
            SurfaceData,
            SurfaceOptions,
            (".gii",),
            prefixes=("lh.", "rh."),
            wraps=Surface,
        ),
    )
}


def item_type(category: Category) -> ItemType:
    """Returns the registry entry of a category."""
    return REGISTRY[category]
