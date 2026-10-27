"""Items, their categories, and the names and layout of the views."""

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from larmor.types import Volume
from larmor.viewer.model.options import ItemOptions

if TYPE_CHECKING:
    from larmor.viewer.model.matlab_import import MatlabPart
    from larmor.viewer.model.registration import Registration

#: The number that names an item in its viewer, as in `Item.id`.
type ItemId = int


class Category(StrEnum):
    """A kind of item the viewer shows, such as volumes or tractograms.

    The category of an item decides how its file is read and how it is drawn,
    and each category has its own group in the Layers panel.
    """

    VOLUMES = "volumes"
    OVERLAYS = "overlays"
    TRACTOGRAMS = "tractograms"
    FODS = "fods"
    PEAKS = "peaks"
    SURFACES = "surfaces"
    REGIONS = "regions"  # spheres, boxes and masks that filter tractograms


class ViewKind(StrEnum):
    """One of the four views."""

    AXIAL = "axial"
    CORONAL = "coronal"
    SAGITTAL = "sagittal"
    THREE_D = "3d"


# The four views, in the order of ViewKind.
VIEW_KINDS: tuple[ViewKind, ...] = tuple(ViewKind)
# Each view's name, as the views and the dialogs show it.
VIEW_NAMES: dict[ViewKind, str] = {
    ViewKind.AXIAL: "Axial",
    ViewKind.CORONAL: "Coronal",
    ViewKind.SAGITTAL: "Sagittal",
    ViewKind.THREE_D: "3D",
}
# The views as the window lays them out, row by row.
VIEW_GRID: tuple[tuple[ViewKind, ...], ...] = (
    (ViewKind.AXIAL, ViewKind.CORONAL),
    (ViewKind.SAGITTAL, ViewKind.THREE_D),
)
# Pixels between neighbouring views. The window draws the gap as two 1 px
# borders, and the off-screen renderer and the Screenshots panel use the same gap.
VIEW_GAP = 2


@dataclass
class Item:
    """One loaded file or object.

    Attributes:
        id: The item's id in its viewer.
        category: Which kind of item it is.
        name: The file's name, or the name given from Python.
        source: The file, or None for an object given from Python.
        data: The `Volume`, `Tractogram` or surface data it shows; for a region,
            its `Sphere`, `Box` or `MaskRegion`.
        options: The item's options.
        value_range: The data's minimum and maximum in scaled units for an image,
            the range of all of a surface's values per vertex; None for the others.
        part: What the item was read from inside a MATLAB file; None for any other.
        registration: The registration the item was brought through when loaded.
    """

    id: ItemId
    category: Category
    name: str
    source: Path | None
    data: object
    options: ItemOptions
    value_range: tuple[float, float] | None = None
    part: "MatlabPart | None" = None  # noqa: UP037 (an import cycle)
    registration: "Registration | None" = None  # noqa: UP037 (an import cycle)

    @property
    def volume_count(self) -> int | None:
        """The number of volumes of an image (1 for a 3D one); None for other data."""
        if isinstance(self.data, Volume):
            vol = self.data.data
            if vol.ndim == 4:
                return int(vol.shape[3])
            else:
                return 1
        return None
