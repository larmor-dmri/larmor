"""The events the scene sends, one for every change."""

from dataclasses import dataclass

from larmor.viewer.model.detail import Detail
from larmor.viewer.model.item import Item
from larmor.viewer.model.options import OptionName
from larmor.viewer.model.phantom import PhantomPart
from larmor.viewer.model.view_state import ViewField


@dataclass(frozen=True)
class ItemAdded:
    """An item was added to the scene (at the end of the scene tree)."""

    item: Item


@dataclass(frozen=True)
class ItemRemoved:
    """An item was removed from the scene."""

    item: Item


@dataclass(frozen=True)
class DataChanged:
    """An item's data was replaced; its id, options and place in the scene tree stay."""

    item: Item


@dataclass(frozen=True)
class SourceChanged:
    """An item's file changed, as when it is saved to a new one."""

    item: Item


@dataclass(frozen=True)
class ItemRenamed:
    """An item was given a new name."""

    item: Item


@dataclass(frozen=True)
class OptionChanged:
    """One option of an item changed from `old` to `new`."""

    item: Item
    name: OptionName
    old: object
    new: object


@dataclass(frozen=True)
class ViewStateChanged:
    """One field of the view state changed from `old` to `new`."""

    name: ViewField
    old: object
    new: object


@dataclass(frozen=True)
class PhantomChanged:
    """A part of the phantom editor's state changed (see `Scene.phantom`)."""

    part: PhantomPart


@dataclass(frozen=True)
class BreakdownChanged:
    """The Signal breakdown dock's state changed (see `Scene.breakdown`)."""


@dataclass(frozen=True)
class FilterChanged:
    """A tractogram's filter by its regions finished: `kept` of `total` streamlines.

    `kept` equals `total` when no region applies or the filter is off.
    """

    item: Item
    kept: int
    total: int


@dataclass(frozen=True)
class DetailOpened:
    """A detail window was opened (see `Scene.details`)."""

    detail: Detail


@dataclass(frozen=True)
class DetailChanged:
    """A detail window's rectangle or settings changed; `detail` is the new one."""

    detail: Detail


@dataclass(frozen=True)
class DetailClosed:
    """A detail window was closed."""

    detail: Detail


type Event = (
    ItemAdded
    | ItemRemoved
    | DataChanged
    | SourceChanged
    | ItemRenamed
    | OptionChanged
    | ViewStateChanged
    | PhantomChanged
    | BreakdownChanged
    | FilterChanged
    | DetailOpened
    | DetailChanged
    | DetailClosed
)
