"""The events the scene sends, one for every change."""

from dataclasses import dataclass

from larmor.viewer.model.item import Item
from larmor.viewer.model.options import OptionName
from larmor.viewer.model.phantom import PhantomPart
from larmor.viewer.model.view_state import ViewField


@dataclass(frozen=True)
class ItemAdded:
    """An item was added to the scene (at the end of Layers)."""

    item: Item


@dataclass(frozen=True)
class ItemRemoved:
    """An item was removed from the scene."""

    item: Item


@dataclass(frozen=True)
class DataChanged:
    """An item's data was replaced; its id, options and place in Layers stay."""

    item: Item


@dataclass(frozen=True)
class SourceChanged:
    """An item's file changed, as when it is saved to a new one."""

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


type Event = (
    ItemAdded
    | ItemRemoved
    | DataChanged
    | SourceChanged
    | OptionChanged
    | ViewStateChanged
    | PhantomChanged
    | BreakdownChanged
)
