"""Holds what the viewer shows and sends an event for every change."""

from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from larmor.viewer.model.breakdown import BreakdownState
from larmor.viewer.model.detail import Detail, DetailId
from larmor.viewer.model.events import (
    BreakdownChanged,
    DataChanged,
    DetailChanged,
    DetailClosed,
    DetailOpened,
    Event,
    FilterChanged,
    ItemAdded,
    ItemRemoved,
    ItemRenamed,
    OptionChanged,
    PhantomChanged,
    SourceChanged,
    ViewStateChanged,
)
from larmor.viewer.model.item import Category, Item, ItemId
from larmor.viewer.model.matlab_import import MatlabPart
from larmor.viewer.model.options import ItemOptions, OptionName
from larmor.viewer.model.phantom import PhantomPart, PhantomState
from larmor.viewer.model.registration import Registration
from larmor.viewer.model.view_state import ViewField, ViewState


class Scene:
    """The loaded items in scene tree order and the view state, with change events."""

    def __init__(self) -> None:
        """Creates an empty scene."""
        self._items: dict[ItemId, Item] = {}  # insertion order is scene tree order
        self._next_id: ItemId = 1
        self.view = ViewState()
        # The phantom editor's state; None while no phantom is open.
        self.phantom: PhantomState | None = None
        # What the Signal breakdown dock shows; None while it is closed.
        self.breakdown: BreakdownState | None = None
        # The open detail windows, by id, in the order opened.
        self.details: dict[DetailId, Detail] = {}
        self._next_detail: DetailId = 1
        self._subscribers: list[Callable[[Event], None]] = []

    def subscribe(self, callback: Callable[[Event], None]) -> Callable[[], None]:
        """Calls callback with every event from now on; returns a function to stop."""
        self._subscribers.append(callback)
        return lambda: self._subscribers.remove(callback)

    def _emit(self, event: Event) -> None:
        """Calls every subscriber with the event."""
        # A copy, as a callback may unsubscribe while the loop runs.
        for callback in list(self._subscribers):
            callback(event)

    @property
    def items(self) -> list[Item]:
        """The items, in scene tree order."""
        return list(self._items.values())

    def item(self, item_id: ItemId) -> Item:
        """Returns the item with this id; raises ValueError if there is none."""
        if item_id not in self._items:
            raise ValueError(f"There is no item {item_id}.")
        return self._items[item_id]

    def in_category(self, category: Category) -> list[Item]:
        """Returns the items of a category, in scene tree order."""
        return [item for item in self._items.values() if item.category == category]

    def add(
        self,
        category: Category,
        name: str,
        source: Path | None,
        data: object,
        options: ItemOptions,
        value_range: tuple[float, float] | None = None,
        part: MatlabPart | None = None,
        registration: Registration | None = None,
    ) -> Item:
        """Adds an item at the end of the scene tree and returns it."""
        item = Item(
            self._next_id,
            category,
            name,
            source,
            data,
            options,
            value_range,
            part,
            registration,
        )
        self._next_id += 1
        self._items[item.id] = item
        self._emit(ItemAdded(item))
        return item

    def remove(self, item_id: ItemId) -> Item:
        """Removes an item and returns it."""
        item = self._items.pop(self.item(item_id).id)
        self._emit(ItemRemoved(item))
        return item

    def replace_data(
        self, item_id: ItemId, data: object, value_range: tuple[float, float] | None
    ) -> None:
        """Gives an item new data, read from no file; sends DataChanged."""
        item = self.item(item_id)
        item.data = data
        item.value_range = value_range
        item.source, item.part, item.registration = None, None, None
        self._emit(DataChanged(item))

    def set_source(self, item_id: ItemId, source: Path) -> None:
        """Makes an item refer to a file holding its data; sends SourceChanged."""
        item = self.item(item_id)
        item.source, item.part, item.registration = source, None, None
        self._emit(SourceChanged(item))

    def rename(self, item_id: ItemId, name: str) -> None:
        """Gives an item a new name; sends ItemRenamed."""
        item = self.item(item_id)
        item.name = name
        self._emit(ItemRenamed(item))

    def set_option(self, item_id: ItemId, name: OptionName, value: object) -> None:
        """Replaces one option of an item; sends OptionChanged if it changes."""
        item = self.item(item_id)
        old = getattr(item.options, name)
        if old == value:
            return
        changes: dict[str, Any] = {name: value}  # the field is chosen at run time
        item.options = replace(item.options, **changes)
        self._emit(OptionChanged(item, name, old, value))

    def set_view(self, name: ViewField, value: object) -> None:
        """Replaces one view-state field; sends ViewStateChanged if it changes."""
        old = getattr(self.view, name)
        if old == value:
            return
        changes: dict[str, Any] = {name: value}
        self.view = replace(self.view, **changes)
        self._emit(ViewStateChanged(name, old, value))

    def set_phantom(self, state: PhantomState | None, part: PhantomPart) -> None:
        """Replaces the phantom editor's state; sends PhantomChanged."""
        self.phantom = state
        self._emit(PhantomChanged(part))

    def set_breakdown(self, state: BreakdownState | None) -> None:
        """Replaces the Signal breakdown dock's state; sends BreakdownChanged."""
        self.breakdown = state
        self._emit(BreakdownChanged())

    def filtered(self, item_id: ItemId, kept: int, total: int) -> None:
        """Says a tractogram's filter by its regions finished; sends FilterChanged."""
        self._emit(FilterChanged(self.item(item_id), kept, total))

    def detail(self, detail_id: DetailId) -> Detail:
        """Returns the detail window with this id; raises ValueError if none."""
        if detail_id not in self.details:
            raise ValueError(f"There is no detail window {detail_id}.")
        return self.details[detail_id]

    def add_detail(self, detail: Detail) -> Detail:
        """Adds a detail window, given its next id; sends DetailOpened."""
        detail = replace(detail, id=self._next_detail)
        self._next_detail += 1
        self.details[detail.id] = detail
        self._emit(DetailOpened(detail))
        return detail

    def set_detail(self, detail: Detail) -> None:
        """Replaces a detail window by its id; sends DetailChanged if it changes."""
        if self.detail(detail.id) == detail:
            return
        self.details[detail.id] = detail
        self._emit(DetailChanged(detail))

    def remove_detail(self, detail_id: DetailId) -> None:
        """Removes a detail window; sends DetailClosed."""
        detail = self.details.pop(self.detail(detail_id).id)
        self._emit(DetailClosed(detail))

    def to_dict(self) -> dict[str, object]:
        """Returns the scene as plain data for a session file."""
        return {
            "items": [
                {
                    "id": i.id,  # the view state names items by id
                    "category": i.category,
                    "name": i.name,
                    "source": None if i.source is None else str(i.source),
                    "part": None if i.part is None else i.part.to_json(),
                    "registration": (
                        None if i.registration is None else i.registration.to_json()
                    ),
                    "options": asdict(i.options),
                }
                for i in self._items.values()
            ],
            "view": asdict(self.view),
        }
