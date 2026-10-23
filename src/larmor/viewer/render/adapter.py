"""Defines the base class of the adapters, which draw one item each."""

from larmor.viewer.model.events import Event
from larmor.viewer.model.item import Item, ViewKind
from larmor.viewer.model.viewer import Viewer
from larmor.viewer.render.view import ViewRenderer


class Adapter:
    """Draws one item in the views."""

    def __init__(
        self,
        item: Item,
        views: dict[ViewKind, ViewRenderer],
        viewer: Viewer,
        order: float,
    ) -> None:
        """Creates the adapter that draws item in views."""
        self.item = item
        self.views = views
        self.viewer = viewer
        self.order = order

    def on_event(self, event: Event) -> None:
        """Updates the drawing for a scene event."""
        pass

    def close(self) -> None:
        """Removes the item's objects from the views."""
        pass
