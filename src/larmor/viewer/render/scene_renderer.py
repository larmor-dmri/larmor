"""Keeps an adapter for each drawn item, and the views' other drawings."""

from collections.abc import Callable

from larmor._log import LarmorLogger
from larmor.viewer.model.events import (
    DataChanged,
    Event,
    ItemAdded,
    ItemRemoved,
    SourceChanged,
    ViewStateChanged,
)
from larmor.viewer.model.item import Category, Item, ItemId, ViewKind
from larmor.viewer.model.regions import mask_ids
from larmor.viewer.model.view_state import ViewField
from larmor.viewer.model.viewer import Viewer
from larmor.viewer.render.adapter import Adapter
from larmor.viewer.render.annotations import Annotations
from larmor.viewer.render.colour_bars import ColourBars
from larmor.viewer.render.fod_adapter import FodAdapter
from larmor.viewer.render.label_regions import LabelRegionGuides
from larmor.viewer.render.navigation import Navigation
from larmor.viewer.render.peaks_adapter import PeaksAdapter
from larmor.viewer.render.phantom_guides import PhantomGuides
from larmor.viewer.render.regions_guides import DetailBoxes, RegionGuides
from larmor.viewer.render.surface_adapter import SurfaceAdapter
from larmor.viewer.render.tractogram_adapter import (
    SLAB_BUDGET,
    TractogramAdapter,
)
from larmor.viewer.render.view import ViewRenderer
from larmor.viewer.render.view_labels import ViewLabels
from larmor.viewer.render.volume_adapter import VolumeAdapter

ADAPTERS: dict[Category, type[Adapter]] = {
    Category.VOLUMES: VolumeAdapter,
    Category.OVERLAYS: VolumeAdapter,
    Category.TRACTOGRAMS: TractogramAdapter,
    Category.FODS: FodAdapter,
    Category.PEAKS: PeaksAdapter,
    Category.SURFACES: SurfaceAdapter,
}

RANK: dict[Category, float] = {
    Category.VOLUMES: 0,
    Category.OVERLAYS: 1,
    Category.FODS: 1.5,  # over the slices, under the lines
    Category.TRACTOGRAMS: 2,
    Category.PEAKS: 2,
    Category.SURFACES: 2,
}


class SceneRenderer:
    """Keeps an adapter per drawn item, and passes the scene's events to them."""

    def __init__(self, viewer: Viewer, views: dict[ViewKind, ViewRenderer]) -> None:
        """Creates the adapters of the items already in the scene, and subscribes."""
        self.viewer = viewer
        self.views = views
        self.adapters: dict[ItemId, Adapter] = {}
        self._draw_cap = TractogramAdapter.draw_cap  # the cap its lines were built with
        self._tube_limit = TractogramAdapter.tube_limit  # likewise for the tubes
        self._notice_subscribers: list[Callable[[str], None]] = []
        self._noticed: dict[ItemId, str] = {}  # the notices that apply, by item
        self.colour_bars = ColourBars(viewer, views)
        self.navigation = Navigation(viewer, views)
        self.annotations = Annotations(viewer, views)
        self.view_labels = ViewLabels(viewer, views)
        self.phantom_guides = PhantomGuides(viewer, views)
        self.region_guides = RegionGuides(viewer, views)
        self.label_regions = LabelRegionGuides(viewer, views)
        self.detail_boxes = DetailBoxes(viewer, views)
        for view in views.values():
            view.show_overlay(viewer.scene.view.overlay_visible)
        self._unsubscribe = viewer.scene.subscribe(self._on_event)
        for item in viewer.scene.items:
            self._add(item)
        self._give_notices()

    def subscribe_notices(self, callback: Callable[[str], None]) -> Callable[[], None]:
        """Calls callback with each notice for the user; returns a function to stop.

        A notice says what the views draw differently from what was asked, such as
        lines in place of tubes; it is given once, when it starts to apply. A new
        subscriber is first given the notices that already apply.

        Returns:
            A function that stops the notices.
        """
        self._notice_subscribers.append(callback)
        for notice in self._noticed.values():
            callback(notice)
        return lambda: self._notice_subscribers.remove(callback)

    def _give_notices(self) -> None:
        """Logs and hands out the notices of the adapters that start to have one."""
        notices: dict[ItemId, str] = {}
        for item_id, adapter in self.adapters.items():
            if isinstance(adapter, TractogramAdapter):
                notice = adapter.tube_notice()
                if notice is not None:
                    notices[item_id] = notice
        for item_id, notice in notices.items():
            if item_id not in self._noticed:
                LarmorLogger.info(notice)
                for callback in list(self._notice_subscribers):
                    callback(notice)
        self._noticed = notices

    def _tell(self, text: str) -> None:
        """Logs and hands out a message for the user, such as a filter that failed."""
        LarmorLogger.info(text)
        for callback in list(self._notice_subscribers):
            callback(text)

    def _add(self, item: Item) -> None:
        """Creates the adapter of an item, if its category is drawn."""
        if item.category in ADAPTERS:
            rank = RANK[item.category]
            order = rank * 1_000_000 + float(item.id)
            if item.category == Category.TRACTOGRAMS:
                # Tells the user when its regions cannot be applied.
                self.adapters[item.id] = TractogramAdapter(
                    item, self.views, self.viewer, order, self._tell
                )
            else:
                self.adapters[item.id] = ADAPTERS[item.category](
                    item, self.views, self.viewer, order
                )

    def _on_event(self, event: Event) -> None:
        """Passes a scene event to the views' drawings and the adapters."""
        self.colour_bars.on_event(event)
        self.navigation.on_event(event)
        self.annotations.on_event(event)
        self.view_labels.on_event(event)
        self.phantom_guides.on_event(event)
        self.region_guides.on_event(event)
        self.label_regions.on_event(event)
        self.detail_boxes.on_event(event)
        if (
            isinstance(event, ViewStateChanged)
            and event.name == ViewField.OVERLAY_VISIBLE
        ):
            assert isinstance(event.new, bool)
            for view in self.views.values():
                view.show_overlay(event.new)
            return
        if isinstance(event, ItemAdded):
            self._add(event.item)
            self._give_notices()
            return

        if isinstance(event, (ItemRemoved, DataChanged)):
            self._drop(event)
            return
        if isinstance(event, SourceChanged):
            return  # the same data, so nothing to draw again

        for adapter in self.adapters.values():
            adapter.on_event(event)
        self._give_notices()

    def _drop(self, event: ItemRemoved | DataChanged) -> None:
        """Removes an item's adapter, and draws it again if only its data changed."""
        adapter = self.adapters.pop(event.item.id, None)
        if adapter is not None:
            adapter.close()
        self._noticed.pop(event.item.id, None)
        if isinstance(event, DataChanged):  # drawn again, in the same place
            self._add(event.item)
            # The items drawn with its data, as modulated or coloured by it, too.
            for named_by in self.viewer.scene.naming(event.item.id):
                dependent = self.adapters.pop(named_by.id, None)
                if dependent is not None:
                    dependent.close()
                    self._add(named_by)
            # The tractograms it is a mask of filter their streamlines again.
            for other in self.adapters.values():
                if isinstance(other, TractogramAdapter) and event.item.id in (
                    mask_ids(self.viewer.scene, other.item)
                ):
                    other.refilter()
            self._give_notices()

    def set_tract_limits(self, draw_cap: int, slab_bytes: int, tube_limit: int) -> None:
        """Sets the streamlines drawn, the slab memory and the tube limit."""
        # The class values are shared by windows: compare with this one's own.
        reshape = draw_cap != self._draw_cap or tube_limit != self._tube_limit
        self._draw_cap = TractogramAdapter.draw_cap = draw_cap
        self._tube_limit = TractogramAdapter.tube_limit = tube_limit
        SLAB_BUDGET.limit = slab_bytes  # copies made already stay, until dropped
        for adapter in self.adapters.values():
            if isinstance(adapter, TractogramAdapter):
                if reshape:
                    adapter.reshape()
                else:  # a larger budget can fit copies that did not
                    adapter.request_sorts()
        self._give_notices()

    def close(self) -> None:
        """Stops listening to the scene and removes every item's drawing."""
        self._unsubscribe()
        self._unsubscribe = lambda: None  # a second close does nothing
        for adapter in self.adapters.values():
            adapter.close()
        self.adapters.clear()
        self.colour_bars.close()
        self.annotations.close()
        self.view_labels.close()
        self.phantom_guides.close()
        self.region_guides.close()
        self.label_regions.close()
        self.detail_boxes.close()
