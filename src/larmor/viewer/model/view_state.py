"""The state of the views, shared by all of them."""

from dataclasses import dataclass
from enum import StrEnum

from larmor.viewer.model.camera import EMPTY_FIELD, Cameras, Convention, Vec3
from larmor.viewer.model.item import Category, ItemId, ViewKind


class Tool(StrEnum):
    """The pointer tools of the views, which decide what a click and a drag do."""

    CURSOR = "cursor"  # moves the crosshair
    SPHERE = "sphere"  # places a sphere region
    BOX = "box"  # places a box region
    ZOOM_BOX = "zoom_box"  # opens a rectangle of a 2D view in a detail window
    DRAW_BUNDLE = "draw_bundle"  # adds the points of a phantom bundle


class ViewField(StrEnum):
    """The fields of the view state, as ViewStateChanged names them."""

    CURSOR = "cursor"
    MAIN_VOLUME = "main_volume"
    SELECTED = "selected"
    SELECTED_VIEW = "selected_view"
    MAXIMISED = "maximised"
    CONVENTION = "convention"
    OVERLAY_VISIBLE = "overlay_visible"
    CROSSHAIR_VISIBLE = "crosshair_visible"
    CAMERAS = "cameras"
    FIELD_OF_VIEW = "field_of_view"
    TOOL = "tool"


@dataclass(frozen=True)
class ViewState:
    """The state of the views, shared by all of them."""

    cursor: tuple[float, float, float] = (0.0, 0.0, 0.0)  # world mm (RAS)
    main_volume: ItemId | None = None
    selected: ItemId | Category | None = None  # an item, a category, or nothing
    selected_view: ViewKind = ViewKind.AXIAL  # the one last clicked
    maximised: ViewKind | None = None
    convention: Convention = Convention.RAS
    overlay_visible: bool = True  # text, labels and colour bars
    crosshair_visible: bool = True
    cameras: Cameras = Cameras()
    # The world box (lowest and highest corner, mm) the 2D views show at zoom 1.
    field_of_view: tuple[Vec3, Vec3] = EMPTY_FIELD
    tool: Tool = Tool.CURSOR
