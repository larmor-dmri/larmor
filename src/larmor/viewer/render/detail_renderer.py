"""Draws a detail window: one rectangle of a 2D view, filling its own canvas."""

from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import pygfx as gfx
from PIL import Image, ImageDraw
from rendercanvas import BaseRenderCanvas

from larmor.types import Grid
from larmor.viewer.model.camera import Camera2D, screen_axes
from larmor.viewer.model.detail import (
    VIEW_AXIS,
    Detail,
    DetailId,
    FigureBackground,
    detail_field,
    plane_axes,
    scale_bar_length,
    screen_millimetres,
)
from larmor.viewer.model.events import (
    DataChanged,
    DetailChanged,
    Event,
    ItemAdded,
    ItemRemoved,
    OptionChanged,
    ViewStateChanged,
)
from larmor.viewer.model.item import Category, Item, ItemId
from larmor.viewer.model.scene import Scene
from larmor.viewer.model.view_state import ViewField, ViewState
from larmor.viewer.model.viewer import Viewer
from larmor.viewer.render.adapter import Adapter
from larmor.viewer.render.fod_adapter import FodAdapter
from larmor.viewer.render.offscreen import (
    OffscreenRenderer,
    check_image,
    draw_view,
    image_canvas,
    straight_alpha,
)
from larmor.viewer.render.peaks_adapter import PeaksAdapter
from larmor.viewer.render.regions_guides import DETAIL_COLOUR
from larmor.viewer.render.view import Vec3, ViewRenderer
from larmor.viewer.render.view_3d import light_glyphs
from larmor.viewer.render.volume_adapter import VolumeAdapter
from larmor.viewer.render.worker import flush

# The order the categories are drawn in, under to over.
RANK: dict[Category, float] = {
    Category.VOLUMES: 0,
    Category.OVERLAYS: 1,
    Category.FODS: 1.5,
    Category.PEAKS: 2,
}
GRID_COLOUR = "#6e6e6e"  # the voxels' edges
BAR_COLOUR = "#ffffff"
BAR_WIDTH = 3.0  # px
BAR_INSET = 0.05  # of the picture's width and height, from its lower left corner
FONT_SIZE = 13
# A figure's margins and the gap between its two pictures, logical pixels.
FIGURE_MARGIN = 24
FIGURE_GAP = 48
FIGURE_LINE = 2  # px per scale, the outlines and connectors


class _DetailVolume(VolumeAdapter):
    """Draws an image's slice in a detail window, never smoothed."""

    smooth = False


class _DetailScene:
    """Stands for the scene to a detail's adapters, its cursor in the detail's slice."""

    def __init__(self, scene: Scene, position: Vec3) -> None:
        """Wraps scene, with the cursor at position."""
        self._scene = scene
        self.position = position

    @property
    def view(self) -> ViewState:
        """The scene's view state, with the cursor in the detail's slice."""
        return replace(self._scene.view, cursor=self.position)

    def item(self, item_id: ItemId) -> Item:
        """Returns the scene's item with this id."""
        return self._scene.item(item_id)


class _DetailViewer:
    """Stands for the viewer to a detail's adapters, which read only its scene."""

    def __init__(self, scene: _DetailScene) -> None:
        """Holds the stand-in scene."""
        self.scene = scene


class DetailRenderer:
    """Draws one detail window's rectangle so that it fills the canvas.

    The rectangle is the field of a 2D view at zoom 1. Images are drawn without
    smoothing, FOD glyphs only for the rectangle at the detail's glyph detail,
    and peaks; the voxels' edges and a scale bar are drawn on request. It has no
    crosshair and no orientation letters.

    Args:
        viewer: The viewer whose scene is drawn.
        detail_id: The detail window.
        canvas: The canvas to draw into.
    """

    def __init__(
        self, viewer: Viewer, detail_id: DetailId, canvas: BaseRenderCanvas
    ) -> None:
        """Creates the view and the drawings, and follows the scene."""
        self.viewer = viewer
        self.detail_id = detail_id
        self.detail = viewer.scene.detail(detail_id)
        self.view = ViewRenderer(canvas, self.detail.view)
        light_glyphs(self.view)
        self._scene = _DetailScene(viewer.scene, viewer.detail_position(detail_id))
        # The adapters read the scene's cursor: theirs is in the detail's slice.
        self._proxy = cast("Viewer", _DetailViewer(self._scene))
        self.adapters: dict[ItemId, Adapter] = {}
        self.grid_lines = gfx.Line(
            gfx.Geometry(positions=np.zeros((2, 3), np.float32)),
            gfx.LineSegmentMaterial(
                thickness=1.0,
                color=GRID_COLOUR,
                depth_test=False,
            ),
        )
        self.bar = gfx.Line(
            gfx.Geometry(positions=np.zeros((2, 3), np.float32)),
            gfx.LineMaterial(thickness=BAR_WIDTH, color=BAR_COLOUR, depth_test=False),
        )
        self.bar_text = gfx.Text(
            text="",
            font_size=FONT_SIZE,
            screen_space=True,
            anchor="bottom-left",
            material=gfx.TextMaterial(
                color=BAR_COLOUR, outline_color="#000", outline_thickness=0.25
            ),
        )
        self.view.guides.add(self.grid_lines, self.bar, self.bar_text)
        for item in viewer.scene.items:
            self._add(item)
        self._place()
        self._unsubscribe = viewer.scene.subscribe(self._on_event)

    def close(self) -> None:
        """Stops following the scene and removes the drawings."""
        self._unsubscribe()
        self._unsubscribe = lambda: None
        for adapter in self.adapters.values():
            adapter.close()
        self.adapters.clear()

    def _add(self, item: Item) -> None:
        """Creates the adapter of an item, if a detail draws its category."""
        if item.category not in RANK:
            return
        order = RANK[item.category] * 1_000_000 + float(item.id)
        views = {self.detail.view: self.view}
        adapter: Adapter
        if item.category == Category.FODS:
            adapter = FodAdapter(
                item, views, self._proxy, order, self._field(), self.detail.glyph_detail
            )
        elif item.category == Category.PEAKS:
            adapter = PeaksAdapter(item, views, self._proxy, order)
        else:
            adapter = _DetailVolume(item, views, self._proxy, order)
        self.adapters[item.id] = adapter

    def _on_event(self, event: Event) -> None:
        """Follows the items, the cursor while linked, and the detail's settings."""
        if isinstance(event, (ItemAdded, ItemRemoved, DataChanged)):
            self._follow_item(event)
        elif isinstance(event, OptionChanged):
            for adapter in self.adapters.values():
                adapter.on_event(event)
        elif isinstance(event, ViewStateChanged):
            if event.name == ViewField.CURSOR and self.detail.linked:
                self._move_to(self.viewer.scene.view.cursor)
            elif event.name in (ViewField.CONVENTION, ViewField.MAIN_VOLUME):
                for adapter in self.adapters.values():
                    adapter.on_event(event)
                self._place()
        elif isinstance(event, DetailChanged) and event.detail.id == self.detail_id:
            self._follow_detail(event.detail)

    def _follow_detail(self, detail: Detail) -> None:
        """Shows the detail's new rectangle, slice or settings."""
        self.detail = detail
        self._move_to(self.viewer.detail_position(self.detail_id))
        for adapter in self.adapters.values():
            if isinstance(adapter, FodAdapter):
                adapter.set_box(self._field(), detail.glyph_detail)
        self._place()

    def _follow_item(self, event: ItemAdded | ItemRemoved | DataChanged) -> None:
        """Adds, drops or draws again an item's adapter."""
        if not isinstance(event, ItemAdded):
            adapter = self.adapters.pop(event.item.id, None)
            if adapter is not None:
                adapter.close()
        if not isinstance(event, ItemRemoved):
            self._add(event.item)

    def _move_to(self, position: Vec3) -> None:
        """Shows the slice through position, telling the adapters the cursor moved."""
        old = self._scene.position
        if old == position:
            return
        self._scene.position = position
        moved = ViewStateChanged(ViewField.CURSOR, old, position)
        for adapter in self.adapters.values():
            adapter.on_event(moved)
        self._place()

    def _grid(self) -> Grid | None:
        """Returns the views' grid, whose voxels the rectangle counts."""
        return self.viewer._reference_grid()

    def _field(self) -> tuple[Vec3, Vec3] | None:
        """Returns the world box the detail shows; None without a grid."""
        grid = self._grid()
        if grid is None:
            return None
        return detail_field(
            grid, self.detail.view, self.detail.box, self._scene.position
        )

    def _place(self) -> None:
        """Points the camera at the rectangle and draws the grid and the bar."""
        field = self._field()
        grid = self._grid()
        if field is None or grid is None:
            return
        self.view.set_camera(
            field=field,
            camera=Camera2D(),
            convention=self.viewer.scene.view.convention,
        )
        self._draw_grid(grid)
        self._draw_bar(grid, field)
        self.view.request_draw()

    def _draw_grid(self, grid: Grid) -> None:
        """Draws the voxels' edges inside the rectangle, if asked."""
        self.grid_lines.visible = self.detail.voxel_grid
        if not self.detail.voxel_grid:
            return
        across, first, second = plane_axes(grid, self.detail.view)
        inverse = np.linalg.inv(grid.affine)
        position = np.asarray(self._scene.position, dtype=float)
        depth = (inverse[:3, :3] @ position + inverse[:3, 3])[across]
        i0, j0, ni, nj = self.detail.box
        ends = []
        # Voxel k spans k - 0.5 to k + 0.5.
        i_edges = [k - 0.5 for k in range(i0, i0 + ni + 1)]
        j_edges = [k - 0.5 for k in range(j0, j0 + nj + 1)]
        for i in i_edges:
            for j in (j_edges[0], j_edges[-1]):
                ends.append(_voxel(across, first, second, depth, i, j))
        for j in j_edges:
            for i in (i_edges[0], i_edges[-1]):
                ends.append(_voxel(across, first, second, depth, i, j))
        voxels = np.array(ends)
        world = voxels @ grid.affine[:3, :3].T + grid.affine[:3, 3]
        world[:, VIEW_AXIS[self.detail.view]] = position[VIEW_AXIS[self.detail.view]]
        self.grid_lines.geometry = gfx.Geometry(positions=world.astype(np.float32))

    def _draw_bar(self, grid: Grid, field: tuple[Vec3, Vec3]) -> None:
        """Draws a scale bar of a round length at the lower left, if asked."""
        shown = self.detail.scale_bar
        self.bar.visible = self.bar_text.visible = shown
        if not shown:
            return
        width, height = screen_millimetres(grid, self.detail.view, self.detail.box)
        length = scale_bar_length(width)
        right, up = screen_axes(self.detail.view, self.viewer.scene.view.convention)
        low, high = np.asarray(field[0]), np.asarray(field[1])
        centre = (low + high) / 2
        r, u = np.asarray(right), np.asarray(up)
        corner = centre - r * width / 2 - u * height / 2
        start = corner + r * width * BAR_INSET + u * height * BAR_INSET
        end = start + r * length
        axis = VIEW_AXIS[self.detail.view]
        start[axis] = end[axis] = self._scene.position[axis]
        self.bar.geometry = gfx.Geometry(
            positions=np.array([start, end], dtype=np.float32)
        )
        self.bar_text.set_text(f"{length:g} mm")
        self.bar_text.local.position = tuple(start + u * height * BAR_INSET / 2)


def _voxel(
    across: int, first: int, second: int, depth: float, i: float, j: float
) -> np.ndarray:
    """Returns a voxel point from its place along the two in-plane axes and across."""
    point = np.zeros(3)
    point[first], point[second], point[across] = i, j, depth
    return point


def render_detail(
    viewer: Viewer,
    detail: Detail,
    size: tuple[int, int],
    scale: float,
    transparent: bool = False,
) -> np.ndarray:
    """Draws a detail's picture off screen.

    Args:
        viewer: The viewer whose scene is drawn.
        detail: The detail window.
        size: The picture's width and height, logical pixels.
        scale: Pixels per logical pixel.
        transparent: Whether the background is transparent.

    Returns:
        The picture, rows from the top, (height, width, 4) uint8 RGBA.
    """
    canvas = image_canvas(size, scale)
    renderer = DetailRenderer(viewer, detail.id, canvas)
    straight_alpha(renderer.view.renderer)
    try:
        flush([canvas])
        return draw_view(renderer.view, transparent)
    finally:
        renderer.close()


def save_detail(
    viewer: Viewer, detail: Detail, path: Path, scale: float, transparent: bool
) -> Path:
    """Writes a detail's picture, `scale` times the window's, to an image file.

    Returns:
        The path written.

    Raises:
        ValueError: If the extension is not an image format, or is JPEG with a
            transparent background.
        OSError: If the file cannot be written.
    """
    check_image(path, transparent)
    image = render_detail(viewer, detail, detail.size, scale, transparent)
    Image.fromarray(image if transparent else image[..., :3].copy()).save(path)
    return path


def save_figure(
    viewer: Viewer,
    detail: Detail,
    path: Path,
    scale: float,
    background: FigureBackground,
) -> Path:
    """Writes a figure: the detail's view with its rectangle, and the detail beside.

    The view is drawn whole, at zoom 1, as high as the detail's picture; two lines
    join the rectangle's right corners to the detail's left ones.

    Returns:
        The path written.

    Raises:
        ValueError: If the extension is not an image format.
        OSError: If the file cannot be written.
    """
    check_image(path, False)
    width, height = detail.size
    with OffscreenRenderer(viewer, (height, height), scale) as parent:
        view = parent.views[detail.view]
        view.set_camera(camera=Camera2D())
        parent.scene_renderer.detail_boxes.show(False)
        parent.flush()
        whole = parent.render(detail.view, overlay=False)
        grid = viewer._reference_grid()
        assert grid is not None
        low, high = detail_field(
            grid, detail.view, detail.box, viewer.detail_position(detail.id)
        )
        corners = [view.to_screen(p) for p in _box_corners(low, high)]
    part = render_detail(viewer, detail, (width, height), scale)[..., :3]
    ink = DETAIL_COLOUR
    fill = (255, 255, 255) if background == FigureBackground.WHITE else (0, 0, 0)
    margin, gap = round(FIGURE_MARGIN * scale), round(FIGURE_GAP * scale)
    line = max(round(FIGURE_LINE * scale), 1)
    size = (
        margin * 2 + whole.shape[1] + gap + part.shape[1],
        margin * 2 + whole.shape[0],
    )
    figure = Image.new("RGB", size, fill)
    figure.paste(Image.fromarray(whole), (margin, margin))
    left = margin + whole.shape[1] + gap
    figure.paste(Image.fromarray(part), (left, margin))
    draw = ImageDraw.Draw(figure)
    xs = [margin + x * scale for x, _ in corners]
    ys = [margin + y * scale for _, y in corners]
    box = (min(xs), min(ys), max(xs), max(ys))
    draw.rectangle(box, outline=ink, width=line)
    inset = (left, margin, left + part.shape[1] - 1, margin + part.shape[0] - 1)
    draw.rectangle(inset, outline=ink, width=line)
    draw.line([(box[2], box[1]), (inset[0], inset[1])], fill=ink, width=line)
    draw.line([(box[2], box[3]), (inset[0], inset[3])], fill=ink, width=line)
    figure.save(path)
    return path


def _box_corners(low: Vec3, high: Vec3) -> list[np.ndarray]:
    """Returns the eight corners of a world box."""
    return [
        np.array([(low, high)[i][0], (low, high)[j][1], (low, high)[k][2]])
        for i in (0, 1)
        for j in (0, 1)
        for k in (0, 1)
    ]
