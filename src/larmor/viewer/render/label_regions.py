"""Draws the label regions: their voxels on the slices and their surfaces in 3D."""

import colorsys

import numpy as np
import pygfx as gfx

from larmor.types import Volume
from larmor.viewer.model.detail import VIEW_AXIS
from larmor.viewer.model.events import (
    DataChanged,
    Event,
    ItemAdded,
    ItemRemoved,
    OptionChanged,
    ViewStateChanged,
)
from larmor.viewer.model.item import Category, Item, ItemId, ViewKind
from larmor.viewer.model.label_regions import slice_cells
from larmor.viewer.model.options import LabelMode, RegionOptions, RegionShape
from larmor.viewer.model.view_state import ViewField
from larmor.viewer.model.viewer import Viewer
from larmor.viewer.render.labels_3d import style_opacity
from larmor.viewer.render.surfaces import LabelMesh, mask_surface
from larmor.viewer.render.view import ViewRenderer
from larmor.viewer.render.view_3d import ORDER_SURFACE
from larmor.viewer.render.worker import Worker

LINE_WIDTH = 1.5  # px, as the other regions' outlines
SELECTED_WIDTH = 2.5  # px, the selected region's outline
FILL_SHARE = 0.35  # the cells' opacity, a share of the region's 2D opacity
# The view-state fields whose change moves the slices' drawing.
FOLLOWED = (ViewField.CURSOR, ViewField.SELECTED, ViewField.CONVENTION)
# What a region's surfaces are made from: the parcellation, its version and the
# two sets.
type SurfaceKey = tuple[ItemId, int, tuple[int, ...], tuple[int, ...]]


class LabelRegionGuides:
    """Draws each visible label region: its cells on every slice, surfaces in 3D.

    Set A is drawn in the region's colour and set B in a lighter shade of it.
    The surfaces are made in the background and kept until the sets change.
    """

    def __init__(self, viewer: Viewer, views: dict[ViewKind, ViewRenderer]) -> None:
        """Adds a group to each 2D view's guides; the 3D view's waits for a region."""
        self.viewer = viewer
        self.views = views
        self.groups: dict[ViewKind, gfx.Group] = {}
        for kind, view in views.items():
            group = gfx.Group()
            if kind in VIEW_AXIS:
                view.guides.add(group)
            self.groups[kind] = group
        # Each parcellation's labels, by item id with the volume they came from.
        self._labels: dict[ItemId, tuple[Volume, np.ndarray]] = {}
        # Each region's surfaces, for the sets they were made of.
        self._surfaces: dict[ItemId, tuple[SurfaceKey, list[LabelMesh | None]]] = {}
        self._asked: dict[ItemId, SurfaceKey] = {}
        self._versions: dict[ItemId, int] = {}
        # One worker per region, so that one region's request does not replace
        # another's.
        self._workers: dict[ItemId, Worker] = {}
        self._drawing = False
        self._draw()

    def on_event(self, event: Event) -> None:
        """Draws again when a region or parcellation changes, or the slices do."""
        if isinstance(event, DataChanged):
            self._labels.pop(event.item.id, None)
            self._versions[event.item.id] = self._versions.get(event.item.id, 0) + 1
            redraw = bool(self._regions())
        elif isinstance(event, (ItemAdded, ItemRemoved, OptionChanged)):
            redraw = event.item.category == Category.REGIONS
            if isinstance(event, ItemRemoved):
                self._labels.pop(event.item.id, None)
                self._surfaces.pop(event.item.id, None)
                self._asked.pop(event.item.id, None)
                worker = self._workers.pop(event.item.id, None)
                if worker is not None:
                    worker.close()
        elif isinstance(event, ViewStateChanged):
            redraw = event.name in FOLLOWED and bool(self._regions())
        else:
            redraw = False
        if redraw:
            self._draw()

    def close(self) -> None:
        """Removes the drawings from the views and drops surfaces being made."""
        for worker in self._workers.values():
            worker.close()
        for kind, group in self.groups.items():
            view = self.views[kind]
            if kind in VIEW_AXIS:
                view.guides.remove(group)
            elif group.parent is not None:
                view.remove(group)
            view.request_draw()
        self.groups.clear()

    def _regions(self) -> list[Item]:
        """Returns the visible label regions that name a parcellation."""
        return [
            r
            for r in self.viewer.scene.in_category(Category.REGIONS)
            if isinstance(r.options, RegionOptions)
            and r.options.visible
            and r.options.shape == RegionShape.LABELS
            and r.options.parcellation is not None
        ]

    def _parcellation(self, item_id: ItemId) -> tuple[Volume, np.ndarray]:
        """Returns a parcellation's volume and labels, found once per data."""
        from larmor.tractography import parcellation_labels

        volume = self.viewer.scene.item(item_id).data
        assert isinstance(volume, Volume)
        known = self._labels.get(item_id)
        if known is None or known[0] is not volume:
            known = (volume, parcellation_labels(volume))
            self._labels[item_id] = known
        return known

    def _draw(self) -> None:
        """Draws every label region again in every view."""
        self._drawing = True
        try:
            self._draw_regions()
        finally:
            self._drawing = False

    def _draw_regions(self) -> None:
        """Draws every label region again in every view, as `_draw`."""
        selected = self.viewer.scene.view.selected
        cursor = self.viewer.scene.view.cursor
        regions = self._regions()
        for kind, group in self.groups.items():
            group.clear()
            for region in regions:
                o = region.options
                assert isinstance(o, RegionOptions) and o.parcellation is not None
                volume, labels = self._parcellation(o.parcellation)
                sets = _sets(o)
                if kind in VIEW_AXIS:
                    axis = VIEW_AXIS[kind]
                    for chosen, colour in zip(sets, _colours(o.colour), strict=False):
                        cut = slice_cells(
                            labels, volume.affine, chosen, axis, cursor[axis]
                        )
                        group.add(
                            *_cut(
                                cut.cells, cut.edges, o, colour, region.id == selected
                            )
                        )
                else:
                    self._add_surfaces(group, region, o, volume, labels, sets)
            if kind not in VIEW_AXIS:
                if group.children and group.parent is None:
                    self.views[kind].add(group)
                elif not group.children and group.parent is not None:
                    self.views[kind].remove(group)
            self.views[kind].request_draw()

    def _add_surfaces(
        self,
        group: gfx.Group,
        region: Item,
        o: RegionOptions,
        volume: Volume,
        labels: np.ndarray,
        sets: tuple[tuple[int, ...], ...],
    ) -> None:
        """Adds a region's surfaces to the 3D group, asking for them if not made."""
        assert o.parcellation is not None
        a, b = sets[0], sets[1] if len(sets) > 1 else ()
        key: SurfaceKey = (o.parcellation, self._versions.get(o.parcellation, 0), a, b)
        made = self._surfaces.get(region.id)
        if (made is None or made[0] != key) and self._asked.get(region.id) != key:
            self._ask(region.id, key, labels)
        made = self._surfaces.get(region.id)
        if made is not None:
            # Old surfaces stay shown until the new ones are made.
            for surface, colour in zip(made[1], _colours(o.colour), strict=False):
                if surface is not None:
                    group.add(_mesh(surface, volume.affine, colour, o.opacity_3d))

    def _ask(self, region_id: ItemId, key: SurfaceKey, labels: np.ndarray) -> None:
        """Makes a region's surfaces in the background, then draws again."""
        three_d = [v for k, v in self.views.items() if k not in VIEW_AXIS]
        if not three_d:
            return
        self._asked[region_id] = key
        _, _, a, b = key

        def work() -> list[LabelMesh | None]:
            """Finds the surface of each set's voxels."""
            return [
                mask_surface(np.isin(labels, chosen).astype(np.float32), 0.5)
                if chosen
                else None
                for chosen in (a, b)
            ]

        def done(surfaces: list[LabelMesh | None]) -> None:
            """Keeps the surfaces and draws them, unless already drawing."""
            self._surfaces[region_id] = (key, surfaces)
            if not self._drawing:
                self._draw()

        worker = self._workers.get(region_id)
        if worker is None:
            worker = Worker(three_d[0].canvas)
            self._workers[region_id] = worker
        worker.submit(work, done)


def _sets(o: RegionOptions) -> tuple[tuple[int, ...], ...]:
    """Returns the sets a region draws: A and B, or A alone when it stays within."""
    if o.label_mode == LabelMode.STAY_WITHIN:
        return (o.labels_a,)
    return (o.labels_a, o.labels_b)


def _colours(colour: str) -> tuple[str, str]:
    """Returns the colours of set A and set B: the colour and a lighter shade."""
    r, g, b = (int(colour[i : i + 2], 16) / 255 for i in (1, 3, 5))
    h, lightness, s = colorsys.rgb_to_hls(r, g, b)
    lighter = colorsys.hls_to_rgb(h, lightness + (1 - lightness) * 0.55, s)
    return colour, "#" + "".join(f"{round(c * 255):02x}" for c in lighter)


def _cut(
    cells: np.ndarray,
    edges: np.ndarray,
    o: RegionOptions,
    colour: str,
    chosen: bool,
) -> list[gfx.WorldObject]:
    """Returns the filled cells and their outline on a slice."""
    objects: list[gfx.WorldObject] = []
    if len(cells):
        n = len(cells)
        quads = np.arange(4 * n, dtype=np.int32).reshape(n, 4)
        faces = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
        fill = gfx.MeshBasicMaterial(
            color=colour,
            opacity=o.opacity_2d * FILL_SHARE,
            side="both",
            depth_test=False,
        )
        fill.alpha_mode = "blend"
        geometry = gfx.Geometry(
            positions=cells.reshape(-1, 3).astype(np.float32), indices=faces
        )
        objects.append(gfx.Mesh(geometry, fill))
    if len(edges):
        line = gfx.LineSegmentMaterial(
            thickness=SELECTED_WIDTH if chosen else LINE_WIDTH,
            color=colour,
            opacity=o.opacity_2d,
            depth_test=False,
        )
        line.alpha_mode = "solid" if o.opacity_2d >= 1 else "blend"
        positions = edges.reshape(-1, 3).astype(np.float32)
        objects.append(gfx.Line(gfx.Geometry(positions=positions), line))
    return objects


def _mesh(
    surface: LabelMesh, affine: np.ndarray, colour: str, opacity: float
) -> gfx.Mesh:
    """Returns a set's surface as a translucent mesh in world coordinates."""
    geometry = gfx.Geometry(
        positions=surface.vertices, indices=surface.faces, normals=surface.normals
    )
    material = gfx.MeshPhongMaterial(color=colour, side="both")
    style_opacity(material, opacity)
    mesh = gfx.Mesh(geometry, material)
    mesh.render_order = ORDER_SURFACE
    mesh.local.matrix = affine
    return mesh
