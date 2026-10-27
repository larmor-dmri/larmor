"""Draws surfaces as a lit mesh in 3D, and as cut lines in the 2D views."""

import threading
from functools import partial

import numpy as np
import pygfx as gfx

from larmor.surfaces import vertex_normals
from larmor.viewer.model.events import Event, OptionChanged, ViewStateChanged
from larmor.viewer.model.item import Item, ViewKind
from larmor.viewer.model.options import Colormap, Colouring, SurfaceOptions
from larmor.viewer.model.surface import SurfaceData
from larmor.viewer.model.view_state import ViewField
from larmor.viewer.model.viewer import Viewer
from larmor.viewer.render.adapter import Adapter
from larmor.viewer.render.colour_space import physical_colours
from larmor.viewer.render.geometry import (
    PlaneCut,
    colormap_colours,
    normal_lines,
    normal_subset,
    plane_cut,
    sample_cut,
)
from larmor.viewer.render.labels_3d import style_opacity
from larmor.viewer.render.tractogram_adapter import QUEUE_2D, SLAB_AXIS
from larmor.viewer.render.view import ViewRenderer
from larmor.viewer.render.view_3d import ORDER_SURFACE
from larmor.viewer.render.volume_adapter import COLORMAP_MAPS
from larmor.viewer.render.worker import Worker

# What a view's cut is made from, the plane's position and whether the wall is hidden.
type CutKey = tuple[float, bool]
# What the colours per vertex are made from, the values' name, colormap and range.
type ColourKey = tuple[str, Colormap, float, float]
# What the normals in 3D are made from, their count, length and whether the wall is
# hidden.
type NormalsKey = tuple[int, float, bool]
# What a view's normals are made from, its cut, the spacing and the length.
type CutNormalsKey = tuple[PlaneCut, float, float]

NORMALS_WIDTH = 1.5  # px, the normals' lines in 3D


class SurfaceAdapter(Adapter):
    """Draws a mesh in the 3D view, and in each 2D view the line its plane cuts.

    The mesh's vertex normals are computed once, in the background, when it is
    first shown, and shared by the mesh and the normals drawn in 3D and along the
    cuts. A 2D view's cut is made again in the background only when its plane
    moves or the medial wall is hidden or shown.
    """

    def __init__(
        self,
        item: Item,
        views: dict[ViewKind, ViewRenderer],
        viewer: Viewer,
        order: float,
    ) -> None:
        """Adds an empty line to each 2D view; the mesh comes once it is shown."""
        super().__init__(item, views, viewer, order)
        assert isinstance(item.data, SurfaceData)
        self.mesh: gfx.Mesh | None = None
        self._mesh_asked = False
        self._mesh_worker: Worker | None = None
        self._faces: dict[bool, np.ndarray] = {}  # the faces drawn, by wall hidden
        self._indices: dict[bool, gfx.Buffer] = {}  # likewise, for the mesh
        self._colours: tuple[ColourKey, np.ndarray] | None = None
        self._mesh_colours: np.ndarray | None = None  # the colours in its buffer
        self.cuts: dict[ViewKind, PlaneCut | None] = {}
        # The cut and colours each view's line was made from, compared by identity.
        self._drawn: dict[ViewKind, tuple[PlaneCut | None, np.ndarray | None]] = {}
        self._asked: dict[ViewKind, CutKey | None] = {}
        self._workers: dict[ViewKind, Worker] = {}
        self.lines: dict[ViewKind, gfx.Line] = {}
        self._normals: np.ndarray | None = None  # per vertex, once computed
        self._normals_lock = threading.Lock()  # the workers share the computation
        self.normals_3d: gfx.Line | None = None
        self._normals_asked: NormalsKey | None = None
        self._normals_worker: Worker | None = None
        self.cut_normals: dict[ViewKind, gfx.Line] = {}
        self._cut_normals_asked: dict[ViewKind, CutNormalsKey | None] = {}
        self._cut_normals_workers: dict[ViewKind, Worker] = {}
        for kind, view in views.items():
            if kind not in SLAB_AXIS:
                continue
            self.lines[kind] = self._add_2d_line(view, order)
            self.cut_normals[kind] = self._add_2d_line(view, order)
            self.cuts[kind] = None
            self._drawn[kind] = (None, None)
            self._asked[kind] = None
            self._cut_normals_asked[kind] = None
            self._workers[kind] = Worker(view.canvas)
            self._cut_normals_workers[kind] = Worker(view.canvas)
        if ViewKind.THREE_D in views:
            self._mesh_worker = Worker(views[ViewKind.THREE_D].canvas)
            self._normals_worker = Worker(views[ViewKind.THREE_D].canvas)
        self._show()

    def _add_2d_line(self, view: ViewRenderer, order: float) -> gfx.Line:
        """Adds an empty set of segments to a 2D view, above the slices."""
        material = gfx.LineSegmentMaterial(thickness_space="screen", depth_test=False)
        material.render_queue = QUEUE_2D
        obj = gfx.Line(_line_geometry(None, None), material)
        obj.render_order = order
        view.add(obj)
        return obj

    def on_event(self, event: Event) -> None:
        """Follows this item's options and the cursor."""
        if isinstance(event, OptionChanged) and event.item.id == self.item.id:
            self._show()
        elif isinstance(event, ViewStateChanged) and event.name == ViewField.CURSOR:
            self._request_cuts()

    def close(self) -> None:
        """Removes the mesh and the lines, and drops the work in progress."""
        for kind, obj in self.lines.items():
            self._workers[kind].close()
            self._cut_normals_workers[kind].close()
            self.views[kind].remove(obj)
            self.views[kind].remove(self.cut_normals[kind])
        for worker in (self._mesh_worker, self._normals_worker):
            if worker is not None:
                worker.close()
        if self.mesh is not None:
            self.views[ViewKind.THREE_D].remove(self.mesh)
        if self.normals_3d is not None:
            self.views[ViewKind.THREE_D].remove(self.normals_3d)

    def _options(self) -> SurfaceOptions:
        """Returns the item's surface options."""
        o = self.item.options
        assert isinstance(o, SurfaceOptions)
        return o

    def _data(self) -> SurfaceData:
        """Returns the item's surface data."""
        data = self.item.data
        assert isinstance(data, SurfaceData)
        return data

    def _hide_wall(self) -> bool:
        """Returns whether the medial wall is left out (asked, and the cortex known)."""
        return self._options().hide_medial_wall and self._data().cortex is not None

    def _drawn_faces(self, hide: bool) -> np.ndarray:
        """Returns the faces drawn, all or those with every vertex on the cortex."""
        if hide not in self._faces:
            data = self._data()
            faces = data.surface.faces
            if hide:
                assert data.cortex is not None
                faces = np.ascontiguousarray(faces[data.cortex[faces].all(axis=1)])
            self._faces[hide] = faces
        return self._faces[hide]

    def _vertex_normals(self) -> np.ndarray:
        """Returns each vertex's unit normal, computed by the first worker to ask."""
        with self._normals_lock:
            if self._normals is None:
                self._normals = vertex_normals(self._data().surface)
            return self._normals

    def _vertex_colours(self) -> np.ndarray | None:
        """Returns the colour of each vertex by its values; None for the flat colour."""
        o = self._options()
        if o.colour_by is None:
            return None
        key: ColourKey = (o.colour_by, o.colormap, o.minimum, o.maximum)
        if self._colours is None or self._colours[0] != key:
            values = self._data().surface.vertex_data[o.colour_by]
            table = COLORMAP_MAPS[o.colormap].texture.data
            self._colours = (key, colormap_colours(values, table, o.minimum, o.maximum))
        return self._colours[1]

    # Drawing

    def _show(self) -> None:
        """Shows the options on the mesh and the lines, and asks for what is missing."""
        o = self._options()
        colours = self._vertex_colours()
        flat = gfx.Color(o.colour).rgb
        if self.mesh is not None:
            self._show_mesh(colours)
        elif o.visible:
            self._request_mesh()
        for kind, obj in self.lines.items():
            material = obj.material
            material.color_mode = "uniform" if colours is None else "vertex"
            material.color = flat if colours is None else (1.0, 1.0, 1.0)
            material.thickness = o.line_width
            material.opacity = o.opacity_2d
            material.alpha_mode = "solid" if o.opacity_2d >= 1 else "blend"
            obj.visible = o.visible and o.cut_lines
            self._colour_cut(kind, colours)
            normals = self.cut_normals[kind]
            _style_normals(normals.material, o)
            normals.material.thickness = o.line_width
            normals.visible = o.visible and o.cut_lines and o.show_normals
            self.views[kind].request_draw()
        if self.normals_3d is not None:
            _style_normals(self.normals_3d.material, o)
            self.normals_3d.visible = o.visible and o.show_normals
            self.views[ViewKind.THREE_D].request_draw()
        self._request_cuts()
        self._request_normals()
        for kind in self.cut_normals:
            self._request_cut_normals(kind)

    def _show_mesh(self, colours: np.ndarray | None) -> None:
        """Sets the mesh's colours, faces, opacity and visibility."""
        assert self.mesh is not None
        o = self._options()
        material = self.mesh.material
        geometry = self.mesh.geometry
        if colours is None:
            material.color_mode = "uniform"
            material.color = gfx.Color(o.colour).rgb
        else:
            material.color_mode = "vertex"
            material.color = (1.0, 1.0, 1.0)
            if self._mesh_colours is not colours:
                self._mesh_colours = colours
                geometry.colors.data[:] = physical_colours(colours)  # linear
                geometry.colors.update_full()
        hide = self._hide_wall()
        if hide not in self._indices:
            self._indices[hide] = gfx.Buffer(self._drawn_faces(hide))
        if geometry.indices is not self._indices[hide]:
            geometry.indices = self._indices[hide]
        style_opacity(material, o.opacity_3d)
        self.mesh.visible = o.visible
        self.views[ViewKind.THREE_D].request_draw()

    def _request_mesh(self) -> None:
        """Asks for the mesh's normals, once."""
        if self._mesh_asked or self._mesh_worker is None:
            return
        self._mesh_asked = True
        self._mesh_worker.submit(
            self._vertex_normals,
            self._take_mesh,
            self._mesh_failed,
        )

    def _take_mesh(self, normals: np.ndarray) -> None:
        """Adds the mesh to the 3D view, with its normals."""
        surface = self._data().surface
        geometry = gfx.Geometry(
            positions=surface.vertices,
            normals=normals,
            colors=np.ones((surface.num_vertices, 3), np.float32),
            indices=surface.faces,
        )
        self._indices[False] = geometry.indices
        # Pickable, so a click in 3D finds the point on it.
        material = gfx.MeshPhongMaterial(side="both", pick_write=True)
        self.mesh = gfx.Mesh(geometry, material)
        self.mesh.render_order = ORDER_SURFACE
        self.views[ViewKind.THREE_D].add(self.mesh)
        self._show_mesh(self._vertex_colours())

    def _mesh_failed(self) -> None:
        """Forgets a failed request, so the mesh is asked for again."""
        self._mesh_asked = False  # asked again next time

    def _request_cuts(self) -> None:
        """Asks for the cut of each 2D view whose plane or faces changed, if shown."""
        o = self._options()
        if not (o.visible and o.cut_lines):
            return
        hide = self._hide_wall()
        cursor = self.viewer.scene.view.cursor
        for kind in self.lines:
            axis = SLAB_AXIS[kind]
            key: CutKey = (float(cursor[axis]), hide)
            if key == self._asked[kind]:
                continue
            self._asked[kind] = key
            self._workers[kind].submit(
                partial(self._cut, axis, key),
                partial(self._take_cut, kind),
                partial(self._cut_failed, kind),
            )

    def _cut(self, axis: int, key: CutKey) -> PlaneCut:
        """Returns one view's cut; runs on the worker."""
        position, hide = key
        vertices = self._data().surface.vertices
        return plane_cut(vertices, self._drawn_faces(hide), axis, position)

    def _take_cut(self, kind: ViewKind, cut: PlaneCut) -> None:
        """Draws a view's new cut and asks for its normals."""
        self.cuts[kind] = cut
        self._colour_cut(kind, self._vertex_colours())
        self._request_cut_normals(kind)
        self.views[kind].request_draw()

    def _colour_cut(self, kind: ViewKind, colours: np.ndarray | None) -> None:
        """Gives a view's line its cut, coloured by the vertices' colours if any."""
        cut = self.cuts[kind]
        drawn_cut, drawn_colours = self._drawn[kind]
        if drawn_cut is not cut or drawn_colours is not colours:
            self._drawn[kind] = (cut, colours)
            self.lines[kind].geometry = _line_geometry(cut, colours)

    def _cut_failed(self, kind: ViewKind) -> None:
        """Forgets a failed request, so the cut is asked for again."""
        self._asked[kind] = None  # asked again next time

    # Normals

    def _request_normals(self) -> None:
        """Asks for the normals in 3D, if shown and changed."""
        o = self._options()
        if self._normals_worker is None or not (o.visible and o.show_normals):
            return
        key: NormalsKey = (o.normals_count, o.normals_length, self._hide_wall())
        if key == self._normals_asked:
            return
        self._normals_asked = key
        self._normals_worker.submit(
            partial(self._normals_3d, key),
            self._take_normals,
            self._normals_failed,
        )

    def _normals_3d(self, key: NormalsKey) -> gfx.Geometry:
        """Returns the segments of the normals in 3D; runs on the worker."""
        count, length, hide = key
        data = self._data()
        vertices = data.surface.vertices
        chosen = normal_subset(len(vertices), count, data.cortex if hide else None)
        positions, colours = normal_lines(
            vertices[chosen], self._vertex_normals()[chosen], length
        )
        return _segments_geometry(positions, colours)

    def _take_normals(self, geometry: gfx.Geometry) -> None:
        """Draws the new normals in the 3D view."""
        if self.normals_3d is None:
            material = gfx.LineSegmentMaterial(thickness=NORMALS_WIDTH)
            self.normals_3d = gfx.Line(geometry, material)
            self.normals_3d.render_order = ORDER_SURFACE
            self.views[ViewKind.THREE_D].add(self.normals_3d)
        else:
            self.normals_3d.geometry = geometry
        o = self._options()
        _style_normals(self.normals_3d.material, o)
        self.normals_3d.visible = o.visible and o.show_normals
        self.views[ViewKind.THREE_D].request_draw()

    def _normals_failed(self) -> None:
        """Forgets a failed request, so the normals are asked for again."""
        self._normals_asked = None  # asked again next time

    def _request_cut_normals(self, kind: ViewKind) -> None:
        """Asks for the normals along a view's cut, if shown and changed."""
        o = self._options()
        cut = self.cuts[kind]
        if cut is None or not (o.visible and o.cut_lines and o.show_normals):
            return
        key: CutNormalsKey = (cut, o.normals_spacing, o.normals_length)
        asked = self._cut_normals_asked[kind]
        # The cut by identity: a new cut is a new object.
        if asked is not None and asked[0] is cut and asked[1:] == key[1:]:
            return
        if asked is None or asked[0] is not cut:
            # The normals drawn belong to another slice, so none until the new ones.
            self.cut_normals[kind].geometry = _line_geometry(None, None)
        self._cut_normals_asked[kind] = key
        self._cut_normals_workers[kind].submit(
            partial(self._cut_normals, SLAB_AXIS[kind], key),
            partial(self._take_cut_normals, kind),
            partial(self._cut_normals_failed, kind),
        )

    def _cut_normals(self, axis: int, key: CutNormalsKey) -> gfx.Geometry:
        """Returns the segments of the normals along a view's cut, on the worker."""
        cut, spacing, length = key
        points, normals = sample_cut(cut, spacing, self._vertex_normals())
        return _segments_geometry(*normal_lines(points, normals, length, axis))

    def _take_cut_normals(self, kind: ViewKind, geometry: gfx.Geometry) -> None:
        """Draws a view's new normals along its cut."""
        self.cut_normals[kind].geometry = geometry
        self.views[kind].request_draw()

    def _cut_normals_failed(self, kind: ViewKind) -> None:
        """Forgets a failed request, so the normals are asked for again."""
        self._cut_normals_asked[kind] = None  # asked again next time


def _style_normals(material: gfx.LineSegmentMaterial, o: SurfaceOptions) -> None:
    """Colours the normals by direction, from their geometry, or in one colour."""
    single = o.normals_colouring == Colouring.SINGLE
    material.color_mode = "uniform" if single else "vertex"
    material.color = gfx.Color(o.normals_colour).rgb if single else (1.0, 1.0, 1.0)


def _segments_geometry(positions: np.ndarray, colours: np.ndarray) -> gfx.Geometry:
    """Returns segments with colours at their ends; one empty pair if there are none."""
    if len(positions) == 0:
        positions = np.full((2, 3), np.nan, np.float32)
        colours = np.ones((2, 3), np.float32)
    return gfx.Geometry(positions=positions, colors=colours)


def _line_geometry(cut: PlaneCut | None, colours: np.ndarray | None) -> gfx.Geometry:
    """Returns a cut's segments with colours at their ends; one empty pair if none."""
    if cut is None or len(cut.positions) == 0:
        empty = np.full((2, 3), np.nan, np.float32)
        return gfx.Geometry(positions=empty, colors=np.ones((2, 3), np.float32))
    rgb = (
        np.ones((len(cut.positions), 3), np.float32)
        if colours is None
        else cut.interpolate(colours)
    )
    return gfx.Geometry(positions=cut.positions, colors=rgb)
