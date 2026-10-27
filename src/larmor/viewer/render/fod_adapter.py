"""Draws FODs as glyphs on the current slices of the 2D views."""

import threading
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import numpy as np
import pygfx as gfx

from larmor.types import Volume
from larmor.viewer.model.camera import view_directions
from larmor.viewer.model.events import Event, OptionChanged, ViewStateChanged
from larmor.viewer.model.item import Item, ViewKind
from larmor.viewer.model.options import (
    Colouring,
    FodOptions,
    GlyphScaling,
    OptionName,
)
from larmor.viewer.model.view_state import ViewField
from larmor.viewer.model.viewer import Viewer
from larmor.viewer.render.adapter import Adapter
from larmor.viewer.render.geometry import slice_index
from larmor.viewer.render.glyphs import (
    Glyphs,
    empty_glyphs,
    image_amplitude,
    repeated,
    slice_glyphs,
    sphere,
)
from larmor.viewer.render.tractogram_adapter import QUEUE_2D
from larmor.viewer.render.view import Vec3, ViewRenderer
from larmor.viewer.render.volume_3d import PLANES
from larmor.viewer.render.worker import Worker

# Options whose change makes the glyphs again; the others change the material.
REBUILDING = (
    OptionName.BASIS,
    OptionName.SCALE,
    OptionName.SCALING,
    OptionName.THRESHOLD,
    OptionName.ANISOTROPY,
    OptionName.DETAIL,
)
# A thread per 2D view, so the three views' glyphs are made together.
_POOL = ThreadPoolExecutor(max_workers=len(PLANES), thread_name_prefix="larmor-fod")

# The voxels glyphs are built for along a slice's two other axes, lowest first,
# as `slice_glyphs` takes them; None for the whole slice.
type Crop = tuple[tuple[int, int], tuple[int, int]] | None
# What a view's glyphs are made from, the slice (voxel axis and index, None outside
# the grid), the options and the crop.
type Key = tuple[
    tuple[int, int] | None, float, GlyphScaling, float, float, str, int, Crop
]


class FodAdapter(Adapter):
    """Draws the glyphs of the FOD's slice nearest the cursor, one mesh per 2D view.

    A view's mesh is made again in the background only when its slice or a glyph
    option changes. A hidden FOD makes nothing until it is shown. The meshes are
    moved towards the camera, so the depth test keeps them over the slices.
    """

    def __init__(
        self,
        item: Item,
        views: dict[ViewKind, ViewRenderer],
        viewer: Viewer,
        order: float,
        box: tuple[Vec3, Vec3] | None = None,
        detail: int | None = None,
    ) -> None:
        """Adds an empty mesh to each 2D view and asks for its glyphs.

        With box, a world box (lowest and highest corner, mm), only the glyphs in
        it are built, as a detail window needs; detail replaces the item's
        sphere detail.
        """
        super().__init__(item, views, viewer, order)
        assert isinstance(item.data, Volume)
        self.box = box
        self.detail = detail
        self.meshes: dict[ViewKind, gfx.Mesh] = {}
        self._workers: dict[ViewKind, Worker] = {}
        self._asked: dict[ViewKind, Key | None] = {}  # the glyphs last asked for
        self._counts: dict[ViewKind, int] = {}  # of the glyphs drawn
        # Per view, the sphere detail, the glyphs they hold, colours and faces.
        self._repeated: dict[ViewKind, tuple[int, int, gfx.Buffer, gfx.Buffer]] = {}
        # The image's high amplitude, by basis and sphere detail.
        self._amplitudes: dict[tuple[str, int], float] = {}
        self._amplitudes_lock = threading.Lock()
        for kind in PLANES:
            if kind not in views:
                continue
            material = glyph_material()
            material.render_queue = QUEUE_2D  # the slices' queue, for Layers order
            mesh = gfx.Mesh(empty_geometry(), material)
            mesh.render_order = order
            mesh.visible = False
            self.meshes[kind] = mesh
            self._workers[kind] = Worker(views[kind].canvas, _POOL)
            self._asked[kind] = None
            self._counts[kind] = 0
            views[kind].add(mesh)
        self._place()
        self._show()
        self._request()

    # Events

    def on_event(self, event: Event) -> None:
        """Follows this item's options, the cursor and the convention."""
        if isinstance(event, OptionChanged) and event.item.id == self.item.id:
            if event.name in (OptionName.SCALE, OptionName.VISIBLE):
                self._place()
            if event.name in REBUILDING or event.name == OptionName.VISIBLE:
                self._request()
            self._show()
        elif isinstance(event, ViewStateChanged):
            if event.name == ViewField.CURSOR:
                self._request()  # only the views whose slice changed
            elif event.name == ViewField.CONVENTION:
                self._place()

    def set_box(self, box: tuple[Vec3, Vec3] | None, detail: int | None) -> None:
        """Builds the glyphs in another world box, at another sphere detail."""
        self.box, self.detail = box, detail
        self._request()

    def close(self) -> None:
        """Removes the meshes and drops the work in progress."""
        for kind, mesh in self.meshes.items():
            self._workers[kind].close()
            self.views[kind].remove(mesh)

    # Drawing

    def _options(self) -> FodOptions:
        """Returns the item's FOD options."""
        o = self.item.options
        assert isinstance(o, FodOptions)
        return o

    def _show(self) -> None:
        """Applies the colour, opacity and visibility to the meshes."""
        o = self._options()
        for kind, mesh in self.meshes.items():
            material = mesh.material
            if o.colouring == Colouring.SINGLE:
                material.color_mode = "uniform"
                material.color = o.colour
            else:
                material.color_mode = "vertex"
            material.opacity = o.opacity_2d
            material.alpha_mode = "solid" if o.opacity_2d >= 1 else "blend"
            # Written while blending too, so a glyph's front hides its back.
            material.depth_write = True
            mesh.visible = o.visible and self._counts[kind] > 0
            self.views[kind].request_draw()

    def _place(self) -> None:
        """Moves each mesh towards its view's camera, clear of the slice's plane."""
        volume = self.item.data
        assert isinstance(volume, Volume)
        voxel = float(np.linalg.norm(volume.affine[:3, :3], axis=0).max())
        # Further than the largest glyph reaches out of the plane.
        distance = (self._options().scale + 1.0) * voxel
        convention = self.viewer.scene.view.convention
        for kind, mesh in self.meshes.items():
            forward, _ = view_directions(kind, convention)
            mesh.local.position = tuple(-distance * np.asarray(forward))
            self.views[kind].request_draw()

    def _key(self, kind: ViewKind) -> Key:
        """Returns what a view's glyphs are made from now."""
        volume = self.item.data
        assert isinstance(volume, Volume)
        cursor = self.viewer.scene.view.cursor
        where = slice_index(volume.affine, volume.grid.shape, PLANES[kind], cursor)
        o = self._options()
        crop = None
        if self.box is not None and where is not None:
            crop = _crop(volume.affine, where[0], self.box)
        return (
            where,
            o.scale,
            o.scaling,
            o.threshold,
            o.anisotropy,
            o.basis,
            o.detail if self.detail is None else self.detail,
            crop,
        )

    def _request(self) -> None:
        """Asks for the glyphs of each view whose slice or options changed."""
        if not self._options().visible:
            return
        for kind, worker in self._workers.items():
            key = self._key(kind)
            if key == self._asked[kind]:
                continue
            self._asked[kind] = key
            # The cursor's plane when asked; a later move within the slice stays
            # within the mesh's shift towards the camera.
            axis = PLANES[kind]
            plane = (axis, float(self.viewer.scene.view.cursor[axis]))
            worker.submit(
                partial(self._build, key, plane),
                partial(self._take, kind, key[6]),
                partial(self._failed, kind, key),
            )

    def _build(self, key: Key, plane: tuple[int, float]) -> Glyphs:
        """Makes a slice's glyphs; runs in the background."""
        where, scale, scaling, threshold, anisotropy, basis, detail, crop = key
        volume = self.item.data
        assert isinstance(volume, Volume)
        if where is None:
            return empty_glyphs()
        largest = None
        if scaling == GlyphScaling.IMAGE or threshold > 0:
            largest = self._image_amplitude(basis, detail)
        voxel = float(np.linalg.norm(volume.affine[:3, :3], axis=0).min())
        return slice_glyphs(
            volume,
            *where,
            basis,
            detail,
            radius=scale * voxel / 2,
            threshold=threshold * largest if largest is not None else 0.0,
            image_amplitude=largest if scaling == GlyphScaling.IMAGE else None,
            anisotropy=anisotropy,
            cursor_plane=plane,
            crop=crop,
        )

    def _image_amplitude(self, basis: str, detail: int) -> float:
        """Returns the image's high amplitude, computed once per basis and detail."""
        with self._amplitudes_lock:
            if (basis, detail) not in self._amplitudes:
                volume = self.item.data
                assert isinstance(volume, Volume)
                self._amplitudes[basis, detail] = image_amplitude(volume, basis, detail)
            return self._amplitudes[basis, detail]

    def _take(self, kind: ViewKind, detail: int, glyphs: Glyphs) -> None:
        """Puts a view's new glyphs in its mesh."""
        mesh = self.meshes[kind]
        if glyphs.count == 0:
            mesh.geometry = empty_geometry()
        else:
            colours, faces = self._repeated_buffers(kind, detail, glyphs.count)
            mesh.geometry = gfx.Geometry(
                positions=glyphs.positions,
                normals=glyphs.normals,
                colors=colours,
                indices=faces,
            )
        self._counts[kind] = glyphs.count
        mesh.visible = self._options().visible and glyphs.count > 0
        self.views[kind].request_draw()

    def _repeated_buffers(
        self, kind: ViewKind, detail: int, n: int
    ) -> tuple[gfx.Buffer, gfx.Buffer]:
        """Returns a view's colour and face buffers, uploaded again only when they grow.

        A bigger pair serves fewer glyphs, as the faces' draw range picks the first n.
        """
        held = self._repeated.get(kind)
        if held is None or held[0] != detail or held[1] < n:
            capacity = n * 5 // 4  # room for a slice somewhat larger
            colours, faces = repeated(detail, capacity)
            held = (detail, capacity, gfx.Buffer(colours), gfx.Buffer(faces))
            self._repeated[kind] = held
        _, _, colours_buffer, faces_buffer = held
        faces_buffer.draw_range = (0, n * len(sphere(detail).faces))
        return colours_buffer, faces_buffer

    def _failed(self, kind: ViewKind, key: Key) -> None:
        """Forgets a failed request, so the glyphs are asked for again."""
        if self._asked[kind] == key:
            self._asked[kind] = None  # asked again next time


def _crop(affine: np.ndarray, axis: int, box: tuple[Vec3, Vec3]) -> Crop:
    """Returns the voxels of a grid within a world box, along the two axes but axis.

    A voxel counts when its centre is inside the box.
    """
    inverse = np.linalg.inv(affine)
    corners = np.array(
        [
            [box[i][0], box[j][1], box[k][2]]
            for i in (0, 1)
            for j in (0, 1)
            for k in (0, 1)
        ]
    )
    voxels = corners @ inverse[:3, :3].T + inverse[:3, 3]
    first, second = (a for a in range(3) if a != axis)
    low = np.ceil(voxels.min(axis=0) - 1e-6).astype(int)
    high = np.floor(voxels.max(axis=0) + 1e-6).astype(int) + 1
    return (
        (int(low[first]), int(high[first])),
        (int(low[second]), int(high[second])),
    )


def empty_geometry() -> gfx.Geometry:
    """Returns an empty mesh's geometry, one degenerate triangle as pygfx needs one."""
    point = np.zeros((3, 3), np.float32)
    return gfx.Geometry(
        positions=point,
        normals=np.tile(np.array([0, 0, 1], np.float32), (3, 1)),
        colors=point,
        indices=np.array([[0, 1, 2]], np.uint32),
    )


def glyph_material() -> gfx.MeshPhongMaterial:
    """Returns the glyphs' material, vertex colours with a clear highlight."""
    return gfx.MeshPhongMaterial(
        color_mode="vertex", shininess=40, specular="#bbbbbb", pick_write=False
    )
