"""Draws an image in the 3D view as its slices, a volume rendering or its surfaces."""

from collections.abc import Mapping
from dataclasses import dataclass, field

import pygfx as gfx

from larmor.types import Label, Volume
from larmor.viewer.model.camera import Vec3
from larmor.viewer.model.item import Category, Item, ViewKind
from larmor.viewer.model.options import OverlayOptions, VolumeOptions
from larmor.viewer.render.labels_3d import Labels3D
from larmor.viewer.render.overlay_slice import OverlaySliceMaterial
from larmor.viewer.render.surfaces import LabelMesh, mask_surface
from larmor.viewer.render.tissue_slice import (
    NO_ZERO,
    TissueSliceMaterial,
    background_cutoff,
)
from larmor.viewer.render.view import ViewRenderer
from larmor.viewer.render.view_3d import ORDER_RENDERING, ORDER_SURFACE
from larmor.viewer.render.volume_blend import (
    Gradient,
    VolumeBlendMaterial,
    volume_gradient,
)
from larmor.viewer.render.worker import Worker

# Distance between samples along a ray, in voxels, at full quality and while the
# view is dragged.
STEP = 0.5
INTERACTIVE_STEP = 1.0

# The slice planes, by the 2D view whose slice each one is.
PLANES: dict[ViewKind, int] = {
    ViewKind.SAGITTAL: 0,
    ViewKind.CORONAL: 1,
    ViewKind.AXIAL: 2,
}


@dataclass(frozen=True)
class Look:
    """Holds how the image's slices look, as the 2D views draw them."""

    texture: gfx.Texture
    clim: tuple[float, float]
    cmap: gfx.TextureMap
    interpolation: str
    colour: tuple[float, float, float]  # the colormap's top, a surface's colour
    labels: bool = False  # shown as labels, a surface per chosen label
    table: Mapping[int, Label] = field(default_factory=dict)  # the labels' lookup table
    # DTI colours (RGBA), which the slices show in place of the texture's values,
    # with no volume rendering.
    dti: gfx.Texture | None = None


class Volume3D:
    """Draws one image in the 3D view.

    A volume shows its slices and a volume rendering; an overlay shows its slices
    and the surface where it crosses its threshold or, shown as labels, the
    surfaces of its chosen labels.
    """

    def __init__(
        self, item: Item, view: ViewRenderer, order: float, texture: gfx.Texture
    ) -> None:
        """Creates the objects on the image's texture; the surface comes later."""
        self.item = item
        self.view = view
        self.slices: dict[ViewKind, gfx.Volume] = {}
        self.rendering: gfx.Volume | None = None
        self.surface: gfx.Mesh | None = None
        # The volume index and level (stored units) of the last surface asked for.
        self._surface_key: tuple[int, float] | None = None
        self._mesh_key: tuple[int, float] | None = None  # the volume and level shown
        self._worker = Worker(view.canvas)
        # Computes the gradient of the shown texture, which lights the rendering.
        self._gradient_worker = Worker(view.canvas)
        self._gradient_of: gfx.Texture | None = None  # the texture last asked for
        self._colour = (1.0, 1.0, 1.0)
        # The background's cut-off of each volume of the image, by its index, found
        # when the volume is first shown; and the image they were found in.
        self._cutoffs: dict[int, float] = {}
        self._cutoffs_of: object = item.data
        self._labels = False
        self.labels: Labels3D | None = None
        volume = item.data
        assert isinstance(volume, Volume)
        overlay = item.category == Category.OVERLAYS
        for kind in PLANES:
            if overlay:
                # Drawn over the volume's slice, a little nearer the camera, with
                # nothing where the overlay is transparent.
                material = OverlaySliceMaterial()
                material.depth_compare = "<="
                material.alpha_test = 0.01
            else:
                # Nothing where a magnitude image is background, which would
                # otherwise hide the rendering as black planes.
                material = TissueSliceMaterial()
            material.pick_write = True  # a click in 3D finds the point on it
            obj = gfx.Volume(gfx.Geometry(grid=texture), material)
            obj.render_order = order
            obj.local.matrix = volume.affine
            self.slices[kind] = obj
        if overlay:
            self.labels = Labels3D(item, view)
        else:
            material = VolumeBlendMaterial()
            material.interpolation = "linear"
            self.rendering = gfx.Volume(gfx.Geometry(grid=texture), material)
            self.rendering.render_order = ORDER_RENDERING
            self.rendering.local.matrix = volume.affine
            view.canvas.add_event_handler(self._prepare, "before_draw")

    def show(self, look: Look, cursor: Vec3) -> None:
        """Follows the item's options, the slices as in 2D and the rest in 3D."""
        o = self.item.options
        assert isinstance(o, VolumeOptions)
        texture, clim, cmap = look.texture, look.clim, look.cmap
        cutoff = float("-inf")
        if look.dti is not None:
            texture, clim, cmap = look.dti, (0.0, 1.0), None
            cutoff = NO_ZERO  # black alone is their background
        elif self.rendering is not None:
            cutoff = self._background_cutoff(texture, o.volume_index)
        for kind, obj in self.slices.items():
            if obj.geometry.grid is not texture:
                obj.geometry = gfx.Geometry(grid=texture)
            if isinstance(obj.material, TissueSliceMaterial):
                obj.material.cutoff = cutoff
            obj.material.plane = slice_plane(kind, cursor)
            obj.material.clim = clim
            obj.material.map = cmap
            obj.material.interpolation = look.interpolation
            self._place(obj, o.visible and o.slices_3d)
        if self.rendering is not None:
            self._show_rendering(self.rendering, look, o)
        if isinstance(o, OverlayOptions):
            self._colour = look.colour
            self._labels = look.labels
            self._show_surface(o)
            assert self.labels is not None
            self.labels.show(o, look.labels, look.table)
        self.view.request_draw()

    def _show_rendering(self, obj: gfx.Volume, look: Look, o: VolumeOptions) -> None:
        """Follows the item's options in the volume rendering."""
        old = obj.geometry.grid
        if old is not look.texture:
            obj.geometry = gfx.Geometry(grid=look.texture)
            # A 4D image's next volume keeps the old lighting until its own
            # gradient is ready, rather than flashing unlit at each step.
            if old.size != look.texture.size:
                obj.material.gradient = None
        obj.material.clim = look.clim
        obj.material.map = look.cmap  # its low end is transparent already
        obj.material.opacity = o.opacity_3d
        # A rendering of one component would hide the DTI colours' slices.
        shown = o.visible and o.opacity_3d > 0 and look.dti is None
        self._place(obj, shown)
        if shown and self._gradient_of is not look.texture:
            self._request_gradient(look.texture)

    def follow_cursor(self, old: Vec3, new: Vec3) -> None:
        """Moves the slices whose coordinate the cursor changed."""
        moved = False
        for kind, obj in self.slices.items():
            axis = PLANES[kind]
            if old[axis] != new[axis]:
                obj.material.plane = slice_plane(kind, new)
                moved = moved or obj.parent is not None
        if moved:
            self.view.request_draw()

    def close(self) -> None:
        """Removes the objects from the view and drops any surface being computed."""
        self._worker.close()
        self._gradient_worker.close()
        if self.labels is not None:
            self.labels.close()
        if self.rendering is not None:
            self.view.canvas.remove_event_handler(self._prepare, "before_draw")
        for obj in [*self.slices.values(), self.rendering, self.surface]:
            if obj is not None and obj.parent is not None:
                self.view.remove(obj)

    def _background_cutoff(self, texture: gfx.Texture, index: int) -> float:
        """Returns the volume's background cut-off, found the first time it is shown."""
        if self._cutoffs_of is not self.item.data:
            self._cutoffs, self._cutoffs_of = {}, self.item.data
        if index not in self._cutoffs:
            self._cutoffs[index] = background_cutoff(texture.data)
        return self._cutoffs[index]

    def _prepare(self, event: object = None) -> None:
        """Takes fewer samples per ray while the view is dragged, before it draws."""
        assert self.rendering is not None
        step = INTERACTIVE_STEP if self.view.interacting else STEP
        if self.rendering.material.step_size != step:
            self.rendering.material.step_size = step

    def _request_gradient(self, texture: gfx.Texture) -> None:
        """Lights the rendering once the gradient of its new texture is ready."""
        assert self.rendering is not None
        self._gradient_of = texture
        values = texture.data
        self._gradient_worker.submit(
            lambda: volume_gradient(values),
            lambda gradient: self._set_gradient(texture, gradient),
            lambda: self._forget_gradient(texture),
        )

    def _set_gradient(self, texture: gfx.Texture, gradient: Gradient) -> None:
        """Lights the rendering with the gradient, if it still shows that texture."""
        assert self.rendering is not None
        if self.rendering.geometry.grid is texture:
            self.rendering.material.gradient = gradient
            self.view.request_draw()

    def _forget_gradient(self, texture: gfx.Texture) -> None:
        """Lets a gradient whose work failed be asked for again."""
        if self._gradient_of is texture:
            self._gradient_of = None

    def _place(self, obj: gfx.WorldObject, shown: bool) -> None:
        """Adds an object to the view when it is first shown, and shows or hides it."""
        obj.visible = shown
        if shown and obj.parent is None:
            self.view.add(obj)

    def _show_surface(self, o: OverlayOptions) -> None:
        """Shows the surface, computing it first when its volume or level changed."""
        # Hidden, not discarded: shown again at the same volume and level, it needs
        # no new computation; a mesh of another level stays hidden until replaced.
        shown = self._surface_shown(o)
        key = (o.volume_index, self._level(o))
        if self.surface is not None:
            m = self.surface.material
            m.color = self._colour
            m.opacity = o.opacity_3d
            self.surface.visible = shown and self._mesh_key == key
        if not shown:
            return  # computed when it is first shown
        volume = self.item.data
        assert isinstance(volume, Volume)
        if key == self._surface_key:
            return
        # Set now, so the same surface is not asked for again while it is computed;
        # cleared if the work fails, so that asking again tries again.
        self._surface_key = key
        index, stored = key
        data = volume.data[..., index] if volume.data.ndim == 4 else volume.data
        self._worker.submit(
            lambda: mask_surface(data, stored),
            lambda surface: self._set_surface(surface, key),
            lambda: self._forget_surface(key),
        )

    def _surface_shown(self, o: OverlayOptions) -> bool:
        """Returns whether the surface is drawn; shown as labels, the labels' are."""
        return o.visible and o.opacity_3d > 0 and o.surface_3d and not self._labels

    def _level(self, o: OverlayOptions) -> float:
        """Returns the surface's level in stored units, the threshold or mid-range."""
        volume = self.item.data
        assert isinstance(volume, Volume)
        assert self.item.value_range is not None
        lo, hi = self.item.value_range
        level = o.threshold if o.threshold > lo else (lo + hi) / 2
        return (level - volume.intercept) / volume.slope

    def _forget_surface(self, key: tuple[int, float]) -> None:
        """Lets a surface whose work failed be asked for again."""
        if self._surface_key == key:
            self._surface_key = None

    def _set_surface(self, surface: LabelMesh | None, key: tuple[int, float]) -> None:
        """Puts a newly computed surface in the view, in place of the old one."""
        if self.surface is not None:
            self.view.remove(self.surface)
            self.surface = None
        if surface is not None:
            o = self.item.options
            assert isinstance(o, OverlayOptions)
            volume = self.item.data
            assert isinstance(volume, Volume)
            geometry = gfx.Geometry(
                positions=surface.vertices,
                indices=surface.faces,
                normals=surface.normals,
            )
            material = gfx.MeshPhongMaterial(
                color=self._colour,
                opacity=o.opacity_3d,
                alpha_mode="blend",
                depth_write=False,
                side="both",
            )
            mesh = gfx.Mesh(geometry, material)
            mesh.render_order = ORDER_SURFACE
            mesh.local.matrix = volume.affine
            mesh.visible = self._surface_shown(o)
            self.surface = mesh
            self._mesh_key = key
            self.view.add(mesh)
        self.view.request_draw()


def slice_plane(kind: ViewKind, cursor: Vec3) -> tuple[float, float, float, float]:
    """Returns the plane of a 2D view's slice through the cursor, as pygfx takes it."""
    axis = PLANES[kind]
    normal = [0.0, 0.0, 0.0]
    normal[axis] = -1.0
    return (normal[0], normal[1], normal[2], cursor[axis])
