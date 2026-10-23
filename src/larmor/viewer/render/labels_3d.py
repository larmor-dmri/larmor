"""Draws a parcellation's chosen labels in the 3D view, as one mesh of surfaces."""

import threading
from collections.abc import Mapping

import pygfx as gfx

from larmor.types import Label, Volume
from larmor.viewer.model.item import Item
from larmor.viewer.model.labels import label_rgb
from larmor.viewer.model.options import OverlayOptions
from larmor.viewer.render.colour_space import physical_colours
from larmor.viewer.render.surfaces import (
    ColouredSurface,
    LabelMesh,
    label_surfaces,
    merge_surfaces,
)
from larmor.viewer.render.view import ViewRenderer
from larmor.viewer.render.view_3d import ORDER_SURFACE
from larmor.viewer.render.worker import Worker

# What a mesh is made from, the volume index, the labels and the lookup table.
type Key = tuple[int, tuple[int, ...], str]


class Labels3D:
    """Draws the surfaces of an overlay's chosen labels, each in its label's colour.

    All of them are one mesh with a colour per vertex. A label's surface is
    computed once, in the background, and kept while the label stays chosen.
    """

    def __init__(self, item: Item, view: ViewRenderer) -> None:
        """Creates the drawing with no mesh, made when labels are first shown."""
        self.item = item
        self.view = view
        self.mesh: gfx.Mesh | None = None
        # Each label's surface by (volume index, label), filled by the work itself,
        # so a request replaced while it ran still leaves its surfaces behind.
        self._cache: dict[tuple[int, int], LabelMesh | None] = {}
        self._lock = threading.Lock()
        self._asked: Key | None = None  # the last mesh asked for
        self._shown = False  # whether the options last said to show it
        self._worker = Worker(view.canvas)

    def show(self, o: OverlayOptions, labels: bool, table: Mapping[int, Label]) -> None:
        """Follows the options; `labels` says whether the overlay is shown as labels.

        A mesh being replaced stays shown until its successor is ready, so adding a
        label does not make the others blink.
        """
        shown = (
            labels
            and o.visible
            and o.opacity_3d > 0
            and o.surface_3d
            and bool(o.surface_labels)
        )
        self._shown = shown
        if self.mesh is not None:
            style_opacity(self.mesh.material, o.opacity_3d)
            self.mesh.visible = shown
        if not shown:
            return  # computed when shown
        key: Key = (o.volume_index, o.surface_labels, o.lookup_table)
        if key == self._asked:
            return
        # Set now, so the same mesh is not asked for again while it is computed;
        # cleared if the work fails, so that asking again tries again.
        self._asked = key
        index, chosen, _ = key
        volume = self.item.data
        assert isinstance(volume, Volume)
        colours = {label: label_rgb(label, table) for label in chosen}

        def work() -> ColouredSurface | None:
            """Makes the missing labels' surfaces and merges the chosen ones."""
            with self._lock:
                missing = [k for k in chosen if (index, k) not in self._cache]
            made = label_surfaces(volume, index, missing)
            with self._lock:
                self._cache.update({(index, k): s for k, s in made.items()})
                surfaces = [self._cache[index, k] for k in chosen]
            parts = [(k, s) for k, s in zip(chosen, surfaces, strict=True) if s]
            return merge_surfaces([s for _, s in parts], [colours[k] for k, _ in parts])

        self._worker.submit(
            work,
            lambda merged: self._set_mesh(merged, index, chosen),
            lambda: self._forget(key),
        )

    def close(self) -> None:
        """Removes the mesh and drops any surfaces being computed."""
        self._worker.close()
        if self.mesh is not None and self.mesh.parent is not None:
            self.view.remove(self.mesh)

    def _forget(self, key: Key) -> None:
        """Lets a mesh whose work failed be asked for again."""
        if self._asked == key:
            self._asked = None

    def _set_mesh(
        self, merged: ColouredSurface | None, index: int, chosen: tuple[int, ...]
    ) -> None:
        """Puts a newly made mesh in the view, in place of the old one."""
        with self._lock:  # the labels no longer chosen are dropped
            kept = {(index, k) for k in chosen}
            self._cache = {key: s for key, s in self._cache.items() if key in kept}
        if self.mesh is not None:
            self.view.remove(self.mesh)
            self.mesh = None
        if merged is not None:
            o = self.item.options
            volume = self.item.data
            assert isinstance(o, OverlayOptions) and isinstance(volume, Volume)
            s = merged.surface
            geometry = gfx.Geometry(
                positions=s.vertices,
                indices=s.faces,
                normals=s.normals,
                colors=physical_colours(merged.colours),  # linear, as for any mesh
            )
            material = gfx.MeshPhongMaterial(color_mode="vertex", side="both")
            style_opacity(material, o.opacity_3d)
            mesh = gfx.Mesh(geometry, material)
            mesh.render_order = ORDER_SURFACE
            mesh.local.matrix = volume.affine
            mesh.visible = self._shown
            self.mesh = mesh
            self.view.add(mesh)
        self.view.request_draw()


def style_opacity(material: gfx.Material, opacity: float) -> None:
    """Sets a material's opacity; opaque at 1, so surfaces hide each other."""
    opaque = opacity >= 1.0
    mode = "solid" if opaque else "blend"
    if material.alpha_mode != mode:
        material.alpha_mode = mode
        material.depth_write = opaque
    material.opacity = opacity
