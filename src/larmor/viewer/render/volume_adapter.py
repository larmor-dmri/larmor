"""Draws volumes and overlays as slices in the 2D views and as objects in 3D."""

from typing import ClassVar

import numpy as np
import pygfx as gfx

from larmor._log import LarmorLogger
from larmor.types import Label, Volume
from larmor.viewer.model.events import Event, OptionChanged, ViewStateChanged
from larmor.viewer.model.item import Category, Item, ItemId, ViewKind
from larmor.viewer.model.labels import MAX_LABELS, label_rgb, lookup_table
from larmor.viewer.model.options import (
    COLORMAPS,
    Colormap,
    OptionName,
    OverlayOptions,
    Scope,
    VolumeOptions,
    option_fields,
)
from larmor.viewer.model.view_state import ViewField
from larmor.viewer.model.viewer import Viewer
from larmor.viewer.render.adapter import Adapter
from larmor.viewer.render.view import Vec3, ViewRenderer
from larmor.viewer.render.volume_3d import PLANES, Look, Volume3D, slice_plane
from larmor.viewer.render.worker import Worker

VIEWS_2D: tuple[ViewKind, ...] = (ViewKind.AXIAL, ViewKind.CORONAL, ViewKind.SAGITTAL)
SLICE_AXIS: dict[ViewKind, int] = PLANES
# Colormaps of our own, as pygfx's gray fades to transparent at its dark end.
_OWN_MAPS = {
    Colormap.GRAY: gfx.cm.create_colormap([(0, 0, 0, 1), (1, 1, 1, 1)]),
    Colormap.GRAY_R: gfx.cm.create_colormap([(1, 1, 1, 1), (0, 0, 0, 1)]),
    # Linear over [min, max], so centred on 0 when min = -max.
    Colormap.BLUE_RED: gfx.cm.create_colormap(
        [
            (0.3, 0.6, 1, 1),
            (0, 0, 0.6, 1),
            (0.1, 0.1, 0.1, 1),
            (0.6, 0, 0, 1),
            (1, 0.4, 0.2, 1),
        ]
    ),
}
COLORMAP_MAPS = {
    name: _OWN_MAPS[name] if name in _OWN_MAPS else getattr(gfx.cm, name)
    for name in COLORMAPS
}


class VolumeAdapter(Adapter):
    """Draws an image from one texture, as a slice in each 2D view and in 3D.

    It draws in the views it is given, so a detail window gives it one 2D view.
    """

    # Whether the slices may be smoothed; a detail window's never are.
    smooth: ClassVar[bool] = True

    def __init__(
        self,
        item: Item,
        views: dict[ViewKind, ViewRenderer],
        viewer: Viewer,
        order: float,
    ) -> None:
        """Uploads the image once and adds its slices and 3D objects to the views."""
        super().__init__(item, views, viewer, order)
        assert isinstance(item.data, Volume)
        vol = item.data
        self._values = self._build_texture()
        # Shown as labels, the distinct values and a texture of their indices.
        self._labels: tuple[np.ndarray, gfx.Texture] | None = None
        self._labels_counted = False
        # The lookup table's path, its labels, and the colours of the image's labels.
        self._label_map: tuple[str, dict[int, Label], gfx.TextureMap] | None = None
        self.slices: dict[ViewKind, gfx.Volume] = {}
        # DTI colours, and the modulating item last asked for, wrapped so that ()
        # means none asked.
        self._dti: gfx.Texture | None = None
        self._dti_asked: tuple[ItemId | None] | tuple[()] = ()
        self._dti_worker = Worker(next(iter(views.values())).canvas)

        cursor = viewer.scene.view.cursor
        for kind in VIEWS_2D:
            if kind not in views:
                continue
            material = gfx.VolumeSliceMaterial(plane=_plane(kind, cursor))
            material.depth_test = item.category == Category.VOLUMES
            obj = gfx.Volume(gfx.Geometry(grid=self._values), material)
            obj.render_order = order
            obj.local.matrix = vol.affine
            self.slices[kind] = obj
            self.views[kind].add(obj)
        # The same texture, drawn in the 3D view.
        self.three_d: Volume3D | None = None
        if ViewKind.THREE_D in views:
            self.three_d = Volume3D(item, views[ViewKind.THREE_D], order, self._values)

        self._show_options()

    def _show_options(self, scope: Scope = Scope.BOTH) -> None:
        """Shows the options in the views they apply to (`scope`)."""
        v = self.item.data
        o = self.item.options
        assert isinstance(v, Volume) and isinstance(o, VolumeOptions)
        dti = self._dti_texture() if o.dti else None
        labels = None
        if isinstance(o, OverlayOptions) and o.labels and not o.dti:
            labels = self._count_labels()
        if isinstance(o, OverlayOptions) and labels is not None:
            values, texture = labels
            # Value k of the index texture falls on the centre of colour k.
            clim = (-0.5, len(values) - 0.5)
            table, cmap = self._label_colormap(
                values * v.slope + v.intercept, o.lookup_table
            )
            interpolation = "nearest"  # never blend two labels
        else:
            table = {}
            texture = self._values
            lo, hi = o.minimum, o.maximum
            cmap = COLORMAP_MAPS[o.colormap]
            if isinstance(o, OverlayOptions):
                assert self.item.value_range is not None  # an image has one
                cmap, (lo, hi) = _threshold_cmap(
                    cmap, lo, hi, o.threshold, self.item.value_range
                )
            # The GPU holds stored values; the options are in scaled units.
            clim = ((lo - v.intercept) / v.slope, (hi - v.intercept) / v.slope)
            interpolation = "linear" if o.interpolate else "nearest"

        if self.three_d is not None and scope != Scope.TWO_D:
            top = COLORMAP_MAPS[o.colormap].texture.data[-1]
            colour = (float(top[0]), float(top[1]), float(top[2]))
            look = Look(
                texture,
                clim,
                cmap,
                interpolation,
                colour,
                labels is not None,
                table,
                dti,
            )
            self.three_d.show(look, self.viewer.scene.view.cursor)
        if scope == Scope.THREE_D:
            return

        # The colours are the texture's own: no colormap, and values already 0 to 1.
        slice_map = (texture, clim, cmap) if dti is None else (dti, (0.0, 1.0), None)
        for obj in self.slices.values():
            if obj.geometry.grid is not slice_map[0]:
                obj.geometry = gfx.Geometry(grid=slice_map[0])
            obj.visible = o.visible
            obj.material.opacity = o.opacity_2d
            obj.material.clim = slice_map[1]
            obj.material.map = slice_map[2]
            obj.material.interpolation = interpolation if self.smooth else "nearest"

        for kind in self.slices:
            self.views[kind].request_draw()

    def _count_labels(self) -> tuple[np.ndarray, gfx.Texture] | None:
        """Returns the distinct values, a texture of their indices; None if too many.

        They are made on first use.
        """
        if not self._labels_counted:
            self._labels_counted = True
            volume = self.item.data
            opts = self.item.options
            assert isinstance(opts, VolumeOptions)
            assert isinstance(volume, Volume)
            volume_index = opts.volume_index
            data = _volume_at(volume.data, volume_index)
            values, index = np.unique(data, return_inverse=True)
            if len(values) > MAX_LABELS:
                LarmorLogger.warning(
                    "%s has %d distinct values, more than the %d a parcellation "
                    "can show; it is shown as an image.",
                    self.item.name,
                    len(values),
                    MAX_LABELS,
                )
            else:
                # Indices in their own uint16 texture, as label values can be large.
                index = index.reshape(data.shape).astype(np.uint16)
                self._labels = (
                    values,
                    gfx.Texture(np.ascontiguousarray(index.T), dim=3),
                )
        return self._labels

    def _label_colormap(
        self, values: np.ndarray, path: str
    ) -> tuple[dict[int, Label], gfx.TextureMap]:
        """Returns the table and a colour per label, made again for a new table."""
        if self._label_map is None or self._label_map[0] != path:
            table = lookup_table(path)
            self._label_map = (path, table, _label_colours(values, table))
        return self._label_map[1], self._label_map[2]

    def _dti_texture(self) -> gfx.Texture | None:
        """Returns the DTI colours' texture; asks for it if missing or stale."""
        o = self.item.options
        assert isinstance(o, VolumeOptions)
        if self._dti_asked != (o.modulate,):
            self._dti_asked = (o.modulate,)
            vectors = self.item.data
            assert isinstance(vectors, Volume)
            weights: Volume | None = None
            if o.modulate is not None:
                data = self.viewer.scene.item(o.modulate).data
                assert isinstance(data, Volume)
                weights = data
            overlay = self.item.category == Category.OVERLAYS
            key = o.modulate
            self._dti_worker.submit(
                lambda: dti_colours(vectors, weights, overlay),
                self._set_dti,
                lambda: self._forget_dti(key),
            )
        # The old colours stay until the new ones are ready; at first, the image.
        return self._dti

    def _set_dti(self, colours: np.ndarray) -> None:
        """Shows newly made DTI colours."""
        self._dti = gfx.Texture(colours, dim=3)
        self._show_options()

    def _forget_dti(self, key: ItemId | None) -> None:
        """Lets colours whose work failed be asked for again."""
        if self._dti_asked == (key,):
            self._dti_asked = ()

    def _drop_dti(self) -> None:
        """Frees the DTI colours once they are off, and drops any being made."""
        self._dti_worker.cancel()
        self._dti = None
        self._dti_asked = ()

    def close(self) -> None:
        """Removes the slices and the 3D objects from the views."""
        self._dti_worker.close()
        for kind, obj in self.slices.items():
            self.views[kind].remove(obj)
        if self.three_d is not None:
            self.three_d.close()

    def on_event(self, event: Event) -> None:
        """Follows this item's options and the cursor."""
        if isinstance(event, OptionChanged):
            if event.item.id == self.item.id:
                if event.name == OptionName.VOLUME_INDEX:
                    self._values = self._build_texture()
                    # Each volume has its own labels, so count them again.
                    self._labels = None
                    self._labels_counted = False
                    self._label_map = None
                if event.name == OptionName.DTI and not event.new:
                    self._drop_dti()
                _, info = option_fields(type(self.item.options))[event.name]
                self._show_options(info.scope)

        elif isinstance(event, ViewStateChanged) and event.name == ViewField.CURSOR:
            assert isinstance(event.old, tuple) and isinstance(event.new, tuple)
            self._follow_cursor(event.old, event.new)

    def _follow_cursor(self, old: Vec3, new: Vec3) -> None:
        """Moves the slices whose coordinate the cursor changed."""
        for kind, obj in self.slices.items():
            axis = SLICE_AXIS[kind]
            if old[axis] != new[axis]:
                obj.material.plane = _plane(kind, new)
                self.views[kind].request_draw()
        if self.three_d is not None:
            self.three_d.follow_cursor(old, new)

    def _build_texture(self) -> gfx.Texture:
        """Returns the texture of the shown volume of the image."""
        volume = self.item.data
        assert isinstance(volume, Volume)
        opts = self.item.options
        assert isinstance(opts, VolumeOptions)
        return gfx.Texture(
            _texture_data(_volume_at(volume.data, opts.volume_index)),
            dim=3,
        )


def _volume_at(data: np.ndarray, idx: int) -> np.ndarray:
    """Returns the 3D volume at an index of the fourth axis, or the 3D data itself."""
    if data.ndim == 4:
        return data[..., idx]
    else:
        return data


def _label_colours(values: np.ndarray, table: dict[int, Label]) -> gfx.TextureMap:
    """Returns a colour per value, from the table or built-in colours.

    The value 0 is transparent, as is NaN.
    """
    colours = np.empty((len(values), 4), np.float32)
    for k, value in enumerate(values):
        label = round(float(value)) if np.isfinite(value) else 0
        colours[k] = (*label_rgb(label, table), 0.0 if label == 0 else 1.0)
    return gfx.cm.create_colormap(colours, n=len(colours))


def dti_colours(
    vectors: Volume, weights: Volume | None, overlay: bool = False
) -> np.ndarray:
    """Returns RGBA per voxel, the absolute vector components times the weights.

    Each channel is clipped to [0, 1], as are the weights; NaN counts as 0. An
    overlay's black voxels are transparent.

    Returns:
        The colours, in a texture's (z, y, x, 4) order.
    """
    x, y, z = vectors.data.shape[:3]
    out = np.empty((z, y, x, 4), np.float32)
    rgb = out[..., :3]
    rgb[...] = vectors.data.transpose(2, 1, 0, 3)
    if vectors.slope != 1 or vectors.intercept != 0:
        rgb *= vectors.slope
        rgb += vectors.intercept
    np.abs(rgb, out=rgb)
    np.nan_to_num(rgb, copy=False, nan=0.0)
    np.minimum(rgb, 1.0, out=rgb)
    if weights is not None:
        # A 4D image of one volume weighs as its 3D volume.
        three_d = weights.data.reshape(weights.data.shape[:3])
        w = np.asarray(three_d.T, dtype=np.float32) * weights.slope
        w += weights.intercept
        np.nan_to_num(w, copy=False, nan=0.0)
        rgb *= np.clip(w, 0.0, 1.0)[..., None]
    if overlay:
        out[..., 3] = rgb.max(axis=-1) > 0
    else:
        out[..., 3] = 1.0
    return out


def _texture_data(data: np.ndarray) -> np.ndarray:
    """Returns a volume as a texture's float32 data, in (z, y, x) order."""
    # Float32, as the GPU interpolates it in hardware, unlike an integer texture.
    return np.ascontiguousarray(data.T, dtype=np.float32)


def _plane(kind: ViewKind, cursor: Vec3) -> tuple[float, float, float, float]:
    """Returns the plane of a 2D view's slice through the cursor."""
    return slice_plane(kind, cursor)


def _threshold_cmap(
    cmap: gfx.TextureMap,
    minimum: float,
    maximum: float,
    threshold: float,
    data_range: tuple[float, float],
) -> tuple[gfx.TextureMap, tuple[float, float]]:
    """Returns a colormap transparent at or below the threshold, and its range.

    The range is the colour range widened to take in the threshold; between the
    threshold and the colour range, the copy repeats the end colours. The shader
    clamps values outside the range onto the end entries, so those entries decide
    whether data beyond the range is hidden or shown.

    Returns:
        The colormap and its range.
    """
    table = np.asarray(cmap.texture.data, dtype=np.float32)
    n = len(table)
    lo, hi = min(minimum, threshold), max(maximum, threshold)
    centres = lo + (np.arange(n) + 0.5) * (hi - lo) / n  # each entry's value
    span = maximum - minimum
    where = (centres - minimum) / span if span else (centres > minimum).astype(float)
    colours = table[np.clip((where * n).astype(int), 0, n - 1)]  # a copy
    colours[centres < threshold, 3] = 0.0
    # Values at the threshold too, so a mask's zeros vanish at a threshold of 0.
    if lo == threshold and data_range[0] <= threshold:
        colours[0, 3] = 0.0
    if hi == threshold and data_range[1] >= threshold:
        colours[-1, 3] = table[-1, 3]
    return gfx.cm.create_colormap(colours, n=n), (lo, hi)
