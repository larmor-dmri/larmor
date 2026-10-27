"""Carries out the Viewer's phantom actions on its scene."""

import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from larmor._log import LarmorLogger
from larmor.simulation.bundles import centreline_spline
from larmor.types import Grid, Tractogram, Volume
from larmor.viewer.model.breakdown import BreakdownTool
from larmor.viewer.model.item import Category, Item, ItemId
from larmor.viewer.model.options import (
    Colormap,
    Colouring,
    OptionError,
    OptionName,
    option_fields,
)
from larmor.viewer.model.phantom import (
    GEOMETRY_FIELDS,
    PROXY_NAMES,
    SEED_LIMIT,
    BundleField,
    BundleSettings,
    CompartmentSettings,
    Draft,
    GeometrySettings,
    ItemSource,
    LoadedPhantom,
    PhantomPart,
    PhantomState,
    ProxyMap,
    RunInputs,
    RunOutputs,
    ScanField,
    SimulationOutputs,
    TissuePreset,
    TissueSettings,
    Vec3,
    build_geometry,
    build_phantom,
    build_protocol,
    build_streamlines,
    build_tissue,
    bundle_value,
    custom_settings,
    default_tissues,
    drawable,
    options_json,
    preset_defaults,
    problems,
    proxy_volumes,
    scan_value,
    state_from_json,
    state_json,
    viewer_block,
    with_bundle_value,
)
from larmor.viewer.model.view_state import Tool, ViewField

if TYPE_CHECKING:
    from larmor.simulation import Phantom
    from larmor.viewer.model.scene import Scene
    from larmor.viewer.model.viewer import Viewer

# Runs a function later, once per frame in a window; see `PhantomEditor.schedule`.
type Schedule = Callable[[Callable[[], None]], None]
# The bundles and maps as `build_phantom` takes them.
type ShownBundles = list[tuple[BundleSettings, Tractogram, Path | None]]
type ShownMaps = list[tuple[str, Volume, Path | None]]

# The most control points one freehand stroke gives.
STROKE_POINTS = 12
# The colormap each proxy map starts with.
PROXY_COLORMAPS = {
    ProxyMap.FILL: Colormap.HOT,
    ProxyMap.PARTITION: Colormap.VIRIDIS,
    ProxyMap.EXPECTED_B0: Colormap.GRAY,
}


class _Rebuild(Enum):
    """Work an edit asks for, done at most once per frame."""

    PREVIEW = auto()
    PROXIES = auto()
    BREAKDOWN = auto()


class PhantomEditor:
    """Carries out the Viewer's phantom actions on its scene.

    The state lives in `scene.phantom`; this keeps only the rebuilds waiting for
    the next frame.
    """

    def __init__(self, viewer: Viewer) -> None:
        """Edits the phantom of viewer's scene."""
        self.viewer = viewer
        # Runs the rebuilds later, once per frame in a window; None runs them at once.
        self.schedule: Schedule | None = None
        self._pending: set[_Rebuild] = set()
        # Whether the item of the bundle being edited shows other streamlines than
        # its own, so that a cancel draws them again.
        self._edited = False
        # The Signal breakdown dock's voxel, followed through every edit.
        self.breakdown = BreakdownTool(self)

    # State

    @property
    def state(self) -> PhantomState | None:
        """The phantom; None if none is open."""
        return self.viewer.scene.phantom

    def require(self) -> PhantomState:
        """Returns the phantom; raises ValueError if none is open."""
        state = self.viewer.scene.phantom
        if state is None:
            raise ValueError("No phantom is open; start one with new_phantom.")
        return state

    def _set(self, state: PhantomState | None, part: PhantomPart) -> None:
        """Puts a new phantom state in the scene.

        The breakdown is asked for again unless only the proxy maps or the output
        folder changed.
        """
        scene = self.viewer.scene
        scene.set_phantom(state, part)
        # The bundle tool is chosen exactly while a bundle is being drawn.
        drawing = state is not None and state.draft is not None
        if drawing and scene.view.tool != Tool.DRAW_BUNDLE:
            scene.set_view(ViewField.TOOL, Tool.DRAW_BUNDLE)
        elif not drawing and scene.view.tool == Tool.DRAW_BUNDLE:
            scene.set_view(ViewField.TOOL, Tool.CURSOR)
        if part not in (PhantomPart.PROXIES, PhantomPart.OUTPUT):
            self._request(_Rebuild.BREAKDOWN)

    def new(self, grid: Grid) -> None:
        """Starts a phantom on grid, with the default tissues and scan."""
        self.close()
        self._set(PhantomState(grid, default_tissues()), PhantomPart.ALL)
        self.viewer._frame(grid)

    def close(self) -> None:
        """Drops the phantom and the items it made; other items stay."""
        state = self.state
        if state is None:
            return
        self._pending.clear()
        self.breakdown.forget()
        owned = state.owned()
        self._set(None, PhantomPart.ALL)
        for item_id in owned:
            if any(i.id == item_id for i in self.viewer.scene.items):
                self.viewer.remove(item_id)

    def set_grid(self, grid: Grid) -> None:
        """Moves the phantom onto another grid; bundles keep their world position."""
        state = self.require()
        for tissue, item_id in state.tissue_maps:
            data = self.viewer.scene.item(item_id).data
            assert isinstance(data, Volume)
            if not data.grid.matches(grid):
                raise ValueError(
                    f"The map of {tissue} is not on that grid; clear it first."
                )
        self._set(replace(state, grid=grid), PhantomPart.GRID)
        self.viewer._frame(grid)
        self._request(_Rebuild.PROXIES)

    # Tissues

    def set_tissue(
        self, name: str, preset: TissuePreset, values: Mapping[str, float] | None
    ) -> None:
        """Adds a tissue or changes one; the bundles using it follow."""
        state = self.require()
        if not name.strip():
            raise ValueError("A tissue needs a name.")
        old = next((t for t in state.tissues if t.name == name), None)
        if preset == TissuePreset.CUSTOM:
            start = (
                custom_settings(old)
                if old is not None
                else custom_settings(TissueSettings(name, TissuePreset.FREE_WATER))
            )
            settings = replace(start, name=name)
        else:
            known = preset_defaults(preset)
            kept = dict(old.values) if old is not None and old.preset == preset else {}
            for key, value in (values or {}).items():
                if key not in known:
                    raise ValueError(
                        f"The {preset.value} preset has no parameter {key!r}; it has "
                        f"{', '.join(known)}."
                    )
                kept[key] = float(value)
            settings = TissueSettings(name, preset, tuple(kept.items()))
        self._replace_tissue(state, old, settings)

    def set_compartments(
        self, name: str, compartments: Sequence[Mapping[str, Any]]
    ) -> None:
        """Makes a tissue of these compartments, as a custom tissue."""
        state = self.require()
        try:
            made = tuple(CompartmentSettings.from_json(c) for c in compartments)
        except (KeyError, TypeError) as e:
            raise ValueError(f"A compartment of {name} is incomplete: {e}") from e
        old = next((t for t in state.tissues if t.name == name), None)
        settings = TissueSettings(name, TissuePreset.CUSTOM, compartments=made)
        self._replace_tissue(state, old, settings)

    def _replace_tissue(
        self,
        state: PhantomState,
        old: TissueSettings | None,
        settings: TissueSettings,
    ) -> None:
        """Puts settings in place of the old tissue, or adds it if old is None."""
        try:
            tissue = build_tissue(settings)  # refused here, before it is kept
        except TypeError as e:
            raise ValueError(f"Tissue {settings.name}: {e}") from e
        if tissue.is_oriented and self._unoriented_use(state, settings.name):
            raise ValueError(
                f"{settings.name} fills a tissue map or the background, which have "
                "no fibre direction; it cannot have a stick or a zeppelin."
            )
        if old is None:
            tissues = (*state.tissues, settings)
        else:
            tissues = tuple(settings if t is old else t for t in state.tissues)
        self._set(replace(state, tissues=tissues), PhantomPart.TISSUES)
        self._request(_Rebuild.PROXIES)

    def _unoriented_use(self, state: PhantomState, name: str) -> bool:
        """Tells whether a tissue fills the background or a tissue map."""
        return state.background == name or any(t == name for t, _ in state.tissue_maps)

    def remove_tissue(self, name: str) -> None:
        """Removes a tissue that nothing uses."""
        state = self.require()
        state.tissue(name)
        users = [b for b in state.bundles if b.tissue == name]
        if users or self._unoriented_use(state, name):
            raise ValueError(
                f"{name} is in use; give its bundles another tissue first."
            )
        if state.draft is not None and state.draft.tissue == name:
            raise ValueError(f"The bundle being drawn uses {name}.")
        tissues = tuple(t for t in state.tissues if t.name != name)
        self._set(replace(state, tissues=tissues), PhantomPart.TISSUES)

    def set_tissue_map(self, tissue: str, item_id: ItemId | None) -> None:
        """Makes an overlay the map of a tissue's fractions, or clears it."""
        state = self.require()
        settings = state.tissue(tissue)
        maps = tuple((t, i) for t, i in state.tissue_maps if t != tissue)
        if item_id is not None:
            item = self.viewer.scene.item(item_id)
            data = item.data
            if item.category != Category.OVERLAYS or not isinstance(data, Volume):
                raise ValueError(f"{item.name} is not an overlay.")
            if data.data.ndim != 3 or not data.grid.matches(state.grid):
                raise ValueError(
                    f"{item.name} must be a 3D image on the phantom's grid "
                    f"{state.grid.shape}."
                )
            values = data.as_float()
            if values.min() < -1e-6 or values.max() > 1 + 1e-6:
                raise ValueError(f"{item.name} must hold fractions between 0 and 1.")
            if build_tissue(settings).is_oriented:
                raise ValueError(
                    f"{tissue} needs a fibre direction, which a map has not; use it "
                    "in a bundle."
                )
            maps = (*maps, (tissue, item_id))
        self._set(replace(state, tissue_maps=maps), PhantomPart.TISSUES)
        self._request(_Rebuild.PROXIES)

    def set_background(self, tissue: str | None) -> None:
        """Fills what the bundles and maps leave with a tissue, or with nothing."""
        state = self.require()
        if tissue is not None and build_tissue(state.tissue(tissue)).is_oriented:
            raise ValueError(f"{tissue} needs a fibre direction; it cannot fill space.")
        self._set(replace(state, background=tissue), PhantomPart.TISSUES)
        self._request(_Rebuild.PROXIES)

    # Scan

    def set_scan(self, name: ScanField, value: object) -> None:
        """Changes one setting of the scan."""
        state = self.require()
        change: dict[str, Any] = {name.value: scan_value(name, value)}
        scan = replace(state.scan, **change)
        self._set(replace(state, scan=scan), PhantomPart.SCAN)
        if ProxyMap.EXPECTED_B0 in (k for k, _ in state.proxies):
            self._request(_Rebuild.PROXIES)

    def show_box(self, shown: bool) -> None:
        """Outlines the grid's box in the views, or not."""
        state = self.require()
        self._set(replace(state, show_box=shown), PhantomPart.GRID)

    def set_output(self, folder: Path | None) -> None:
        """Sets the folder a run writes to."""
        state = self.require()
        self._set(replace(state, output=folder), PhantomPart.OUTPUT)

    # Drawing

    def start_bundle(self) -> None:
        """Starts drawing a new bundle; the one being drawn is finished first."""
        if self.state is None:
            self.new(self.viewer._phantom_grid())
        state = self.require()
        if state.draft is not None:
            if drawable(state.draft.points):
                self.finish()
            else:
                self.cancel()
            state = self.require()
        last = next(
            (b.geometry for b in reversed(state.bundles) if b.geometry is not None),
            GeometrySettings(),
        )
        geometry = replace(last, control_points=(), seed=random.randrange(SEED_LIMIT))
        names = [t.name for t in state.tissues]
        tissue = "WM" if "WM" in names else names[0]
        self._set(
            replace(state, draft=Draft(geometry=geometry, tissue=tissue)),
            PhantomPart.DRAFT,
        )

    def draft(self) -> tuple[PhantomState, Draft]:
        """Returns the phantom and the bundle being drawn; raises ValueError if none."""
        state = self.require()
        if state.draft is None:
            raise ValueError("No bundle is being drawn; start one with start_bundle.")
        return state, state.draft

    def set_points(self, points: Sequence[Vec3]) -> None:
        """Replaces the control points of the bundle being drawn."""
        state, draft = self.draft()
        draft = replace(draft, points=tuple(points))
        self._set(replace(state, draft=draft), PhantomPart.DRAFT)
        self._request(_Rebuild.PREVIEW)

    def finish(self) -> Item:
        """Makes the bundle being drawn a bundle of the phantom."""
        state, draft = self.draft()
        if not drawable(draft.points):
            raise ValueError(
                "A bundle needs two or more control points, none twice in a row."
            )
        geometry = replace(draft.geometry, control_points=draft.points)
        build_geometry(geometry)  # refused here, before anything changes
        if _Rebuild.PREVIEW in self._pending or draft.item_id is None:
            self._pending.discard(_Rebuild.PREVIEW)
            self._rebuild_preview()
            state, draft = self.draft()
        assert draft.item_id is not None
        if draft.editing is not None:
            bundle = replace(draft.editing, tissue=draft.tissue, geometry=geometry)
            bundles = tuple(
                bundle if b.item_id == bundle.item_id else b for b in state.bundles
            )
        else:
            bundle = BundleSettings(draft.item_id, draft.tissue, geometry)
            bundles = (*state.bundles, bundle)
        self._edited = False
        self._set(replace(state, bundles=bundles, draft=None), PhantomPart.BUNDLES)
        self._request(_Rebuild.PROXIES)
        return self.viewer.scene.item(bundle.item_id)

    def cancel(self) -> None:
        """Stops drawing: a new bundle is dropped, an edited one drawn as it was."""
        state, draft = self.draft()
        self._pending.discard(_Rebuild.PREVIEW)
        self._set(replace(state, draft=None), PhantomPart.DRAFT)
        if draft.editing is None:
            if draft.item_id is not None:
                self.viewer.remove(draft.item_id)
        elif self._edited and draft.editing.geometry is not None:
            streamlines = build_streamlines(draft.editing.geometry)
            self.viewer.replace_data(draft.editing.item_id, streamlines)
        self._edited = False
        self._request(_Rebuild.PROXIES)

    def edit(self, item_id: ItemId) -> None:
        """Puts a drawn bundle back into draw mode; its item changes in place."""
        state = self.require()
        bundle = state.bundle(item_id)
        if bundle.geometry is None:
            raise ValueError("A loaded tractogram used as a bundle cannot be redrawn.")
        if state.draft is not None:
            self.cancel()
            state = self.require()
        draft = Draft(
            bundle.geometry.control_points,
            bundle.geometry,
            bundle.tissue,
            item_id,
            bundle,
        )
        self._edited = False
        self._set(replace(state, draft=draft), PhantomPart.DRAFT)

    def use_tractogram(self, item_id: ItemId, tissue: str) -> None:
        """Makes a loaded tractogram a bundle of the phantom, as it is."""
        state = self.require()
        state.tissue(tissue)
        item = self.viewer.scene.item(item_id)
        if item.category != Category.TRACTOGRAMS:
            raise ValueError(f"{item.name} is not a tractogram.")
        if item_id in state.owned():
            raise ValueError(f"{item.name} is a bundle of the phantom already.")
        bundles = (*state.bundles, BundleSettings(item_id, tissue))
        self._set(replace(state, bundles=bundles), PhantomPart.BUNDLES)
        self._request(_Rebuild.PROXIES)

    def set_bundle(
        self, item_id: ItemId | None, name: BundleField, value: object
    ) -> None:
        """Changes one setting of a bundle, or of the one being drawn if None."""
        state = self.require()
        if name == BundleField.TISSUE:
            state.tissue(str(value))
        if item_id is None or (
            state.draft is not None and state.draft.item_id == item_id
        ):
            state, draft = self.draft()
            checked = bundle_value(name, value)
            if name == BundleField.TISSUE:
                assert isinstance(checked, str)
                draft = replace(draft, tissue=checked)
            elif name in GEOMETRY_FIELDS:
                change: dict[str, Any] = {name.value: checked}
                draft = replace(draft, geometry=replace(draft.geometry, **change))
            elif draft.editing is not None:
                draft = replace(
                    draft, editing=with_bundle_value(draft.editing, name, checked)
                )
            else:
                raise ValueError(f"{name.value} is set once the bundle is finished.")
            self._set(replace(state, draft=draft), PhantomPart.DRAFT)
            if name in GEOMETRY_FIELDS:
                self._request(_Rebuild.PREVIEW)
            self._request(_Rebuild.PROXIES)
            return
        bundle = with_bundle_value(state.bundle(item_id), name, value)
        bundles = tuple(bundle if b.item_id == item_id else b for b in state.bundles)
        self._set(replace(state, bundles=bundles), PhantomPart.BUNDLES)
        if name in GEOMETRY_FIELDS:
            assert bundle.geometry is not None
            index = bundles.index(bundle)
            self.viewer.replace_data(item_id, build_streamlines(bundle.geometry, index))
        self._request(_Rebuild.PROXIES)

    # Proxy maps

    def show_proxy(self, kind: ProxyMap, shown: bool) -> Item | None:
        """Adds a proxy map's overlay, or removes it."""
        state = self.require()
        present = state.proxy(kind)
        if not shown:
            if present is not None:
                self.viewer.remove(present)
            return None
        if present is not None:
            return self.viewer.scene.item(present)
        volume = proxy_volumes(self._proxy_phantom(state), state.scan, [kind])[kind]
        item = self.viewer.add(volume, Category.OVERLAYS, name=PROXY_NAMES[kind])
        self.viewer.set(item.id, OptionName.COLORMAP, PROXY_COLORMAPS[kind])
        state = self.require()
        proxies = (*state.proxies, (kind, item.id))
        self._set(replace(state, proxies=proxies), PhantomPart.PROXIES)
        return item

    def _proxy_phantom(self, state: PhantomState) -> Phantom:
        """Builds the phantom as the items show it, the bundle being drawn included."""
        return build_phantom(state, *self.shown_parts(state)[:2])

    def shown_parts(
        self, state: PhantomState
    ) -> tuple[ShownBundles, ShownMaps, list[str]]:
        """Returns the parts `build_phantom` takes as the items show them, and names.

        The bundle being drawn is included, as the last bundle if it is new.

        Returns:
            The bundles with their streamlines and files, the tissue maps with
            their volumes and files, and the name of each bundle's item.
        """
        scene = self.viewer.scene
        draft = state.draft
        bundles: ShownBundles = []
        names = []
        for bundle in state.bundles:
            if draft is not None and draft.editing is not None:
                if draft.item_id == bundle.item_id:
                    bundle = replace(bundle, tissue=draft.tissue)
            item = scene.item(bundle.item_id)
            assert isinstance(item.data, Tractogram)
            bundles.append((bundle, item.data, item.source))
            names.append(item.name)
        if draft is not None and draft.editing is None and draft.item_id is not None:
            item = scene.item(draft.item_id)
            assert isinstance(item.data, Tractogram)
            geometry = replace(draft.geometry, control_points=draft.points)
            bundles.append(
                (BundleSettings(draft.item_id, draft.tissue, geometry), item.data, None)
            )
            names.append(item.name)
        maps: ShownMaps = []
        for tissue, item_id in state.tissue_maps:
            item = scene.item(item_id)
            assert isinstance(item.data, Volume)
            maps.append((tissue, item.data, item.source))
        return bundles, maps, names

    # Rebuilds, at most once per frame

    def _request(self, kind: _Rebuild) -> None:
        """Asks for a rebuild, run at once or at the next frame; skips needless ones."""
        if kind == _Rebuild.PROXIES and not (self.state and self.state.proxies):
            return
        if kind == _Rebuild.BREAKDOWN and self.viewer.scene.breakdown is None:
            return
        if self.schedule is None:
            self._run(kind)
            return
        first = not self._pending
        self._pending.add(kind)
        if first:
            self.schedule(self.flush)

    def flush(self) -> None:
        """Does the rebuilds asked for since the last frame."""
        pending, self._pending = self._pending, set()
        if _Rebuild.PREVIEW in pending:
            self._rebuild_preview()
            pending.add(_Rebuild.PROXIES)  # in this frame too, not the next
        if _Rebuild.PROXIES in pending:
            self._rebuild_proxies()
        if _Rebuild.BREAKDOWN in pending:
            self.breakdown.refresh()

    def _run(self, kind: _Rebuild) -> None:
        """Runs a rebuild at once; a new preview also rebuilds the proxy maps."""
        if kind == _Rebuild.BREAKDOWN:
            self.breakdown.refresh()
            return
        if kind == _Rebuild.PREVIEW:
            self._rebuild_preview()
        self._rebuild_proxies()

    def _rebuild_preview(self) -> None:
        """Draws the bundle being drawn again, from its points."""
        state = self.state
        if state is None or state.draft is None:
            return
        draft = state.draft
        if not drawable(draft.points):
            if draft.item_id is not None and draft.editing is None:
                self.viewer.remove(draft.item_id)  # forgets it in the draft
            return
        geometry = replace(draft.geometry, control_points=draft.points)
        try:
            streamlines = build_streamlines(geometry, len(state.bundles))
        except ValueError as e:
            LarmorLogger.warning(f"The bundle cannot be drawn: {e}")
            return
        if draft.item_id is not None:
            self._edited = draft.editing is not None
            self.viewer.replace_data(draft.item_id, streamlines)
            return
        name = f"Bundle {len(state.bundles) + 1}"
        item = self.viewer.add(streamlines, Category.TRACTOGRAMS, name=name)
        self.viewer.set(item.id, OptionName.COLOURING, Colouring.PER_TRACTOGRAM)
        state = self.require()
        assert state.draft is not None
        self._set(
            replace(state, draft=replace(state.draft, item_id=item.id)),
            PhantomPart.DRAFT,
        )

    def _rebuild_proxies(self) -> None:
        """Makes the proxy maps shown again from the phantom as it is now."""
        state = self.state
        if state is None or not state.proxies:
            return
        kinds = [k for k, _ in state.proxies]
        try:
            volumes = proxy_volumes(self._proxy_phantom(state), state.scan, kinds)
        except (OSError, TypeError, ValueError) as e:
            LarmorLogger.warning(f"The proxy maps are not updated: {e}")
            return
        for kind, item_id in state.proxies:
            self.viewer.replace_data(item_id, volumes[kind])

    # Items removed, files and sessions

    def forget(self, item_id: ItemId) -> None:
        """Drops an item that left the scene from the phantom."""
        state = self.state
        if state is None:
            return
        bundles = tuple(b for b in state.bundles if b.item_id != item_id)
        maps = tuple((t, i) for t, i in state.tissue_maps if i != item_id)
        proxies = tuple((k, i) for k, i in state.proxies if i != item_id)
        draft = state.draft
        if draft is not None and draft.item_id == item_id:
            draft = None if draft.editing is not None else replace(draft, item_id=None)
        if (bundles, maps, proxies, draft) == (
            state.bundles,
            state.tissue_maps,
            state.proxies,
            state.draft,
        ):
            return
        changed = replace(
            state, bundles=bundles, tissue_maps=maps, proxies=proxies, draft=draft
        )
        part = PhantomPart.BUNDLES if bundles != state.bundles else PhantomPart.ALL
        self._set(changed, part)
        if bundles != state.bundles or maps != state.tissue_maps:
            self._request(_Rebuild.PROXIES)

    def save(self, path: Path) -> None:
        """Writes the phantom to a phantom file, with the viewer's settings."""
        import larmor.simulation as sim

        state = self.require()
        scene = self.viewer.scene
        bundles = []
        for bundle in state.bundles:
            item = scene.item(bundle.item_id)
            assert isinstance(item.data, Tractogram)
            if bundle.geometry is None and item.source is None:
                raise ValueError(
                    f"{item.name} has no file; save it (File > Save As) first."
                )
            bundles.append((bundle, item.data, item.source))
        maps = []
        for tissue, item_id in state.tissue_maps:
            item = scene.item(item_id)
            assert isinstance(item.data, Volume)
            if item.source is None:
                raise ValueError(f"The map of {tissue} has no file; save it first.")
            maps.append((tissue, item.data, item.source))
        phantom = build_phantom(state, bundles, maps)
        try:
            protocol = build_protocol(state.scan)
        except OSError, ValueError:
            protocol = None  # the viewer block keeps the scan's settings
        sim.write_phantom(phantom, path, protocol=protocol, viewer=self._block(state))

    def _block(self, state: PhantomState) -> dict[str, object]:
        """Returns the viewer's block of a phantom file for this state."""
        scene = self.viewer.scene
        names = {b.item_id: scene.item(b.item_id).name for b in state.bundles}
        options = {
            b.item_id: options_json(scene.item(b.item_id).options)
            for b in state.bundles
        }
        return viewer_block(state, names, options)

    def apply(self, loaded: LoadedPhantom) -> None:
        """Shows a phantom read by `read_phantom_file`, its bundles and maps as items.

        Each bundle is added once, with the streamlines read; none is drawn again.
        """
        self.close()
        state = loaded.state
        self._set(state, PhantomPart.ALL)
        self.viewer._frame(state.grid)
        bundles = []
        for bundle in loaded.bundles:
            item = self.viewer.add(
                bundle.streamlines,
                Category.TRACTOGRAMS,
                name=bundle.name,
                source=bundle.path,
            )
            self._apply_options(item.id, bundle.options)
            settings = replace(bundle.settings, item_id=item.id)
            if settings.geometry is not None and not bundle.options:
                self.viewer.set(item.id, OptionName.COLOURING, Colouring.PER_TRACTOGRAM)
            bundles.append(settings)
        maps = []
        for tissue, volume, path in loaded.maps:
            item = self.viewer.add(
                volume, Category.OVERLAYS, name=path.name, source=path
            )
            maps.append((tissue, item.id))
        state = replace(self.require(), bundles=tuple(bundles), tissue_maps=tuple(maps))
        self._set(state, PhantomPart.ALL)
        for kind in loaded.proxies:
            self.show_proxy(kind, True)

    def _apply_options(self, item_id: ItemId, options: Mapping[str, object]) -> None:
        """Gives an item saved options; those naming items, or refused, are skipped."""
        item = self.viewer.scene.item(item_id)
        for name, (_, info) in option_fields(type(item.options)).items():
            if info.item_ref or name not in options:
                continue
            try:
                self.viewer.set(item_id, OptionName(name), options[name])
            except OptionError, TypeError, ValueError:
                pass

    def session_block(self) -> dict[str, object] | None:
        """Returns the phantom as a session stores it; None without a phantom."""
        state = self.state
        if state is None:
            return None
        scene = self.viewer.scene
        items = {
            str(b.item_id): {
                "name": scene.item(b.item_id).name,
                "options": options_json(scene.item(b.item_id).options),
            }
            for b in state.bundles
        }
        return {"state": state_json(state), "items": items}

    def restore(
        self, block: Mapping[str, Any] | None, ids: Mapping[ItemId, ItemId]
    ) -> list[str]:
        """Restores the phantom of a session whose items are opened.

        Bundles opened from their files are kept; drawn bundles without a file
        are drawn once.

        Returns:
            The problems, one sentence each.
        """
        self._pending.clear()
        self._set(None, PhantomPart.ALL)
        if block is None:
            return []
        try:
            saved = state_from_json(block["state"])
            items = block.get("items", {})
        except (KeyError, TypeError, ValueError) as e:
            return [f"The phantom is not restored: {e!r}."]
        found: list[str] = []
        present = {i.id for i in self.viewer.scene.items}
        state = replace(saved, bundles=(), tissue_maps=(), proxies=())
        self._set(state, PhantomPart.ALL)
        bundles = []
        for index, bundle in enumerate(saved.bundles):
            if ids.get(bundle.item_id) in present:
                bundles.append(replace(bundle, item_id=ids[bundle.item_id]))
                continue
            if bundle.geometry is None:
                found.append(
                    f"A bundle of the phantom was not opened: {bundle.item_id}."
                )
                continue
            extra = items.get(str(bundle.item_id), {})
            item = self.viewer.add(
                build_streamlines(bundle.geometry, index),
                Category.TRACTOGRAMS,
                name=str(extra.get("name") or f"Bundle {index + 1}"),
            )
            self._apply_options(item.id, extra.get("options", {}))
            bundles.append(replace(bundle, item_id=item.id))
        maps = []
        for tissue, item_id in saved.tissue_maps:
            if ids.get(item_id) in present:
                maps.append((tissue, ids[item_id]))
            else:
                found.append(f"The map of {tissue} was not opened.")
        state = replace(self.require(), bundles=tuple(bundles), tissue_maps=tuple(maps))
        self._set(state, PhantomPart.ALL)
        for kind, _ in saved.proxies:
            try:
                self.show_proxy(kind, True)
            except (OSError, ValueError) as e:
                found.append(f"{PROXY_NAMES[kind]} is not shown: {e}")
        return found

    # Runs

    def run_inputs(self) -> RunInputs:
        """Returns what a run simulates; raises ValueError if the phantom cannot run."""
        state = self.require()
        scene = self.viewer.scene
        found = problems(state, item_sources(state, scene))
        if found:
            raise ValueError(" ".join(found))
        bundles = []
        for bundle in state.bundles:
            item = scene.item(bundle.item_id)
            assert isinstance(item.data, Tractogram)
            bundles.append((bundle, item.data, item.source))
        maps = []
        for tissue, item_id in state.tissue_maps:
            item = scene.item(item_id)
            assert isinstance(item.data, Volume)
            maps.append((tissue, item.data, item.source))
        return RunInputs(
            build_phantom(state, bundles, maps),
            build_protocol(state.scan),
            state.scan,
            state,
            self._block(state),
        )

    def finish_run(self, outputs: RunOutputs) -> None:
        """Points the bundles a run wrote at their files, if unchanged since."""
        state = self.state
        if state is None or state.serial != outputs.state.serial:
            return  # another phantom was opened during the run
        files = outputs.files
        scene = self.viewer.scene
        for bundle, path in zip(outputs.state.bundles, files.bundles, strict=False):
            if bundle.geometry is not None and bundle in state.bundles:
                scene.set_source(bundle.item_id, path)
        self.set_output(files.folder)

    def link(self, files: SimulationOutputs) -> None:
        """Points the drawn bundles at a simulation's streamline files, in order."""
        state = self.require()
        scene = self.viewer.scene
        for bundle, path in zip(state.bundles, files.bundles, strict=False):
            if bundle.geometry is not None and path.is_file():
                scene.set_source(bundle.item_id, path)
        self.set_output(files.folder)


def simplify_stroke(
    points: Sequence[Vec3], tolerance: float, most: int = STROKE_POINTS
) -> list[Vec3]:
    """Returns control points for a freehand stroke, evenly spaced along it.

    They are the fewest evenly spaced points, from 2 to `most` and always with
    the stroke's first and last, whose bundle centreline (see
    `centreline_spline`) stays within `tolerance` of the stroke; `most` points
    when no fewer do.

    Args:
        points: The stroke, world mm, in order.
        tolerance: How far the centreline may stray from the stroke, mm.
        most: The most points returned, at least 2.

    Returns:
        The points, in order; one point for a stroke that never moves.
    """
    stroke = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(stroke) == 0:
        return []
    steps = np.linalg.norm(np.diff(stroke, axis=0), axis=1)
    if not np.any(steps > 0):
        return [_vec3(stroke[0])]
    stroke = np.vstack([stroke[:1], stroke[1:][steps > 0]])
    arc = np.concatenate([[0.0], np.cumsum(steps[steps > 0])])
    limit = max(tolerance, 1e-9)
    # The stroke resampled finely enough that nearest samples measure distances.
    count = int(min(np.ceil(arc[-1] / (limit / 4)), 4000)) + 1
    dense = _along(stroke, arc, np.linspace(0.0, arc[-1], count))
    chosen = stroke[[0, -1]]
    for n in range(2, max(most, 2) + 1):
        chosen = _along(stroke, arc, np.linspace(0.0, arc[-1], n))
        chosen[0], chosen[-1] = stroke[0], stroke[-1]
        if _follows(chosen, dense, limit):
            break
    return [_vec3(p) for p in chosen]


def _along(stroke: np.ndarray, arc: np.ndarray, at: np.ndarray) -> np.ndarray:
    """Returns the points of the stroke at the arc lengths `at`, (N, 3).

    Args:
        stroke: The stroke's points, (M, 3).
        arc: The arc length at each of them, (M,), increasing.
        at: The arc lengths wanted, (N,).
    """
    return np.column_stack([np.interp(at, arc, stroke[:, k]) for k in range(3)])


def _follows(control: np.ndarray, dense: np.ndarray, tolerance: float) -> bool:
    """Returns whether the centreline through `control` keeps close to a stroke.

    Every point of the centreline must be within `tolerance` of the finely
    resampled stroke `dense`, and every point of `dense` within `tolerance` of
    the centreline.
    """
    from scipy.spatial import cKDTree

    if np.any(np.linalg.norm(np.diff(control, axis=0), axis=1) == 0):
        return False
    centreline, _ = centreline_spline(control)
    curve = centreline(np.linspace(0.0, 1.0, len(dense)))
    stray = cKDTree(dense).query(curve)[0].max()
    missed = cKDTree(curve).query(dense)[0].max()
    return bool(max(stray, missed) <= tolerance)


def _vec3(point: np.ndarray) -> Vec3:
    """Returns a point as a tuple of three floats."""
    return (float(point[0]), float(point[1]), float(point[2]))


def item_sources(state: PhantomState, scene: Scene) -> dict[ItemId, ItemSource]:
    """Returns what a run needs of the items the loaded bundles and tissue maps name."""
    named: list[tuple[ItemId, type]] = [
        (b.item_id, Tractogram) for b in state.bundles if b.geometry is None
    ]
    named += [(item_id, Volume) for _, item_id in state.tissue_maps]
    found = {}
    for item_id, kind in named:
        try:
            item = scene.item(item_id)
        except ValueError:
            found[item_id] = ItemSource(f"Item {item_id}", False, False)
            continue
        found[item_id] = ItemSource(
            item.name, isinstance(item.data, kind), item.source is not None
        )
    return found
