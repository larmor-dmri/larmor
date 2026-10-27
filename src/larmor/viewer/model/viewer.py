"""Defines the Viewer, whose methods make every change to what the viewer shows."""

import importlib.util
import math
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import replace
from enum import StrEnum
from functools import partial
from numbers import Real
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, TypeIs

import numpy as np

from larmor._log import LarmorLogger
from larmor.io import read_tsf, save_tractogram, save_volume, write_tsf
from larmor.types import GradientTable, Grid, Tractogram, Volume
from larmor.viewer.model.breakdown import BreakdownState
from larmor.viewer.model.camera import (
    PRESETS,
    Camera2D,
    Camera3D,
    CameraPreset,
    Cameras,
    Convention,
    Vec3,
    field_of_view,
    screen_axes,
)
from larmor.viewer.model.detail import (
    VIEW_AXIS,
    Detail,
    DetailId,
    DetailOption,
    FigureBackground,
    clipped_box,
    detail_title,
    picture_size,
)
from larmor.viewer.model.export import (
    DEFAULT_SIZE,
    ExportPreset,
    find_preset,
)
from larmor.viewer.model.item import Category, Item, ItemId, ViewKind
from larmor.viewer.model.labels import labels_in
from larmor.viewer.model.matlab_import import (
    LoadAs,
    MatlabPart,
    MatlabPreset,
    PlannedItem,
    plan_import,
    point_values,
    read_part,
    split_values_spec,
    values_spec,
)
from larmor.viewer.model.options import (
    BOX_SIZE,
    ROLE_COLOURS,
    SPHERE_RADIUS,
    SURFACE_COLOUR,
    SURFACE_COLOURS,
    Colormap,
    Colouring,
    Lengths,
    OptionError,
    OptionName,
    OverlayOptions,
    Position,
    RegionOptions,
    RegionRole,
    RegionShape,
    SurfaceOptions,
    TractogramOptions,
    VolumeOptions,
    option_fields,
    validate,
)
from larmor.viewer.model.phantom import (
    DEFAULT_BOX_MM,
    DEFAULT_VOXEL_MM,
    BundleField,
    LoadedPhantom,
    PhantomState,
    ProxyMap,
    RunInputs,
    RunOutputs,
    ScanField,
    SimulationOutputs,
    TissuePreset,
    box_grid,
    drawable,
    existing_outputs,
    read_phantom_file,
    run_simulation,
    simulation_outputs,
)
from larmor.viewer.model.phantom_editor import PhantomEditor, Schedule, simplify_stroke
from larmor.viewer.model.readout import Readout, closest_axis, read_volume
from larmor.viewer.model.readout import slice_index as volume_slice
from larmor.viewer.model.regions import (
    RegionKey,
    mask_ids,
    mask_regions_of,
    passes_regions,
    region_data,
    region_filter,
    region_key,
)
from larmor.viewer.model.registration import Registration
from larmor.viewer.model.registry import (
    MAX_STREAMLINES,
    SUBSETS,
    Subset,
    item_type,
)
from larmor.viewer.model.scene import Scene
from larmor.viewer.model.session import (
    Session,
    SessionItem,
    SessionReport,
    missing_file,
    read_session,
    region_entries,
    view_state,
    write_session,
)
from larmor.viewer.model.surface import (
    CURVATURE_RANGE,
    CURVATURES,
    SurfaceData,
)
from larmor.viewer.model.view_state import (
    Layout,
    Tool,
    ViewField,
    ViewState,
    Workspace,
)
from larmor.viewer.system import INSTALL_HINT, ViewerUnavailableError

if TYPE_CHECKING:
    # Only for annotations: the model itself never loads pygfx.
    from larmor.viewer.render.offscreen import OffscreenRenderer


class _Mixed:
    """The value of a category's option when its items differ."""

    def __repr__(self) -> str:
        """Returns "MIXED"."""
        return "MIXED"


MIXED: Final = _Mixed()

# The world axis each 2D view moves along when its slice changes (RAS: x, y, z).
_VIEW_AXIS: dict[ViewKind, int] = {
    ViewKind.SAGITTAL: 0,
    ViewKind.CORONAL: 1,
    ViewKind.AXIAL: 2,
}
# How far a 2D view zooms out and in, from the field of view at zoom 1.
ZOOM_RANGE = (0.25, 64.0)
# A rotation video's defaults, a full turn of the 3D view in 12 seconds.
ROTATION_SPEED = 30.0  # degrees per second
ROTATION_DURATION = 12.0  # seconds
ROTATION_FPS = 30
# The colormap of a surface's values other than curvatures, an image's default.
VALUES_COLORMAP: Final = VolumeOptions().colormap
# Called after each frame of a rotation video with the frames drawn and their total;
# returns whether to go on.
type Progress = Callable[[int, int], bool]

# The options a session sets again after the values.
_RESET_BY_VALUES: Final = (
    OptionName.COLOURING,
    OptionName.COLORMAP,
    OptionName.MINIMUM,
    OptionName.MAXIMUM,
)

# The options of a region item that change its data.
_REGION_GEOMETRY: Final = (
    OptionName.CENTRE,
    OptionName.RADIUS,
    OptionName.SIZE,
    OptionName.MASK,
)
# The name a new region of each shape gets, a number after it.
_REGION_NAMES: Final = {
    RegionShape.SPHERE: "Sphere",
    RegionShape.BOX: "Box",
    RegionShape.MASK: "Mask",
}
# The range of a detail window's export scale, times its picture's size.
DETAIL_SCALES: Final = (1.0, 8.0)
# The range of a detail window's glyph detail.
GLYPH_DETAILS: Final = (2, 16)

# The categories whose items `save_item` writes to a file.
SAVABLE: Final = (
    Category.VOLUMES,
    Category.OVERLAYS,
    Category.TRACTOGRAMS,
    Category.FODS,
    Category.PEAKS,
)

# The pointer tools each workspace offers.
WORKSPACE_TOOLS: Final = {
    Workspace.VIEW: (Tool.CURSOR, Tool.SPHERE, Tool.BOX, Tool.ZOOM_BOX),
    Workspace.SIMULATE: (Tool.CURSOR, Tool.DRAW_BUNDLE),
}

# Set on the methods that agents may call; see `agent_action`.
AGENT_ACTION = "agent_action"


def agent_action[F: Callable[..., Any]](method: F) -> F:
    """Marks a Viewer method as one that agents can call as a tool.

    The MCP server makes a tool of each marked method, described by its docstring.
    Mark only methods whose arguments are plain values (numbers, text, paths,
    enums, item ids).

    Returns:
        The method itself, marked.
    """
    setattr(method, AGENT_ACTION, True)
    return method


class Viewer:
    """The viewer's state and the actions that change it.

    Every change to what the viewer shows goes through these methods, whether it
    comes from the window, from `lar bv` or from Python, so each behaviour exists
    once. The window and the drawing follow the scene's events. A viewer needs no
    window: `view` opens one on it, and `screenshot` draws a view off screen.

    Positions are world coordinates in mm, RAS+ (x towards the subject's right,
    y anterior, z superior), as in the items' affines. Categories, views and
    option names are enums: `Category`, `ViewKind` and `OptionName`.

    Args:
        scene: The scene to act on; a new empty one if None.

    Attributes:
        scene: The items and the view state; read it, change it only through the
            actions.

    Example:
        >>> viewer = Viewer()
        >>> t1 = viewer.load("t1.nii.gz", Category.VOLUMES)
        >>> viewer.set(t1.id, OptionName.COLORMAP, Colormap.BONE)
        >>> viewer.screenshot(ViewKind.AXIAL, "axial.png")
    """

    def __init__(self, scene: Scene | None = None) -> None:
        """Creates a viewer on a scene, or on a new empty one."""
        self.scene = scene if scene is not None else Scene()
        # Option values that new items of a category start with, in place of the
        # options class's defaults.
        self._defaults: dict[Category, dict[OptionName, object]] = {}
        # The labels in each overlay, by its id and volume index, counted on first use.
        self._labels: dict[tuple[ItemId, int], tuple[int, ...]] = {}
        # Each tractogram's TSF values, by its id, read when its TSF file is set.
        self._tsf_values: dict[ItemId, np.ndarray] = {}
        # Each tractogram's last selection by its regions, with the region options
        # it was made for; None for every streamline.
        self._selections: dict[ItemId, tuple[RegionKey, np.ndarray | None]] = {}
        # How many times each item's data was replaced; unreplaced items are absent.
        self._versions: dict[ItemId, int] = {}
        # Carries out the phantom actions; the phantom itself is `scene.phantom`.
        self._phantom = PhantomEditor(self)
        # How many control points each click or stroke added to the bundle being
        # drawn, so that Backspace takes back a whole stroke.
        self._point_groups: list[int] = []
        # The radius and size the next sphere and box get when none is given,
        # those of the last one added or resized.
        self.region_radius: float = SPHERE_RADIUS
        self.box_size: Lengths = BOX_SIZE

    # Items

    @agent_action
    def load(
        self,
        path: str | Path,
        category: Category,
        *,
        max_streamlines: int | None = MAX_STREAMLINES,
        transforms: list[str] | None = None,
        reference: str | Path | None = None,
        inverse: bool = False,
        interpolation: str | None = None,
        format: str | None = None,
        warp_type: str | None = None,
        fsl_input_path: str | Path | None = None,
        fsl_reference_path: str | Path | None = None,
    ) -> Item:
        """Reads a file into a new item of `category`, as `add` adds it.

        The file is read before the method returns. Volumes, overlays, FODs and
        peaks are NIfTI (.nii, .nii.gz) or MRtrix (.mif, .mif.gz, .mih) images;
        tractograms are TCK, TRK or VTK files; surfaces are GIfTI (.gii) or
        FreeSurfer surfaces (such as lh.white). For a FreeSurfer surface,
        lh.curv in its folder is read as its values "curv", and
        ../label/lh.cortex.label as its cortex, when they exist.

        Args:
            path: The file to read.
            category: Where the item goes: `Category.VOLUMES`, `OVERLAYS` (masks,
                maps, parcellations), `TRACTOGRAMS`, `FODS`, `PEAKS` (from
                `lar peaks`) or `SURFACES`.
            max_streamlines: A tractogram with more streamlines keeps a random
                subset of this size; the same file always gives the same subset.
                None keeps all. Ignored for images.
            transforms: Bring the item into another space through a registration
                computed by ANTs, FSL or MRtrix3: its transform files, listed as
                antsApplyTransforms lists them (an affine, a warp, or a warp and
                the inverse warp its tool wrote as "warp,inverse_warp"). The
                files are those that resample an image into the reference space;
                tractograms and surfaces are brought into the same space. FODs
                cannot be brought through a registration. None loads the file
                as it is.
            reference: With transforms: an image of the reference space, whose
                grid images are resampled onto (needed for images).
            inverse: With transforms: go the other way, from the reference
                space into the space the transforms bring images from.
            interpolation: With transforms, for images: "nearest", "linear" or
                "cubic"; None is linear for volumes and nearest for overlays and
                peaks (whose directions are reoriented).
            format: With transforms: the tool that wrote them, "ants", "fsl" or
                "mrtrix"; None guesses from the files (a 4D NIfTI warp can be
                FSL's or MRtrix3's, so it needs format).
            warp_type: With transforms: the warp fields' type, "deformation" (the
                default) or "displacement" for MRtrix3, "relative" (the default)
                or "absolute" for FSL; checked against the fields' values.
            fsl_input_path: With FSL transforms: the input image of the FLIRT or
                FNIRT run; defaults to the item when it is an image (to the
                reference with inverse). Tractograms and surfaces need it.
            fsl_reference_path: With FSL transforms: the run's reference image;
                defaults to `reference` (to the item with inverse). Tractograms
                and surfaces need it with inverse.

        Returns:
            The new item; its `id` names it in the other actions.

        Raises:
            TypeError: If category is not a Category.
            ValueError: If the file's extension is not one the category reads,
                the file cannot be read as that format, an FOD's number of
                coefficients fits no SH layout, or the registration cannot be
                applied (a transform file it cannot read, an image without a
                reference, an FOD).
            OSError: If the file cannot be opened.
        """
        _require(category, Category)
        p = Path(path).absolute()  # not resolved, to keep symlinks as given
        registration = None
        if transforms:
            registration = Registration.from_files(
                list(transforms),
                reference,
                inverse,
                format,
                interpolation,
                warp_type,
                fsl_input_path,
                fsl_reference_path,
            )
        data = item_type(category).load(p, None, max_streamlines)
        if registration is not None:
            data = registration.apply(data, category)
        return self.add(
            data, category, name=p.name, source=p, registration=registration
        )

    @agent_action
    def load_matlab(
        self,
        path: str | Path,
        preset: MatlabPreset,
        *,
        reference: str | Path | None = None,
        variables: dict[str, LoadAs] | None = None,
        tractogram_id: ItemId | None = None,
        max_streamlines: int | None = MAX_STREAMLINES,
    ) -> list[Item]:
        """Reads what a MATLAB .mat file holds into new items, as chosen.

        The presets read MRIToolkit's results. `TENSOR_FIT` (a DTI or DKI fit)
        adds FA, MD and the eigenvalues as volumes, the first eigenvector as a
        volume with DTI colours modulated by the FA, and the DWI. `FOD_FIT` (a GRL
        or mFOD result) adds its FODs (SH, MRtrix basis) and its fractions and
        residuals as volumes. `TRACTOGRAPHY` adds the tracts as a tractogram.
        `MANUAL` reads the variables named in `variables` (paths as in
        `read_matlab`, such as "FA", "DT{1}" or "GRL_Results.FOD") each as its
        `LoadAs`: an image (MRIToolkit's layout) as a volume, overlay, FODs or
        peaks; a cell of (N, 3) points as a tractogram; a struct with vertices
        and faces as a surface; or a cell of one value per point as the values
        `tractogram_id` is coloured by. A choice that does not fit its variable's
        shape is refused. The file is read before the method returns; sessions
        reopen the items from the same variables.

        Args:
            path: The .mat file.
            preset: `MatlabPreset.TENSOR_FIT`, `FOD_FIT`, `TRACTOGRAPHY` or `MANUAL`.
            reference: The grid of files that store none (FOD results, tracts, and
                images of files without a header): the MRIToolkit fit (.mat) they
                came from, or a NIfTI or MIF file of its image.
            variables: For `MANUAL`, what to read each variable as.
            tractogram_id: The tractogram item coloured by a variable read as
                `LoadAs.POINT_VALUES`.
            max_streamlines: As in `load`.

        Returns:
            The new items, in the order added.

        Raises:
            ValueError: If the preset does not fit the file, a reference is
                needed and not given, or a variable cannot be read as chosen.
            OSError: If a file cannot be opened.
        """
        _require(preset, MatlabPreset)
        p = Path(path).absolute()
        ref = None if reference is None else Path(reference).absolute()
        planned, values = plan_import(p, preset, ref, variables, tractogram_id)
        # Everything is read, and the values checked, before anything is added.
        read = [read_part(p, plan.part, None, max_streamlines) for plan in planned]
        if values:
            assert tractogram_id is not None
            target = self.scene.item(tractogram_id).data
            if not isinstance(target, Tractogram):
                raise ValueError(f"Item {tractogram_id} is not a tractogram.")
            for variable in values:
                point_values(p, variable, target, SUBSETS.get(target))
        items: list[Item | None] = [
            self.add(data, plan.category, name=plan.name, source=p, part=plan.part)
            for plan, data in zip(planned, read, strict=True)
        ]
        for problem in self._apply_import(planned, items):
            LarmorLogger.warning(problem)
        for variable in values:
            assert tractogram_id is not None
            self.set(tractogram_id, OptionName.VALUES_FILE, values_spec(p, variable))
        return [i for i in items if i is not None]

    def _apply_import(
        self, planned: list[PlannedItem], items: list[Item | None]
    ) -> list[str]:
        """Gives the items a MATLAB import added their first options.

        Args:
            planned: The import's plan.
            items: The item added for each planned one, None where it failed.

        Returns:
            The options that could not be set, one sentence each.
        """
        problems = []
        for plan, item in zip(planned, items, strict=True):
            if item is None:
                continue
            options = dict(plan.options)
            if plan.modulate_by is not None:
                source = items[plan.modulate_by]
                if source is not None:
                    options[OptionName.MODULATE] = source.id
            for name, value in options.items():
                try:
                    self.set(item.id, name, value)
                except (OptionError, ValueError) as e:
                    problems.append(f"{item.name}: {e}")
        return problems

    def add(
        self,
        data: object,
        category: Category,
        *,
        name: str | None = None,
        source: Path | None = None,
        part: MatlabPart | None = None,
        registration: Registration | None = None,
    ) -> Item:
        """Adds an object already in memory, without copying it.

        The viewer never changes the object. A volume becomes the main volume,
        the only visible one. An image starts with its minimum and maximum at its
        data's range (scaled by its slope and intercept), and an overlay with its
        threshold at its minimum. A surface's colour comes from its name (white
        yellow, pial red, others light grey), and one with values "curv" is
        coloured by them. The first item with a grid sets the 2D views' field of
        view to that grid's box and puts the cursor at its centre.

        Args:
            data: A `Volume` for volumes and overlays (3D, or 4D with volumes along
                the last axis), a `Tractogram` for tractograms, for FODs a 4D
                `Volume` with SH coefficients along the last axis: even degrees
                only ((lmax + 1)(lmax + 2) / 2 coefficients) or, for an
                asymmetric FOD, all degrees ((lmax + 1)^2), and for peaks a 4D
                `Volume` with x, y and z of each peak's direction scaled by its
                amplitude (3 volumes per peak, NaN or 0 where there are fewer
                peaks), in world coordinates, and for surfaces a `Surface`.
            category: Where the item goes: `Category.VOLUMES`, `OVERLAYS`,
                `TRACTOGRAMS`, `FODS`, `PEAKS` or `SURFACES`.
            name: The name shown in Layers; "Volume 2" and so on if None.
            source: The file the object came from, if any.
            part: What it was read from inside a MATLAB file `source`, if so; a
                session reopens it from there.
            registration: The registration `data` was brought through when read
                from `source`, if so; a session applies it again.

        Returns:
            The new item; its `id` names it in the other actions.

        Raises:
            TypeError: If category is not a Category.
            ValueError: If data is not of the category's type, an FOD's number
                of coefficients fits no SH layout, or a peaks image is not 4D with
                a multiple of 3 volumes.
        """
        had_grid = self._reference_grid() is not None
        _require(category, Category)
        if category == Category.REGIONS:
            raise ValueError(
                "Regions are added with add_sphere, add_box or add_mask, not add."
            )
        kind = item_type(category)

        data = _checked_data(category, data)

        given_name = name
        if name is None:  # "Volume 2"
            number = len(self.scene.in_category(category)) + 1
            name = f"{kind.label[:-1]} {number}"

        defaults: dict[str, Any] = {
            str(name): value for name, value in self._defaults.get(category, {}).items()
        }
        options = replace(kind.options(), **defaults)
        if category == Category.VOLUMES:
            # Hidden first, so that no event sees two visible volumes.
            options = replace(options, visible=False)

        value_range = None
        if isinstance(options, VolumeOptions):  # volumes and overlays
            assert isinstance(data, Volume)
            value_range = _value_range(data)
            if isinstance(options, OverlayOptions):
                options = replace(options, threshold=value_range[0])
            options = replace(options, minimum=value_range[0], maximum=value_range[1])
        if isinstance(options, SurfaceOptions):
            assert isinstance(data, SurfaceData)
            options, value_range = _surface_options(data, options, given_name)

        item = self.scene.add(
            category, name, source, data, options, value_range, part, registration
        )
        if category == Category.VOLUMES:
            self._show_volume(item)
        if not had_grid:
            grid = self._reference_grid()
            if grid is not None:
                self._frame(grid)
        return item

    def _frame(self, grid: Grid) -> None:
        """Fits the views to a grid: its box, new cameras, the cursor at its centre."""
        # A new scene: the cameras of the one before do not fit this grid.
        cameras = replace(
            self.scene.view.cameras,
            axial=Camera2D(),
            coronal=Camera2D(),
            sagittal=Camera2D(),
            three_d=Camera3D(),
        )
        self.scene.set_view(ViewField.CAMERAS, cameras)
        self.scene.set_view(ViewField.FIELD_OF_VIEW, field_of_view(grid))
        self.move_cursor(_centre(grid))

    @agent_action
    def remove(self, target: ItemId | Category) -> None:
        """Removes an item, or every item of a category.

        When the main volume is removed, the next volume in Layers order becomes
        the main volume, and is shown if the removed one was. The views keep
        their field of view.

        Args:
            target: An item's id, or a Category to remove all its items.

        Raises:
            TypeError: If target is neither an item id nor a Category.
            ValueError: If there is no item with that id.
        """
        _require_target(target)
        if isinstance(target, Category):
            # The main volume goes last, so that no other volume is shown on the way.
            main = self.scene.view.main_volume
            items = sorted(self.scene.in_category(target), key=lambda i: i.id == main)
            for item in items:
                self.remove(item.id)
        else:
            item = self.scene.item(target)
            # A mask region goes with its image.
            for region in mask_regions_of(self.scene, target):
                self.remove(region.id)
            if self.scene.view.selected == target:
                self.scene.set_view(ViewField.SELECTED, None)
            # Before it goes, so that no option ever names an item not in the scene.
            self._drop_references(target)
            if item.id == self.scene.view.main_volume:
                self._pass_main_volume(item)
            self.scene.remove(target)
            self._phantom.forget(target)
            self._labels = {k: v for k, v in self._labels.items() if k[0] != target}
            self._tsf_values.pop(target, None)
            self._selections.pop(target, None)
            self._versions.pop(target, None)

    def _drop_references(self, target: ItemId) -> None:
        """Removes an item's id from every option of the other items that names it."""
        for other in self.scene.items:
            for name, (_, info) in option_fields(other.options).items():
                named = getattr(other.options, name)
                if info.item_ref and isinstance(named, tuple) and target in named:
                    kept = tuple(i for i in named if i != target)
                    self.scene.set_option(other.id, name, kept)
                elif info.item_ref and named == target:
                    self.scene.set_option(other.id, name, None)

    def _pass_main_volume(self, item: Item) -> None:
        """Makes the next volume the main one, shown if item was, before item goes."""
        others = [
            v for v in self.scene.in_category(Category.VOLUMES) if v.id != item.id
        ]
        if not others:
            self.scene.set_view(ViewField.MAIN_VOLUME, None)
        elif item.options.visible:
            self._show_volume(others[0])
        else:
            self.scene.set_view(ViewField.MAIN_VOLUME, others[0].id)

    def clear(self) -> None:
        """Removes every item, closes the phantom and the detail windows.

        The views stay as they are.
        """
        self._phantom.close()
        for detail_id in list(self.scene.details):
            self.scene.remove_detail(detail_id)
        for category in Category:
            self.remove(category)

    @agent_action
    def rename(self, item_id: ItemId, name: str) -> None:
        """Gives an item a new name, as the scene tree and sessions show it.

        Args:
            item_id: The item.
            name: The new name; spaces at its ends are dropped.

        Raises:
            ValueError: If there is no item with that id, or the name is empty.
        """
        text = str(name).strip()
        if not text:
            raise ValueError("A name needs at least one character.")
        self.scene.rename(item_id, text)

    @agent_action
    def save_item(self, item_id: ItemId, path: str | Path) -> Item:
        """Writes an item's data to a file, which the item then refers to.

        The format follows the extension. Tractograms are written as TCK, TRK,
        VTK or FIB; a TRK is written for the tractogram's reference grid, else
        for the grid the 2D views slice along. Volumes, overlays, FODs and peaks
        are written as NIfTI (.nii, .nii.gz) or MIF (.mif, .mif.gz, .mih).
        Afterwards the item refers to the new file, so a session saved later
        reopens it from there, also when it was added without a file. A
        tractogram read as a random subset of a larger file is written as that
        subset. Surfaces cannot be saved.

        Args:
            item_id: The item to save.
            path: The file to write; an existing file is replaced.

        Returns:
            The item, now referring to the file.

        Raises:
            ValueError: If there is no item with that id, it is a surface, or the
                extension is not one its category writes.
            OSError: If the file cannot be written.

        Example:
            >>> viewer.save_item(bundle.id, "bundle.tck")
        """
        item = self.scene.item(item_id)
        p = Path(path).absolute()  # not resolved, to keep symlinks as given
        data = item.data
        if isinstance(data, Tractogram):
            subset = SUBSETS.get(data)
            if (
                subset is not None
                and item.source is not None
                and p.resolve() == item.source.resolve()
            ):
                raise ValueError(
                    f"{item.name} holds {data.num_streamlines:,} of the "
                    f"{subset.total:,} streamlines of {p.name}; saving it there "
                    "would lose the others. Choose another file."
                )
            reference = data.reference or self._reference_grid()
            save_tractogram(data, p, reference=reference)
            SUBSETS.pop(data, None)  # the new file holds just these streamlines
            if subset is not None:
                self._save_subset_values(item, p)
        elif isinstance(data, Volume):
            save_volume(data, p)
        else:
            label = item_type(item.category).label.lower()
            raise ValueError(f"{item.name}: {label} cannot be saved to a file.")
        self.scene.set_source(item_id, p)
        return item

    def _save_subset_values(self, item: Item, path: Path) -> None:
        """Writes the TSF values of a subset saved to path beside it, and uses them.

        The TSF file the values were read from holds a value per point of the
        whole file, which no longer matches the saved streamlines.
        """
        o = item.options
        values = self._tsf_values.get(item.id)
        if not isinstance(o, TractogramOptions) or not o.values_file or values is None:
            return
        assert isinstance(item.data, Tractogram)
        name = path.name
        for extension in item_type(item.category).extensions:
            if name.lower().endswith(extension):
                name = name[: -len(extension)]
                break
        tsf = path.with_name(f"{name}.tsf")
        write_tsf(values, item.data.offsets, tsf)
        self.scene.set_option(item.id, OptionName.VALUES_FILE, str(tsf))

    def replace_data(self, item_id: ItemId, data: object) -> None:
        """Gives an item new data, keeping its id, options and place in Layers.

        The data is checked as `add` checks it, and is not copied. The item then
        refers to no file: a session leaves it out until it is saved with
        `save_item`. Options that depend on the data follow it: an image's
        minimum, maximum and threshold keep their place in the data's range,
        mapped from the old range onto the new one (min stays at most max); a
        volume index past the new volumes becomes the last; overlay labels the new
        data lacks are dropped; a tractogram's TSF values, one per old point, are
        dropped; a surface's values to colour by that the new data lacks are
        cleared.

        Args:
            item_id: The item.
            data: The new data, of the item's category's type.

        Raises:
            ValueError: If there is no item with that id, or data is not of the
                category's type (or, for FODs and peaks, not of their layout).
        """
        item = self.scene.item(item_id)
        data = _checked_data(item.category, data)
        o = item.options
        new_range = item.value_range
        if isinstance(o, VolumeOptions):
            assert isinstance(data, Volume)
            new_range = _value_range(data)
        elif isinstance(o, SurfaceOptions):
            assert isinstance(data, SurfaceData)
            _, new_range = _surface_options(data, o, item.name)
        # These options first, each valid for the old data and the new, so that the
        # drawing rebuilt for the new data never sees options that do not fit it.
        # The range options come after the data, as they fit only the new range.
        changes, ranged = _changes_for_data(item, data, new_range)
        for name, value in changes.items():
            self.scene.set_option(item_id, name, value)
        self._forget_data(item_id)
        self.scene.replace_data(item_id, data, new_range)
        # The maximum first when the window moves up, so that min <= max throughout.
        names = sorted(ranged, key=lambda n: n != OptionName.MAXIMUM)
        if ranged.get(OptionName.MINIMUM, 0.0) <= getattr(o, OptionName.MAXIMUM, 0.0):
            names.reverse()
        for name in names:
            self.scene.set_option(item_id, name, ranged[name])

    def _forget_data(self, item_id: ItemId) -> None:
        """Drops what was derived from an item's old data and bumps its version."""
        self._labels = {k: v for k, v in self._labels.items() if k[0] != item_id}
        self._tsf_values.pop(item_id, None)
        self._versions[item_id] = self._versions.get(item_id, 0) + 1
        # Its own selection, and those of the tractograms it is a mask of.
        for other in list(self._selections):
            tract = self.scene.item(other)
            if other == item_id or item_id in mask_ids(self.scene, tract):
                del self._selections[other]

    # Sessions

    @agent_action
    def save_session(self, path: str | Path) -> list[str]:
        """Saves the items and the state of the views to a session file.

        The file is JSON (`.lar.json` is the usual extension). It holds each
        item's category, name, file (its absolute path and its path from the
        session file's folder, so a project folder can be moved) and options,
        and the view state: cursor, cameras, main volume, selection, maximised
        view, convention and the overlay and crosshair switches, and the phantom
        being edited, if any, whose bundles and proxy maps are made again when
        the session opens. The application's settings are not saved. Items
        added from Python without a file cannot be saved and are left out, with
        a warning in the log.

        Args:
            path: The session file to write; an existing file is replaced.

        Returns:
            The names of the items left out, in Layers order.

        Raises:
            OSError: If the file cannot be written.
            ValueError: If an option holds a value JSON cannot store, such as
                an infinite number.

        Example:
            >>> skipped = viewer.save_session("project/view.lar.json")
        """
        state = self.scene.phantom
        owned = state.owned() if state is not None else set()
        # Regions have no file: the session holds them in a list of their own.
        owned |= {r.id for r in self.scene.in_category(Category.REGIONS)}
        block = self._phantom.session_block()
        skipped = write_session(
            self.scene, Path(path), block, owned, region_entries(self.scene)
        )
        for name in skipped:
            LarmorLogger.warning(f"{name} has no file, so the session leaves it out.")
        return skipped

    @agent_action
    def load_session(
        self, path: str | Path, *, max_streamlines: int | None = MAX_STREAMLINES
    ) -> SessionReport:
        """Replaces the scene with the one a session file holds.

        Each file is looked for at its path from the session file's folder,
        then at its absolute path, and read before the method returns, as
        `load` reads it. A file that is missing or cannot be read, and an
        option or a view-state field that is no longer allowed, are reported
        (and logged), and the rest of the session still opens.

        Args:
            path: A session file written by `save_session`.
            max_streamlines: A tractogram with more streamlines keeps a random
                subset of this size, as in `load`; None keeps all.

        Returns:
            The items opened and the problems met, one sentence each.

        Raises:
            SessionError: If the file is not a session, or was saved by a newer
                Larmor; the scene is then unchanged.
            OSError: If the session file cannot be read.

        Example:
            >>> report = viewer.load_session("project/view.lar.json")
            >>> for problem in report.problems:
            ...     print(problem)
        """
        session = read_session(Path(path))
        self.clear()
        ids: dict[ItemId, ItemId] = {}
        problems = list(session.problems)
        for entry in session.items:
            source = entry.locate()
            if source is None:
                problems.append(missing_file(entry))
                continue
            try:
                if entry.part is not None:
                    data = read_part(source, entry.part, None, max_streamlines)
                else:
                    data = item_type(entry.category).load(source, None, max_streamlines)
                if entry.registration is not None:
                    data = entry.registration.apply(data, entry.category)
                item = self.add(
                    data,
                    entry.category,
                    name=entry.name,
                    source=source,
                    part=entry.part,
                    registration=entry.registration,
                )
            except Exception as e:  # any failure is reported, and the rest opens
                problems.append(f"Cannot open {source}: {e}")
                continue
            ids[entry.id] = item.id
        problems += self._apply_session(session, ids)
        for problem in problems:
            LarmorLogger.warning(problem)
        items = tuple(self.scene.item(i) for i in ids.values())
        return SessionReport(items, tuple(problems))

    def _apply_session(
        self, session: Session, ids: Mapping[ItemId, ItemId]
    ) -> list[str]:
        """Gives a session's items, once added, their options, then sets its view.

        `load_session` calls it once the files are read; the window calls it once
        its loader has read them in the background.

        Args:
            session: The session, from `read_session`.
            ids: The id each session item had when saved, mapped to the id of the
                item added for it; items missing from it were not opened.

        Returns:
            The options and view-state fields refused, one sentence each.
        """
        problems: list[str] = []
        # Only the items still in the scene: one may be removed while files load.
        present = {item.id for item in self.scene.items}
        ids = {old: new for old, new in ids.items() if new in present}
        entries = [e for e in session.items if e.id in ids]
        names = {e.id: e.name for e in session.items}
        # The regions next, as the tractograms' options name them.
        added, refused = self._restore_regions(session, ids)
        problems += refused
        ids = {**ids, **added}
        names |= {e.id: e.name for e in session.regions}
        # Options naming an item wait until every item has its other options.
        for refs in (False, True):
            for entry in entries:
                problems += self._restore_options(entry, ids, names, refs)
        # Choosing a tractogram's values resets its colouring and range, and the
        # volume is set last, as it names an item: the saved ones are set again.
        for entry in entries:
            self._restore_reset_by_values(entry, ids)
        problems += self._phantom.restore(session.phantom, ids)
        state, refused = view_state(session.view, ids, self.scene.view)
        problems += refused
        try:
            self.set_view_state(state)
        except (TypeError, ValueError) as e:
            problems.append(f"The view is not restored: {e}")
        return problems

    def _restore_options(
        self,
        entry: SessionItem,
        ids: Mapping[ItemId, ItemId],
        names: Mapping[ItemId, str],
        refs: bool,
    ) -> list[str]:
        """Sets an opened item's saved options, either those naming items or the rest.

        Returns:
            The options refused or the items dropped from them, one sentence each.
        """
        problems: list[str] = []
        item = self.scene.item(ids[entry.id])
        for name, (_, info) in option_fields(item.options).items():
            if info.item_ref != refs or name not in entry.options:
                continue
            value = entry.options[name]
            if refs and isinstance(value, list):
                # Items' ids; those of items not opened are dropped.
                for i in value:
                    if not _known_id(i, ids):
                        name_of = names.get(i) if _is_id(i) else None
                        missing = name_of or f"item {i!r}"
                        problems.append(
                            f"{item.name}: {info.label}: {missing} is "
                            "dropped, as it was not opened."
                        )
                value = [ids[i] for i in value if _known_id(i, ids)]
            elif refs and value is not None:
                if (
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value not in ids
                ):
                    continue  # names an item not opened: stays None
                value = ids[value]
            try:
                self.set(item.id, name, value)
            except (OptionError, TypeError, ValueError) as e:
                problems.append(f"{item.name}: {info.label} is ignored: {e}")
        return problems

    def _restore_reset_by_values(
        self, entry: SessionItem, ids: Mapping[ItemId, ItemId]
    ) -> None:
        """Sets again the saved options that choosing a tractogram's values resets."""
        item = self.scene.item(ids[entry.id])
        o = item.options
        if not isinstance(o, TractogramOptions) or (
            o.values_volume is None and not o.values_file
        ):
            return
        for name in _RESET_BY_VALUES:
            if name in entry.options:
                try:
                    self.set(item.id, name, entry.options[name])
                except OptionError, TypeError, ValueError:
                    pass  # reported when the options were first set

    def _restore_regions(
        self, session: Session, ids: Mapping[ItemId, ItemId]
    ) -> tuple[dict[ItemId, ItemId], list[str]]:
        """Adds a session's regions, a mask region's image named through `ids`.

        Returns:
            The id each region had when saved, mapped to the new region's id, and
            the regions or options refused, one sentence each.
        """
        added: dict[ItemId, ItemId] = {}
        problems: list[str] = []
        for entry in session.regions:
            changes: dict[str, Any] = {}
            for key, value in entry.options.items():
                try:
                    name = OptionName(key)
                    if name == OptionName.MASK:
                        changes[name] = ids.get(value) if _is_id(value) else None

                        continue

                    changes[name] = validate(RegionOptions, name, value)
                except (OptionError, TypeError, ValueError) as e:
                    problems.append(f"{entry.name}: {key} is ignored: {e}")
            options = replace(RegionOptions(), **changes)
            if options.shape == RegionShape.MASK and options.mask is None:
                problems.append(
                    f"{entry.name} is dropped, as the image it is a mask of was not "
                    "opened."
                )
                continue
            added[entry.id] = self._add_region(
                options, entry.name, apply=False, defaults=False
            )

        return added, problems

    # Phantom

    @property
    def phantom(self) -> PhantomState | None:
        """The phantom being edited, as plain values; None if no phantom is open."""
        return self.scene.phantom

    @property
    def drawing(self) -> bool:
        """Whether a bundle is being drawn, so clicks in the 2D views add points."""
        state = self.scene.phantom
        return state is not None and state.draft is not None

    def set_scheduler(self, schedule: Schedule | None) -> None:
        """Runs the phantom's rebuilds through schedule, or at once if None.

        The window uses it; a script has no need to. The window passes a
        function that runs its argument at the next frame, so that several point
        edits in one frame draw the bundle being drawn only once.

        Args:
            schedule: Called with a function to run later; None runs it at once.
        """
        self._phantom.schedule = schedule

    def flush_phantom(self) -> None:
        """Does the phantom's rebuilds that wait for the next frame, now.

        The window uses it with `set_scheduler`; a script has no need to.
        """
        self._phantom.flush()

    @agent_action
    def new_phantom(
        self,
        size_mm: tuple[float, float, float] = DEFAULT_BOX_MM,
        voxel_mm: float = DEFAULT_VOXEL_MM,
        *,
        grid_of: ItemId | None = None,
    ) -> None:
        """Starts a phantom to simulate, replacing the one open.

        A phantom is a voxel grid with named tissues (WM, GM and CSF to start
        with), bundles drawn in the 2D views (each an ordinary tractogram item)
        or loaded tractograms used as bundles, tissue maps and a scan. The
        views are fitted to its grid. The items the old phantom made are removed.

        Args:
            size_mm: The grid's box (x, y, z) in mm, centred on the origin.
            voxel_mm: The voxel size in mm.
            grid_of: Use the grid of this loaded image instead of a box.

        Raises:
            ValueError: If a size is not positive, or grid_of is not an image.
        """
        self._phantom.new(self._grid_of(grid_of, size_mm, voxel_mm))

    @agent_action
    def close_phantom(self) -> None:
        """Closes the phantom: its bundles and proxy maps are removed."""
        self._phantom.close()

    @agent_action
    def set_phantom_grid(
        self,
        size_mm: tuple[float, float, float] = DEFAULT_BOX_MM,
        voxel_mm: float = DEFAULT_VOXEL_MM,
        *,
        grid_of: ItemId | None = None,
    ) -> None:
        """Moves the phantom onto a new grid; bundles keep their place in mm.

        Args:
            size_mm: The grid's box (x, y, z) in mm, centred on the origin.
            voxel_mm: The voxel size in mm.
            grid_of: Use the grid of this loaded image instead of a box.

        Raises:
            ValueError: If no phantom is open, a size is not positive, grid_of
                is not an image, or a tissue map is not on the new grid.
        """
        self._phantom.set_grid(self._grid_of(grid_of, size_mm, voxel_mm))

    def _grid_of(
        self, item_id: ItemId | None, size_mm: Sequence[float], voxel_mm: float
    ) -> Grid:
        """Returns the grid of an image item, or a box grid if item_id is None."""
        if item_id is None:
            return box_grid(size_mm, voxel_mm)
        data = self.scene.item(item_id).data
        if not isinstance(data, Volume):
            raise ValueError(f"{self.scene.item(item_id).name} is not an image.")
        return data.grid

    def _phantom_grid(self) -> Grid:
        """Returns the grid a phantom started by a key gets, the views' or a box."""
        grid = self._reference_grid()
        return grid if grid is not None else box_grid(DEFAULT_BOX_MM, DEFAULT_VOXEL_MM)

    @agent_action
    def set_tissue(
        self,
        name: str,
        preset: TissuePreset,
        values: dict[str, float] | None = None,
    ) -> None:
        """Adds a named tissue, or changes one; every bundle using it follows.

        Args:
            name: The tissue's name, such as "WM".
            preset: `TissuePreset.STANDARD_MODEL` (stick, zeppelin, free water),
                `SANDI` (sphere, powder-averaged stick, ball), `NODDI`
                (Watson-dispersed sticks, zeppelin, free water), `FREE_WATER`
                (free water alone) or `CUSTOM` (its compartments, see
                `set_compartments`; a preset tissue turned custom keeps its
                compartments).
            values: Parameters of the preset to change, by the names of
                `larmor.simulation.standard_model_tissue`, `sandi_tissue`,
                `noddi_tissue` or `free_water_tissue` (such as "intra_t2_ms" or
                "proton_density"); the others keep their values.

        Raises:
            ValueError: If no phantom is open, a parameter is unknown, or the
                values make no valid tissue.
        """
        _require(preset, TissuePreset)
        self._phantom.set_tissue(name, preset, values)

    @agent_action
    def set_compartments(self, name: str, compartments: list[dict[str, Any]]) -> None:
        """Makes a tissue of compartments, as a custom tissue.

        Args:
            name: The tissue's name; a new one is added.
            compartments: One dict each: "kind" (the value of a
                `larmor.simulation.CompartmentModel`: stick, zeppelin, ball,
                powder_stick, sphere, watson_stick, cylinder, dot or
                free_water), "fraction", "diffusivity" (mm^2/s), "t1_ms",
                "t2_ms", "t2star_ms", and optionally
                "proton_density", "perpendicular_diffusivity" (mm^2/s),
                "radius_um" and "kappa". The fractions sum to 1.

        Raises:
            ValueError: If no phantom is open, or the compartments make no valid
                tissue.
        """
        self._phantom.set_compartments(name, compartments)

    @agent_action
    def remove_tissue(self, name: str) -> None:
        """Removes a tissue that no bundle, map or background uses.

        Args:
            name: The tissue's name.

        Raises:
            ValueError: If no phantom is open, or the tissue is unknown or used.
        """
        self._phantom.remove_tissue(name)

    @agent_action
    def set_tissue_map(self, tissue: str, item_id: ItemId | None) -> None:
        """Makes an overlay the map of a tissue's fraction in each voxel.

        The bundles fill each voxel first; the maps share what they leave.

        Args:
            tissue: The tissue, one without a fibre direction (such as GM).
            item_id: A 3D overlay on the phantom's grid with values in [0, 1];
                None clears the tissue's map.

        Raises:
            ValueError: If no phantom is open, or the tissue or overlay is not one
                a map can use.
        """
        self._phantom.set_tissue_map(tissue, item_id)

    @agent_action
    def set_background_tissue(self, tissue: str | None) -> None:
        """Fills what the bundles and maps leave of each voxel with a tissue.

        Args:
            tissue: A tissue without a fibre direction, or None for no signal.

        Raises:
            ValueError: If no phantom is open or the tissue cannot fill space.
        """
        self._phantom.set_background(tissue)

    @agent_action
    def set_scan(self, name: ScanField, value: object) -> None:
        """Changes one setting of the phantom's scan.

        Args:
            name: `ScanField.SHELLS` ((b-value s/mm^2, directions) pairs),
                `NUM_B0`, `GRADIENT_SEED` (of the directions), `GRADIENT_FILES`
                (bvals and bvecs paths, directions in world coordinates, used
                instead of the shells; None for the shells), `SMALL_DELTA_MS`,
                `BIG_DELTA_MS`, `TE_MS` (one or several echo times), `TR_MS`,
                `SEQUENCE` (`EchoKind`), `SNR` (None for no noise) or `SEED` (of
                the noise).
            value: The new value.

        Raises:
            TypeError: If name is not a ScanField.
            ValueError: If no phantom is open or the value is refused.
        """
        _require(name, ScanField)
        self._phantom.set_scan(name, value)

    @agent_action
    def start_bundle(self) -> None:
        """Starts drawing a new bundle: strokes and clicks in the 2D views add points.

        Starts a phantom on the views' grid if none is open, and chooses the
        `Tool.DRAW_BUNDLE` tool. A bundle being drawn is finished first if it
        has two points, else dropped. The new bundle starts with the last
        bundle's settings and a new seed.
        """
        self._phantom.start_bundle()

    @agent_action
    def add_bundle_point(self, position: tuple[float, float, float]) -> None:
        """Adds a control point at the end of the bundle being drawn.

        The bundle is drawn again from its points, once per frame in a window.

        Args:
            position: The point (x, y, z), world mm.

        Raises:
            ValueError: If no bundle is being drawn, or position is not three
                finite numbers inside the phantom's grid box.
        """
        state, draft = self._phantom.draft()
        point = _position(position)
        if not state.contains(point):
            raise ValueError(
                "The point ({:.1f}, {:.1f}, {:.1f}) mm is outside the phantom's "
                "grid; draw inside the box.".format(*point)
            )
        self._phantom.set_points((*draft.points, point))
        self._point_groups = [*self._current_groups(draft.points), 1]

    @agent_action
    def add_bundle_stroke(self, points: list[tuple[float, float, float]]) -> int:
        """Adds the control points of a stroke, as a freehand drag draws one.

        The stroke becomes the fewest points evenly spaced along it, at most 12
        and always its first and last, whose centreline stays within about one
        voxel of the views' grid of the stroke; those inside the phantom's grid
        box are added at the end of the bundle being drawn.

        Args:
            points: The stroke's points (x, y, z), world mm, in order.

        Returns:
            The number of control points added.

        Raises:
            ValueError: If no bundle is being drawn, a point is not three finite
                numbers, or no point of the stroke is inside the grid box.
        """
        state, draft = self._phantom.draft()
        stroke = [_position(p) for p in points]
        grid = self._reference_grid() or state.grid
        tolerance = float(np.linalg.norm(grid.affine[:3, :3], axis=0).min())
        kept = [p for p in simplify_stroke(stroke, tolerance) if state.contains(p)]
        if not kept:
            raise ValueError(
                "The stroke is outside the phantom's grid; draw inside the box."
            )
        self._phantom.set_points((*draft.points, *kept))
        self._point_groups = [*self._current_groups(draft.points), len(kept)]
        return len(kept)

    @agent_action
    def remove_bundle_point(self) -> None:
        """Removes the last point or stroke added to the bundle being drawn.

        A freehand stroke is taken back whole, a point placed by a click alone.

        Raises:
            ValueError: If no bundle is being drawn.
        """
        _, draft = self._phantom.draft()
        if not draft.points:
            return
        groups = self._current_groups(draft.points)
        last = groups.pop() if groups else 1
        self._phantom.set_points(draft.points[:-last])
        self._point_groups = groups

    def _current_groups(self, points: Sequence[object]) -> list[int]:
        """Returns the point groups if they still add up to `points`, else one each."""
        if sum(self._point_groups) == len(points):
            return list(self._point_groups)
        return [1] * len(points)

    @agent_action
    def set_bundle_points(self, points: list[tuple[float, float, float]]) -> None:
        """Replaces the control points of the bundle being drawn.

        Args:
            points: The points (x, y, z), world mm, in order.

        Raises:
            ValueError: If no bundle is being drawn, or a point is not three
                finite numbers.
        """
        self._phantom.set_points([_position(p) for p in points])

    @agent_action
    def finish_bundle(self) -> Item:
        """Makes the bundle being drawn a bundle of the phantom, and leaves draw mode.

        Returns:
            The bundle's tractogram item, coloured per tractogram.

        Raises:
            ValueError: If no bundle is being drawn, or it has fewer than two
                points.
        """
        return self._phantom.finish()

    @agent_action
    def cancel_bundle(self) -> None:
        """Leaves draw mode: a new bundle is dropped, an edited one kept as it was.

        Raises:
            ValueError: If no bundle is being drawn.
        """
        self._phantom.cancel()

    @agent_action
    def edit_bundle(self, item_id: ItemId) -> None:
        """Draws a finished bundle again: its points come back into draw mode.

        Its item is changed in place, keeping its id and options.

        Args:
            item_id: The bundle's tractogram item.

        Raises:
            ValueError: If it is not a drawn bundle of the phantom.
        """
        self._phantom.edit(item_id)

    @agent_action
    def use_as_bundle(self, item_id: ItemId, tissue: str = "WM") -> None:
        """Makes a loaded tractogram a bundle of the phantom, its streamlines as is.

        Args:
            item_id: The tractogram item.
            tissue: The tissue that fills it.

        Raises:
            ValueError: If no phantom is open, or the item or tissue is unknown.
        """
        self._phantom.use_tractogram(item_id, tissue)

    @agent_action
    def set_bundle(
        self, item_id: ItemId | None, name: BundleField, value: object
    ) -> None:
        """Changes one setting of a bundle; a drawn one is drawn again.

        Args:
            item_id: The bundle's tractogram item, or None for the bundle being
                drawn.
            name: `BundleField.TISSUE` (a tissue's name), `NUM_STREAMLINES`,
                `RADIUS` (mm), `NUM_POINTS`, `FAN_START`, `FAN_END`,
                `FAN_SHARPNESS`, `DISPERSION` (mm), `SEED`, `STREAMLINE_AREA`
                (mm^2 per streamline, None for the default: the bundle's
                cross-section shared by its streamlines) or `DENSEST_FILL` (the
                fill of a loaded tractogram's dense voxels, in (0, 1]).
            value: The new value.

        Raises:
            TypeError: If name is not a BundleField.
            ValueError: If the bundle or the value is refused.
        """
        _require(name, BundleField)
        self._phantom.set_bundle(item_id, name, value)

    @agent_action
    def show_proxy(self, kind: ProxyMap, shown: bool = True) -> Item | None:
        """Shows or hides a map of what the phantom gives, without a run.

        The maps are overlays updated as the phantom changes: `ProxyMap.FILL`
        (volume 0 the fraction of each voxel all bundles fill, then one volume
        per bundle), `PARTITION` (one volume per tissue's fraction, then the
        background) and `EXPECTED_B0` (the noise-free signal at b = 0 and the
        first echo time).

        Args:
            kind: The map.
            shown: Show it, or remove it.

        Returns:
            The overlay item when shown, else None.

        Raises:
            TypeError: If kind is not a ProxyMap.
            ValueError: If no phantom is open or it cannot be made.
        """
        _require(kind, ProxyMap)
        return self._phantom.show_proxy(kind, shown)

    @agent_action
    def show_grid_box(self, shown: bool = True) -> None:
        """Outlines the phantom's grid box in the views, or hides it.

        The box is shown only in the Simulate workspace, and always while a
        bundle is drawn whatever this says; a session keeps the choice.

        Args:
            shown: Outline it, or hide it.

        Raises:
            ValueError: If no phantom is open.
        """
        self._phantom.show_box(bool(shown))

    # Signal breakdown

    @property
    def breakdown(self) -> BreakdownState | None:
        """What the Signal breakdown dock shows; None while it is closed."""
        return self.scene.breakdown

    def show_breakdown(self, shown: bool = True) -> None:
        """Starts following one voxel's signal, factor by factor, or stops.

        While shown, the breakdown follows the voxel under the pointer
        (`hover_breakdown`) or the pinned one (`pin_breakdown`), and every edit
        of the phantom's tissues, bundles, maps and scan, without a simulation.

        Args:
            shown: Follow the signal, or stop.
        """
        self._phantom.breakdown.show(bool(shown))

    def hover_breakdown(self, position: tuple[float, float, float]) -> None:
        """Shows the voxel under the pointer, unless a voxel is pinned.

        Does nothing while the breakdown is not shown, outside the phantom's
        grid, or in the voxel already shown, so it can follow every mouse move.

        Args:
            position: The world position (x, y, z) in mm, RAS+.
        """
        self._phantom.breakdown.hover(_position(position))

    def pin_breakdown(self, position: tuple[float, float, float] | None) -> None:
        """Pins the breakdown to the voxel at a position, or unpins it.

        Pinning shows the breakdown if it is not shown. Unpinned, it follows the
        pointer again.

        Args:
            position: The world position (x, y, z) in mm, RAS+; None unpins.

        Raises:
            ValueError: If no phantom is open or the position is outside its grid.
        """
        self._phantom.breakdown.pin(None if position is None else _position(position))

    def set_breakdown_volume(self, volume: int) -> None:
        """Shows the breakdown at another volume of the scan: its b-value and direction.

        Args:
            volume: The volume's index in the scan's gradient table.

        Raises:
            ValueError: If the breakdown is not shown or there is no such volume.
        """
        self._phantom.breakdown.set_volume(int(volume))

    def breakdown_gradients(self) -> GradientTable | None:
        """Returns the phantom scan's gradient table, the volumes a breakdown can show.

        Returns:
            The table; None without a phantom or with a scan that cannot be made.
        """
        return self._phantom.breakdown.gradients()

    @agent_action
    def set_run_folder(self, folder: str | Path | None) -> None:
        """Sets the folder a run of the phantom writes its outputs to.

        Args:
            folder: The folder, or None to be asked at the next run.

        Raises:
            ValueError: If no phantom is open.
        """
        self._phantom.set_output(None if folder is None else Path(folder).absolute())

    @agent_action
    def check_phantom(self) -> list[str]:
        """Returns what stops the phantom from running, one sentence each.

        Returns:
            The problems; empty if it can run.

        Raises:
            ValueError: If no phantom is open.
        """
        try:
            self._phantom.run_inputs()
        except ValueError as e:
            return [str(e)]
        return []

    @agent_action
    def save_phantom(self, path: str | Path) -> None:
        """Writes the phantom to a phantom file, with the viewer's settings.

        The file is `write_phantom`'s JSON: grid, tissues, bundles (geometries,
        or the files of loaded tractograms), tissue maps (their files),
        background and scan, plus a viewer block with the tissue presets, the
        scan settings, the bundles' names and options and the proxy maps.

        Args:
            path: The file to write, such as "scene.phantom.json".

        Raises:
            ValueError: If no phantom is open, or a loaded bundle or a map has
                no file.
            OSError: If the file cannot be written.
        """
        self._phantom.save(Path(path).absolute())

    @agent_action
    def load_phantom(self, path: str | Path) -> None:
        """Opens a phantom file, replacing the phantom open.

        Each bundle is drawn once and added as a tractogram item; the tissue
        maps are added as overlays.

        Args:
            path: A phantom file.

        Raises:
            ValueError: If it is not a phantom file or holds invalid values.
            OSError: If a file cannot be read.
        """
        self.apply_phantom(read_phantom_file(Path(path).absolute()))

    def apply_phantom(self, loaded: LoadedPhantom) -> None:
        """Shows a phantom file that the window read in the background.

        The window reads a phantom file and draws its bundles on another
        thread, then calls this; a script uses `load_phantom` instead.

        Args:
            loaded: The phantom file read, with its bundles drawn.
        """
        self._phantom.apply(loaded)

    @agent_action
    def run_phantom(self, folder: str | Path, *, overwrite: bool = False) -> list[Item]:
        """Simulates the phantom, saves the simulation folder and opens it.

        The folder is the one `larmor.simulation.save_simulation` writes. The
        DWI (with its .bval and .bvec) opens as a volume, the tissue masks as
        overlays and the ground-truth fibre directions as peaks; the bundles'
        items then refer to the streamline files written, so a session keeps
        everything. In the window, Run (Ctrl+R) does this in the background,
        with a Cancel button.

        Args:
            folder: The output folder; made if missing.
            overwrite: Replace outputs already in the folder.

        Returns:
            The items opened.

        Raises:
            ValueError: If no phantom is open or it cannot run.
            FileExistsError: If outputs exist and overwrite is False.
        """
        inputs = self.prepare_run(folder, overwrite=overwrite)
        outputs = run_simulation(inputs, Path(folder).absolute(), lambda *_: None)
        items = self._open_outputs(outputs.files)
        self.finish_run(outputs)
        return items

    @agent_action
    def open_simulation(self, folder: str | Path) -> list[Item]:
        """Opens a simulation folder, its phantom in the phantom editor.

        The folder is one `lar simulate` or Run writes. Its phantom file opens
        in the phantom editor, replacing the phantom open, with each bundle as
        a tractogram item referring to the bundle's streamline file; the DWI
        opens as a volume, the tissue masks as overlays and the ground-truth
        fibre directions as peaks.

        Args:
            folder: The simulation folder, holding simulation.json.

        Returns:
            The items opened: the bundles, the DWI, the masks and the peaks.

        Raises:
            ValueError: If it is not a simulation folder, or its phantom file
                is not valid.
            OSError: If a file cannot be read.
        """
        files = simulation_outputs(Path(folder))
        self.apply_phantom(read_phantom_file(files.phantom))
        self.link_simulation(files)
        state = self.scene.phantom
        assert state is not None
        bundles = [self.scene.item(b.item_id) for b in state.bundles]
        return bundles + self._open_outputs(files)

    def link_simulation(self, files: SimulationOutputs) -> None:
        """Points the phantom's drawn bundles at a simulation's streamline files.

        Afterwards each drawn bundle's item refers to the .tck file the
        simulation wrote, so a session saved later reopens it from there.

        Args:
            files: The simulation folder's files; its phantom is the one open.
        """
        self._phantom.link(files)

    def _open_outputs(self, files: SimulationOutputs) -> list[Item]:
        """Loads a simulation's DWI, masks and peaks as items."""
        items = [self.load(p, Category.VOLUMES) for p in files.dwi]
        items += [self.load(p, Category.OVERLAYS) for p in files.masks]
        if files.peaks is not None:
            items.append(self.load(files.peaks, Category.PEAKS))
        return items

    def prepare_run(self, folder: str | Path, *, overwrite: bool = False) -> RunInputs:
        """Checks the phantom and the folder, and takes what a run simulates.

        The window uses it with `finish_run` to run a simulation in the
        background; a script uses `run_phantom` instead.

        Args:
            folder: The output folder.
            overwrite: Replace outputs already in the folder.

        Returns:
            The run's inputs, which can be simulated on any thread.

        Raises:
            ValueError: If no phantom is open or it cannot run.
            FileExistsError: If outputs exist and overwrite is False.
        """
        inputs = self._phantom.run_inputs()
        existing = existing_outputs(inputs, Path(folder).absolute())
        if existing and not overwrite:
            names = ", ".join(p.name for p in existing)
            raise FileExistsError(f"{folder} holds outputs already: {names}.")
        return inputs

    def finish_run(self, outputs: RunOutputs) -> None:
        """Points the bundles at the files a run wrote, and keeps its folder.

        The window calls it after a run started with `prepare_run`.

        Args:
            outputs: The files the run wrote.
        """
        self._phantom.finish_run(outputs)

    # Options

    @agent_action
    def set(self, target: ItemId | Category, name: OptionName, value: object) -> None:
        """Changes one option of an item, or of every item of a category.

        Showing a volume makes it the main volume and hides the others; setting
        the Volumes category's visibility shows or hides the main volume only.
        A new volume index drops the surface labels its volume does not hold.
        Minimum, maximum and threshold are in the image's scaled units. DTI
        colours need a 4D image with 3 volumes (a direction's x, y and z), and
        the image that modulates them must be one of `item_choices`. A
        surface's values to colour by must be one of `data_choices`; a new one
        also sets the colormap and range: for "curv" and "sulc" white to black
        from -0.3 to 0.3, for others the image default over the values' range.
        A tractogram's TSF file (`VALUES_FILE`) must hold a value per point of
        its streamlines, and its sampled volume (`VALUES_VOLUME`) must be one of
        `item_choices`; setting either clears the other, colours the tractogram
        by value and sets the range to the values' (for a volume, its data's).

        Args:
            target: An item's id, or a Category to change all its items.
            name: The option, such as `OptionName.COLORMAP`.
            value: The new value, of the option's type: a bool, a number, a
                `Colormap`, a path for a lookup table ("" for the built-in
                colours) or a TSF file ("" for none), label numbers for
                `SURFACE_LABELS`, which must be among the overlay's `labels`,
                or another item's id (or None) for an option that names an
                item, such as `OptionName.MODULATE` or `VALUES_VOLUME`, or a
                name (or None) for `OptionName.COLOUR_BY`. A tractogram's
                `REGIONS` are region items' ids. A region's `CENTRE` is
                (x, y, z) in world coordinates and its `SIZE` three lengths,
                mm; its shape is set when it is added.

        Raises:
            TypeError: If target or name is of the wrong type.
            OptionError: If the items have no such option, the value is outside
                the option's range or choices, the option (such as the volume
                index) is set per item and target is a category, DTI colours are
                asked of an image without 3 volumes, the item named is not
                one the option can name, a surface has no such values, or a TSF
                file cannot be read or does not match the tractogram.
            ValueError: If there is no item with that id.

        Example:
            >>> viewer.set(Category.OVERLAYS, OptionName.OPACITY_2D, 0.5)
        """
        _require_target(target)
        _require(name, OptionName)
        if isinstance(target, Category):
            opts = item_type(target).options
            value = validate(opts, name, value)

            _, info = option_fields(opts)[name]
            if info.per_item:
                raise OptionError(f"{info.label} is set per item, not category.")

            items = self.scene.in_category(target)
            if target == Category.VOLUMES and name == OptionName.VISIBLE:
                main = self.scene.view.main_volume
                items = [i for i in items if i.id == main]
        else:
            item = self.scene.item(target)
            value = self._item_value(item, name, value)
            items = [item]
        for item in items:
            self._apply_option(item, name, value)

    def _item_value(self, item: Item, name: OptionName, value: object) -> object:
        """Returns value checked for one item's option; reads a TSF file it names."""
        value = validate(type(item.options), name, value)
        if name == OptionName.VOLUME_INDEX:
            _check_volume_index(item, value)
        if name == OptionName.SURFACE_LABELS:
            self._check_labels(item, value)
        if name == OptionName.DTI and value:
            _check_dti(item)
        if name == OptionName.HIDE_MEDIAL_WALL and value:
            _check_cortex(item)
        if name == OptionName.VALUES_FILE and value:
            value = self._read_tsf(item, str(value))
        _, info = option_fields(type(item.options))[name]
        if info.data_key and value is not None:
            self._check_data_key(item, name, value)
        if isinstance(item.options, RegionOptions):
            _check_region_option(item, name, value)
        if info.item_ref and value is not None:
            self._check_item_refs(item, name, value)
        return value

    def _check_data_key(self, item: Item, name: OptionName, value: object) -> None:
        """Raises OptionError if the item's data has no values of that name."""
        _, info = option_fields(type(item.options))[name]
        names = self.data_choices(item.id, name)
        if value not in names:
            raise OptionError(
                f"{info.label}: {item.name} has no values {value!r}; it has "
                f"{', '.join(map(repr, names)) or 'none'}."
            )

    def _check_item_refs(self, item: Item, name: OptionName, value: object) -> None:
        """Raises OptionError if an item the option names is not one of its choices."""
        named = value if isinstance(value, tuple) else (value,)
        choices = [i.id for i in self.item_choices(item.id, name)]
        for one in named:
            assert isinstance(one, int)
            if one not in choices:
                raise OptionError(_item_ref_error(self.scene, item, name, one))

    def _check_labels(self, item: Item, value: object) -> None:
        """Raises OptionError if an overlay lacks one of the surface labels."""
        assert isinstance(value, tuple)
        missing = sorted(set(value) - set(self.labels(item.id)))
        if missing:
            numbers = ", ".join(map(str, missing))
            raise OptionError(f"{item.name} has no label {numbers}.")

    def _apply_option(self, item: Item, name: OptionName, value: object) -> None:
        """Sets a checked option on one item, with the changes it brings first."""
        if isinstance(item.options, RegionOptions):
            self._apply_region_option(item, name, value)
            return
        if name == OptionName.VOLUME_INDEX:
            assert isinstance(value, int)
            # First, so that no surface is made of a label the volume lacks.
            self._keep_present_labels(item, value)
        if (
            name == OptionName.COLOUR_BY
            and value is not None
            and value != getattr(item.options, name)
        ):
            assert isinstance(item.data, SurfaceData) and isinstance(value, str)
            # First, so that the values are never shown with another's range.
            for other, v in _colour_by(item.data, value).items():
                self.scene.set_option(item.id, OptionName(other), v)
        if (
            name in (OptionName.VALUES_FILE, OptionName.VALUES_VOLUME)
            and value not in (None, "")
            and value != getattr(item.options, name)
        ):
            # First, so that the values are never shown with another's range.
            self._colour_by_values(item, name, value)
        if name == OptionName.VALUES_FILE and not value:
            self._tsf_values.pop(item.id, None)
        if item.category == Category.VOLUMES and name == OptionName.VISIBLE and value:
            self._show_volume(item)
        else:
            self.scene.set_option(item.id, name, value)

    def _read_tsf(self, item: Item, path: str) -> str:
        """Reads a tractogram's TSF file, keeps its values, returns its full path.

        Returns:
            The full path of the file.

        Raises:
            OptionError: If the file cannot be read or does not match the item.
        """
        o = item.options
        assert isinstance(item.data, Tractogram) and isinstance(o, TractogramOptions)
        matlab = split_values_spec(path)
        if matlab is not None:
            full = values_spec(*matlab)
        else:
            full = str(Path(path).absolute())
        if full == o.values_file and item.id in self._tsf_values:
            return full  # read already
        subset = SUBSETS.get(item.data)
        try:
            if matlab is not None:
                values = point_values(*matlab, item.data, subset)
            elif subset is None:
                values, _ = read_tsf(full, tractogram=item.data)
            else:
                values = _subset_values(full, item.data, subset)
        except (OSError, ValueError) as e:
            raise OptionError(f"TSF file: {e}") from e
        self._tsf_values[item.id] = values
        return full

    def _colour_by_values(self, item: Item, name: OptionName, value: object) -> None:
        """Clears a tractogram's other values, colours it by these, sets the range."""
        if name == OptionName.VALUES_FILE:
            self.scene.set_option(item.id, OptionName.VALUES_VOLUME, None)
            low, high = _finite_range(self._tsf_values[item.id])
        else:
            self.scene.set_option(item.id, OptionName.VALUES_FILE, "")
            self._tsf_values.pop(item.id, None)
            assert isinstance(value, int)
            source = self.scene.item(value).value_range
            low, high = source if source is not None else (0.0, 1.0)
        item.value_range = (low, high)
        self.scene.set_option(item.id, OptionName.MINIMUM, low)
        self.scene.set_option(item.id, OptionName.MAXIMUM, high)
        self.scene.set_option(item.id, OptionName.COLOURING, Colouring.VALUES)

    def _tsf_point_values(self, item_id: ItemId) -> np.ndarray | None:
        """Returns a tractogram's TSF values, one per point; None without a TSF file.

        Args:
            item_id: The tractogram.

        Returns:
            The values, or None.

        Raises:
            ValueError: If there is no item with that id.
        """
        self.scene.item(item_id)
        return self._tsf_values.get(item_id)

    # Regions

    def _apply_region_option(self, item: Item, name: OptionName, value: object) -> None:
        """Sets a checked option of a region item, its data and colour following it.

        The data is replaced before the option's event, so that whoever follows
        the event sees the region as it now is. A colour that is the old role's
        follows the role.
        """
        o = item.options
        assert isinstance(o, RegionOptions)
        if name == OptionName.ROLE and value != o.role:
            assert isinstance(value, RegionRole)
            if o.colour == ROLE_COLOURS[o.role]:
                self.scene.set_option(item.id, OptionName.COLOUR, ROLE_COLOURS[value])
        if name in _REGION_GEOMETRY and value != getattr(o, name):
            changes: dict[str, Any] = {name: value}
            item.data = region_data(replace(o, **changes))
        if name == OptionName.RADIUS and o.shape == RegionShape.SPHERE:
            assert isinstance(value, float)
            self.region_radius = value
        if name == OptionName.SIZE and o.shape == RegionShape.BOX:
            assert isinstance(value, Lengths)
            self.box_size = value
        self.scene.set_option(item.id, name, value)

    def _add_region(
        self,
        options: RegionOptions,
        name: str | None,
        apply: bool,
        defaults: bool = True,
    ) -> ItemId:
        """Adds a region item with these options, applied to every tractogram if asked.

        With defaults, the values set by `set_default` for regions replace those
        of the options.

        Returns:
            The new region's id.
        """
        if name is None:
            label = _REGION_NAMES[options.shape]
            count = sum(
                1
                for r in self.scene.in_category(Category.REGIONS)
                if isinstance(r.options, RegionOptions)
                and r.options.shape == options.shape
            )
            name = f"{label} {count + 1}"
        if defaults:
            changes: dict[str, Any] = {
                str(n): v for n, v in self._defaults.get(Category.REGIONS, {}).items()
            }
            options = replace(options, **changes)
        item = self.scene.add(
            Category.REGIONS, name, None, region_data(options), options
        )
        if apply:
            for tract in self.scene.in_category(Category.TRACTOGRAMS):
                self.apply_region(tract.id, item.id)
        return item.id

    @agent_action
    def add_sphere(
        self,
        centre: tuple[float, float, float] | None = None,
        radius: float | None = None,
        role: RegionRole = RegionRole.INCLUDE,
        name: str | None = None,
        *,
        apply: bool = False,
    ) -> ItemId:
        """Adds a sphere region, an item of `Category.REGIONS`.

        A region filters the streamlines of the tractograms it is applied to
        (`apply_region`): a point is inside the sphere when it is at most the
        radius from the centre. The sphere starts in its role's colour.

        Args:
            centre: Its centre (x, y, z), world mm; the cursor if None.
            radius: Its radius in mm, at least 0.1; the last sphere's radius
                (5 mm at first) if None.
            role: `RegionRole.INCLUDE` keeps the streamlines that pass through
                it, `EXCLUDE` drops them, `WITHIN` keeps those that stay inside.
            name: Its name in Layers; "Sphere 1" and so on if None.
            apply: Apply it to every tractogram in the scene.

        Returns:
            The region's id.

        Raises:
            TypeError: If role is not a RegionRole.
            ValueError: If centre is not three finite numbers.
            OptionError: If the radius is not a number of at least 0.1.
        """
        _require(role, RegionRole)
        where = _position(self.scene.view.cursor if centre is None else centre)
        size = self.region_radius if radius is None else radius
        checked = validate(RegionOptions, OptionName.RADIUS, size)
        assert isinstance(checked, float)
        options = RegionOptions(
            shape=RegionShape.SPHERE,
            role=role,
            colour=ROLE_COLOURS[role],
            centre=Position(where),
            radius=checked,
        )
        self.region_radius = checked
        return self._add_region(options, name, apply)

    @agent_action
    def add_box(
        self,
        centre: tuple[float, float, float] | None = None,
        size: tuple[float, float, float] | None = None,
        role: RegionRole = RegionRole.INCLUDE,
        name: str | None = None,
        *,
        apply: bool = False,
    ) -> ItemId:
        """Adds a box region, its edges along the world axes.

        As `add_sphere`; a point is inside the box when it is at most half the
        size from the centre along each of x, y and z.

        Args:
            centre: Its centre (x, y, z), world mm; the cursor if None.
            size: Its edges' lengths along x, y and z, mm, each positive; the
                last box's size (10 mm a side at first) if None.
            role: `RegionRole.INCLUDE`, `EXCLUDE` or `WITHIN`, as for a sphere.
            name: Its name in Layers; "Box 1" and so on if None.
            apply: Apply it to every tractogram in the scene.

        Returns:
            The region's id.

        Raises:
            TypeError: If role is not a RegionRole.
            ValueError: If centre is not three finite numbers.
            OptionError: If size is not three positive numbers.
        """
        _require(role, RegionRole)
        where = _position(self.scene.view.cursor if centre is None else centre)
        lengths = validate(
            RegionOptions, OptionName.SIZE, self.box_size if size is None else size
        )
        assert isinstance(lengths, Lengths)
        options = RegionOptions(
            shape=RegionShape.BOX,
            role=role,
            colour=ROLE_COLOURS[role],
            centre=Position(where),
            size=lengths,
        )
        self.box_size = lengths
        return self._add_region(options, name, apply)

    @agent_action
    def add_mask(
        self,
        mask_id: ItemId,
        role: RegionRole = RegionRole.INCLUDE,
        name: str | None = None,
        *,
        apply: bool = False,
    ) -> ItemId:
        """Adds a mask region: the voxels of an image whose value is not zero.

        A point is inside the mask when it lies in a voxel whose value rounds to
        a whole number other than 0. The region is removed with its image.

        Args:
            mask_id: The image: a volume or an overlay of one volume, on any grid.
            role: `RegionRole.INCLUDE`, `EXCLUDE` or `WITHIN`, as for a sphere.
            name: Its name in Layers; the image's name and "(mask)" if None.
            apply: Apply it to every tractogram in the scene.

        Returns:
            The region's id.

        Raises:
            TypeError: If role is not a RegionRole.
            ValueError: If there is no item with that id.
            OptionError: If the item is not an image of one volume.
        """
        _require(role, RegionRole)
        mask = self.scene.item(mask_id)
        if mask not in self._images_of_one_volume():
            raise OptionError(f"{mask.name} is not a volume or overlay of one volume.")
        options = RegionOptions(
            shape=RegionShape.MASK, role=role, colour=ROLE_COLOURS[role], mask=mask_id
        )
        return self._add_region(options, name or f"{mask.name} (mask)", apply)

    @agent_action
    def apply_region(
        self, tract_id: ItemId, region_id: ItemId, applied: bool = True
    ) -> None:
        """Applies a region to a tractogram's streamlines, or stops applying it.

        With the filter on (`OptionName.REGION_FILTER`, on by default), only the
        streamlines that pass through every include region applied, through no
        exclude one, and stay within every within one are drawn. A streamline
        passes through a region when one of its points is inside it, and stays
        within it when all are. The subset and the points per streamline then
        apply to the streamlines kept. One region can filter several tractograms.

        Args:
            tract_id: The tractogram.
            region_id: The region item.
            applied: Apply it, or stop applying it.

        Raises:
            ValueError: If there is no item with either id.
            OptionError: If the first item is not a tractogram or the second not
                a region.
        """
        regions = self._tractogram_options(tract_id).regions
        if self.scene.item(region_id).category != Category.REGIONS:
            raise OptionError(f"{self.scene.item(region_id).name} is not a region.")
        if applied:
            kept = (*regions, region_id)
        else:
            kept = tuple(r for r in regions if r != region_id)
        self.set(tract_id, OptionName.REGIONS, kept)

    @agent_action
    def clear_regions(self, tract_id: ItemId) -> None:
        """Stops applying every region to a tractogram, so all are drawn.

        The region items stay.

        Args:
            tract_id: The tractogram.

        Raises:
            ValueError: If there is no item with that id.
            OptionError: If the item is not a tractogram.
        """
        self._tractogram_options(tract_id)
        self.set(tract_id, OptionName.REGIONS, ())

    @agent_action
    def save_kept(self, tract_id: ItemId, path: str | Path) -> Path:
        """Writes the streamlines a tractogram's regions keep to a file.

        The selection is made with `larmor.tractography.edit`, the same tests
        as the views', over every streamline loaded (not only the subset
        drawn). With the filter off or no region applied, all are written. The
        item itself does not change.

        Args:
            tract_id: The tractogram.
            path: The file, TCK, TRK, VTK or FIB by its extension; an existing
                file is replaced.

        Returns:
            The path written.

        Raises:
            ValueError: If there is no item with that id or the extension is not
                one a tractogram is written as.
            OptionError: If the item is not a tractogram.
            OSError: If the file cannot be written.
        """
        self._tractogram_options(tract_id)
        item = self.scene.item(tract_id)
        tractogram = item.data
        assert isinstance(tractogram, Tractogram)
        regions = region_filter(self.scene, item)
        kept = tractogram
        if regions is not None:
            from larmor.tractography import EditOptions, edit

            options = EditOptions(
                include=regions.include,
                exclude=regions.exclude,
                within=regions.within,
                ends_only=regions.ends_only,
            )
            kept = edit(tractogram, options)
        p = Path(path).absolute()
        save_tractogram(
            kept, p, reference=tractogram.reference or self._reference_grid()
        )
        return p

    def region_selection(self, item_id: ItemId) -> np.ndarray | None:
        """Returns which of a tractogram's streamlines its regions keep.

        It runs the same filter as the drawing, once per change of the
        regions: the result is kept until they change.

        Args:
            item_id: The tractogram's id.

        Returns:
            (S,) bool, one per streamline of the item's tractogram, True for those
            kept; None if the options keep every streamline (the filter is off
            or there are no regions).

        Raises:
            ValueError: If there is no item with that id.
            OptionError: If the item is not a tractogram.
        """
        key, work = self._selection_work(item_id)
        selection = work()
        self._remember_selection(item_id, key, selection)
        return selection

    def _selection_work(
        self, item_id: ItemId
    ) -> tuple[RegionKey, Callable[[], np.ndarray | None]]:
        """Returns the work giving `region_selection`, to run on any thread.

        The regions are read now. The work only filters the streamlines, or
        gives the selection kept for these regions; `_remember_selection`, on the
        main thread, keeps its result.

        Args:
            item_id: The tractogram.

        Returns:
            The region options the work is for, and the work.

        Raises:
            ValueError: If there is no item with that id.
            OptionError: If the item is not a tractogram.
        """
        key = self._region_key(item_id)
        kept = self._selections.get(item_id)
        if kept is not None and kept[0] == key:
            selection = kept[1]
            return key, lambda: selection
        item = self.scene.item(item_id)
        regions = region_filter(self.scene, item)
        tractogram = item.data
        assert isinstance(tractogram, Tractogram)
        if regions is None:
            return key, lambda: None
        return key, partial(passes_regions, tractogram, regions)

    def _remember_selection(
        self, item_id: ItemId, key: RegionKey, selection: np.ndarray | None
    ) -> None:
        """Keeps a tractogram's selection for the region options `key`.

        Sends FilterChanged when the selection is new. Ignored for an item no
        longer in the scene.

        Args:
            item_id: The tractogram.
            key: The region options the selection is for.
            selection: The selection, or None for no regions.
        """
        present = any(i.id == item_id for i in self.scene.items)
        # Not a selection made for data replaced since.
        if not present or key != self._region_key(item_id):
            return
        known = self._selections.get(item_id)
        self._selections[item_id] = (key, selection)
        if known is not None and known[0] == key:
            return
        data = self.scene.item(item_id).data
        assert isinstance(data, Tractogram)
        total = data.num_streamlines
        kept = total if selection is None else int(np.count_nonzero(selection))
        self.scene.filtered(item_id, kept, total)

    @agent_action
    def kept_streamlines(self, item_id: ItemId) -> int:
        """Counts how many of a tractogram's streamlines its regions keep.

        Counted over the streamlines loaded, before the subset is taken.

        Args:
            item_id: The tractogram.

        Returns:
            The number of streamlines kept.

        Raises:
            ValueError: If there is no item with that id.
            OptionError: If the item is not a tractogram.
        """
        selection = self.region_selection(item_id)
        if selection is None:
            data = self.scene.item(item_id).data
            assert isinstance(data, Tractogram)
            return data.num_streamlines
        return int(np.count_nonzero(selection))

    def _region_key(self, item_id: ItemId) -> RegionKey:
        """Returns what a tractogram's selection by its regions depends on.

        Returns:
            Its region options, and how many times its data and each mask's data
            were replaced, so that a selection made before a replace is not used.

        Raises:
            ValueError: If there is no item with that id.
            OptionError: If the item is not a tractogram.
        """
        self._tractogram_options(item_id)
        item = self.scene.item(item_id)
        versions = tuple(
            self._versions.get(i, 0) for i in (item_id, *mask_ids(self.scene, item))
        )
        return (*region_key(self.scene, item), versions)

    def _tractogram_options(self, item_id: ItemId) -> TractogramOptions:
        """Returns a tractogram's options; raises OptionError for another item."""
        item = self.scene.item(item_id)
        if not isinstance(item.options, TractogramOptions):
            raise OptionError(f"{item.name} is not a tractogram.")
        return item.options

    def _keep_present_labels(self, item: Item, index: int) -> None:
        """Drops the chosen labels that the overlay's volume `index` does not hold."""
        o = item.options
        if isinstance(o, OverlayOptions) and o.surface_labels:
            present = set(self._labels_at(item, index))
            kept = tuple(label for label in o.surface_labels if label in present)
            if kept != o.surface_labels:
                self.scene.set_option(item.id, OptionName.SURFACE_LABELS, kept)

    @agent_action
    def labels(self, item_id: ItemId) -> tuple[int, ...]:
        """Returns the labels in an overlay's shown volume.

        The surfaces of the 3D view are chosen among these (`SURFACE_LABELS`).

        Args:
            item_id: An overlay's id.

        Returns:
            The values other than 0 in the volume at the overlay's volume index,
            in scaled units rounded to whole numbers, sorted; none for an image
            with more distinct values than a parcellation can show (8192).

        Raises:
            ValueError: If there is no item with that id, or it is not an overlay.
        """
        item = self.scene.item(item_id)
        if item.category != Category.OVERLAYS:
            raise ValueError(f"{item.name} is not an overlay, so it has no labels.")
        assert isinstance(item.options, OverlayOptions)
        return self._labels_at(item, item.options.volume_index)

    def _labels_at(self, item: Item, index: int) -> tuple[int, ...]:
        """Returns the labels in an overlay's volume `index`, counted once."""
        assert isinstance(item.data, Volume)
        key = (item.id, index)
        if key not in self._labels:
            self._labels[key] = labels_in(item.data, index)
        return self._labels[key]

    @agent_action
    def set_default(self, category: Category, name: OptionName, value: object) -> None:
        """Sets the value an option starts with on the items added to a category.

        Items already in the scene keep their value. The window uses it for the
        settings of new images and tractograms.

        Args:
            category: The category whose new items start with the value.
            name: The option.
            value: Its starting value, checked as `set` checks it.

        Raises:
            TypeError: If category or name is not its enum.
            OptionError: If the category has no such option, the option is set
                per item, or the value is not allowed.
        """
        _require(category, Category)
        _require(name, OptionName)
        opts = item_type(category).options
        value = validate(opts, name, value)
        _, info = option_fields(opts)[name]
        if info.per_item:  # its rules hold for one item, so not for every new one
            raise OptionError(f"{info.label} is set per item, not as a default.")
        self._defaults.setdefault(category, {})[name] = value

    def _category_value(self, category: Category, name: OptionName) -> object:
        """Returns the value an option has on all of a category's items.

        Args:
            category: The category.
            name: The option.

        Returns:
            The items' shared value; `MIXED` if they differ; the option's default
            if the category is empty. For the Volumes category's visibility, the
            main volume's.

        Raises:
            TypeError: If category or name is of the wrong type.
            OptionError: If the category's items have no such option.
        """
        _require(category, Category)
        _require(name, OptionName)
        kind = item_type(category)
        if name not in option_fields(kind.options):
            raise OptionError(f"{kind.options.__name__} has no option '{name}'.")
        items = self.scene.in_category(category)
        if category == Category.VOLUMES and name == OptionName.VISIBLE:
            main = self.scene.view.main_volume
            items = [i for i in items if i.id == main]
        values = {getattr(i.options, name) for i in items}
        if not values:
            return getattr(kind.options(), name)
        return values.pop() if len(values) == 1 else MIXED

    @agent_action
    def item_choices(self, item_id: ItemId, name: OptionName) -> list[Item]:
        """Returns the items an option that names an item can name, for one item.

        For `OptionName.MODULATE`: the other volumes and overlays that are 3D
        and on the same grid (shape and affine) as the item. For
        `OptionName.VALUES_VOLUME` and a region's `MASK`: the volumes and
        overlays with one volume (3D, or 4D with a single volume), on any grid.
        For a tractogram's `REGIONS`: the region items.

        Args:
            item_id: The item whose option it is.
            name: An option that names an item, such as `OptionName.MODULATE`.

        Returns:
            The items, in Layers order.

        Raises:
            TypeError: If item_id or name is of the wrong type.
            ValueError: If there is no item with that id.
            OptionError: If the item has no such option, or it names no item.
        """
        _require(name, OptionName)
        item = self.scene.item(item_id)
        known = option_fields(item.options)
        if name not in known or not known[name][1].item_ref:
            raise OptionError(f"{name} of {item.name} is not an option naming an item.")
        if name == OptionName.REGIONS:
            return self.scene.in_category(Category.REGIONS)
        if isinstance(item.data, Tractogram) or item.category == Category.REGIONS:
            return self._images_of_one_volume()
        assert isinstance(item.data, Volume)
        grid = item.data.grid
        return [
            other
            for other in self.scene.items
            if other.id != item.id
            and other.category in (Category.VOLUMES, Category.OVERLAYS)
            and isinstance(other.data, Volume)
            and other.data.data.ndim == 3
            and other.data.grid.shape[:3] == grid.shape[:3]
            and np.allclose(other.data.affine, grid.affine)
        ]

    def _images_of_one_volume(self) -> list[Item]:
        """Returns the volumes and overlays with one volume, in Layers order."""
        return [
            other
            for other in self.scene.items
            if other.category in (Category.VOLUMES, Category.OVERLAYS)
            and other.volume_count == 1
        ]

    @agent_action
    def data_choices(self, item_id: ItemId, name: OptionName) -> list[str]:
        """Returns the names an option that names some of an item's data can take.

        For `OptionName.COLOUR_BY`: the surface's values per vertex, one number
        per vertex each, such as "curv" or "thickness".

        Args:
            item_id: The item whose option it is.
            name: An option that names some of the item's data.

        Returns:
            The names, in the data's order; None, for no data, is always allowed.

        Raises:
            TypeError: If item_id or name is of the wrong type.
            ValueError: If there is no item with that id.
            OptionError: If the item has no such option, or it names no data.
        """
        _require(name, OptionName)
        item = self.scene.item(item_id)
        known = option_fields(item.options)
        if name not in known or not known[name][1].data_key:
            raise OptionError(f"{name} of {item.name} is not an option naming data.")
        assert isinstance(item.data, SurfaceData)  # only surfaces have one yet
        return item.data.values()

    def _show_volume(self, item: Item) -> None:
        """Makes item the main volume and the only visible volume."""
        for other in self.scene.in_category(Category.VOLUMES):
            if other.id != item.id:
                self.scene.set_option(other.id, OptionName.VISIBLE, False)
        self.scene.set_option(item.id, OptionName.VISIBLE, True)
        self.scene.set_view(ViewField.MAIN_VOLUME, item.id)

    # Views

    @agent_action
    def select(self, target: ItemId | Category | None) -> None:
        """Selects an item or a category, as a click in the scene tree does.

        The selection is what the inspector's Item tab shows, and the image the left
        and right arrows step through.

        Args:
            target: An item's id, a Category, or None to select nothing.

        Raises:
            TypeError: If target is neither an item id nor a Category.
            ValueError: If there is no item with that id.
        """
        if target is not None:
            _require_target(target)
        if target is not None and not isinstance(target, Category):
            self.scene.item(target)
        self.scene.set_view(ViewField.SELECTED, target)

    @agent_action
    def move_cursor(self, position: tuple[float, float, float]) -> None:
        """Moves the cursor, and so the crosshair and the slices, to a position.

        Args:
            position: The world position (x, y, z) in mm, RAS+. It may lie
                outside the images.

        Raises:
            ValueError: If position is not three finite numbers.
        """
        self.scene.set_view(ViewField.CURSOR, _position(position))

    @agent_action
    def step_slice(self, view: ViewKind, n: int) -> None:
        """Moves a 2D view's slice by n steps along the view's axis.

        A step is the reference grid's voxel size along the world axis closest
        to the view's (x for sagittal, y for coronal, z for axial), or 1 mm with
        no grid. Positive n moves towards right, anterior or superior.

        Args:
            view: `ViewKind.AXIAL`, `CORONAL` or `SAGITTAL`.
            n: The number of steps, negative to go back.

        Raises:
            TypeError: If view is not a ViewKind.
            ValueError: If view is the 3D view.
        """
        _require(view, ViewKind)
        if view not in _VIEW_AXIS:
            raise ValueError(f"The {view} view has no slices.")
        axis = _VIEW_AXIS[view]
        grid = self._reference_grid()
        step = 1.0 if grid is None else _step_along(grid, axis)
        cursor = list(self.scene.view.cursor)
        cursor[axis] += n * step
        self.move_cursor((cursor[0], cursor[1], cursor[2]))

    @agent_action
    def select_view(self, view: ViewKind) -> None:
        """Makes a view the selected one, as a click in it does; shortcuts act there.

        Args:
            view: The view.

        Raises:
            TypeError: If view is not a ViewKind.
        """
        _require(view, ViewKind)
        self.scene.set_view(ViewField.SELECTED_VIEW, view)

    @agent_action
    def step_volume(self, n: int) -> None:
        """Moves a 4D image's volume index by n, as the left and right arrows do.

        The image is the selected one if it is 4D, else the main volume. The
        index stops at the first and the last volume; without a main volume,
        nothing happens.

        Args:
            n: The number of volumes, negative to go back.

        Raises:
            TypeError: If n is not a whole number.
        """
        if isinstance(n, bool) or not isinstance(n, int):
            raise TypeError(f"A number of volumes is a whole number; got {n!r}.")
        selected = self.scene.view.selected
        target: ItemId | None = self.scene.view.main_volume
        if isinstance(selected, int):
            chosen = self.scene.item(selected)
            count = chosen.volume_count
            # A 3D image has nothing to step, nor an FOD or a peaks image.
            if isinstance(chosen.options, VolumeOptions) and count and count > 1:
                target = selected
        if target is None:
            return
        item = self.scene.item(target)
        count = item.volume_count
        assert count is not None and isinstance(item.options, VolumeOptions)
        index = min(max(item.options.volume_index + n, 0), count - 1)
        self.set(target, OptionName.VOLUME_INDEX, index)

    @agent_action
    def zoom(self, view: ViewKind, factor: float, *, about: Vec3 | None = None) -> None:
        """Zooms a view in or out, keeping a point where it is on the screen.

        The zoom stays within `ZOOM_RANGE` (0.25 to 64 times the field of view at
        zoom 1). The 3D view always zooms about its centre.

        Args:
            view: The view to zoom.
            factor: How much to multiply the zoom by: above 1 zooms in, below 1
                zooms out.
            about: The world position (mm) that stays in place, such as the
                point under the mouse; the view's centre if None. Not allowed
                for the 3D view.

        Raises:
            TypeError: If view is not a ViewKind.
            ValueError: If factor is not a positive number, about is not three
                finite numbers, or about is given for the 3D view.
        """
        _require(view, ViewKind)
        if not _is_number(factor) or factor <= 0:
            raise ValueError(f"A zoom factor is a positive number; got {factor!r}.")
        if view == ViewKind.THREE_D:
            if about is not None:
                raise ValueError("The 3D view zooms about its centre only.")
            camera_3d = self.scene.view.cameras.three_d
            zoom = _clamp_zoom(camera_3d.zoom * float(factor))
            self._set_camera(view, replace(camera_3d, zoom=zoom))
            return
        camera = self._camera_2d(view)
        zoom = _clamp_zoom(camera.zoom * float(factor))
        pan = camera.pan
        if about is not None:
            # The point stays where it is on the screen: its offset from the centre
            # shrinks in mm as the zoom grows.
            right, up = screen_axes(view, self.scene.view.convention)
            offset = np.asarray(_position(about)) - np.asarray(self.view_centre(view))
            keep = 1.0 - camera.zoom / zoom
            pan = (
                pan[0] + float(offset @ right) * keep,
                pan[1] + float(offset @ up) * keep,
            )
        self._set_camera(view, Camera2D(zoom, pan))

    @agent_action
    def pan(self, view: ViewKind, right: float, up: float) -> None:
        """Moves what a 2D view shows, as a drag does.

        Args:
            view: `ViewKind.AXIAL`, `CORONAL` or `SAGITTAL`.
            right: How far the view's centre moves towards the screen's right, in
                mm (negative: left).
            up: How far it moves towards the screen's top, in mm (negative: down).

        Raises:
            TypeError: If view is not a ViewKind.
            ValueError: If view is the 3D view, or right or up is not a finite
                number.
        """
        camera = self._camera_2d(view)
        if not (_is_number(right) and _is_number(up)):
            raise ValueError(
                f"A pan is two finite numbers (mm); got {right!r}, {up!r}."
            )
        pan = (camera.pan[0] + float(right), camera.pan[1] + float(up))
        self._set_camera(view, Camera2D(camera.zoom, pan))

    @agent_action
    def reset_camera(self, view: ViewKind) -> None:
        """Shows a view's whole field of view again, at zoom 1 and centred.

        The 3D view also returns to its starting direction.

        Args:
            view: The view to reset.

        Raises:
            TypeError: If view is not a ViewKind.
        """
        _require(view, ViewKind)
        if view == ViewKind.THREE_D:
            self._set_camera(view, Camera3D())
            return
        self._camera_2d(view)
        self._set_camera(view, Camera2D())

    @agent_action
    def rotate_camera(self, right: float, up: float) -> None:
        """Turns the 3D view's scene, as a drag in that view does.

        The turn is about the camera's own axes, so it never locks up when
        looking straight down.

        Args:
            right: Degrees to turn the scene towards the screen's right (about
                the screen's up axis).
            up: Degrees to turn it towards the screen's top (about the screen's
                right axis).

        Raises:
            ValueError: If right or up is not a finite number.
        """
        if not (_is_number(right) and _is_number(up)):
            raise ValueError(
                f"A rotation is two finite angles (degrees); got {right!r}, {up!r}."
            )
        camera = self.scene.view.cameras.three_d
        self._set_camera(ViewKind.THREE_D, camera.rotated(float(right), float(up)))

    @agent_action
    def set_camera(self, view: ViewKind, preset: CameraPreset) -> None:
        """Moves the 3D view's camera to a standard position, keeping its zoom.

        Args:
            view: `ViewKind.THREE_D`, the only view with camera positions.
            preset: Axial (from above), coronal (from the front) or sagittal
                (from the subject's right).

        Raises:
            TypeError: If view is not a ViewKind or preset not a CameraPreset.
            ValueError: If view is a 2D view.
        """
        _require(view, ViewKind)
        _require(preset, CameraPreset)
        if view != ViewKind.THREE_D:
            raise ValueError(f"The {view} view has no camera positions.")
        forward, up = PRESETS[preset]
        camera = self.scene.view.cameras.three_d
        self._set_camera(view, replace(camera, forward=forward, up=up))

    @agent_action
    def set_convention(self, convention: Convention) -> None:
        """Shows the subject's right on the screen's right or on its left.

        Each 2D view stays centred on the same place.

        Args:
            convention: `Convention.RAS` (neurological, right on the right) or
                `Convention.LAS` (radiological, right on the left).

        Raises:
            TypeError: If convention is not a Convention.
        """
        _require(convention, Convention)
        # A pan is along the screen's axes; where an axis flips, so must the pan,
        # so each view stays centred on the same place.
        old = self.scene.view.convention
        cameras = self.scene.view.cameras
        for view in _VIEW_AXIS:
            camera = cameras.of(view)
            assert isinstance(camera, Camera2D)
            signs = [
                float(np.dot(a, b))
                for a, b in zip(
                    screen_axes(view, old), screen_axes(view, convention), strict=True
                )
            ]
            pan = (camera.pan[0] * signs[0], camera.pan[1] * signs[1])
            cameras = cameras.with_camera(view, replace(camera, pan=pan))
        self.scene.set_view(ViewField.CAMERAS, cameras)
        self.scene.set_view(ViewField.CONVENTION, convention)

    @agent_action
    def view_centre(self, view: ViewKind) -> Vec3:
        """Returns the world position at the centre of a 2D view, in its slice plane.

        Args:
            view: `ViewKind.AXIAL`, `CORONAL` or `SAGITTAL`.

        Returns:
            The position (x, y, z) in mm, RAS+.

        Raises:
            TypeError: If view is not a ViewKind.
            ValueError: If view is the 3D view.
        """
        camera = self._camera_2d(view)
        lo, hi = self.scene.view.field_of_view
        centre = (np.asarray(lo) + np.asarray(hi)) / 2
        right, up = screen_axes(view, self.scene.view.convention)
        centre += camera.pan[0] * np.asarray(right) + camera.pan[1] * np.asarray(up)
        centre[_VIEW_AXIS[view]] = self.scene.view.cursor[_VIEW_AXIS[view]]
        return (float(centre[0]), float(centre[1]), float(centre[2]))

    @agent_action
    def readout(self) -> Readout:
        """Returns what the crosshair reports, as the 2D views show it.

        Returns:
            The cursor's position (mm, RAS+) and, with a main volume, its name,
            the nearest voxel (i, j, k, with the volume index for a 4D image) and
            the value there in scaled units, interpolated if the volume is shown
            interpolated; voxel and value are None outside its grid.
        """
        position = self.scene.view.cursor
        main = self.scene.view.main_volume
        if main is None:
            return Readout(position)
        item = self.scene.item(main)
        assert isinstance(item.data, Volume) and isinstance(item.options, VolumeOptions)
        o = item.options
        voxel, value = read_volume(item.data, position, o.volume_index, o.interpolate)
        return Readout(position, item.name, voxel, value)

    def _camera_2d(self, view: ViewKind) -> Camera2D:
        """Returns a 2D view's camera; raises ValueError for the 3D view."""
        _require(view, ViewKind)
        camera = self.scene.view.cameras.of(view)
        if not isinstance(camera, Camera2D):
            raise ValueError(f"The {view} view is not a 2D view.")
        return camera

    def _set_camera(self, view: ViewKind, camera: Camera2D | Camera3D) -> None:
        """Replaces one view's camera."""
        cameras = self.scene.view.cameras.with_camera(view, camera)
        self.scene.set_view(ViewField.CAMERAS, cameras)

    @agent_action
    def toggle_maximised(self, view: ViewKind) -> None:
        """Makes a view fill the window, or, if it already does, shows all four.

        The view filling the window is also selected, so the shortcuts act on it.

        Args:
            view: The view.

        Raises:
            TypeError: If view is not a ViewKind.
        """
        _require(view, ViewKind)
        maximised = None if self.scene.view.maximised == view else view
        if maximised is not None:
            self.select_view(view)
        self.scene.set_view(ViewField.MAXIMISED, maximised)

    def set_view_state(self, state: ViewState) -> None:
        """Sets the whole state of the views at once, as a session restores it.

        The main volume keeps its visibility and the other volumes are hidden,
        so at most the main volume is shown. Zooms are kept within `ZOOM_RANGE`
        and the 3D camera's directions are made unit and at right angles.

        Args:
            state: The new view state; its items must be in the scene.

        Raises:
            TypeError: If state is not a ViewState, or one of its fields is not
                of its type.
            ValueError: If the main volume is not a volume in the scene, there
                are volumes and no main volume, the selected item is not in the
                scene, or a position, camera or field of view is not finite.
        """
        _check_view_types(state)
        cursor = _position(state.cursor)
        lo, hi = (_position(corner) for corner in state.field_of_view)
        if not all(a < b for a, b in zip(lo, hi, strict=True)):
            raise ValueError(f"A field of view runs from low to high; got {lo}, {hi}.")
        cameras = _checked_cameras(state.cameras)
        volumes = self.scene.in_category(Category.VOLUMES)
        main = state.main_volume
        if main is None and volumes:
            raise ValueError("While there are volumes, one is the main volume.")
        if main is not None and main not in [v.id for v in volumes]:
            raise ValueError(f"The main volume {main} is not a volume in the scene.")
        if state.selected is not None:
            _require_target(state.selected)
            if not isinstance(state.selected, Category):
                self.scene.item(state.selected)

        self.scene.set_view(ViewField.CONVENTION, state.convention)
        self.scene.set_view(ViewField.FIELD_OF_VIEW, (lo, hi))
        self.scene.set_view(ViewField.CAMERAS, cameras)
        if main is not None:
            for volume in volumes:
                if volume.id != main:
                    self.scene.set_option(volume.id, OptionName.VISIBLE, False)
            self.scene.set_view(ViewField.MAIN_VOLUME, main)
        self.scene.set_view(ViewField.SELECTED, state.selected)
        self.scene.set_view(ViewField.SELECTED_VIEW, state.selected_view)
        self.scene.set_view(ViewField.MAXIMISED, state.maximised)
        self.scene.set_view(ViewField.OVERLAY_VISIBLE, state.overlay_visible)
        self.scene.set_view(ViewField.CROSSHAIR_VISIBLE, state.crosshair_visible)
        # Drawing a bundle is not restored, so neither is its tool.
        drawing = state.tool == Tool.DRAW_BUNDLE and not self.drawing
        self.scene.set_view(ViewField.TOOL, Tool.CURSOR if drawing else state.tool)
        self.scene.set_view(ViewField.WORKSPACE, state.workspace)
        self.scene.set_view(ViewField.LAYOUT, state.layout)
        self.move_cursor(cursor)

    @agent_action
    def toggle_overlay(self) -> None:
        """Shows or hides the text and colour bars drawn over the views.

        The text is the orientation letters at the edges and the crosshair's
        readout.
        """
        visible = not self.scene.view.overlay_visible
        self.scene.set_view(ViewField.OVERLAY_VISIBLE, visible)

    @agent_action
    def set_tool(self, tool: Tool) -> None:
        """Chooses the pointer tool of the views, which decides what a click does.

        `Tool.CURSOR` moves the crosshair, `SPHERE` and `BOX` place regions,
        `ZOOM_BOX` opens a rectangle of a 2D view in a detail window, and
        `DRAW_BUNDLE` draws a phantom bundle: choosing it starts one, as
        `start_bundle` does, and choosing another tool while drawing finishes
        the bundle if it has two points, else drops it.

        Args:
            tool: The tool.
        """
        tool = Tool(tool)
        if tool == Tool.DRAW_BUNDLE:
            if not self.drawing:
                self.start_bundle()  # which chooses the tool
            return
        if self.drawing:
            _, draft = self._phantom.draft()
            if drawable(draft.points):
                self.finish_bundle()
            else:
                self.cancel_bundle()
        self.scene.set_view(ViewField.TOOL, tool)

    # Detail windows

    @agent_action
    def open_detail(self, view: ViewKind, box: tuple[int, int, int, int]) -> DetailId:
        """Opens a detail window: a rectangle of a 2D view, drawn on its own.

        The window shows the rectangle's voxels filling its picture, with the
        slices drawn without smoothing, FOD glyphs built for the rectangle alone
        at the window's own sphere detail (12 at first), and optional voxel
        edges and scale bar. It follows the crosshair's slice while linked
        (`set_detail`). In the window the detail is shown in its own floating
        window, and its rectangle is outlined in the view.

        Args:
            view: `ViewKind.AXIAL`, `CORONAL` or `SAGITTAL`.
            box: The rectangle in the main volume's voxels: the lowest index and
                the size along each of the two voxel axes in the view's plane,
                the lower axis first, (i, j, size i, size j). The part inside the
                grid is kept.

        Returns:
            The detail window's id.

        Raises:
            TypeError: If view is not a ViewKind.
            ValueError: If view is the 3D view, nothing with a grid is loaded,
                box is not four whole numbers with positive sizes, or it lies
                outside the grid.
        """
        grid, box = self._detail_box(view, box)
        detail = Detail(
            0, view, box, self.scene.view.cursor, size=picture_size(grid, view, box)
        )
        return self.scene.add_detail(detail).id

    @agent_action
    def move_detail(self, detail_id: DetailId, box: tuple[int, int, int, int]) -> None:
        """Gives a detail window another rectangle of its view.

        Args:
            detail_id: The detail window.
            box: The rectangle, as for `open_detail`.

        Raises:
            ValueError: If there is no such detail window, or the rectangle is
                refused as in `open_detail`.
        """
        detail = self.scene.detail(detail_id)
        grid, box = self._detail_box(detail.view, box)
        size = detail.size
        if box[2:] != detail.box[2:]:
            size = picture_size(grid, detail.view, box)
        self.scene.set_detail(replace(detail, box=box, size=size))

    @agent_action
    def close_detail(self, detail_id: DetailId) -> None:
        """Closes a detail window.

        Args:
            detail_id: The detail window.

        Raises:
            ValueError: If there is no such detail window.
        """
        self.scene.remove_detail(detail_id)

    @agent_action
    def set_detail(
        self, detail_id: DetailId, name: DetailOption, value: object
    ) -> None:
        """Changes one setting of a detail window.

        Args:
            detail_id: The detail window.
            name: `DetailOption.LINKED` (follow the crosshair's slice; unlinked,
                it keeps the slice it shows), `VOXEL_GRID` (draw the voxels'
                edges), `SCALE_BAR` or `GLYPH_DETAIL` (the sphere detail of its
                FOD glyphs, 2 to 16).
            value: True or False, or a whole number for `GLYPH_DETAIL`.

        Raises:
            TypeError: If name is not a DetailOption, or value is not of its type.
            ValueError: If there is no such detail window, or the glyph detail is
                out of range.
        """
        _require(name, DetailOption)
        detail = self.scene.detail(detail_id)
        if name == DetailOption.GLYPH_DETAIL:
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"A glyph detail is a whole number; got {value!r}.")
            low, high = GLYPH_DETAILS
            if not low <= value <= high:
                raise ValueError(
                    f"A glyph detail is between {low} and {high}; got {value}."
                )
            self.scene.set_detail(replace(detail, glyph_detail=value))
            return
        if not isinstance(value, bool):
            raise TypeError(f"{name} is True or False; got {value!r}.")
        if name == DetailOption.LINKED:
            # Unlinked, it keeps the slice it shows now.
            position = self.detail_position(detail_id)
            self.scene.set_detail(replace(detail, linked=value, position=position))
        elif name == DetailOption.VOXEL_GRID:
            self.scene.set_detail(replace(detail, voxel_grid=value))
        else:
            self.scene.set_detail(replace(detail, scale_bar=value))

    def resize_detail(self, detail_id: DetailId, size: tuple[int, int]) -> None:
        """Records the size of a detail window's picture, as the window shows it.

        Args:
            detail_id: The detail window.
            size: Its width and height, logical pixels, each at least 1.

        Raises:
            ValueError: If there is no such detail window or a size is below 1.
        """
        width, height = (int(v) for v in size)
        if width < 1 or height < 1:
            raise ValueError(f"A picture's size is at least 1 x 1; got {size}.")
        detail = self.scene.detail(detail_id)
        self.scene.set_detail(replace(detail, size=(width, height)))

    def detail_position(self, detail_id: DetailId) -> Vec3:
        """Returns a point of the slice a detail window shows: the cursor if linked.

        Args:
            detail_id: The detail window.

        Returns:
            The point (x, y, z), world mm.

        Raises:
            ValueError: If there is no such detail window.
        """
        detail = self.scene.detail(detail_id)
        return self.scene.view.cursor if detail.linked else detail.position

    def detail_title(self, detail_id: DetailId) -> str:
        """Returns a detail window's title: the image, view, slice and size.

        The image is the selected FOD, else the first visible FOD, else the main
        volume.

        Args:
            detail_id: The detail window.

        Returns:
            The title, such as "Detail - wmfod.mif - coronal y 60 - 12 x 12 voxels".

        Raises:
            ValueError: If there is no such detail window.
        """
        detail = self.scene.detail(detail_id)
        selected = self.scene.view.selected
        fods = [i for i in self.scene.in_category(Category.FODS) if i.options.visible]
        named: Item | None = None
        if isinstance(selected, int) and self.scene.item(selected) in fods:
            named = self.scene.item(selected)
        elif fods:
            named = fods[0]
        elif self.scene.view.main_volume is not None:
            named = self.scene.item(self.scene.view.main_volume)
        axis = VIEW_AXIS[detail.view]
        slice_text = "xyz"[axis]
        grid = self._reference_grid()
        if grid is not None:
            index, _ = volume_slice(grid, self.detail_position(detail_id), axis)
            slice_text += f" {index}"
        return detail_title(
            None if named is None else named.name, detail.view, slice_text, detail.box
        )

    @agent_action
    def save_detail(
        self,
        detail_id: DetailId,
        path: str | Path,
        scale: float = 1.0,
        *,
        transparent: bool = False,
    ) -> Path:
        """Draws a detail window's picture off screen and writes it to a file.

        Args:
            detail_id: The detail window.
            path: The image file; .png or .tif keep a transparent background.
                An existing file is replaced.
            scale: The image's size as a multiple of the window's picture, 1 to
                8.
            transparent: Whether the background is transparent.

        Returns:
            The path written.

        Raises:
            ViewerUnavailableError: If the viewer's dependencies are not installed.
            ValueError: If there is no such detail window, scale is out of range,
                or the extension is not an image format (or is JPEG with a
                transparent background).
            OSError: If the file cannot be written.
        """
        detail = self.scene.detail(detail_id)
        _check_screenshot(None, transparent)
        _check_detail_scale(scale)
        from larmor.viewer.render.detail_renderer import save_detail

        return save_detail(self, detail, Path(path), float(scale), transparent)

    @agent_action
    def save_detail_figure(
        self,
        detail_id: DetailId,
        path: str | Path,
        scale: float = 1.0,
        background: FigureBackground = FigureBackground.WHITE,
    ) -> Path:
        """Writes a figure of a detail: its view with the rectangle, and the detail.

        The view is drawn whole (zoom 1) with the rectangle outlined, and the
        detail beside it at the view's height, with two lines joining the
        rectangle's corners to the detail's.

        Args:
            detail_id: The detail window.
            path: The image file, such as .png or .tif; an existing file is
                replaced.
            scale: The size as a multiple of the window's picture, 1 to 8.
            background: `FigureBackground.WHITE` or `BLACK`, around and between
                the two pictures.

        Returns:
            The path written.

        Raises:
            ViewerUnavailableError: If the viewer's dependencies are not installed.
            TypeError: If background is not a FigureBackground.
            ValueError: If there is no such detail window, scale is out of range,
                or the extension is not an image format.
            OSError: If the file cannot be written.
        """
        detail = self.scene.detail(detail_id)
        _require(background, FigureBackground)
        _check_screenshot(None, False)
        _check_detail_scale(scale)
        from larmor.viewer.render.detail_renderer import save_figure

        return save_figure(self, detail, Path(path), float(scale), background)

    def _detail_box(
        self, view: ViewKind, box: object
    ) -> tuple[Grid, tuple[int, int, int, int]]:
        """Returns the views' grid and a detail's rectangle, checked and clipped.

        Raises:
            TypeError: If view is not a ViewKind.
            ValueError: As for `open_detail`.
        """
        _require(view, ViewKind)
        if view not in VIEW_AXIS:
            raise ValueError(f"The {view} view has no slices to show in detail.")
        grid = self._reference_grid()
        if grid is None:
            raise ValueError("Open an image first: a detail is a rectangle of voxels.")
        values = tuple(box) if isinstance(box, Iterable) else ()
        if len(values) != 4 or not all(_is_id(v) for v in values):
            raise ValueError(f"A detail's box is four whole numbers; got {box!r}.")
        i, j, size_i, size_j = (int(v) for v in values)
        if size_i < 1 or size_j < 1:
            raise ValueError(f"A detail's box has positive sizes; got {box!r}.")
        clipped = clipped_box(grid, view, (i, j, size_i, size_j))
        if clipped is None:
            raise ValueError(f"The box {box!r} lies outside the grid.")
        return grid, clipped

    @agent_action
    def set_workspace(self, workspace: Workspace) -> None:
        """Shows the window's View or Simulate workspace.

        A pointer tool the new workspace does not offer goes back to the cursor.

        Args:
            workspace: `Workspace.VIEW` for the scene tree and the inspector, or
                `Workspace.SIMULATE` for the phantom editor's steps.

        Raises:
            TypeError: If workspace is not a Workspace.
        """
        _require(workspace, Workspace)
        if self.scene.view.tool not in WORKSPACE_TOOLS[workspace]:
            self.scene.set_view(ViewField.TOOL, Tool.CURSOR)
        self.scene.set_view(ViewField.WORKSPACE, workspace)

    @agent_action
    def set_layout(self, layout: Layout) -> None:
        """Lays the four views out in a grid, as one large and three small, or alone.

        Args:
            layout: `Layout.GRID` (two by two), `Layout.ONE_PLUS_THREE` (the
                selected view large on the left, the others stacked on the
                right) or `Layout.ONE` (the selected view only).

        Raises:
            TypeError: If layout is not a Layout.
        """
        _require(layout, Layout)
        self.scene.set_view(ViewField.LAYOUT, layout)

    @agent_action
    def toggle_crosshair(self) -> None:
        """Shows or hides the crosshair in the 2D views."""
        visible = not self.scene.view.crosshair_visible
        self.scene.set_view(ViewField.CROSSHAIR_VISIBLE, visible)

    @agent_action
    def screenshot(
        self,
        view: ViewKind | None,
        path: str | Path,
        *,
        size: tuple[int, int] = (800, 800),
        scale: float = 1.0,
        overlay: bool | None = None,
        transparent: bool = False,
    ) -> Path:
        """Draws one view, or all of them, off screen, as the window shows them.

        Needs no window and no Qt, only the viewer's drawing library (pygfx) and
        a GPU or a software renderer. With a window open, the image is drawn
        separately, at the given size, and the window is not changed. To draw
        many images of one scene, an `OffscreenRenderer` (in
        `larmor.viewer.render.offscreen`) is faster: it is set up once.

        Args:
            view: The view to draw, or None for the whole window's layout: the
                maximised view if there is one, else the four views in the
                window's 2 x 2 grid (axial, coronal; sagittal, 3D), with a thin
                grey line between them.
            path: The image file to write; its extension sets the format (.png,
                .jpg, .tif, ...). An existing file is replaced.
            size: Each view's width and height in logical pixels, the units of
                the window's views; text and lines keep their size in them.
            scale: Pixels per logical pixel in the image, for example 2 for twice
                the resolution with the same layout.
            overlay: Whether to draw the information overlay (text, labels and
                colour bars); None follows `toggle_overlay`'s switch, which the
                screenshot leaves as it is.
            transparent: Whether the background is transparent (an RGBA image)
                rather than the views' background colour; needs PNG or TIFF.

        Returns:
            The path written. The image is RGB, or RGBA if transparent; each view
            in it is size times scale pixels.

        Raises:
            ViewerUnavailableError: If the viewer's dependencies are not installed.
            TypeError: If view is neither a ViewKind nor None, or overlay or
                transparent is not a bool.
            ValueError: If size or scale is not positive, the image is larger
                than the GPU draws, or the extension is not an image format (or
                is JPEG with a transparent background).
            OSError: If the file cannot be written.

        Example:
            >>> viewer.screenshot(ViewKind.AXIAL, "axial.png", size=(600, 600))
            >>> viewer.screenshot(None, "all.png", scale=2, transparent=True)
        """
        if view is not None:
            _require(view, ViewKind)
        _check_screenshot(overlay, transparent)
        preset = ExportPreset("", size, scale, overlay=overlay, transparent=transparent)
        return self.export_image(preset, path, view)

    @agent_action
    def screenshot_views(
        self,
        path: str | Path,
        *,
        size: tuple[int, int] = (800, 800),
        scale: float = 1.0,
        overlay: bool | None = None,
        transparent: bool = False,
    ) -> tuple[Path, ...]:
        """Draws the three 2D views off screen, each into its own file.

        The images are those `screenshot` draws of each 2D view, drawn with one
        renderer, so they are faster than three screenshots.

        Args:
            path: The name the files are made from, the view's name added to
                its stem: "fig.png" gives "fig_axial.png", "fig_coronal.png"
                and "fig_sagittal.png". Existing files are replaced.
            size: Each view's width and height in logical pixels.
            scale: Pixels per logical pixel in the images.
            overlay: Whether to draw the information overlay; None follows
                `toggle_overlay`'s switch, which is left as it is.
            transparent: Whether the background is transparent (RGBA images);
                needs PNG or TIFF.

        Returns:
            The paths written: axial, coronal, sagittal.

        Raises:
            ViewerUnavailableError: If the viewer's dependencies are not installed.
            TypeError: If overlay or transparent is not a bool.
            ValueError: As for `screenshot`.
            OSError: If a file cannot be written.

        Example:
            >>> viewer.screenshot_views("fig.png", scale=2)
        """
        _check_screenshot(overlay, transparent)
        preset = ExportPreset("", size, scale, overlay=overlay, transparent=transparent)
        return self.export_view_images(preset, path)

    @agent_action
    def record_rotation(
        self,
        path: str | Path,
        *,
        size: tuple[int, int] = (800, 800),
        scale: float = 1.0,
        speed: float = ROTATION_SPEED,
        duration: float = ROTATION_DURATION,
        fps: int = ROTATION_FPS,
        overlay: bool | None = None,
        transparent: bool = False,
        on_progress: Progress | None = None,
    ) -> Path | None:
        """Records the 3D view turning, off screen, as an animated WebP or GIF.

        The scene turns about the screen's up axis, starting from the 3D view's
        camera, which is left as it is: with a window open, its 3D view does not
        move. The frames are written as they are drawn, except that a GIF keeps
        every frame in memory until it is written.

        Args:
            path: The file to write: .webp or .gif. An existing file is replaced
                once the video is complete, and kept if it is cancelled.
            size: The 3D view's width and height in logical pixels.
            scale: Pixels per logical pixel in the frames.
            speed: Degrees the scene turns per second, towards the screen's
                right; negative turns it to the left.
            duration: Seconds of video; the default, at the default speed, is
                one full turn.
            fps: Frames per second, a whole number; a GIF takes at most 50. GIF
                frame times are whole hundredths of a second, so at 30 fps its
                frames last 30 or 40 ms.
            overlay: Whether to draw the information overlay; None follows
                `toggle_overlay`'s switch.
            transparent: Whether the background is transparent; needs WebP.
            on_progress: Called after each frame with the number of frames drawn
                and their total, for example to show progress; it returns True
                to go on, or False to cancel.

        Returns:
            The path written, or None if on_progress cancelled the video.

        Raises:
            ViewerUnavailableError: If the viewer's dependencies are not installed.
            TypeError: If overlay or transparent is not a bool, speed or
                duration is not a real number, or fps is not a whole number.
            ValueError: If size or scale is not positive, the frames are larger
                than the GPU draws, the extension is neither .webp nor .gif (or
                is .gif with a transparent background or fps above 50), speed
                is not finite, duration is not positive, or fps is not positive.
            OSError: If the file cannot be written.

        Example:
            >>> viewer.record_rotation("turn.webp", size=(600, 600))
            >>> viewer.record_rotation("half.gif", speed=45, duration=4, fps=20)
        """
        _check_screenshot(overlay, transparent)
        preset = ExportPreset("", size, scale, overlay=overlay, transparent=transparent)
        return self.export_video(
            preset,
            path,
            speed=speed,
            duration=duration,
            fps=fps,
            on_progress=on_progress,
        )

    @agent_action
    def export(
        self, preset: str, path: str | Path, view: ViewKind | None = None
    ) -> Path:
        """Draws one view, or the window's layout, off screen with an export preset.

        The preset sets the size, the scale, the background, the information
        overlay, the transparency, the 3D camera and the options given to the
        items while the image is drawn; the scene is as it was afterwards. The
        shipped presets are "Screen", "Figure", "Talk" and "Website", and the
        Export sheet saves others in the settings folder.

        Args:
            preset: The preset's name.
            path: The image file to write; its extension sets the format (.png,
                .jpg, .tif, ...). An existing file is replaced.
            view: The view to draw, or None for the window's layout, as in
                `screenshot`.

        Returns:
            The path written.

        Raises:
            ViewerUnavailableError: If the viewer's dependencies are not installed.
            TypeError: If view is neither a ViewKind nor None.
            ValueError: If there is no such preset, the image is larger than
                the GPU draws, or the extension is not an image format (or is
                JPEG with a transparent background).
            OSError: If the file cannot be written.

        Example:
            >>> viewer.export("Website", "slide.png", ViewKind.THREE_D)
        """
        if view is not None:
            _require(view, ViewKind)
        _check_screenshot(None, False)
        return self.export_image(find_preset(preset), path, view)

    @agent_action
    def export_views(self, preset: str, path: str | Path) -> tuple[Path, ...]:
        """Draws the three 2D views off screen with an export preset, a file each.

        Args:
            preset: The preset's name, as for `export`.
            path: The name the files are made from, as in `screenshot_views`.

        Returns:
            The paths written: axial, coronal, sagittal.

        Raises:
            ViewerUnavailableError: If the viewer's dependencies are not installed.
            ValueError: As for `export`.
            OSError: If a file cannot be written.
        """
        _check_screenshot(None, False)
        return self.export_view_images(find_preset(preset), path)

    @agent_action
    def export_rotation(
        self,
        preset: str,
        path: str | Path,
        *,
        speed: float = ROTATION_SPEED,
        duration: float = ROTATION_DURATION,
        fps: int = ROTATION_FPS,
    ) -> Path | None:
        """Records the 3D view turning, off screen, with an export preset.

        The turn starts from the preset's 3D camera, or from the 3D view's if
        the preset has none.

        Args:
            preset: The preset's name, as for `export`.
            path: The file to write, .webp or .gif, as in `record_rotation`.
            speed: Degrees the scene turns per second.
            duration: Seconds of video.
            fps: Frames per second, a whole number; a GIF takes at most 50.

        Returns:
            The path written.

        Raises:
            ViewerUnavailableError: If the viewer's dependencies are not installed.
            TypeError: As for `record_rotation`.
            ValueError: If there is no such preset, or as for `record_rotation`.
            OSError: If the file cannot be written.
        """
        _check_screenshot(None, False)
        return self.export_video(
            find_preset(preset), path, speed=speed, duration=duration, fps=fps
        )

    def export_image(
        self,
        preset: ExportPreset,
        path: str | Path,
        view: ViewKind | None = None,
        *,
        size: tuple[int, int] | None = None,
    ) -> Path:
        """Draws one view, or the window's layout, off screen as a preset says.

        Args:
            preset: The preset.
            path: The image file to write.
            view: The view, or None for the window's layout.
            size: Each view's size in logical pixels when the preset has none;
                `DEFAULT_SIZE` if None.

        Returns:
            The path written.

        Raises:
            ValueError: As for `export`.
            OSError: If the file cannot be written.
        """
        from larmor.viewer.render.offscreen import check_image

        # Before the renderer, which starts the GPU and uploads the scene.
        check_image(Path(path), preset.transparent)
        with self._export_renderer(preset, size) as renderer:
            return renderer.save(
                view, path, overlay=preset.overlay, transparent=preset.transparent
            )

    def export_view_images(
        self,
        preset: ExportPreset,
        path: str | Path,
        *,
        size: tuple[int, int] | None = None,
    ) -> tuple[Path, ...]:
        """Draws the three 2D views off screen as a preset says, each into a file.

        Args:
            preset: The preset.
            path: The name the files are made from, as in `screenshot_views`.
            size: Each view's size in logical pixels when the preset has none;
                `DEFAULT_SIZE` if None.

        Returns:
            The paths written: axial, coronal, sagittal.

        Raises:
            ValueError: As for `export`.
            OSError: If a file cannot be written.
        """
        from larmor.viewer.render.offscreen import check_image

        check_image(Path(path), preset.transparent)
        with self._export_renderer(preset, size) as renderer:
            return renderer.save_views(
                path, overlay=preset.overlay, transparent=preset.transparent
            )

    def export_video(
        self,
        preset: ExportPreset,
        path: str | Path,
        *,
        size: tuple[int, int] | None = None,
        speed: float = ROTATION_SPEED,
        duration: float = ROTATION_DURATION,
        fps: int = ROTATION_FPS,
        on_progress: Progress | None = None,
    ) -> Path | None:
        """Records the 3D view turning, off screen, as a preset says.

        Args:
            preset: The preset.
            path: The file to write, .webp or .gif.
            size: The 3D view's size in logical pixels when the preset has none;
                `DEFAULT_SIZE` if None.
            speed: Degrees the scene turns per second.
            duration: Seconds of video.
            fps: Frames per second.
            on_progress: Called after each frame, as in `record_rotation`.

        Returns:
            The path written, or None if on_progress cancelled the video.

        Raises:
            ValueError: As for `record_rotation`.
            OSError: If the file cannot be written.
        """
        from larmor.viewer.render.offscreen import check_video

        check_video(Path(path), speed, duration, fps, preset.transparent)
        with self._export_renderer(preset, size) as renderer:
            return renderer.save_rotation(
                path,
                speed,
                duration,
                fps,
                overlay=preset.overlay,
                transparent=preset.transparent,
                on_progress=on_progress,
            )

    @contextmanager
    def _export_renderer(
        self, preset: ExportPreset, size: tuple[int, int] | None
    ) -> Iterator[OffscreenRenderer]:
        """Yields an off-screen renderer that draws as a preset says."""
        # Imported here: the model itself never loads pygfx.
        from larmor.viewer.render.annotations import CROSSHAIR_COLOUR
        from larmor.viewer.render.offscreen import OffscreenRenderer

        shape = preset.size or size or DEFAULT_SIZE
        with self.preset_options(preset):
            with OffscreenRenderer(self, shape, preset.scale) as renderer:
                if preset.background is not None:
                    renderer.set_colours(preset.background, CROSSHAIR_COLOUR)
                yield renderer

    @contextmanager
    def preset_options(self, preset: ExportPreset) -> Iterator[None]:
        """Gives the items a preset's options and the 3D view its camera, for a block.

        Each item of a category the preset names takes the options it has of
        the preset's overrides; an option an item refuses is left as it is.
        Every option and the camera are set back at the end of the block.

        Args:
            preset: The preset.
        """
        saved: list[tuple[ItemId, OptionName, object]] = []
        camera = self.scene.view.cameras.three_d
        try:
            for category, values in preset.overrides.items():
                for item in self.scene.in_category(category):
                    for name, value in values.items():
                        old = getattr(item.options, name, None)
                        if old is None or old == value:
                            continue
                        try:
                            self.set(item.id, name, value)
                        except OptionError:
                            continue
                        saved.append((item.id, name, old))
            if preset.camera is not None:
                self._set_camera(ViewKind.THREE_D, preset.camera)
            yield
        finally:
            present = {i.id for i in self.scene.items}
            for item_id, name, old in reversed(saved):
                if item_id in present:
                    self.set(item_id, name, old)
            if preset.camera is not None:
                self._set_camera(ViewKind.THREE_D, camera)

    def _reference_grid(self) -> Grid | None:
        """Returns the grid the 2D views slice along, whose voxel is a slice step.

        Returns:
            The main volume's grid; else the phantom's; else the first overlay's;
            else the first FOD's; else the first peaks image's; else the grid of
            the first surface that names the volume it was made on; else a 1 mm
            grid over the tractograms' points and the surfaces' vertices; None if
            nothing is loaded.
        """
        main = self.scene.view.main_volume
        if main is not None:
            data = self.scene.item(main).data
            assert isinstance(data, Volume)
            return data.grid
        if self.scene.phantom is not None:
            return self.scene.phantom.grid
        for category in (Category.OVERLAYS, Category.FODS, Category.PEAKS):
            for item in self.scene.in_category(category):
                assert isinstance(item.data, Volume)
                return item.data.grid
        surfaces = [
            i.data.surface for i in self.scene.items if isinstance(i.data, SurfaceData)
        ]
        for surface in surfaces:
            if surface.reference is not None:
                return surface.reference
        tractograms = [
            i.data for i in self.scene.items if isinstance(i.data, Tractogram)
        ]
        points = [t.points for t in tractograms if t.num_points > 0]
        points += [s.vertices for s in surfaces if s.num_vertices > 0]
        if not points:
            return None
        lo = np.floor(np.min([p.min(axis=0) for p in points], axis=0))
        hi = np.ceil(np.max([p.max(axis=0) for p in points], axis=0))
        shape = tuple(int(n) for n in hi - lo + 1)
        affine = np.eye(4)
        affine[:3, 3] = lo
        return Grid((shape[0], shape[1], shape[2]), affine)


# mypy checks the types of these arguments in our code; these checks are for callers
# from plain Python and from agents (MCP), whose text would otherwise be misread.
def _require(value: object, kind: type[StrEnum]) -> None:
    """Raises TypeError unless value is a member of the enum `kind`."""
    if not isinstance(value, kind):
        name = kind.__name__
        article = "an" if name[0] in "AEIOU" else "a"
        example = f"{name}.{next(iter(kind)).name}"
        raise TypeError(f"Expected {article} {name}, such as {example}; got {value!r}.")


def _check_screenshot(overlay: object, transparent: object) -> None:
    """Checks a screenshot's switches, and that the viewer can draw off screen."""
    if overlay is not None and not isinstance(overlay, bool):
        raise TypeError(f"overlay must be True, False or None; got {overlay!r}.")
    if not isinstance(transparent, bool):
        raise TypeError(f"transparent must be True or False; got {transparent!r}.")
    if importlib.util.find_spec("pygfx") is None:
        raise ViewerUnavailableError(
            f"Screenshots need the viewer's dependencies. {INSTALL_HINT}"
        )


def _check_detail_scale(scale: object) -> None:
    """Raises ValueError unless scale is a number within DETAIL_SCALES."""
    low, high = DETAIL_SCALES
    error = ValueError(f"A scale is between {low:g} and {high:g}; got {scale!r}.")
    if not _is_number(scale):
        raise error
    assert isinstance(scale, Real)
    if not low <= float(scale) <= high:
        raise error


def _is_number(value: object) -> bool:
    """Tells whether value is a finite real number (not a bool)."""
    if isinstance(value, bool) or not isinstance(value, Real):
        return False
    return math.isfinite(float(value))


def _clamp_zoom(zoom: float) -> float:
    """Returns the zoom kept within ZOOM_RANGE."""
    return min(max(zoom, ZOOM_RANGE[0]), ZOOM_RANGE[1])


def _position(position: object) -> Vec3:
    """Three finite numbers as a world position; raises ValueError otherwise."""
    values = tuple(position) if isinstance(position, Iterable) else ()
    if len(values) != 3 or not all(_is_number(v) for v in values):
        raise ValueError(f"A position is three finite numbers (mm); got {position!r}.")
    x, y, z = (float(v) for v in values)
    return (x, y, z)


def _checked_cameras(cameras: object) -> Cameras:
    """Returns the cameras with zooms in range and the 3D directions orthonormal."""
    if not isinstance(cameras, Cameras):
        raise TypeError(f"Expected Cameras; got {cameras!r}.")
    for view in _VIEW_AXIS:
        camera = cameras.of(view)
        if not isinstance(camera, Camera2D):
            raise TypeError(f"The {view} view has a Camera2D; got {camera!r}.")
        if not (_is_number(camera.zoom) and camera.zoom > 0):
            raise ValueError(f"A zoom is a positive number; got {camera.zoom!r}.")
        values: tuple[object, ...] = tuple(camera.pan)
        if len(values) != 2 or not all(_is_number(p) for p in values):
            raise ValueError(f"A pan is two finite numbers (mm); got {camera.pan!r}.")
        pan = (float(camera.pan[0]), float(camera.pan[1]))
        cameras = cameras.with_camera(view, Camera2D(_clamp_zoom(camera.zoom), pan))
    c3 = cameras.three_d
    if not isinstance(c3, Camera3D):
        raise TypeError(f"The 3D view has a Camera3D; got {c3!r}.")
    if not (_is_number(c3.zoom) and c3.zoom > 0):
        raise ValueError(f"A zoom is a positive number; got {c3.zoom!r}.")
    f, u = np.asarray(_position(c3.forward)), np.asarray(_position(c3.up))
    # An orthonormal pair is kept exactly, so a saved camera comes back the same.
    unit = np.allclose(np.linalg.norm([f, u], axis=1), 1.0, rtol=0, atol=1e-12)
    if not (unit and abs(f @ u) < 1e-12):
        u = u - f * (u @ f) / max(float(f @ f), 1e-300)
        if np.linalg.norm(f) < 1e-9 or np.linalg.norm(u) < 1e-9:
            raise ValueError("The 3D camera's forward and up are distinct directions.")
        f, u = f / np.linalg.norm(f), u / np.linalg.norm(u)
    c3 = Camera3D(
        (float(f[0]), float(f[1]), float(f[2])),
        (float(u[0]), float(u[1]), float(u[2])),
        _clamp_zoom(c3.zoom),
    )
    return cameras.with_camera(ViewKind.THREE_D, c3)


def _require_target(target: object) -> None:
    """Raises TypeError unless target is an item id or a Category."""
    is_id = isinstance(target, int) and not isinstance(target, bool)
    if not (is_id or isinstance(target, Category)):
        raise TypeError(
            "Expected an item id or a Category, such as Category.VOLUMES; "
            f"got {target!r}."
        )


def _step_along(grid: Grid, axis: int) -> float:
    """Returns the voxel size along the grid axis closest to world axis `axis`."""
    return float(np.linalg.norm(grid.affine[:3, closest_axis(grid, axis)]))


def _centre(grid: Grid) -> tuple[float, float, float]:
    """Returns the world position of the grid's centre voxel."""
    # A whole voxel: on an even axis the exact middle lies between two voxels, and a
    # slice there would blend them.
    voxel = np.array(grid.shape[:3]) // 2
    x, y, z = grid.affine[:3, :3] @ voxel + grid.affine[:3, 3]
    return (float(x), float(y), float(z))


def _checked_data(category: Category, data: object) -> object:
    """Returns an item's data for an object, checked as its category needs.

    Returns:
        The data, as `ItemType.data` gives it.

    Raises:
        ValueError: If the object is not of the category's type, an FOD's number
            of coefficients fits no SH layout, or a peaks image is not 4D with a
            multiple of 3 volumes.
    """
    data = item_type(category).data(data)
    if category == Category.FODS:
        assert isinstance(data, Volume)
        # Imported here: larmor.sd loads scipy, which the model avoids.
        from larmor.sd import sh_layout

        sh_layout(data.data.shape[3] if data.data.ndim == 4 else 1)
    if category == Category.PEAKS:
        assert isinstance(data, Volume)
        _check_peaks(data)
    return data


def _follow(
    value: float,
    old: tuple[float, float] | None,
    new: tuple[float, float] | None,
    end: int,
) -> float:
    """Returns a value set within an old data range, carried over to a new range.

    Args:
        value: The value, such as a window's minimum.
        old: The old range; None if there was none.
        new: The new range; None if there is none.
        end: The end of the new range taken when the old range has no width:
            0 for the low end, 1 for the high end.

    Returns:
        The value at the same place in the new range as in the old one, kept
        inside the new range.
    """
    if old is None or new is None:
        return value
    width = old[1] - old[0]
    if width <= 0:
        return new[end]
    mapped = new[0] + (value - old[0]) / width * (new[1] - new[0])
    return min(max(mapped, new[0]), new[1])


def _check_peaks(volume: Volume) -> None:
    """Raises ValueError unless the image holds 3 volumes (x, y, z) per peak."""
    shape = volume.data.shape
    if len(shape) != 4 or shape[3] % 3 != 0:
        raise ValueError(
            f"A peaks image is 4D with 3 volumes (x, y, z) per peak; got shape {shape}."
        )


def _changes_for_data(
    item: Item, data: object, new_range: tuple[float, float] | None
) -> tuple[dict[OptionName, object], dict[OptionName, float]]:
    """Returns the options new data changes, and the range options it moves.

    The first dict holds options valid for the old data and the new; the second
    the minimum, maximum and threshold mapped from the old range onto new_range.
    """
    o = item.options
    changes: dict[OptionName, object] = {}
    ranged: dict[OptionName, float] = {}
    if isinstance(o, VolumeOptions):
        assert isinstance(data, Volume)
        count = data.data.shape[3] if data.data.ndim == 4 else 1
        index = min(o.volume_index, count - 1)
        changes[OptionName.VOLUME_INDEX] = index
        old = item.value_range
        ranged[OptionName.MINIMUM] = _follow(o.minimum, old, new_range, 0)
        ranged[OptionName.MAXIMUM] = _follow(o.maximum, old, new_range, 1)
        if isinstance(o, OverlayOptions):
            ranged[OptionName.THRESHOLD] = _follow(o.threshold, old, new_range, 0)
            if o.surface_labels:
                present = set(labels_in(data, index))
                changes[OptionName.SURFACE_LABELS] = tuple(
                    label for label in o.surface_labels if label in present
                )
    elif isinstance(o, TractogramOptions) and o.values_file:
        changes[OptionName.VALUES_FILE] = ""
        if o.colouring == Colouring.VALUES:
            changes[OptionName.COLOURING] = TractogramOptions().colouring
    elif isinstance(o, SurfaceOptions) and o.colour_by is not None:
        assert isinstance(data, SurfaceData)
        if o.colour_by not in data.values():
            changes[OptionName.COLOUR_BY] = None
    return changes, ranged


def _check_volume_index(item: Item, value: object) -> None:
    """Raises OptionError if a volume index is past the item's last volume."""
    count = item.volume_count
    assert isinstance(count, int)
    assert isinstance(value, int)
    if value >= count:
        raise OptionError(
            f"Volume index must be between 0 and {count - 1}; got {value}."
        )


def _check_view_types(state: ViewState) -> None:
    """Raises TypeError if a view state or one of its enums or switches is mistyped."""
    if not isinstance(state, ViewState):
        raise TypeError(f"Expected a ViewState; got {state!r}.")
    _require(state.selected_view, ViewKind)
    if state.maximised is not None:
        _require(state.maximised, ViewKind)
    _require(state.convention, Convention)
    _require(state.tool, Tool)
    _require(state.workspace, Workspace)
    _require(state.layout, Layout)
    for flag in (state.overlay_visible, state.crosshair_visible):
        if not isinstance(flag, bool):
            raise TypeError(f"A switch is True or False; got {flag!r}.")


def _check_dti(item: Item) -> None:
    """Raises OptionError unless the image holds 3 volumes: a direction's x, y, z."""
    assert isinstance(item.data, Volume)
    shape = item.data.data.shape
    if len(shape) != 4 or shape[3] != 3:
        raise OptionError(
            f"DTI colours need a 4D image with 3 volumes (x, y and z of the main "
            f"direction); {item.name} has shape {shape}."
        )


def _check_cortex(item: Item) -> None:
    """Raises OptionError unless the surface has a cortex mask, which finds the wall."""
    assert isinstance(item.data, SurfaceData)
    if item.data.cortex is None:
        raise OptionError(
            f"{item.name} has no cortex label (label/lh.cortex.label beside its "
            "folder), so its medial wall is unknown."
        )


def _is_id(value: object) -> TypeIs[int]:
    """Tells whether a session's value can be an item's id: a whole number."""
    return isinstance(value, int) and not isinstance(value, bool)


def _known_id(value: object, ids: Mapping[ItemId, ItemId]) -> bool:
    """Tells whether a session's value is the id of an item opened."""
    return _is_id(value) and value in ids


def _check_region_option(item: Item, name: OptionName, value: object) -> None:
    """Raises OptionError for a region's shape that changes, or a mask on a shape."""
    o = item.options
    assert isinstance(o, RegionOptions)
    if name == OptionName.SHAPE and value != o.shape:
        raise OptionError(f"{item.name}: a region's shape is set when it is added.")
    if name == OptionName.MASK and o.shape != RegionShape.MASK:
        raise OptionError(f"{item.name} is a {o.shape}, so it names no mask.")


def _item_ref_error(scene: Scene, item: Item, name: OptionName, value: int) -> str:
    """Returns why an option of item cannot name the item `value`, for the user."""
    _, info = option_fields(item.options)[name]
    if value == item.id:
        return f"{info.label}: {item.name} cannot name itself."
    try:
        other = scene.item(value)
    except ValueError:
        return f"{info.label}: there is no item {value}."
    if name == OptionName.REGIONS:
        return f"{info.label}: {other.name} is not a region."
    if isinstance(item.data, Tractogram) or item.category == Category.REGIONS:
        return f"{info.label}: {other.name} is not a volume or overlay of one volume."
    return (
        f"{info.label}: {other.name} is not a 3D volume or overlay on the grid of "
        f"{item.name}."
    )


def _value_range(volume: Volume) -> tuple[float, float]:
    """Returns the range of an image's finite values, in scaled units.

    With no finite voxel, the stored values are taken to range from 0 to 1.
    """
    data = volume.data
    raw = np.min(data), np.max(data)
    if not np.isfinite(raw).all():
        # NaN or infinite voxels, as in an ADC map, are left out.
        finite = data[np.isfinite(data)]
        raw = (finite.min(), finite.max()) if finite.size else (0.0, 1.0)
    a, b = (float(v) * volume.slope + volume.intercept for v in raw)
    return (min(a, b), max(a, b))


def _subset_values(path: str, tractogram: Tractogram, subset: Subset) -> np.ndarray:
    """Returns a TSF file's values of the streamlines a tractogram kept of its file.

    Returns:
        The values.

    Raises:
        ValueError: If the file does not match the file the tractogram was read
            from, streamline by streamline.
    """
    values, offsets = read_tsf(path)
    count = len(offsets) - 1
    if count != subset.total:
        raise ValueError(
            f"{path} has values for {count} streamlines, but the tractogram's file "
            f"has {subset.total}."
        )
    starts = offsets[subset.indices]
    lengths = offsets[subset.indices + 1] - starts
    if not np.array_equal(lengths, np.diff(tractogram.offsets)):
        raise ValueError(
            f"{path} has a different number of values than points for some "
            "streamlines of the tractogram."
        )
    owner = np.repeat(np.arange(len(lengths)), lengths)
    index = starts[owner] + np.arange(len(owner)) - tractogram.offsets[owner]
    kept: np.ndarray = values[index]
    return kept


def _finite_range(values: np.ndarray) -> tuple[float, float]:
    """Returns the range of the finite values; 0 to 1 if none is finite."""
    finite = values[np.isfinite(values)]
    if not finite.size:
        return (0.0, 1.0)
    return (float(finite.min()), float(finite.max()))


def _surface_options(
    data: SurfaceData, options: SurfaceOptions, name: str | None
) -> tuple[SurfaceOptions, tuple[float, float]]:
    """Returns a new surface's options and value range.

    The colour follows the name, and a surface with curvatures is coloured by them.
    """
    lowered = (name or "").lower()
    colour = next(
        (c for key, c in SURFACE_COLOURS.items() if key in lowered), SURFACE_COLOUR
    )
    options = replace(options, colour=colour)
    if "curv" in data.values():
        options = replace(options, colour_by="curv", **_colour_by(data, "curv"))
    ranges = [CURVATURE_RANGE, *(data.value_range(v) for v in data.values())]
    return options, (min(r[0] for r in ranges), max(r[1] for r in ranges))


def _colour_by(data: SurfaceData, name: str) -> dict[str, Any]:
    """Returns the colormap and range a surface's values `name` start with."""
    if name in CURVATURES:
        lo, hi = CURVATURE_RANGE
        return {"colormap": Colormap.GRAY_R, "minimum": lo, "maximum": hi}
    lo, hi = data.value_range(name)
    return {"colormap": VALUES_COLORMAP, "minimum": lo, "maximum": hi}
