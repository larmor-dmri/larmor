"""Holds the phantom editor's state as plain values and builds a simulation from it."""

import inspect
import itertools
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields, replace
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from larmor.io import read_bvals_bvecs
from larmor.types import GradientTable, Grid, Tractogram, Volume
from larmor.viewer.jobs import SAVING
from larmor.viewer.model.item import ItemId

if TYPE_CHECKING:
    from larmor.simulation import (
        Bundle,
        BundleGeometry,
        Phantom,
        Protocol,
        Tissue,
        VoxelPartition,
    )

# larmor.simulation is imported only in the functions that need it, so opening the
# viewer does not load it.

type Vec3 = tuple[float, float, float]
# Called by a long job with the steps done, their total and the stage.
type Report = Callable[[int, int, str], None]

# The size of a new phantom's box and its voxels, in mm.
DEFAULT_BOX_MM: Final = (60.0, 60.0, 60.0)
DEFAULT_VOXEL_MM: Final = 2.0
# The most voxels a phantom's grid may have.
MAX_GRID_VOXELS: Final = 256**3
# The upper bound of the bundle seeds the editor draws.
SEED_LIMIT: Final = 2**31
# The name of a simulation folder's description file, and its format name.
SIMULATION_INFO: Final = "simulation.json"
SIMULATION_FORMAT: Final = "larmor-simulation"


# The serial numbers of phantoms, see `PhantomState.serial`.
_SERIALS = itertools.count(1)


class TissuePreset(StrEnum):
    """How a tissue is made, from a preset's parameters or by compartments."""

    STANDARD_MODEL = "standard_model"  # white matter: stick, zeppelin, free water
    SANDI = "sandi"  # grey matter: sphere, powder-averaged stick, ball
    NODDI = "noddi"  # Watson-dispersed sticks, zeppelin, free water
    FREE_WATER = "free_water"  # CSF: free water alone
    CUSTOM = "custom"


class CompartmentKind(StrEnum):
    """The diffusion model of a custom tissue's compartment."""

    STICK = "stick"
    ZEPPELIN = "zeppelin"
    BALL = "ball"
    POWDER_STICK = "powder_stick"
    SPHERE = "sphere"
    WATSON_STICK = "watson_stick"
    CYLINDER = "cylinder"
    DOT = "dot"
    FREE_WATER = "free_water"


class BvecFrame(StrEnum):
    """The frame of a bvecs file, as `larmor.io.read_bvals_bvecs` names it."""

    FSL = "fsl"  # along the grid's axes, x flipped for a positive determinant
    WORLD = "world"


class EchoKind(StrEnum):
    """The scan's kind of echo, which decides the relaxation with T2 or T2*."""

    SPIN_ECHO = "spin_echo"
    GRADIENT_ECHO = "gradient_echo"


class ProxyMap(StrEnum):
    """A map of the phantom the editor shows without a simulation.

    It is an overlay that follows every edit of the phantom.
    """

    FILL = "fill"  # volume 0 all bundles, then one volume per bundle
    PARTITION = "partition"  # one volume per tissue, then the background
    EXPECTED_B0 = "expected_b0"  # the signal at b = 0, without noise


# The item name of each proxy map.
PROXY_NAMES: Final = {
    ProxyMap.FILL: "Bundle fill",
    ProxyMap.PARTITION: "Tissue partition",
    ProxyMap.EXPECTED_B0: "Expected b0",
}


class PhantomPart(StrEnum):
    """The part of the phantom that changed, as the PhantomChanged event says."""

    ALL = "all"  # a new phantom, a loaded one, or none
    GRID = "grid"
    TISSUES = "tissues"
    BUNDLES = "bundles"
    DRAFT = "draft"
    SCAN = "scan"
    PROXIES = "proxies"
    OUTPUT = "output"


class PhantomStep(StrEnum):
    """A step to a first simulation, in the order the guided editor shows them."""

    GRID = "grid"
    BUNDLE = "bundle"
    TISSUE = "tissue"
    SCAN = "scan"
    RUN = "run"


class ScanField(StrEnum):
    """A setting of the scan, as `Viewer.set_scan` names it."""

    SHELLS = "shells"
    NUM_B0 = "num_b0"
    GRADIENT_SEED = "gradient_seed"
    GRADIENT_FILES = "gradient_files"
    BVEC_FRAME = "bvec_frame"
    SMALL_DELTA_MS = "small_delta_ms"
    BIG_DELTA_MS = "big_delta_ms"
    TE_MS = "te_ms"
    TR_MS = "tr_ms"
    SEQUENCE = "sequence"
    SNR = "snr"
    NOISE_REFERENCE = "noise_reference"
    SEED = "seed"


class BundleField(StrEnum):
    """A setting of a bundle, as `Viewer.set_bundle` names it."""

    TISSUE = "tissue"
    NUM_STREAMLINES = "num_streamlines"
    RADIUS = "radius"
    NUM_POINTS = "num_points"
    FAN_START = "fan_start"
    FAN_END = "fan_end"
    FAN_SHARPNESS = "fan_sharpness"
    DISPERSION = "dispersion"
    SEED = "seed"
    STREAMLINE_AREA = "streamline_area"
    DENSEST_FILL = "densest_fill"


# The bundle settings that change the streamlines, so the bundle is drawn again.
GEOMETRY_FIELDS: Final = (
    BundleField.NUM_STREAMLINES,
    BundleField.RADIUS,
    BundleField.NUM_POINTS,
    BundleField.FAN_START,
    BundleField.FAN_END,
    BundleField.FAN_SHARPNESS,
    BundleField.DISPERSION,
    BundleField.SEED,
)


@dataclass(frozen=True)
class CompartmentSettings:
    """One compartment of a custom tissue (see `larmor.simulation.Compartment`)."""

    kind: CompartmentKind
    fraction: float
    diffusivity: float  # mm^2/s
    t1_ms: float
    t2_ms: float
    t2star_ms: float
    proton_density: float = 1.0
    perpendicular_diffusivity: float = 0.0  # mm^2/s, the zeppelin's
    radius_um: float = 0.0  # the sphere's and the cylinder's
    kappa: float = 0.0  # the Watson-dispersed sticks'

    def to_json(self) -> dict[str, object]:
        """Returns the compartment as JSON values."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> CompartmentSettings:
        """Reads a compartment from `to_json`'s values; raises KeyError, ValueError."""
        values = {f.name: data[f.name] for f in fields(cls) if f.name in data}
        values["kind"] = CompartmentKind(str(values.get("kind", data.get("model"))))
        numbers = {k: _number(v) for k, v in values.items() if k != "kind"}
        return cls(kind=values["kind"], **numbers)


@dataclass(frozen=True)
class TissueSettings:
    """A named tissue, as a preset with its parameters or as custom compartments."""

    name: str
    preset: TissuePreset
    values: tuple[tuple[str, float], ...] = ()  # preset parameters, others default
    compartments: tuple[CompartmentSettings, ...] = ()  # CUSTOM only

    def value(self, key: str) -> float:
        """Returns a preset parameter as set, else the preset's default."""
        given = dict(self.values)
        if key in given:
            return given[key]
        return preset_defaults(self.preset)[key]

    def to_json(self) -> dict[str, object]:
        """Returns the tissue as JSON values."""
        return {
            "name": self.name,
            "preset": self.preset.value,
            "values": dict(self.values),
            "compartments": [c.to_json() for c in self.compartments],
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> TissueSettings:
        """Reads a tissue from `to_json`'s values.

        Returns:
            TissueSettings: The tissue.

        Raises:
            KeyError: If a value is missing.
            TypeError: If a value is of the wrong type.
            ValueError: If a value is invalid.
        """
        values = data.get("values", {})
        if not isinstance(values, dict):
            raise TypeError("a tissue's values must be a JSON object")
        return cls(
            str(data["name"]),
            TissuePreset(data["preset"]),
            tuple((str(k), _number(v)) for k, v in values.items()),
            tuple(CompartmentSettings.from_json(c) for c in data["compartments"]),
        )


@dataclass(frozen=True)
class GeometrySettings:
    """A drawn bundle's shape (see `larmor.simulation.BundleGeometry`)."""

    control_points: tuple[Vec3, ...] = ()
    num_streamlines: int = 1000
    radius: float = 2.0  # mm
    num_points: int = 100
    fan_start: float = 0.0
    fan_end: float = 0.0
    fan_sharpness: float = 2.0
    dispersion: float = 0.0  # mm
    seed: int = 0

    def to_json(self) -> dict[str, object]:
        """Returns the geometry as the JSON values `BundleGeometry.to_dict` writes."""
        values = {f.name: getattr(self, f.name) for f in fields(self)}
        values["control_points"] = [list(p) for p in self.control_points]
        return values

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> GeometrySettings:
        """Reads a geometry from `to_json`'s values; raises KeyError or ValueError."""
        values: dict[str, Any] = {
            f.name: data[f.name] for f in fields(cls) if f.name in data
        }
        values["control_points"] = tuple(
            _vec3(p) for p in data.get("control_points", [])
        )
        seed = data.get("seed")
        values["seed"] = int(seed) if seed is not None else 0
        for name in ("num_streamlines", "num_points"):
            if name in values:
                values[name] = int(values[name])
        return cls(**values)


@dataclass(frozen=True)
class BundleSettings:
    """A bundle of the phantom, with its tractogram item, tissue and fill.

    A drawn bundle has a geometry; a loaded tractogram used as a bundle has none.
    """

    item_id: ItemId
    tissue: str = "WM"
    geometry: GeometrySettings | None = None
    streamline_area: float | None = None  # mm^2; None for the library's default
    densest_fill: float = 0.7  # the fill of the densest voxels, when area is None

    def to_json(self) -> dict[str, object]:
        """Returns the bundle as JSON values, its item named by its id."""
        return {
            "item": self.item_id,
            "tissue": self.tissue,
            "geometry": None if self.geometry is None else self.geometry.to_json(),
            "streamline_area": self.streamline_area,
            "densest_fill": self.densest_fill,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> BundleSettings:
        """Reads a bundle from `to_json`'s values.

        Returns:
            BundleSettings: The bundle.

        Raises:
            KeyError: If a value is missing.
            TypeError: If a value is of the wrong type.
            ValueError: If a value is invalid.
        """
        geometry = data.get("geometry")
        area = data.get("streamline_area")
        return cls(
            int(data["item"]),
            str(data["tissue"]),
            None if geometry is None else GeometrySettings.from_json(geometry),
            None if area is None else _number(area),
            _number(data.get("densest_fill", 0.7)),
        )


@dataclass(frozen=True)
class GradientRows:
    """A gradient table kept exactly as it was read, such as a phantom file's.

    Attributes:
        bvals: The b-value of each volume, s/mm^2.
        bvecs: The direction of each volume, in world coordinates.
        b0_threshold: The b-value at or below which a volume counts as b = 0.
        shell_epsilon: The gap between b-values that starts a new shell.
    """

    bvals: tuple[float, ...]
    bvecs: tuple[Vec3, ...]
    b0_threshold: float = 50.0
    shell_epsilon: float = 80.0

    @classmethod
    def of(cls, table: GradientTable) -> GradientRows:
        """Returns the rows of a gradient table."""
        return cls(
            tuple(float(b) for b in table.bvals),
            tuple(_vec3([float(x) for x in v]) for v in table.bvecs),
            float(table.b0_threshold),
            float(table.shell_epsilon),
        )

    def table(self) -> GradientTable:
        """Returns the rows as a gradient table."""
        return GradientTable(
            np.array(self.bvals, dtype=float),
            np.array(self.bvecs, dtype=float).reshape(-1, 3),
            b0_threshold=self.b0_threshold,
            shell_epsilon=self.shell_epsilon,
        )

    def to_json(self) -> dict[str, object]:
        """Returns the rows as JSON values."""
        return {
            "bvals": list(self.bvals),
            "bvecs": [list(v) for v in self.bvecs],
            "b0_threshold": self.b0_threshold,
            "shell_epsilon": self.shell_epsilon,
        }

    @classmethod
    def from_json(cls, data: object) -> GradientRows:
        """Reads rows from `to_json`'s values.

        Returns:
            The rows.

        Raises:
            KeyError: If a value is missing.
            TypeError: If a value is of the wrong type.
            ValueError: If a value is invalid.
        """
        if not isinstance(data, dict):
            raise TypeError("a gradient table must be a JSON object")
        rows = cls(
            tuple(_number(b) for b in data["bvals"]),
            tuple(_vec3(v) for v in data["bvecs"]),
            _number(data.get("b0_threshold", 50.0)),
            _number(data.get("shell_epsilon", 80.0)),
        )
        rows.table()  # refused here if the rows do not make a table
        return rows


@dataclass(frozen=True)
class ScanSettings:
    """The scan, with its gradients, its timings and its noise.

    The gradients come from the gradient files if there are any, else from the
    table read with a phantom file, else from the shells.
    """

    shells: tuple[tuple[float, int], ...] = ((1000.0, 32),)  # (b s/mm^2, directions)
    num_b0: int = 1
    gradient_seed: int = 0  # of the shells' directions
    gradient_files: tuple[Path, Path] | None = None  # bvals, bvecs
    bvec_frame: BvecFrame = BvecFrame.FSL  # of the bvecs file
    small_delta_ms: float = 15.0
    big_delta_ms: float = 30.0
    te_ms: tuple[float, ...] = (80.0,)
    tr_ms: float = 3000.0
    sequence: EchoKind = EchoKind.SPIN_ECHO
    snr: float | None = None  # None for no noise
    noise_reference: str = "WM"  # the tissue whose b = 0 signal the SNR is of
    seed: int = 0  # of the noise
    table: GradientRows | None = None  # a phantom file's gradients, as read

    def to_json(self) -> dict[str, object]:
        """Returns the scan as JSON values."""
        files = self.gradient_files
        return {
            "shells": [[b, n] for b, n in self.shells],
            "num_b0": self.num_b0,
            "gradient_seed": self.gradient_seed,
            "gradient_files": None if files is None else [str(p) for p in files],
            "bvec_frame": self.bvec_frame.value,
            "small_delta_ms": self.small_delta_ms,
            "big_delta_ms": self.big_delta_ms,
            "te_ms": list(self.te_ms),
            "tr_ms": self.tr_ms,
            "sequence": self.sequence.value,
            "snr": self.snr,
            "noise_reference": self.noise_reference,
            "seed": self.seed,
            "table": None if self.table is None else self.table.to_json(),
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> ScanSettings:
        """Reads a scan from `to_json`'s values.

        Gradient files saved without a frame are taken in the world frame.

        Returns:
            ScanSettings: The scan.

        Raises:
            ValueError: If a value is refused.
        """
        values: dict[str, Any] = {}
        for name in ScanField:
            if name in data:
                values[name.value] = scan_value(name, data[name])
        if data.get("gradient_files") is not None and "bvec_frame" not in data:
            values["bvec_frame"] = BvecFrame.WORLD
        if data.get("table") is not None:
            try:
                values["table"] = GradientRows.from_json(data["table"])
            except (KeyError, TypeError, ValueError) as e:
                raise ValueError(f"Scan table: {e}") from e
        return replace(cls(), **values)


@dataclass(frozen=True)
class Draft:
    """The bundle being drawn, or a finished one being edited.

    Attributes:
        points: The control points clicked so far, world mm.
        geometry: The other geometry settings.
        tissue: The tissue that will fill it.
        item_id: The preview tractogram item, once it has two points; for an edit,
            the bundle's own item.
        editing: The bundle being edited, as it was; None for a new bundle.
    """

    points: tuple[Vec3, ...] = ()
    geometry: GeometrySettings = GeometrySettings()
    tissue: str = "WM"
    item_id: ItemId | None = None
    editing: BundleSettings | None = None


@dataclass(frozen=True, eq=False)
class PhantomState:
    """Everything the phantom editor holds and a run needs, as plain values.

    Attributes:
        grid: The voxel grid.
        tissues: The named tissues.
        bundles: The finished bundles, in the order they were made.
        tissue_maps: The maps of tissue fractions, as (tissue name, overlay item).
        background: The tissue that fills what remains of each voxel; None for no
            signal.
        scan: The scan.
        draft: The bundle being drawn; None outside draw mode.
        proxies: (map, overlay item) pairs of the proxy maps shown.
        output: The folder a run writes to; None until chosen.
        show_box: Whether the grid's box is outlined in the views of the
            Simulate workspace; it is always outlined while a bundle is drawn.
        serial: A number that tells phantoms apart; a new or loaded phantom gets
            a new one, and an edit keeps it.
    """

    grid: Grid
    tissues: tuple[TissueSettings, ...]
    bundles: tuple[BundleSettings, ...] = ()
    tissue_maps: tuple[tuple[str, ItemId], ...] = ()
    background: str | None = None
    scan: ScanSettings = ScanSettings()
    draft: Draft | None = None
    proxies: tuple[tuple[ProxyMap, ItemId], ...] = ()
    output: Path | None = None
    show_box: bool = True
    serial: int = field(default_factory=lambda: next(_SERIALS))

    def tissue(self, name: str) -> TissueSettings:
        """Returns the tissue with this name; raises ValueError if there is none."""
        for tissue in self.tissues:
            if tissue.name == name:
                return tissue
        names = ", ".join(t.name for t in self.tissues)
        raise ValueError(f"The phantom has no tissue {name!r}; it has {names}.")

    def bundle(self, item_id: ItemId) -> BundleSettings:
        """Returns the bundle drawn as this item; raises ValueError if it is not one."""
        for bundle in self.bundles:
            if bundle.item_id == item_id:
                return bundle
        raise ValueError(f"Item {item_id} is not a bundle of the phantom.")

    def proxy(self, kind: ProxyMap) -> ItemId | None:
        """Returns the overlay item showing a proxy map; None if it is not shown."""
        return next((i for k, i in self.proxies if k == kind), None)

    def contains(self, position: Vec3) -> bool:
        """Tells whether a world position (mm) lies in the grid's box."""
        voxel = np.linalg.solve(self.grid.affine, [*position, 1.0])[:3]
        shape = np.array(self.grid.shape)
        return bool(np.all(voxel >= -0.5) and np.all(voxel <= shape - 0.5))

    def owned(self) -> set[ItemId]:
        """Returns the items the phantom makes: bundles, the draft's and proxy maps."""
        ids = {b.item_id for b in self.bundles} | {i for _, i in self.proxies}
        if self.draft is not None and self.draft.item_id is not None:
            ids.add(self.draft.item_id)
        return ids


def box_grid(size_mm: Sequence[float], voxel_mm: float) -> Grid:
    """Builds a grid of cubic voxels that fills a box of size_mm, centred on 0.

    Returns:
        The grid.

    Raises:
        ValueError: If a size or the voxel size is not positive and finite.
    """
    sizes = [float(s) for s in size_mm]
    if len(sizes) != 3 or not all(math.isfinite(s) and s > 0 for s in sizes):
        raise ValueError(f"A box is three positive sizes in mm, got {size_mm}.")
    if not (math.isfinite(voxel_mm) and voxel_mm > 0):
        raise ValueError(f"The voxel size must be positive, got {voxel_mm}.")
    shape = tuple(max(1, round(s / voxel_mm)) for s in sizes)
    affine = np.diag([voxel_mm, voxel_mm, voxel_mm, 1.0])
    affine[:3, 3] = [-(n - 1) * voxel_mm / 2 for n in shape]
    return Grid((shape[0], shape[1], shape[2]), affine)


def check_grid_size(grid: Grid) -> None:
    """Raises ValueError if the grid has more than MAX_GRID_VOXELS voxels."""
    count = math.prod(grid.shape[:3])
    if count > MAX_GRID_VOXELS:
        x, y, z = grid.shape[:3]
        raise ValueError(
            f"A grid of {x} x {y} x {z} voxels is too large for a phantom; it may "
            f"have at most {MAX_GRID_VOXELS:,} voxels. Use larger voxels or a "
            "smaller box."
        )


def default_tissues() -> tuple[TissueSettings, ...]:
    """Returns WM (standard model), GM (SANDI) and CSF (free water) at defaults."""
    return (
        TissueSettings("WM", TissuePreset.STANDARD_MODEL),
        TissueSettings("GM", TissuePreset.SANDI),
        TissueSettings("CSF", TissuePreset.FREE_WATER),
    )


def preset_defaults(preset: TissuePreset) -> dict[str, float]:
    """Returns a preset's parameters and their defaults; none for CUSTOM."""
    if preset == TissuePreset.CUSTOM:
        return {}
    signature = inspect.signature(_preset_function(preset))
    return {
        name: float(p.default)
        for name, p in signature.parameters.items()
        if name != "name" and isinstance(p.default, (int, float))
    }


def _preset_function(preset: TissuePreset) -> Callable[..., Tissue]:
    """Returns the simulation's function that makes a preset's tissue."""
    import larmor.simulation as sim

    functions: dict[TissuePreset, Callable[..., Tissue]] = {
        TissuePreset.STANDARD_MODEL: sim.standard_model_tissue,
        TissuePreset.SANDI: sim.sandi_tissue,
        TissuePreset.NODDI: sim.noddi_tissue,
        TissuePreset.FREE_WATER: sim.free_water_tissue,
    }
    return functions[preset]


def build_tissue(settings: TissueSettings) -> Tissue:
    """Builds the simulation's tissue for these settings.

    Returns:
        The tissue.

    Raises:
        ValueError: If a value is out of range or the fractions do not sum to 1.
    """
    import larmor.simulation as sim

    if settings.preset != TissuePreset.CUSTOM:
        function = _preset_function(settings.preset)
        return function(name=settings.name, **dict(settings.values))
    return sim.Tissue(
        settings.name,
        tuple(
            sim.Compartment(
                sim.CompartmentModel(c.kind.value),
                c.fraction,
                c.diffusivity,
                t1_ms=c.t1_ms,
                t2_ms=c.t2_ms,
                t2star_ms=c.t2star_ms,
                proton_density=c.proton_density,
                perpendicular_diffusivity=c.perpendicular_diffusivity,
                radius_um=c.radius_um,
                kappa=c.kappa,
            )
            for c in settings.compartments
        ),
    )


def tissue_settings(tissue: Tissue) -> TissueSettings:
    """Returns a simulation tissue as custom settings, one per compartment."""
    return TissueSettings(
        tissue.name,
        TissuePreset.CUSTOM,
        compartments=tuple(
            CompartmentSettings(
                CompartmentKind(c.model.value),
                c.fraction,
                c.diffusivity,
                c.t1_ms,
                c.t2_ms,
                c.t2star_ms,
                c.proton_density,
                c.perpendicular_diffusivity,
                c.radius_um,
                c.kappa,
            )
            for c in tissue.compartments
        ),
    )


def custom_settings(settings: TissueSettings) -> TissueSettings:
    """Returns a tissue as custom compartments, starting from its preset's."""
    if settings.preset == TissuePreset.CUSTOM:
        return settings
    return tissue_settings(build_tissue(settings))


def build_geometry(settings: GeometrySettings) -> BundleGeometry:
    """Builds the simulation's bundle geometry for these settings.

    Returns:
        The geometry.

    Raises:
        ValueError: If there are fewer than two control points, two in a row are
            the same, or a value is out of range.
    """
    import larmor.simulation as sim

    values = {f.name: getattr(settings, f.name) for f in fields(settings)}
    values["control_points"] = np.array(settings.control_points, dtype=np.float64)
    return sim.BundleGeometry(**values)


def build_streamlines(settings: GeometrySettings, index: int = 0) -> Tractogram:
    """Builds a drawn bundle's streamlines, as the simulation draws them."""
    import larmor.simulation as sim

    return sim.create_bundle(build_geometry(settings), index=index)


def drawable(points: Sequence[Vec3]) -> bool:
    """Tells whether control points make a bundle, two or more, none twice in a row."""
    if len(points) < 2:
        return False
    steps = np.diff(np.asarray(points, dtype=float), axis=0)
    return bool(np.all(np.linalg.norm(steps, axis=1) > 0))


def build_gradients(scan: ScanSettings, grid: Grid) -> GradientTable:
    """Builds the scan's gradient table, from its files, its table or its shells.

    Args:
        scan: The scan.
        grid: The phantom's grid, which bvecs files in the FSL frame refer to.

    Returns:
        The gradient table, directions in world coordinates.

    Raises:
        ValueError: If the files cannot be read or a shell is out of range.
    """
    if scan.gradient_files is not None:
        bvals, bvecs = scan.gradient_files
        return read_bvals_bvecs(bvals, bvecs, frame=scan.bvec_frame.value, grid=grid)
    if scan.table is not None:
        return scan.table.table()
    import larmor.simulation as sim

    return sim.create_gradient_table(
        list(scan.shells), num_b0=scan.num_b0, seed=scan.gradient_seed
    )


def build_protocol(scan: ScanSettings, grid: Grid) -> Protocol:
    """Builds the simulation's protocol for the scan on the phantom's grid.

    Returns:
        The protocol.

    Raises:
        ValueError: If the gradients cannot be made or the timings do not fit.
    """
    import larmor.simulation as sim

    return sim.Protocol(
        build_gradients(scan, grid),
        scan.small_delta_ms,
        scan.big_delta_ms,
        list(scan.te_ms),
        scan.tr_ms,
        sim.PulseSequence(scan.sequence.value),
    )


def scan_settings(protocol: Protocol) -> ScanSettings:
    """Returns a protocol as scan settings, with its gradient table kept as it is.

    The shells, which describe the table, are its b-values rounded to 10 s/mm^2
    with their counts.

    Returns:
        The scan settings.
    """
    bvals = np.asarray(protocol.gradients.bvals, dtype=float)
    b0 = bvals <= protocol.gradients.b0_threshold
    rounded = np.round(bvals[~b0], -1)
    values, counts = np.unique(rounded, return_counts=True)
    return ScanSettings(
        shells=tuple((float(b), int(n)) for b, n in zip(values, counts, strict=True)),
        num_b0=int(b0.sum()),
        small_delta_ms=protocol.small_delta_ms,
        big_delta_ms=protocol.big_delta_ms,
        te_ms=tuple(protocol.te_ms),
        tr_ms=protocol.tr_ms,
        sequence=EchoKind(protocol.sequence.value),
        table=GradientRows.of(protocol.gradients),
    )


def scan_value(name: ScanField, value: object) -> object:
    """Checks a scan setting and returns it as its type.

    Returns:
        The value, of the setting's type.

    Raises:
        ValueError: If the value is not of the setting's kind or out of range.
    """
    try:
        return _SCAN_CONVERTERS.get(name, _positive)(value)
    except (AssertionError, TypeError, ValueError) as e:
        detail = f": {e}" if str(e) else ""
        raise ValueError(f"Scan {name.value}: {value!r} is refused{detail}.") from e


def _shells(value: object) -> tuple[tuple[float, int], ...]:
    """Returns the shells as (b, number of directions) pairs."""
    assert isinstance(value, (list, tuple))
    shells = tuple((_number(b), int(_number(n))) for b, n in value)
    if not shells or any(b <= 0 or n < 1 for b, n in shells):
        raise ValueError("each shell needs b > 0 and at least one direction")
    return shells


def _count_value(value: object) -> int:
    """Returns value as a whole number of at least 0."""
    number = _number(value)
    if number != int(number) or number < 0:
        raise ValueError("it must be a whole number of at least 0")
    return int(number)


def _gradient_files(value: object) -> tuple[Path, Path] | None:
    """Returns the b-values and b-vectors files as absolute paths, or None."""
    if value is None:
        return None
    assert isinstance(value, (list, tuple)) and len(value) == 2
    return (Path(str(value[0])).absolute(), Path(str(value[1])).absolute())


def _echo_times(value: object) -> tuple[float, ...]:
    """Returns one echo time or several as a tuple of positive numbers."""
    times = (
        (_number(value),)
        if not isinstance(value, (list, tuple))
        else tuple(_number(t) for t in value)
    )
    if not times or any(t <= 0 for t in times):
        raise ValueError("echo times must be positive")
    return times


def _tissue_name(value: object) -> str:
    """Returns value as a tissue's name."""
    if not isinstance(value, str) or not value:
        raise ValueError("it must be a tissue's name")
    return value


def _snr(value: object) -> float | None:
    """Returns the SNR as a positive number, or None for no noise."""
    if value is None:
        return None
    snr = _number(value)
    if snr <= 0:
        raise ValueError("it must be positive, or None for no noise")
    return snr


def _positive(value: object) -> float:
    """Returns value as a positive number."""
    number = _number(value)
    if number <= 0:
        raise ValueError("it must be positive")
    return number


# How each scan setting is checked; a setting not listed is a positive number.
_SCAN_CONVERTERS: dict[ScanField, Callable[[object], object]] = {
    ScanField.SHELLS: _shells,
    ScanField.NUM_B0: _count_value,
    ScanField.GRADIENT_SEED: _count_value,
    ScanField.SEED: _count_value,
    ScanField.GRADIENT_FILES: _gradient_files,
    ScanField.BVEC_FRAME: lambda value: BvecFrame(str(value)),
    ScanField.TE_MS: _echo_times,
    ScanField.SEQUENCE: lambda value: EchoKind(str(value)),
    ScanField.SNR: _snr,
    ScanField.NOISE_REFERENCE: _tissue_name,
}
# The scan settings that make the gradients again, so a phantom file's table goes.
GRADIENT_FIELDS: Final = (
    ScanField.SHELLS,
    ScanField.NUM_B0,
    ScanField.GRADIENT_SEED,
    ScanField.GRADIENT_FILES,
)


def bundle_value(name: BundleField, value: object) -> object:
    """Checks a bundle setting and returns it as its type.

    Returns:
        The value, of the setting's type.

    Raises:
        ValueError: If the value is not of the setting's kind or out of range.
    """
    try:
        if name == BundleField.TISSUE:
            if not isinstance(value, str) or not value:
                raise ValueError("it must be a tissue's name")
            return value
        if name == BundleField.STREAMLINE_AREA and value is None:
            return None
        return _bundle_number(name, _number(value))
    except (TypeError, ValueError) as e:
        raise ValueError(f"Bundle {name.value}: {value!r} is refused: {e}.") from e


def _bundle_number(name: BundleField, number: float) -> float | int:
    """Returns a numeric bundle setting checked against its range."""
    if name in (
        BundleField.NUM_STREAMLINES,
        BundleField.NUM_POINTS,
        BundleField.SEED,
    ):
        low = {BundleField.NUM_POINTS: 2, BundleField.SEED: 0}.get(name, 1)
        if number != int(number) or number < low:
            raise ValueError(f"it must be a whole number of at least {low}")
        return int(number)
    if name == BundleField.DENSEST_FILL and not 0 < number <= 1:
        raise ValueError("it must be in (0, 1]")
    if name in (
        BundleField.STREAMLINE_AREA,
        BundleField.FAN_SHARPNESS,
        BundleField.RADIUS,
    ):
        if number <= 0:
            raise ValueError("it must be positive")
    elif number < 0:
        raise ValueError("it must be at least 0")
    return number


def with_bundle_value(
    bundle: BundleSettings, name: BundleField, value: object
) -> BundleSettings:
    """Returns the bundle with one setting changed, as `bundle_value` checks it."""
    value = bundle_value(name, value)
    if name in GEOMETRY_FIELDS:
        if bundle.geometry is None:
            raise ValueError(
                "A loaded tractogram used as a bundle has no geometry to change."
            )
        change: dict[str, Any] = {name.value: value}
        return replace(bundle, geometry=replace(bundle.geometry, **change))
    changes: dict[str, Any] = {name.value: value}
    return replace(bundle, **changes)


# Building the phantom


def build_phantom(
    state: PhantomState,
    bundles: Sequence[tuple[BundleSettings, Tractogram, Path | None]],
    maps: Sequence[tuple[str, Volume, Path | None]],
) -> Phantom:
    """Builds the simulation's phantom.

    Args:
        state: The editor's state, for the grid, tissues and background.
        bundles: Each bundle with the streamlines its item shows and its file;
            a drawn bundle is given as its geometry, which fills its tube.
        maps: Each tissue map's tissue, volume and file.

    Returns:
        The phantom.

    Raises:
        ValueError: If a tissue, a bundle or a map is refused by the simulation.
    """
    import larmor.simulation as sim

    tissues = tuple(build_tissue(t) for t in state.tissues)
    made: list[Bundle] = []
    for bundle, streamlines, path in bundles:
        source: BundleGeometry | Tractogram = streamlines
        if bundle.geometry is not None:
            source = build_geometry(bundle.geometry)
        made.append(
            sim.Bundle(
                source,
                tissue=bundle.tissue,
                path=path if bundle.geometry is None else None,
                streamline_area=bundle.streamline_area,
                densest_fill=bundle.densest_fill,
            )
        )
    return sim.Phantom(
        state.grid,
        tissues=tissues,
        bundles=tuple(made),
        tissue_maps=tuple(sim.TissueMap(t, v, p) for t, v, p in maps),
        background=state.background,
    )


def relaxation_weight(tissue: Tissue, scan: ScanSettings) -> float:
    """Returns the signal of a voxel full of the tissue at b = 0 and the first TE.

    It sums fraction PD (1 - exp(-TR / T1)) exp(-TE / T2) over the compartments,
    with T2* for a gradient echo.

    Returns:
        The weight.
    """
    te = scan.te_ms[0]
    total = 0.0
    for c in tissue.compartments:
        decay = c.t2star_ms if scan.sequence == EchoKind.GRADIENT_ECHO else c.t2_ms
        total += (
            c.fraction
            * c.proton_density
            * (1 - math.exp(-scan.tr_ms / c.t1_ms))
            * math.exp(-te / decay)
        )
    return total


def proxy_volumes(
    phantom: Phantom, scan: ScanSettings, kinds: Sequence[ProxyMap]
) -> dict[ProxyMap, Volume]:
    """Makes the proxy maps of a phantom from its voxel partition.

    Args:
        phantom: The phantom, its bundles given as streamlines.
        scan: The scan, whose timings weight the expected b0.
        kinds: The maps to make.

    Returns:
        Each map as a volume on the phantom's grid.
    """
    import larmor.simulation as sim

    partition: VoxelPartition = sim.partition_voxels(phantom, n_jobs=1)
    affine = phantom.grid.affine
    volumes: dict[ProxyMap, Volume] = {}
    for kind in kinds:
        if kind == ProxyMap.FILL:
            fractions = partition.bundle_fractions
            total = fractions.sum(axis=3, keepdims=True)
            data = np.concatenate([total, fractions], axis=3)
        elif kind == ProxyMap.PARTITION:
            data = np.concatenate(
                [partition.tissue_fractions, partition.background[..., None]], axis=3
            )
        else:
            data = expected_b0(phantom, partition, scan)
        volumes[kind] = Volume(np.ascontiguousarray(data, dtype=np.float32), affine)
    return volumes


def expected_b0(
    phantom: Phantom, partition: VoxelPartition, scan: ScanSettings
) -> np.ndarray:
    """Returns the noise-free signal at b = 0 and the first TE, background included.

    Returns:
        The signal, (X, Y, Z).
    """
    weights = np.array([relaxation_weight(t, scan) for t in phantom.tissues])
    signal: np.ndarray = partition.tissue_fractions @ weights
    if phantom.background is not None:
        signal = signal + partition.background * relaxation_weight(
            phantom.tissue(phantom.background), scan
        )
    return signal


# Checks before a run


@dataclass(frozen=True)
class ItemSource:
    """What a run needs of the item a loaded bundle or a tissue map names.

    Attributes:
        name: The item's name, for the messages.
        present: Whether the item is there and holds its kind of data.
        saved: Whether it has a file, which the run reads.
    """

    name: str
    present: bool
    saved: bool


@dataclass(frozen=True)
class Problem:
    """Something that stops the phantom from running.

    Attributes:
        step: The step it belongs to.
        short: A few words for the step list; empty when the step's own line
            says it, as for a phantom with nothing in it yet.
        text: The full sentence, as a refused run gives it.
    """

    step: PhantomStep
    short: str
    text: str


def run_problems(
    state: PhantomState, items: Mapping[ItemId, ItemSource]
) -> list[Problem]:
    """Checks what stops the phantom from running, fast enough for every edit.

    It checks everything a run checks before it starts, except making the
    gradient directions of the shells, which takes too long: the tissues, the
    items the bundles and maps name, the gradient files and the timings.

    Args:
        state: The phantom.
        items: What a run needs of each item a loaded bundle or a tissue map
            names, by id; a drawn bundle names none.

    Returns:
        The problems, in the order of the steps; empty if it can run.
    """
    found: list[Problem] = []
    if state.draft is not None:
        found.append(Problem(PhantomStep.BUNDLE, "", DRAFT_PROBLEM))
    if not state.bundles and not state.tissue_maps:
        found.append(
            Problem(
                PhantomStep.BUNDLE,
                "",
                "The phantom needs at least one bundle or tissue map.",
            )
        )
    found += _bundle_item_problems(state, items)
    found += _thin_bundle_problems(state)
    found += _tissue_problems(state, items)
    scan = _scan_problem(state.scan, state.grid)
    if scan is not None:
        found.append(Problem(PhantomStep.SCAN, *scan))
    reference = state.scan.noise_reference
    if state.scan.snr is not None and reference not in {t.name for t in state.tissues}:
        found.append(
            Problem(
                PhantomStep.SCAN,
                "The noise reference is gone",
                f"The SNR is that of tissue {reference!r}, which is gone; choose "
                "another noise reference.",
            )
        )
    return found


# The problem of a phantom while a bundle is drawn or edited.
DRAFT_PROBLEM: Final = "A bundle is being drawn; finish or cancel it first."


def _bundle_item_problems(
    state: PhantomState, items: Mapping[ItemId, ItemSource]
) -> list[Problem]:
    """Returns the problems of the tractograms the loaded bundles name."""
    found = []
    for bundle in state.bundles:
        if bundle.geometry is None:
            item = items.get(bundle.item_id)
            if item is None or not item.present:
                found.append(
                    Problem(
                        PhantomStep.BUNDLE,
                        "A bundle's tractogram is gone",
                        f"The tractogram of bundle item {bundle.item_id} is gone.",
                    )
                )
            elif not item.saved:
                found.append(
                    Problem(
                        PhantomStep.BUNDLE,
                        f"{item.name} has no file",
                        f"{item.name} has no file; save it (File > Save As) first.",
                    )
                )
    return found


def _thin_bundle_problems(state: PhantomState) -> list[Problem]:
    """Returns a problem for each drawn bundle too thin to fill on the grid."""
    from larmor.simulation.voxelise import MIN_TUBE_RADIUS

    largest = float(np.linalg.norm(state.grid.affine[:3, :3], axis=0).max())
    least = MIN_TUBE_RADIUS * largest
    return [
        Problem(
            PhantomStep.BUNDLE,
            "A bundle is too thin",
            f"A bundle of radius {b.geometry.radius:g} mm is too thin for voxels of "
            f"{largest:g} mm; give it a radius of at least {least:g} mm.",
        )
        for b in state.bundles
        if b.geometry is not None
        and b.streamline_area is None
        and b.geometry.radius < least
    ]


def _tissue_problems(
    state: PhantomState, items: Mapping[ItemId, ItemSource]
) -> list[Problem]:
    """Returns the problems of the tissues, the tissue maps and the background."""
    found = []
    names = {t.name for t in state.tissues}
    for bundle in state.bundles:
        if bundle.tissue not in names:
            found.append(
                Problem(
                    PhantomStep.TISSUE,
                    f"{bundle.tissue} is gone",
                    f"A bundle uses tissue {bundle.tissue!r}, which is gone.",
                )
            )
    found += _tissue_map_problems(state, items, names)
    if state.background is not None and state.background not in names:
        found.append(
            Problem(
                PhantomStep.TISSUE,
                f"{state.background} is gone",
                f"The background is tissue {state.background!r}, which is gone.",
            )
        )
    for settings in state.tissues:
        try:
            build_tissue(settings)
        except (KeyError, TypeError, ValueError) as e:
            found.append(
                Problem(
                    PhantomStep.TISSUE,
                    f"{settings.name} is not valid",
                    f"Tissue {settings.name}: {e}",
                )
            )
    return found


def _tissue_map_problems(
    state: PhantomState, items: Mapping[ItemId, ItemSource], names: set[str]
) -> list[Problem]:
    """Returns the problems of the tissue maps; names are the tissues that exist."""
    found = []
    for tissue, item_id in state.tissue_maps:
        item = items.get(item_id)
        if tissue not in names:
            found.append(
                Problem(
                    PhantomStep.TISSUE,
                    f"{tissue} is gone",
                    f"A tissue map is of tissue {tissue!r}, which is gone.",
                )
            )
        if item is None or not item.present:
            found.append(
                Problem(
                    PhantomStep.TISSUE,
                    f"The map of {tissue} is gone",
                    f"The map of {tissue} is gone.",
                )
            )
        elif not item.saved:
            found.append(
                Problem(
                    PhantomStep.TISSUE,
                    f"The map of {tissue} has no file",
                    f"The map of {tissue} has no file; save it first.",
                )
            )
    return found


def _scan_problem(scan: ScanSettings, grid: Grid) -> tuple[str, str] | None:
    """Checks the scan's files and timings; returns (short, text), None if valid."""
    import larmor.simulation as sim

    files = scan.gradient_files
    if files is not None:
        if not all(path.is_file() for path in files):
            missing = next(path for path in files if not path.is_file())
            return "A gradient file is missing", f"The scan's {missing} is missing."
        try:
            read_bvals_bvecs(files[0], files[1], frame=scan.bvec_frame.value, grid=grid)
        except (OSError, ValueError) as e:
            return "The gradient files are not valid", f"The scan is not valid: {e}"
    # A table of one b = 0 volume checks the timings without making directions.
    table = GradientTable(np.zeros(1), np.zeros((1, 3)))
    try:
        sim.Protocol(
            table,
            scan.small_delta_ms,
            scan.big_delta_ms,
            list(scan.te_ms),
            scan.tr_ms,
            sim.PulseSequence(scan.sequence.value),
        )
    except ValueError as e:
        return "The timings do not fit", f"The scan is not valid: {e}"
    return None


def problems(state: PhantomState, items: Mapping[ItemId, ItemSource]) -> list[str]:
    """Returns what stops the phantom from running, one sentence each."""
    return [problem.text for problem in run_problems(state, items)]


# Where the steps to a first simulation stand


@dataclass(frozen=True)
class StepStatus:
    """Where a step to a first simulation stands.

    Attributes:
        step: The step.
        ready: Whether the step needs nothing more for a run.
        text: A short line on the step, such as "b = 1000, 32 directions".
    """

    step: PhantomStep
    ready: bool
    text: str


# The presets as the step list names them, after the tissue's name.
PRESET_SHORT: Final = {
    TissuePreset.STANDARD_MODEL: "standard model",
    TissuePreset.SANDI: "SANDI",
    TissuePreset.NODDI: "NODDI",
    TissuePreset.FREE_WATER: "free water",
    TissuePreset.CUSTOM: "custom",
}
# The Run step's line while an earlier step has a problem.
NEEDS: Final = {
    PhantomStep.BUNDLE: "Needs a bundle",
    PhantomStep.TISSUE: "Needs a valid tissue",
    PhantomStep.SCAN: "Needs a valid scan",
}


def step_status(
    state: PhantomState, items: Mapping[ItemId, ItemSource]
) -> tuple[StepStatus, ...]:
    """Returns where each step to a first simulation stands, in the steps' order.

    A step is ready when `run_problems` finds nothing in it, so Run is ready
    exactly when `problems` is empty.

    Args:
        state: The phantom.
        items: What a run needs of the items the phantom names, as for
            `run_problems`.

    Returns:
        One status per `PhantomStep`, in its order.
    """
    found = run_problems(state, items)

    def status(step: PhantomStep, text: str) -> StepStatus:
        """Returns a step's status, its problem's short text replacing text."""
        own = [p for p in found if p.step == step]
        if own and own[0].short:
            text = own[0].short
        return StepStatus(step, not own, text)

    waiting = found[0].step if found else None
    run_text = "Ready" if waiting is None else NEEDS[waiting]
    if found and found[0].text == DRAFT_PROBLEM:
        run_text = "Finish the bundle first"
    return (
        status(PhantomStep.GRID, _grid_text(state.grid)),
        status(PhantomStep.BUNDLE, _bundle_text(state)),
        status(PhantomStep.TISSUE, _tissue_text(state)),
        status(PhantomStep.SCAN, _scan_text(state.scan)),
        StepStatus(PhantomStep.RUN, waiting is None, run_text),
    )


def _count(number: int, noun: str) -> str:
    """Returns a count with its noun, plural unless it is 1."""
    return f"{number} {noun}{'' if number == 1 else 's'}"


def _grid_text(grid: Grid) -> str:
    """Returns the grid as "30 x 30 x 30 voxels of 2 mm"."""
    sizes = [round(float(s), 4) for s in np.linalg.norm(grid.affine[:3, :3], axis=0)]
    voxel = (
        f"{sizes[0]:g}" if len(set(sizes)) == 1 else " x ".join(f"{s:g}" for s in sizes)
    )
    x, y, z = grid.shape[:3]
    return f"{x} x {y} x {z} voxels of {voxel} mm"


def _bundle_text(state: PhantomState) -> str:
    """Returns the Bundle step's line, such as "2 bundles"."""
    draft = state.draft
    if draft is not None:
        verb = "Editing" if draft.editing is not None else "Drawing"
        return f"{verb}: {_count(len(draft.points), 'point')}"
    if state.bundles:
        return _count(len(state.bundles), "bundle")
    if state.tissue_maps:
        return "No bundle; tissue maps fill it"
    return "No bundle yet"


def _tissue_text(state: PhantomState) -> str:
    """Returns the tissues in use, or how many there are to choose from."""
    names = {t.name: t for t in state.tissues}
    used = [b.tissue for b in state.bundles] + [t for t, _ in state.tissue_maps]
    if state.draft is not None:
        used.insert(0, state.draft.tissue)
    if state.background is not None:
        used.append(state.background)
    used = [name for name in dict.fromkeys(used) if name in names]
    if len(used) == 1:
        return f"{used[0]} ({PRESET_SHORT[names[used[0]].preset]})"
    if used:
        return ", ".join(used)
    return f"{_count(len(names), 'tissue')} to choose from"


def _scan_text(scan: ScanSettings) -> str:
    """Returns the Scan step's line, such as "b = 1000, 32 directions"."""
    files = scan.gradient_files
    if files is not None:
        return f"From {files[0].name}"
    if scan.table is not None:
        return f"From the phantom file: {_count(len(scan.table.bvals), 'volume')}"
    values = ", ".join(f"{b:g}" for b, _ in scan.shells)
    directions = _count(sum(n for _, n in scan.shells), "direction")
    joint = ", " if len(scan.shells) == 1 else "; "
    return f"b = {values}{joint}{directions}"


# Runs


@dataclass(frozen=True)
class SimulationOutputs:
    """The files of a simulation folder the viewer opens.

    Attributes:
        folder: The folder.
        dwi: The DWI, one file per echo, opened as volumes.
        masks: One mask per tissue, and the mask of all bundles, opened as
            overlays.
        peaks: The ground-truth fibre directions, opened as peaks.
        bundles: Each bundle's streamlines, in the phantom's bundle order.
        phantom: The phantom file, opened in the phantom editor.
    """

    folder: Path
    dwi: tuple[Path, ...]
    masks: tuple[Path, ...]
    peaks: Path | None
    bundles: tuple[Path, ...]
    phantom: Path


def simulation_outputs(folder: Path) -> SimulationOutputs:
    """Reads the files a simulation folder holds from its simulation.json.

    Args:
        folder: A folder written by `larmor.simulation.save_simulation`.

    Returns:
        The files.

    Raises:
        ValueError: If the folder has no simulation.json or it is not one.
        OSError: If simulation.json cannot be read.
    """
    folder = Path(folder).absolute()
    info_path = folder / SIMULATION_INFO
    if not info_path.is_file():
        raise ValueError(f"{folder} holds no {SIMULATION_INFO}: not a simulation.")
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
        if info.get("format") != SIMULATION_FORMAT:
            raise ValueError(f"{info_path} is not a Larmor simulation file.")
        files = info["files"]
        peaks = files.get("peaks")
        masks = list(files.get("masks", {}).values())
        if files.get("bundle_mask") is not None:
            masks.append(files["bundle_mask"])
        return SimulationOutputs(
            folder,
            tuple(folder / p for p in files["dwi"]),
            tuple(folder / p for p in masks),
            None if peaks is None else folder / peaks,
            tuple(folder / p for p in files.get("bundles", [])),
            folder / files["phantom"],
        )
    except (AttributeError, KeyError, TypeError, json.JSONDecodeError) as e:
        raise ValueError(f"{info_path} cannot be read: {e!r}.") from e


@dataclass(frozen=True, eq=False)
class RunInputs:
    """What a run simulates, taken from the editor before it starts."""

    phantom: Phantom
    protocol: Protocol
    scan: ScanSettings
    state: PhantomState  # the editor's state when the run started
    viewer_block: Mapping[str, object]  # the viewer's part of the phantom file


@dataclass(frozen=True, eq=False)
class RunOutputs:
    """What a run wrote, and the editor's state it was made from."""

    files: SimulationOutputs
    state: PhantomState


def holds_other_files(folder: Path) -> bool:
    """Tells whether a folder holds files but no earlier simulation."""
    if not folder.is_dir() or not any(folder.iterdir()):
        return False
    try:
        info = json.loads((folder / SIMULATION_INFO).read_text(encoding="utf-8"))
    except OSError, ValueError:
        return True
    return not (isinstance(info, dict) and info.get("format") == SIMULATION_FORMAT)


def existing_outputs(inputs: RunInputs, folder: Path) -> list[Path]:
    """Returns the files a run would write in folder that exist already."""
    import larmor.simulation as sim

    planned = sim.simulation_files(
        inputs.phantom, inputs.protocol, folder, noise=inputs.scan.snr is not None
    )
    return [p for p in planned.all_paths() if p.exists()]


def run_simulation(
    inputs: RunInputs, folder: Path, report: Report, *, force: bool = False
) -> RunOutputs:
    """Simulates the phantom and saves the simulation folder; runs on any thread.

    The phantom file also gets the viewer's block, so opening it again restores
    the editor's settings.

    Args:
        inputs: What the run simulates.
        folder: The output folder.
        report: Told the progress.
        force: Write into a folder that holds other files, keeping them.

    Returns:
        The files written.
    """
    import larmor.simulation as sim

    scan = inputs.scan
    simulated = sim.simulate(
        inputs.phantom,
        inputs.protocol,
        snr=scan.snr,
        noise_reference=scan.noise_reference,
        seed=scan.seed,
        progress=report,
    )
    # The last chance to cancel, as saving runs to the end once it starts.
    report(1, 1, SAVING)
    written = sim.save_simulation(simulated, folder, force=force)
    data = json.loads(written.phantom.read_text(encoding="utf-8"))
    data["viewer"] = dict(inputs.viewer_block)
    written.phantom.write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return RunOutputs(simulation_outputs(written.folder), inputs.state)


# Saving and loading the editor's state


@dataclass(frozen=True, eq=False)
class LoadedBundle:
    """A bundle read from a phantom file, built and ready to be added."""

    settings: BundleSettings  # its item id is a stand-in until it is added
    streamlines: Tractogram
    name: str
    path: Path | None  # the file of a tractogram used as a bundle
    options: Mapping[str, object]  # the item's options, as saved


@dataclass(frozen=True, eq=False)
class LoadedPhantom:
    """A phantom file read and its bundles built, ready to be shown."""

    state: PhantomState  # without the bundles, maps and proxies, which are added
    bundles: tuple[LoadedBundle, ...]
    maps: tuple[tuple[str, Volume, Path], ...]
    proxies: tuple[ProxyMap, ...]


def viewer_block(
    state: PhantomState,
    names: Mapping[ItemId, str],
    options: Mapping[ItemId, Mapping[str, object]],
) -> dict[str, object]:
    """Returns the viewer's block of a phantom file.

    It holds the tissue presets, the scan, each bundle's name and options, the
    proxy maps shown, the output folder and whether the grid's box shows.
    """
    return {
        "tissues": [t.to_json() for t in state.tissues],
        "scan": state.scan.to_json(),
        "bundles": [
            {
                "name": names.get(b.item_id, ""),
                "options": dict(options.get(b.item_id, {})),
                "densest_fill": b.densest_fill,
            }
            for b in state.bundles
        ],
        "proxies": [k.value for k, _ in state.proxies],
        "output": None if state.output is None else str(state.output),
        "show_box": state.show_box,
    }


def read_phantom_file(path: Path, report: Report | None = None) -> LoadedPhantom:
    """Reads a phantom file and builds each bundle's streamlines once.

    Runs on any thread. The tissue presets, scan settings, bundle names and
    options and the proxy maps come from the file's viewer block when it has
    one; otherwise tissues are custom and the scan comes from the protocol. A
    geometry without a seed gets the one `lar simulate` draws for it from the
    scan's noise seed, so a run gives the same images as `lar simulate` with
    that seed.

    Returns:
        The phantom, its bundles built.

    Raises:
        ValueError: If the file is not a phantom file or holds invalid values.
        FileNotFoundError: If a tractogram or a tissue map cannot be found.
    """
    import larmor.simulation as sim
    from larmor.simulation.phantom import resolve_bundle_seeds

    read = sim.read_phantom(path)
    phantom = read.phantom
    block = read.viewer
    saved_tissues = {
        t.name: t
        for t in (
            TissueSettings.from_json(raw)
            for raw in _list(block.get("tissues"))
            if isinstance(raw, dict)
        )
    }
    tissues = []
    for tissue in phantom.tissues:
        saved = saved_tissues.get(tissue.name)
        tissues.append(
            saved
            if saved is not None and _builds(saved, tissue)
            else tissue_settings(tissue)
        )
    scan = ScanSettings()
    if isinstance(block.get("scan"), dict):
        scan = ScanSettings.from_json(block["scan"])
    elif read.protocol is not None:
        scan = scan_settings(read.protocol)
    phantom = replace(phantom, bundles=resolve_bundle_seeds(phantom, scan.seed))
    saved_bundles = [b for b in _list(block.get("bundles")) if isinstance(b, dict)]
    bundles = []
    count = len(phantom.bundles)
    for index, bundle in enumerate(phantom.bundles):
        extra = saved_bundles[index] if index < len(saved_bundles) else {}
        source = bundle.source
        geometry = None
        if isinstance(source, sim.BundleGeometry):
            geometry = GeometrySettings.from_json(source.to_dict())
            streamlines = build_streamlines(geometry, index)
        else:
            streamlines = source
        settings = BundleSettings(
            0,
            bundle.tissue,
            geometry,
            bundle.streamline_area,
            float(extra.get("densest_fill", bundle.densest_fill)),
        )
        name = str(extra.get("name") or f"Bundle {index + 1}")
        options = extra.get("options", {})
        bundles.append(
            LoadedBundle(
                settings,
                streamlines,
                name,
                bundle.path,
                options if isinstance(options, dict) else {},
            )
        )
        if report is not None:
            report(index + 1, count, "Bundles")
    maps = tuple(
        (m.tissue, m.volume, m.path) for m in phantom.tissue_maps if m.path is not None
    )
    proxies = tuple(
        ProxyMap(p) for p in _list(block.get("proxies")) if p in _PROXY_VALUES
    )
    output = block.get("output")
    state = PhantomState(
        phantom.grid,
        tuple(tissues),
        background=phantom.background,
        scan=scan,
        output=Path(output) if isinstance(output, str) else None,
        show_box=bool(block.get("show_box", True)),
    )
    return LoadedPhantom(state, tuple(bundles), maps, proxies)


def _builds(settings: TissueSettings, tissue: Tissue) -> bool:
    """Tells whether the settings make exactly this tissue."""
    try:
        return build_tissue(settings) == tissue
    except KeyError, TypeError, ValueError:
        return False


def state_json(state: PhantomState) -> dict[str, object]:
    """Returns the state as JSON values for a session, items named by their ids.

    The bundle being drawn is left out.

    Returns:
        The JSON values.
    """
    return {
        "grid": {"shape": list(state.grid.shape), "affine": state.grid.affine.tolist()},
        "tissues": [t.to_json() for t in state.tissues],
        "bundles": [b.to_json() for b in state.bundles],
        "tissue_maps": [[t, i] for t, i in state.tissue_maps],
        "background": state.background,
        "scan": state.scan.to_json(),
        "proxies": [[k.value, i] for k, i in state.proxies],
        "output": None if state.output is None else str(state.output),
        "show_box": state.show_box,
    }


def state_from_json(data: Mapping[str, Any]) -> PhantomState:
    """Reads a state from `state_json`'s values, its item ids as saved.

    Returns:
        The state.

    Raises:
        KeyError, TypeError, ValueError: If a value is missing or malformed.
    """
    grid = Grid(
        tuple(data["grid"]["shape"]), np.asarray(data["grid"]["affine"], dtype=float)
    )
    output = data.get("output")
    return PhantomState(
        grid,
        tuple(TissueSettings.from_json(t) for t in data["tissues"]),
        tuple(BundleSettings.from_json(b) for b in data.get("bundles", [])),
        tuple((str(t), int(i)) for t, i in data.get("tissue_maps", [])),
        data.get("background"),
        ScanSettings.from_json(data.get("scan", {})),
        None,
        tuple((ProxyMap(k), int(i)) for k, i in data.get("proxies", [])),
        Path(output) if isinstance(output, str) else None,
        bool(data.get("show_box", True)),
    )


def options_json(options: object) -> dict[str, object]:
    """Returns an item's options as JSON values, as a session stores them."""
    raw: dict[str, object] = json.loads(json.dumps(asdict(options)))  # type: ignore[call-overload]
    return raw


# Each proxy map by its value.
_PROXY_VALUES: Final = {k.value: k for k in ProxyMap}


def _list(value: object) -> list[object]:
    """Returns value if it is a list, else an empty list."""
    return value if isinstance(value, list) else []


def _number(value: object) -> float:
    """Returns value as a finite float, else raises TypeError or ValueError."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{value!r} is not a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{value!r} is not finite")
    return number


def _vec3(value: object) -> Vec3:
    """Returns a list or tuple of three finite numbers as a Vec3."""
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise TypeError(f"{value!r} is not three numbers")
    x, y, z = (_number(v) for v in value)
    return (x, y, z)
