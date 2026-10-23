"""Phantoms, which place named tissues on a voxel grid by bundles and tissue maps."""

import math
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from larmor.simulation.bundles import BundleGeometry, create_bundle
from larmor.simulation.tissues import Tissue, default_tissues
from larmor.simulation.voxelise import Voxelisation, tube_fill, voxelise
from larmor.types import Grid, Tractogram, Volume

# How far a map value may lie outside [0, 1] and still count as rounding.
_MAP_TOLERANCE = 1e-6


# The percentile of a tractogram bundle's voxel lengths that `densest_fill` sets.
DENSE_PERCENTILE = 95.0

#: Called with (done, total, stage) as a simulation runs; raise in it to cancel.
Progress = Callable[[int, int, str], None]


@dataclass(frozen=True, eq=False)
class Bundle:
    """A fibre bundle of a phantom, and the tissue that fills it.

    A bundle places its tissue in the voxels it passes through, and its
    streamlines give the fibre directions there. The share of a voxel that a
    bundle fills is its occupancy, between 0 and 1. A bundle drawn from a
    geometry fills each voxel by the share of the voxel inside its tube. A
    tractogram, or a geometry with a `streamline_area`, fills each voxel with
    L a / V, capped at 1, where L is the length of its streamlines in the voxel
    (mm), a the cross-section each streamline stands for (mm^2) and V the voxel
    volume (mm^3).

    Args:
        source: A geometry to draw the streamlines from, or a tractogram (for
            example from `load_tractogram`), in world coordinates (mm).
        tissue: The name of the tissue that fills the bundle. Defaults to "WM".
        path: The file a tractogram source was read from; `write_phantom` stores
            it. Defaults to None.
        streamline_area: The cross-section a that each streamline stands for,
            in mm^2. Defaults to None, with which a geometry fills its tube and
            a tractogram gets the value that gives the bundle's dense voxels an
            occupancy of `densest_fill`. With a value, a geometry's occupancy
            dips at random where fewer streamlines happen to pass.
        densest_fill: The occupancy of a tractogram bundle's dense voxels when
            streamline_area is None, in (0, 1]. A voxel is dense when the
            bundle's streamline length in it reaches the 95th percentile of that
            length over the voxels the bundle reaches; denser voxels are capped
            at 1. Not used for a geometry. Defaults to 0.7.

    Raises:
        ValueError: If streamline_area is not positive or densest_fill is not in
            (0, 1].
    """

    source: BundleGeometry | Tractogram
    tissue: str = "WM"
    path: Path | None = None
    streamline_area: float | None = None
    densest_fill: float = 0.7

    def __post_init__(self) -> None:
        """Checks the streamline area and the densest fill."""
        if not 0 < self.densest_fill <= 1:
            raise ValueError(
                f"densest_fill must be in (0, 1], got {self.densest_fill}."
            )
        if self.streamline_area is not None and not (
            math.isfinite(self.streamline_area) and self.streamline_area > 0
        ):
            raise ValueError(
                f"streamline_area must be positive, got {self.streamline_area}."
            )

    def streamlines(self, index: int = 0) -> Tractogram:
        """Returns the streamlines, drawn from the geometry or the tractogram.

        Args:
            index: The bundle's index, stored in streamline_data["bundle"] of
                the streamlines drawn from a geometry. Defaults to 0.

        Returns:
            The streamlines in world coordinates (mm).
        """
        if isinstance(self.source, BundleGeometry):
            return create_bundle(self.source, index=index)
        return self.source


@dataclass(frozen=True, eq=False)
class TissueMap:
    """The fraction of one tissue in each voxel, such as a grey matter probability.

    Args:
        tissue: The tissue's name. Its compartments must not need a fibre
            direction, as a map gives none.
        volume: 3D values in [0, 1] on the phantom's grid.
        path: The file the map was read from; `write_phantom` stores it.
            Defaults to None.
    """

    tissue: str
    volume: Volume
    path: Path | None = None


@dataclass(frozen=True, eq=False)
class Phantom:
    """An object to simulate, made of tissues placed on a voxel grid.

    Each voxel is split between the tissues in three steps, which never scale a
    fraction up. Each bundle first takes its occupancy; where the bundles add up
    to more than 1, they are scaled down to sum to 1. The tissue maps then share
    what the bundles leave, scaled down where they add up to more. The rest goes
    to the background tissue, or gives no signal when `background` is None, so a
    voxel with a grey matter map value of 0.05 and nothing else is 5% grey
    matter and 95% background.

    Args:
        grid: The voxel grid.
        tissues: The named tissues. Defaults to WM, GM and CSF
            (`default_tissues`).
        bundles: The fibre bundles. Defaults to none.
        tissue_maps: At most one map per tissue. Defaults to none.
        background: The name of the tissue that fills what remains, or None for
            no signal. Its compartments must not need a fibre direction. Defaults
            to None.

    Raises:
        ValueError: If tissue names repeat, a bundle, map or the background names
            an unknown tissue, a map or the background uses a tissue that needs a
            fibre direction, two maps share a tissue, or a map is not a 3D volume
            on the grid with values in [0, 1].
    """

    grid: Grid
    tissues: tuple[Tissue, ...] = field(default_factory=default_tissues)
    bundles: tuple[Bundle, ...] = ()
    tissue_maps: tuple[TissueMap, ...] = ()
    background: str | None = None

    def __post_init__(self) -> None:
        """Checks the names, the maps and the background."""
        for name in ("tissues", "bundles", "tissue_maps"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        names = [t.name for t in self.tissues]
        if len(set(names)) != len(names):
            raise ValueError(f"Tissue names must be unique, got {names}.")
        for bundle in self.bundles:
            self.tissue(bundle.tissue)
        mapped = [m.tissue for m in self.tissue_maps]
        if len(set(mapped)) != len(mapped):
            raise ValueError(f"At most one map per tissue, got {mapped}.")
        for tissue_map in self.tissue_maps:
            self._check_unoriented(tissue_map.tissue, "a tissue map")
            self._check_map(tissue_map)
        if self.background is not None:
            self._check_unoriented(self.background, "the background")

    def tissue(self, name: str) -> Tissue:
        """Returns the tissue with this name.

        Args:
            name: The tissue's name.

        Returns:
            The tissue.

        Raises:
            ValueError: If the phantom has no tissue of that name.
        """
        for tissue in self.tissues:
            if tissue.name == name:
                return tissue
        raise ValueError(
            f"The phantom has no tissue {name!r}; it has "
            f"{[t.name for t in self.tissues]}."
        )

    def _check_unoriented(self, name: str, role: str) -> None:
        """Raises ValueError if the tissue needs a fibre direction."""
        if self.tissue(name).is_oriented:
            raise ValueError(
                f"Tissue {name!r} needs a fibre direction, so it cannot be {role}; "
                "use it in a bundle."
            )

    def _check_map(self, tissue_map: TissueMap) -> None:
        """Raises ValueError for a map off the grid or with values outside [0, 1]."""
        volume = tissue_map.volume
        if volume.data.ndim != 3 or not volume.grid.matches(self.grid):
            raise ValueError(
                f"The map of {tissue_map.tissue!r} must be a 3D volume on the "
                f"phantom's grid {self.grid.shape}."
            )
        values = volume.as_float(dtype=np.float64)
        if values.min() < -_MAP_TOLERANCE or values.max() > 1 + _MAP_TOLERANCE:
            raise ValueError(
                f"The map of {tissue_map.tissue!r} must hold fractions in [0, 1]."
            )


@dataclass(frozen=True, eq=False)
class VoxelPartition:
    """How each voxel of a phantom is split between its tissues and fibres.

    This is the ground truth of a simulation. A fibre is one direction of one
    bundle in one voxel; a bundle that bends or fans out in a voxel has several
    fibres there. The fibres of all voxels are stored in flat arrays of length
    F, sorted by voxel.

    Attributes:
        grid: The phantom's grid, with V voxels.
        tissue_names: The phantom's tissue names, in the order of the columns.
        tissue_fractions: The fraction of each tissue in each voxel, from bundles
            and maps, without the background, (X, Y, Z, T).
        background: The rest, 1 - the sum of tissue_fractions, (X, Y, Z).
        bundle_fractions: Each bundle's occupancy, (X, Y, Z, B).
        offsets: Where each voxel's fibres start, (V + 1,) int64. The fibres of
            voxel v, with voxels numbered in C order, are those from offsets[v]
            to offsets[v + 1] - 1.
        fibre_tissue: The tissue index of each fibre, (F,) int64.
        fibre_bundle: The index of each fibre's bundle, (F,) int64.
        fibre_weight: The fraction of its voxel each fibre fills, (F,).
        fibre_direction: Each fibre's unit direction in world coordinates,
            (F, 3).
        streamlines: Each bundle's streamlines, as drawn or given.
    """

    grid: Grid
    tissue_names: tuple[str, ...]
    tissue_fractions: np.ndarray
    background: np.ndarray
    bundle_fractions: np.ndarray
    offsets: np.ndarray
    fibre_tissue: np.ndarray
    fibre_bundle: np.ndarray
    fibre_weight: np.ndarray
    fibre_direction: np.ndarray
    streamlines: tuple[Tractogram, ...]


def partition_voxels(
    phantom: Phantom,
    *,
    seed: int | None = None,
    n_jobs: int = -1,
    progress: Progress | None = None,
) -> VoxelPartition:
    """Splits each voxel of a phantom between its tissues, fibres and background.

    The split follows the rules of `Phantom`. A bundle's streamline directions
    in a voxel are grouped by the nearest of 300 directions, and each group
    gives one fibre, weighted by its share of the bundle's length in the voxel times the
    bundle's occupancy.

    Args:
        phantom: The phantom.
        seed: The seed for the bundle geometries that have none of their own;
            each of them gets a seed drawn from this one, bundle by bundle.
            Defaults to None.
        n_jobs: Number of threads for resampling the streamlines, -1 for all cores.
            Defaults to -1.
        progress: Called with (done, total, "voxelise") after each bundle.
            Defaults to None.

    Returns:
        The partition.
    """
    grid = phantom.grid
    num_voxels = int(np.prod(grid.shape))
    names = tuple(t.name for t in phantom.tissues)
    column = {name: i for i, name in enumerate(names)}
    voxel_volume = abs(float(np.linalg.det(grid.affine[:3, :3])))

    streamlines = []
    # The fill and direction of each geometry bundle filling its tube, by index.
    tubes: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for index, bundle in enumerate(resolve_bundle_seeds(phantom, seed)):
        streamlines.append(bundle.streamlines(index))
        source = bundle.source
        if isinstance(source, BundleGeometry) and bundle.streamline_area is None:
            fan = (source.fan_start, source.fan_end, source.fan_sharpness)
            tubes[index] = tube_fill(source.control_points, source.radius, fan, grid)
        if progress is not None:
            progress(index + 1, len(phantom.bundles) + 1, "voxelise")
    voxels = _with_tube_fibres(voxelise(streamlines, grid, n_jobs=n_jobs), tubes)
    lengths = voxels.bundle_lengths()  # (V, B)

    occupancy = _occupancy(phantom, streamlines, tubes, lengths, voxel_volume)

    fractions = np.zeros((num_voxels, len(names)))
    for k, bundle in enumerate(phantom.bundles):
        fractions[:, column[bundle.tissue]] += occupancy[:, k]
    remainder = np.clip(1 - occupancy.sum(axis=1), 0, None)
    if phantom.tissue_maps:
        maps = np.stack(
            [
                np.clip(m.volume.as_float(dtype=np.float64).ravel(), 0, 1)
                for m in phantom.tissue_maps
            ],
            axis=1,
        )
        mapped = maps.sum(axis=1)
        scale = np.where(mapped > remainder, remainder / np.maximum(mapped, 1e-300), 1)
        for j, tissue_map in enumerate(phantom.tissue_maps):
            fractions[:, column[tissue_map.tissue]] += maps[:, j] * scale
    background = np.clip(1 - fractions.sum(axis=1), 0, None)

    # A fibre's weight: its share of the bundle's length in the voxel, times the
    # bundle's occupancy there.
    lookup = voxels.voxel * len(phantom.bundles) + voxels.bundle
    weight = voxels.length / lengths.ravel()[lookup] * occupancy.ravel()[lookup]
    # Streamlines that wander outside a bundle's tube fill nothing there.
    kept = weight > 0
    voxels = replace(
        voxels,
        voxel=voxels.voxel[kept],
        bundle=voxels.bundle[kept],
        direction=voxels.direction[kept],
        length=voxels.length[kept],
    )
    weight = weight[kept]
    tissue_index = np.array([column[b.tissue] for b in phantom.bundles], dtype=np.int64)
    order = np.argsort(voxels.voxel, kind="stable")
    offsets = np.zeros(num_voxels + 1, dtype=np.int64)
    np.cumsum(np.bincount(voxels.voxel, minlength=num_voxels), out=offsets[1:])
    if progress is not None:
        progress(len(phantom.bundles) + 1, len(phantom.bundles) + 1, "voxelise")
    return VoxelPartition(
        grid=grid,
        tissue_names=names,
        tissue_fractions=fractions.reshape(*grid.shape, len(names)),
        background=background.reshape(grid.shape),
        bundle_fractions=occupancy.reshape(*grid.shape, len(phantom.bundles)),
        offsets=offsets,
        fibre_tissue=tissue_index[voxels.bundle[order]],
        fibre_bundle=voxels.bundle[order].astype(np.int64),
        fibre_weight=weight[order],
        fibre_direction=np.ascontiguousarray(voxels.direction[order]),
        streamlines=tuple(streamlines),
    )


def _occupancy(
    phantom: Phantom,
    streamlines: list[Tractogram],
    tubes: dict[int, tuple[np.ndarray, np.ndarray]],
    lengths: np.ndarray,
    voxel_volume: float,
) -> np.ndarray:
    """Returns the share of each voxel that each bundle fills, shape (V, B).

    A tube bundle fills its tube; another fills its length in the voxel times its
    streamline area. Voxels filled above 1 in total are scaled down to 1.
    """
    occupancy = np.zeros_like(lengths)
    for k, (bundle, tractogram) in enumerate(
        zip(phantom.bundles, streamlines, strict=True)
    ):
        if k in tubes:
            occupancy[:, k] = tubes[k][0]
            continue
        area = _streamline_area(bundle, tractogram, lengths[:, k], voxel_volume)
        occupancy[:, k] = np.minimum(lengths[:, k] * area / voxel_volume, 1.0)
    total = occupancy.sum(axis=1)
    crowded = total > 1
    occupancy[crowded] /= total[crowded, None]
    return occupancy


def _with_tube_fibres(
    voxels: Voxelisation, tubes: dict[int, tuple[np.ndarray, np.ndarray]]
) -> Voxelisation:
    """Adds fibres along the tubes' centrelines where no streamline reaches.

    A tube can fill a voxel at its edge that none of its streamlines reach; a
    fibre along the centreline there gives that fill a direction.

    Returns:
        The voxelisation with those fibres added.
    """
    if not tubes:
        return voxels
    lengths = voxels.bundle_lengths()
    extra_voxel, extra_bundle, extra_direction = [], [], []
    for k, (fill, direction) in tubes.items():
        missing = np.flatnonzero((fill > 0) & (lengths[:, k] == 0))
        extra_voxel.append(missing)
        extra_bundle.append(np.full(len(missing), k, dtype=np.int64))
        extra_direction.append(direction[missing])
    count = sum(len(v) for v in extra_voxel)
    if count == 0:
        return voxels
    return replace(
        voxels,
        voxel=np.concatenate([voxels.voxel, *extra_voxel]),
        bundle=np.concatenate([voxels.bundle, *extra_bundle]),
        direction=np.concatenate([voxels.direction, *extra_direction]),
        # A token length: these fibres alone carry the bundle's fill there.
        length=np.concatenate([voxels.length, np.full(count, 1e-6)]),
    )


def resolve_bundle_seeds(phantom: Phantom, seed: int | None) -> tuple[Bundle, ...]:
    """Returns the phantom's bundles with a seed on every geometry.

    A geometry without a seed gets one drawn from `seed`, bundle by bundle, as
    `partition_voxels` does; a geometry with its own seed, and a tractogram, are
    kept as they are.

    Returns:
        The bundles, in the phantom's order.
    """
    children = np.random.SeedSequence(seed).spawn(len(phantom.bundles))
    bundles = []
    for index, bundle in enumerate(phantom.bundles):
        source = bundle.source
        if isinstance(source, BundleGeometry) and source.seed is None:
            drawn = int(children[index].generate_state(1)[0])
            bundle = replace(bundle, source=replace(source, seed=drawn))
        bundles.append(bundle)
    return tuple(bundles)


def _streamline_area(
    bundle: Bundle, tractogram: Tractogram, lengths: np.ndarray, voxel_volume: float
) -> float:
    """Returns the cross-section each streamline of a bundle stands for, in mm^2."""
    if bundle.streamline_area is not None:
        return bundle.streamline_area
    if isinstance(bundle.source, BundleGeometry):
        geometry = bundle.source
        return math.pi * geometry.radius**2 / geometry.num_streamlines
    reached = lengths[lengths > 0]
    if len(reached) == 0:
        return 0.0
    dense = float(np.percentile(reached, DENSE_PERCENTILE))
    return bundle.densest_fill * voxel_volume / dense
