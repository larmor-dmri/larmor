"""Plans which items to load from a MATLAB `.mat` file, and reads them."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from larmor.io import (
    MatlabEntry,
    MatlabFile,
    Progress,
    load_grid,
    read_matlab,
    read_mritoolkit_map,
    read_mritoolkit_peaks,
    read_mritoolkit_tracts,
    read_mritoolkit_volume,
)
from larmor.io.mritoolkit import MAP_VARIABLES
from larmor.types import Grid, Surface, Tractogram
from larmor.viewer.model.item import Category
from larmor.viewer.model.options import OptionName
from larmor.viewer.model.registry import SUBSET_SEED, SUBSETS, Subset
from larmor.viewer.model.surface import SurfaceData

if TYPE_CHECKING:
    from larmor.sd import MRIToolkitFOD

# The MATLAB classes of numeric and logical arrays.
_NUMERIC = {
    "double",
    "single",
    "int8",
    "uint8",
    "int16",
    "uint16",
    "int32",
    "uint32",
    "int64",
    "uint64",
    "logical",
}


class MatlabPreset(StrEnum):
    """What to load from a MATLAB file."""

    TENSOR_FIT = "tensor_fit"
    FOD_FIT = "fod_fit"
    TRACTOGRAPHY = "tractography"
    MANUAL = "manual"


# The text the import dialog shows for each preset.
PRESET_LABELS: Mapping[MatlabPreset, str] = {
    MatlabPreset.TENSOR_FIT: "Load as tensor fit",
    MatlabPreset.FOD_FIT: "Load as FOD fit",
    MatlabPreset.TRACTOGRAPHY: "Load as tractography",
    MatlabPreset.MANUAL: "Load manually",
}


class LoadAs(StrEnum):
    """What one variable is loaded as, in a manual import."""

    NOTHING = "nothing"
    VOLUME = "volume"
    OVERLAY = "overlay"
    FODS = "fods"
    PEAKS = "peaks"
    TRACTOGRAM = "tractogram"
    POINT_VALUES = "point_values"  # values per point of a loaded tractogram
    SURFACE = "surface"


# The text the import dialog shows for each choice of a manual import.
LOAD_AS_LABELS: Mapping[LoadAs, str] = {
    LoadAs.NOTHING: "Nothing",
    LoadAs.VOLUME: "Volume",
    LoadAs.OVERLAY: "Overlay",
    LoadAs.FODS: "FODs",
    LoadAs.PEAKS: "Peaks",
    LoadAs.TRACTOGRAM: "Tractogram",
    LoadAs.POINT_VALUES: "Values per point",
    LoadAs.SURFACE: "Surface",
}
# The category an item loaded as each choice goes into.
_CATEGORY_OF: Mapping[LoadAs, Category] = {
    LoadAs.VOLUME: Category.VOLUMES,
    LoadAs.OVERLAY: Category.OVERLAYS,
    LoadAs.FODS: Category.FODS,
    LoadAs.PEAKS: Category.PEAKS,
    LoadAs.TRACTOGRAM: Category.TRACTOGRAMS,
    LoadAs.SURFACE: Category.SURFACES,
}


class PartKind(StrEnum):
    """How an item is read from a MATLAB file."""

    MAP = "map"  # a map of an MRIToolkit fit (read_mritoolkit_map)
    FOD = "fod"  # one FOD of a GRL or mFOD result, as SH
    FRACTIONS = "fractions"  # the signal fractions of that result
    RSS = "rss"  # its residuals
    TRACTS = "tracts"  # MRIToolkit's tractography, with its values per point
    IMAGE = "image"  # any image variable, values as stored
    PEAKS = "peaks"  # a peaks image, its vectors taken from MRIToolkit's frame to world
    SH = "sh"  # an FOD image (amplitudes or SH), taken to world SH
    STREAMLINES = "streamlines"  # any cell of (N, 3) points in MRIToolkit's layout
    SURFACE = "surface"  # a struct with vertices (V, 3) and 1-based faces (F, 3)


@dataclass(frozen=True)
class MatlabPart:
    """What an item was read from inside a MATLAB file.

    Attributes:
        kind: How it is read.
        variable: The map's name for MAP; the result struct for FOD, FRACTIONS and
            RSS; the variable's path (as `MatlabFile.read` takes it) otherwise.
        index: Which anisotropic response's FOD, for FOD.
        reference: The file giving the grid (an MRIToolkit fit, a NIfTI or MIF
            file); None when the `.mat` file stores its own.
    """

    kind: PartKind
    variable: str = ""
    index: int = 0
    reference: Path | None = None

    def to_json(self) -> dict[str, object]:
        """Returns the part as JSON values, with the reference as an absolute path."""
        return {
            "kind": self.kind.value,
            "variable": self.variable,
            "index": self.index,
            "reference": None if self.reference is None else str(self.reference),
        }

    @classmethod
    def from_json(cls, raw: object) -> MatlabPart:
        """Reads a part from `to_json`'s values; raises TypeError or ValueError."""
        if not isinstance(raw, dict):
            raise TypeError("its MATLAB part is not a JSON object")
        index = raw.get("index", 0)
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError(f"its MATLAB part's index is not a whole number: {index!r}")
        reference = raw.get("reference")
        return cls(
            PartKind(raw["kind"]),
            str(raw.get("variable", "")),
            index,
            Path(reference) if isinstance(reference, str) else None,
        )


@dataclass(frozen=True)
class PlannedItem:
    """One item an import adds, with what it reads, its category and first options."""

    part: MatlabPart
    category: Category
    name: str
    options: Mapping[OptionName, object] = field(default_factory=dict)
    # The planned item whose id becomes this one's `modulate` option.
    modulate_by: int | None = None


def suggest_preset(f: MatlabFile) -> MatlabPreset:
    """Returns the preset that fits the file's variables; MANUAL if none does."""
    for preset in (
        MatlabPreset.TENSOR_FIT,
        MatlabPreset.FOD_FIT,
        MatlabPreset.TRACTOGRAPHY,
    ):
        if preset_problem(f, preset) is None:
            return preset
    return MatlabPreset.MANUAL


def preset_problem(f: MatlabFile, preset: MatlabPreset) -> str | None:
    """Returns why a preset does not fit the file, or None if it does."""
    if preset == MatlabPreset.TENSOR_FIT:
        missing = [n for n in ("DT", "hdr") if n not in f]
        if missing:
            return f"The file has no {' and '.join(missing)} of an MRIToolkit fit."
    elif preset == MatlabPreset.FOD_FIT:
        if not fod_structs(f):
            return "The file has no struct with an FOD field (a GRL or mFOD result)."
    elif preset == MatlabPreset.TRACTOGRAPHY and "Tracts" not in f:
        return "The file has no Tracts (MRIToolkit's tractography)."
    return None


def fod_structs(f: MatlabFile) -> list[str]:
    """Returns the top-level structs with the fields FOD and fractions."""
    paths = {e.path for e in f.entries()}
    return [n for n in f.names if {f"{n}.FOD", f"{n}.fractions"} <= paths]


def needs_reference(f: MatlabFile, preset: MatlabPreset) -> bool:
    """Tells whether loading the preset needs a grid from another file."""
    return preset != MatlabPreset.TENSOR_FIT and "hdr" not in f


def find_reference(path: Path) -> Path | None:
    """Finds the file holding a `.mat` file's grid: itself, else a fit beside it.

    Beside it, dwi.mat is tried first, then the other v7.3 `.mat` files.

    Returns:
        The file holding the grid, or None.
    """
    path = Path(path).absolute()
    if _has_header(path):
        return path
    beside = sorted(p for p in path.parent.glob("*.mat") if p != path)
    beside.sort(key=lambda p: p.name != "dwi.mat")
    for candidate in beside:
        if candidate.name == "dwi.mat" or _is_v73(candidate):
            if _has_header(candidate):
                return candidate
    return None


def _is_v73(path: Path) -> bool:
    """Tells whether a file is a MATLAB v7.3 file."""
    try:
        with open(path, "rb") as f:
            return f.read(10) == b"MATLAB 7.3"
    except OSError:
        return False


def _has_header(path: Path) -> bool:
    """Tells whether a `.mat` file holds an MRIToolkit header (hdr)."""
    try:
        return "hdr" in read_matlab(path)
    except OSError, ValueError:
        return False


def reference_grid(path: Path, reference: Path | None) -> Grid | None:
    """Reads the grid of a reference (the `.mat` file if None); None if unreadable."""
    try:
        return load_grid(reference if reference is not None else path)
    except OSError, ValueError:
        return None


def plan_preset(
    path: Path, preset: MatlabPreset, reference: Path | None = None
) -> list[PlannedItem]:
    """Returns the items a preset adds from a file.

    Returns:
        The items to add.

    Raises:
        ValueError: If the preset does not fit the file, or needs a reference that
            was not given.
    """
    f = read_matlab(path)
    problem = preset_problem(f, preset) if preset != MatlabPreset.MANUAL else None
    if problem is not None:
        raise ValueError(f"{path.name}: {problem}")
    if preset == MatlabPreset.MANUAL:
        raise ValueError("A manual import names its variables: use plan_manual.")
    if needs_reference(f, preset) and reference is None:
        raise ValueError(
            f"{path.name} stores no grid; give the fit it was made from (.mat) or "
            "a NIfTI or MIF file of its image as the reference."
        )
    stem = path.stem
    if preset == MatlabPreset.TENSOR_FIT:
        # The FA goes last: the last volume added is the one shown.
        has_fa = _map_present(f, "FA")
        names = [n for n in ("MD", "eigenvalues") if _map_present(f, n)]
        items = [
            PlannedItem(MatlabPart(PartKind.MAP, n), Category.VOLUMES, f"{stem} {n}")
            for n in names
        ]
        if "DWI" in f:
            items.append(
                PlannedItem(
                    MatlabPart(PartKind.IMAGE, "DWI"), Category.VOLUMES, f"{stem} DWI"
                )
            )
        if "FE" in f:
            fa = len(items) + 1 if has_fa else None
            items.append(
                PlannedItem(
                    MatlabPart(PartKind.MAP, "FE"),
                    Category.VOLUMES,
                    f"{stem} FE",
                    {OptionName.DTI: True},
                    modulate_by=fa,
                )
            )
        if has_fa:
            fa_part = MatlabPart(PartKind.MAP, "FA")
            items.append(PlannedItem(fa_part, Category.VOLUMES, f"{stem} FA"))
        return items
    if preset == MatlabPreset.FOD_FIT:
        struct = fod_structs(f)[0]
        # Imported here: larmor.sd loads scipy, which the model avoids.
        from larmor.sd import read_mritoolkit_fod_count

        n_fods = read_mritoolkit_fod_count(f, variable=struct)
        items = [
            PlannedItem(
                MatlabPart(PartKind.FOD, struct, i, reference),
                Category.FODS,
                f"{stem} FOD" if n_fods == 1 else f"{stem} FOD {i + 1}",
            )
            for i in range(n_fods)
        ]
        items.append(
            PlannedItem(
                MatlabPart(PartKind.FRACTIONS, struct, 0, reference),
                Category.VOLUMES,
                f"{stem} fractions",
            )
        )
        if f"{struct}.RSS" in {e.path for e in f.entries()}:
            items.append(
                PlannedItem(
                    MatlabPart(PartKind.RSS, struct, 0, reference),
                    Category.VOLUMES,
                    f"{stem} RSS",
                )
            )
        return items
    return [
        PlannedItem(
            MatlabPart(PartKind.TRACTS, "Tracts", 0, reference),
            Category.TRACTOGRAMS,
            stem,
        )
    ]


def _map_present(f: MatlabFile, name: str) -> bool:
    """Tells whether the file holds the variable of an MRIToolkit map."""
    return MAP_VARIABLES[name] in f


def entry_problem(
    entry: MatlabEntry,
    load_as: LoadAs,
    grid: Grid | None,
    entries: Mapping[str, MatlabEntry],
) -> str | None:
    """Returns why a variable cannot be loaded as `load_as`, or None if it can.

    It looks at the shapes and classes only, without reading data.

    Args:
        entry: The variable, as `MatlabFile.entries` lists it.
        load_as: The choice.
        grid: The reference grid an image must fit; None if there is none.
        entries: Every entry of the file, by path, for the elements of cells and
            the fields of structs.

    Returns:
        The reason, or None.
    """
    if load_as == LoadAs.NOTHING:
        return None
    if load_as in (LoadAs.TRACTOGRAM, LoadAs.POINT_VALUES):
        return _cell_problem(entry, load_as, grid, entries)
    if load_as == LoadAs.SURFACE:
        return _surface_problem(entry, entries)
    return _image_problem(entry, load_as, grid)


def _cell_problem(
    entry: MatlabEntry,
    load_as: LoadAs,
    grid: Grid | None,
    entries: Mapping[str, MatlabEntry],
) -> str | None:
    """Returns why a variable cannot be loaded as a tractogram or point values."""
    if entry.matlab_class != "cell":
        return "Not a cell (one array per streamline)."
    if load_as == LoadAs.TRACTOGRAM:
        points = [e for p, e in entries.items() if p.startswith(entry.path + "{")]
        if any(len(e.shape) != 2 or e.shape[1] != 3 for e in points):
            return "Its elements are not (N, 3) point arrays."
        if grid is None:
            return "Needs a reference grid."
    return None


def _surface_problem(
    entry: MatlabEntry, entries: Mapping[str, MatlabEntry]
) -> str | None:
    """Returns why a variable cannot be loaded as a surface."""
    vertices = entries.get(entry.path + ".vertices")
    faces = entries.get(entry.path + ".faces")
    if entry.matlab_class != "struct" or vertices is None or faces is None:
        return "Not a struct with the fields vertices and faces."
    if vertices.shape[1:] != (3,) or faces.shape[1:] != (3,):
        return "Its vertices and faces are not (V, 3) and (F, 3)."
    return None


def _image_problem(
    entry: MatlabEntry, load_as: LoadAs, grid: Grid | None
) -> str | None:
    """Returns why a variable cannot be loaded as an image, FODs or peaks."""
    if entry.matlab_class not in _NUMERIC or entry.sparse:
        return f"A {entry.matlab_class} value, not a numeric image."
    if len(entry.shape) < 3:
        return "Not an image: fewer than three dimensions."
    if grid is None:
        return "Needs a reference grid."
    nx, ny, nz = grid.shape
    if entry.shape[:3] != (ny, nx, nz):
        return (
            f"Its size {_size(entry.shape)} does not fit the reference grid "
            f"{_size(grid.shape)} (MRIToolkit stores it as {ny} x {nx} x {nz})."
        )
    extra = int(np.prod(entry.shape[3:], dtype=np.int64))
    if load_as == LoadAs.FODS:
        from larmor.sd.sphere import DIRS_300

        if len(entry.shape) < 4 or not (_is_sh_count(extra) or extra == len(DIRS_300)):
            return (
                f"{extra} values per voxel is neither a number of SH coefficients "
                f"nor {len(DIRS_300)} amplitudes."
            )
    if load_as == LoadAs.PEAKS and (len(entry.shape) < 4 or extra % 3 != 0):
        return f"{extra} values per voxel is not 3 per peak."
    return None


def _size(shape: tuple[int, ...]) -> str:
    """Returns a shape as text, such as "96 x 96 x 60"."""
    return " x ".join(str(n) for n in shape)


def _is_sh_count(n: int) -> bool:
    """Tells whether n is the size of an SH series of even or all degrees."""
    return any(n in ((d + 1) * (d + 2) // 2, (d + 1) ** 2) for d in range(0, 17, 2))


def plan_manual(
    path: Path,
    choices: Mapping[str, LoadAs],
    reference: Path | None = None,
) -> list[PlannedItem]:
    """Returns the items a manual import adds, one per variable chosen as a category.

    Values per point are not items; `values_spec` names them for a tractogram.

    Returns:
        The items to add.

    Raises:
        ValueError: If a choice does not fit its variable, naming it and why.
    """
    f = read_matlab(path)
    entries = {e.path: e for e in f.entries()}
    grid = reference_grid(path, reference)
    items = []
    for variable, load_as in choices.items():
        if load_as in (LoadAs.NOTHING, LoadAs.POINT_VALUES):
            continue
        if variable not in entries:
            raise ValueError(f"{path.name} has no {variable}.")
        problem = entry_problem(entries[variable], load_as, grid, entries)
        if problem is not None:
            raise ValueError(
                f"{variable} cannot be loaded as {load_as.value}: {problem}"
            )
        kind = {
            LoadAs.TRACTOGRAM: PartKind.STREAMLINES,
            LoadAs.SURFACE: PartKind.SURFACE,
            LoadAs.PEAKS: PartKind.PEAKS,
            LoadAs.FODS: PartKind.SH,
        }.get(load_as, PartKind.IMAGE)
        items.append(
            PlannedItem(
                MatlabPart(kind, variable, 0, reference),
                _CATEGORY_OF[load_as],
                f"{path.stem} {variable}",
            )
        )
    return items


def read_part(
    path: Path, part: MatlabPart, progress: Progress | None, limit: int | None
) -> object:
    """Reads what a part names from a `.mat` file: a Volume, Tractogram or surface."""
    reference = part.reference if part.reference is not None else path
    if part.kind == PartKind.MAP:
        data: object = read_mritoolkit_map(path, part.variable)
    elif part.kind == PartKind.IMAGE:
        data = read_mritoolkit_volume(path, part.variable, reference=reference)
    elif part.kind == PartKind.PEAKS:
        data = read_mritoolkit_peaks(path, part.variable, reference=reference)
    elif part.kind == PartKind.SH:
        from larmor.sd import read_mritoolkit_sh

        data = read_mritoolkit_sh(path, part.variable, reference)
    elif part.kind in (PartKind.FOD, PartKind.FRACTIONS, PartKind.RSS):
        data = _fod_part(path, part, Path(reference))
    elif part.kind in (PartKind.TRACTS, PartKind.STREAMLINES):
        # The viewer shows the points only, and the per-point values of a large
        # tractography take many times longer to read than the points.
        tracts = read_mritoolkit_tracts(
            path, reference, variable=part.variable, values=False
        )
        data = _subset(tracts, limit)
    else:
        data = _surface(read_matlab(path), part.variable)
    if progress is not None:
        progress(1, 1)
    return data


def _fod_part(path: Path, part: MatlabPart, reference: Path) -> object:
    """Returns the FOD, tissue fractions or RSS of a GRL or mFOD result."""
    result = _fod_result(Path(path).absolute(), part.variable, reference)
    if part.kind == PartKind.FOD:
        return result.fods[part.index]
    if part.kind == PartKind.FRACTIONS:
        return result.tissue_fractions
    if result.rss is None:
        raise ValueError(f"{path.name}: {part.variable} has no RSS.")
    return result.rss


def _fod_result(path: Path, struct: str, reference: Path) -> MRIToolkitFOD:
    """Reads a GRL or mFOD result once for all the items of an import."""
    stamps = tuple(p.stat().st_mtime_ns for p in (path, reference))
    return _cached_fod_result(path, struct, reference.absolute(), stamps)


@lru_cache(maxsize=1)
def _cached_fod_result(
    path: Path, struct: str, reference: Path, stamps: tuple[int, ...]
) -> MRIToolkitFOD:
    """Reads a GRL or mFOD result; stamps make a changed file read again."""
    # Imported here: larmor.sd loads scipy, which the model avoids.
    from larmor.sd import read_mritoolkit_fod

    return read_mritoolkit_fod(path, reference, variable=struct)


def _subset(tractogram: Tractogram, limit: int | None) -> Tractogram:
    """Returns the tractogram, or a random subset of `limit` streamlines of it."""
    total = tractogram.num_streamlines
    if limit is None or total <= limit:
        return tractogram
    indices = tractogram.subsample_indices(limit, seed=SUBSET_SEED)
    subset = tractogram[indices]
    SUBSETS[subset] = Subset(indices, total)
    return subset


def _surface(f: MatlabFile, variable: str) -> SurfaceData:
    """Reads a struct with vertices (V, 3) in world mm and 1-based faces (F, 3)."""
    vertices = f.read_array(f"{variable}.vertices")
    faces = f.read_array(f"{variable}.faces")
    if faces.size and faces.min() < 1:
        raise ValueError(f"{f.path.name}: {variable}.faces are not 1-based indices.")
    return SurfaceData(Surface(vertices.astype(np.float32), faces.astype(np.int64) - 1))


def plan_import(
    path: Path,
    preset: MatlabPreset,
    reference: Path | None = None,
    variables: Mapping[str, LoadAs] | None = None,
    tractogram: int | None = None,
) -> tuple[list[PlannedItem], list[str]]:
    """Returns the items an import adds, and the variables that colour `tractogram`.

    Returns:
        The items to add, and the names of the variables that colour the tractogram.

    Raises:
        ValueError: As `plan_preset` and `plan_manual`; also if values per point
            are chosen without the tractogram they colour.
    """
    choices = dict(variables or {})
    if preset == MatlabPreset.MANUAL:
        planned = plan_manual(path, choices, reference)
    else:
        planned = plan_preset(path, preset, reference)
    values = [v for v, a in choices.items() if a == LoadAs.POINT_VALUES]
    if values and tractogram is None:
        raise ValueError("Values per point need the tractogram they colour.")
    return planned, values


def values_spec(path: Path, variable: str) -> str:
    """Returns a tractogram's values-file option naming a cell of per-point values."""
    return f"{Path(path).absolute()}:{variable}"


def split_values_spec(spec: str) -> tuple[Path, str] | None:
    """Returns the file and cell a values-file option names, or None for a TSF file."""
    head, colon, variable = spec.rpartition(":")
    if not colon or not head.lower().endswith(".mat") or not variable:
        return None
    return Path(head), variable


def point_values(
    path: Path, variable: str, tractogram: Tractogram, subset: Subset | None
) -> np.ndarray:
    """Reads a cell of per-point values for a tractogram read from its whole file.

    Returns:
        The values, one per point.

    Raises:
        ValueError: If the cell is not one column of values per point of the
            tractogram's streamlines (of its file's, for a subset).
    """
    try:
        cells = read_matlab(path).read(variable)
    except KeyError as e:
        raise ValueError(f"{path.name}: {e.args[0] if e.args else variable}") from e
    if not isinstance(cells, list) or not all(isinstance(c, np.ndarray) for c in cells):
        raise ValueError(f"{path.name}: {variable} is not a cell of numeric arrays.")
    arrays = [np.asarray(c) for c in cells if isinstance(c, np.ndarray)]
    # One value per point: a column (N x 1) or a row (1 x N) per streamline.
    if any(a.ndim > 2 or (a.ndim == 2 and min(a.shape) > 1) for a in arrays):
        raise ValueError(f"{path.name}: {variable} has more than one value per point.")
    arrays = [a.reshape(-1, 1) for a in arrays]
    if subset is not None:
        if len(arrays) != subset.total:
            raise ValueError(
                f"{path.name}: {variable} has {len(arrays)} streamlines; the file "
                f"of this tractogram has {subset.total}."
            )
        arrays = [arrays[i] for i in subset.indices]
    counts = np.diff(tractogram.offsets)
    if len(arrays) != len(counts) or any(
        len(a) != n for a, n in zip(arrays, counts, strict=True)
    ):
        raise ValueError(
            f"{path.name}: {variable} does not hold one value per point of this "
            "tractogram."
        )
    if not arrays:
        return np.zeros(0)
    return np.concatenate(arrays)[:, 0].astype(np.float64)
