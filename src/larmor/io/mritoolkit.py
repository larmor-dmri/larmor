"""Reads MRIToolkit's DTI and DKI fits, maps and tractography from MATLAB files."""

import itertools
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from larmor._log import LarmorLogger
from larmor.io._frames import map_directions, voxel_axes
from larmor.io.formats import read_mif_header, read_nifti_header
from larmor.io.formats.matlab import MatlabFile, read_matlab
from larmor.io.formats.nifti import affine_from_header
from larmor.types import GradientTable, Grid, Tractogram, Volume

type MatlabSource = str | Path | MatlabFile
type Reference = Grid | str | Path | MatlabFile

# The element orders of the diffusion and kurtosis tensors, in Larmor and in MRIToolkit.
DT_INDICES = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
KT_INDICES = [
    (0, 0, 0, 0),
    (1, 1, 1, 1),
    (2, 2, 2, 2),
    (0, 0, 0, 1),
    (0, 0, 0, 2),
    (1, 1, 1, 0),
    (1, 1, 1, 2),
    (2, 2, 2, 0),
    (2, 2, 2, 1),
    (0, 0, 1, 1),
    (0, 0, 2, 2),
    (1, 1, 2, 2),
    (0, 0, 1, 2),
    (1, 1, 0, 2),
    (2, 2, 0, 1),
]
# Variables of a tractography file whose names start with "Tract" but that are not
# values per point.
_NOT_PER_POINT = {"Tracts", "TractL", "TractMask"}


@dataclass(frozen=True)
class MRIToolkitFit:
    """A DTI or DKI fit read from MRIToolkit's `.mat` file, in world coordinates.

    Attributes:
        dt: Diffusion tensor, (X, Y, Z, 6) float32, in the element order of
            `DTIFit.dt`, in mm^2/s; 0 outside the mask.
        kt: Kurtosis tensor, (X, Y, Z, 15) float32, in the element order of
            `DKIFit.kt`; None for a DTI fit.
        dwi: The diffusion-weighted data, (X, Y, Z, N), in the file's volume order
            (sorted by b-value) and type.
        gradients: The gradient table of `dwi`, from the file's b-matrix.
        mask: The fitted voxels (where the file's tensor is finite), (X, Y, Z) bool.
        maps: The maps the file stores, float32 and 0 outside the mask, each when
            the file has what it needs: "FA" (the usual FA; the file's is sqrt(3)
            times it), "MD", "AD", "RD", "eigenvalues" (X, Y, Z, 3, descending),
            "FE" (the first eigenvector, (X, Y, Z, 3)) and "b0" (the file's
            `DWIB0`).
    """

    dt: Volume
    kt: Volume | None
    dwi: Volume
    gradients: GradientTable
    mask: Volume
    maps: dict[str, Volume] = field(default_factory=dict)


def _open(source: MatlabSource) -> MatlabFile:
    """Returns `source` as an opened `.mat` file."""
    return source if isinstance(source, MatlabFile) else read_matlab(source)


def read_mritoolkit_grid(source: MatlabSource) -> Grid:
    """Reads the image grid of an MRIToolkit fit from the NIfTI header it stores.

    Args:
        source: A `.mat` file with the struct `hdr` (a DTI or DKI fit), or the file
            already opened with `read_matlab`.

    Returns:
        The grid of the NIfTI image the fit was made from.

    Raises:
        ValueError: If the file has no `hdr`.
    """
    f = _open(source)
    if "hdr" not in f:
        raise ValueError(
            f"{f.path}: no NIfTI header (hdr); give the grid of the image the "
            "data belong to, such as the fit's .mat file or a NIfTI file."
        )
    dime = f.read("hdr.dime")
    hist = f.read("hdr.hist")
    if not isinstance(dime, dict) or not isinstance(hist, dict):
        raise ValueError(f"{f.path}: hdr is not a NIfTI header struct.")
    fields = {
        k: np.asarray(v, dtype=np.float64).ravel()
        for k, v in (dime | hist).items()
        if isinstance(v, np.ndarray)
    }
    header = {k: v if v.size > 1 else v[0] for k, v in fields.items()}
    dims = fields["dim"]
    shape = (int(dims[1]), int(dims[2]), int(dims[3]))
    return Grid(shape, affine_from_header(header))


def read_mritoolkit_frame(reference: Reference) -> np.ndarray:
    """Reads the (3, 3) matrix that maps MRIToolkit's vectors to world coordinates.

    MRIToolkit stores gradients, tensors, eigenvectors and FOD directions along
    the NIfTI voxel axes j, i and k, in that order. The columns of the matrix are
    the world directions of those axes on the reference grid.

    Args:
        reference: The grid, as a Grid, an MRIToolkit fit, or a NIfTI or MIF file.

    Returns:
        The (3, 3) matrix.
    """
    return _gradient_frame(_reference_grid(reference))


def _gradient_frame(grid: Grid) -> np.ndarray:
    """Returns the frame matrix of `grid`, as `read_mritoolkit_frame` gives it."""
    frame: np.ndarray = voxel_axes(grid)[:, [1, 0, 2]]
    return frame


def _reference_grid(reference: Reference) -> Grid:
    """Returns the grid of a Grid, an MRIToolkit fit, or a NIfTI or MIF file."""
    if isinstance(reference, Grid):
        return reference
    if isinstance(reference, MatlabFile):
        return read_mritoolkit_grid(reference)
    name = Path(reference).name
    if name.endswith(".mat"):
        return read_mritoolkit_grid(reference)
    if name.endswith((".nii", ".nii.gz")):
        return read_nifti_header(reference)[0]
    if name.endswith((".mif", ".mif.gz", ".mih")):
        return read_mif_header(reference)[0]
    raise ValueError(
        f"{reference}: a reference grid is an MRIToolkit fit (.mat), a NIfTI or a "
        "MIF file."
    )


def _to_nifti_order(array: np.ndarray, grid: Grid, path: str) -> np.ndarray:
    """Returns `array`, in ExploreDTI's layout, in NIfTI voxel order on `grid`.

    With nx, ny the NIfTI sizes, the file's array I holds I[a, b, k] =
    nii[nx-1-b, ny-1-a, k]. `path` names the array in the error message.

    Raises:
        ValueError: If the array's first three axes do not fit the grid.
    """
    nx, ny, nz = grid.shape
    if array.ndim < 3 or array.shape[:3] != (ny, nx, nz):
        raise ValueError(
            f"{path} has the shape {array.shape}; on the grid {grid.shape} an image "
            f"in MRIToolkit's layout starts with ({ny}, {nx}, {nz})."
        )
    return np.ascontiguousarray(np.flip(np.swapaxes(array, 0, 1), axis=(0, 1)))


def _read_image(f: MatlabFile, path: str, grid: Grid) -> np.ndarray:
    """Reads the image `path` of `f`, an array or a cell of 3D arrays, on `grid`."""
    value = f.read(path)
    if isinstance(value, list):
        if not value or not all(isinstance(v, np.ndarray) for v in value):
            raise ValueError(f"{f.path}: {path} is not a cell of image arrays.")
        arrays = [v for v in value if isinstance(v, np.ndarray)]
        try:
            value = np.stack(arrays, axis=-1)
        except ValueError as e:
            raise ValueError(f"{f.path}: the arrays of {path} differ in shape.") from e
    if not isinstance(value, np.ndarray):
        raise ValueError(f"{f.path}: {path} is not a numeric array.")
    return _to_nifti_order(value, grid, f"{f.path}: {path}")


def read_mritoolkit_volume(
    source: MatlabSource, path: str, *, reference: Reference | None = None
) -> Volume:
    """Reads an image of an MRIToolkit `.mat` file as a Volume, its values as stored.

    Args:
        source: The `.mat` file, or the file opened with `read_matlab`.
        path: The variable, or a struct field such as `GRL_Results.fractions`. It
            is an array whose first three axes are the image's, or a cell of 3D
            arrays (such as `DT` or `DWI`), stacked along a fourth.
        reference: The grid the image lies on, as a Grid, an MRIToolkit fit (its
            `hdr`), or a NIfTI or MIF file. Defaults to the file's own `hdr`.

    Returns:
        The image in NIfTI voxel order on the reference grid; axes after the
        third are kept, flattened into one in MATLAB's order.

    Raises:
        ValueError: If the value is not an image of the reference grid's shape, or
            there is no reference and the file has no `hdr`.
    """
    f = _open(source)
    grid = _reference_grid(f if reference is None else reference)
    data = _read_image(f, path, grid)
    if data.ndim > 4:
        data = data.reshape(*data.shape[:3], -1, order="F")
    return Volume(data, grid.affine.copy())


def _rotation_6(q: np.ndarray) -> np.ndarray:
    """Returns the (6, 6) matrix rotating the elements of a diffusion tensor by q."""
    columns = []
    for i, j in DT_INDICES:
        basis = np.zeros((3, 3))
        basis[i, j] = basis[j, i] = 1.0
        rotated = q @ basis @ q.T
        columns.append([rotated[a, b] for a, b in DT_INDICES])
    return np.array(columns).T


def _rotation_15(q: np.ndarray) -> np.ndarray:
    """Returns the (15, 15) matrix rotating the elements of a kurtosis tensor by q."""
    columns = []
    for index in KT_INDICES:
        basis = np.zeros((3, 3, 3, 3))
        for p in set(itertools.permutations(index)):
            basis[p] = 1.0
        rotated = np.einsum("ai,bj,ck,dl,ijkl->abcd", q, q, q, q, basis)
        columns.append([rotated[i] for i in KT_INDICES])
    return np.array(columns).T


def _gradients(b_matrix: np.ndarray, q: np.ndarray) -> GradientTable:
    """Returns the gradient table of a b-matrix in MRIToolkit's frame.

    `b_matrix` (N, 6) holds b * (xx, 2xy, 2xz, yy, 2yz, zz); `q` maps its vectors
    to world coordinates.
    """
    halves = np.array([1.0, 0.5, 0.5, 1.0, 0.5, 1.0])
    full = np.zeros((len(b_matrix), 3, 3))
    for c, (i, j) in enumerate(DT_INDICES):
        full[:, i, j] = full[:, j, i] = b_matrix[:, c] * halves[c]
    values, vectors = np.linalg.eigh(full)
    bvals = values.sum(axis=1)
    bvecs = map_directions(vectors[:, :, -1], q)
    bvecs[bvals <= 0] = 0.0
    return GradientTable(np.maximum(bvals, 0.0), bvecs)


def read_mritoolkit_fit(source: MatlabSource) -> MRIToolkitFit:
    """Reads a DTI or DKI fit of MRIToolkit (`EDTI.PerformDTI_DKIFit`), in world space.

    The tensors and vectors are converted to world coordinates, so the fit can be
    compared directly with the results of `DTIModel` and `DKIModel`.

    Args:
        source: The fit's `.mat` file, or the file opened with `read_matlab`.

    Returns:
        The fit, on the grid of the NIfTI image it was made from.

    Raises:
        ValueError: If the file is not an MRIToolkit fit (it needs `DT`, `DWI`, `b`
            and `hdr`).
    """
    f = _open(source)
    missing = [n for n in ("DT", "DWI", "b", "hdr") if n not in f]
    if missing:
        raise ValueError(
            f"{f.path}: not an MRIToolkit DTI or DKI fit (no {', '.join(missing)}); "
            "read_mritoolkit_tracts reads tractography, larmor.sd.read_mritoolkit_fod "
            "FOD fits, and read_matlab any variable."
        )
    grid = read_mritoolkit_grid(f)
    q = _gradient_frame(grid)
    affine = grid.affine

    dt = _read_image(f, "DT", grid).astype(np.float64)
    mask = np.isfinite(dt).all(axis=-1)
    dt = np.where(mask[..., None], dt, 0.0) @ _rotation_6(q).T
    kt = None
    if "KT" in f:
        raw = _read_image(f, "KT", grid).astype(np.float64)
        raw = np.where(mask[..., None] & np.isfinite(raw), raw, 0.0)
        kt = Volume((raw @ _rotation_15(q).T).astype(np.float32), affine.copy())

    dwi = _read_image(f, "DWI", grid)
    gradients = _gradients(f.read_array("b"), q)
    if len(gradients.bvals) != dwi.shape[-1]:
        raise ValueError(
            f"{f.path}: {dwi.shape[-1]} volumes in DWI but {len(gradients.bvals)} "
            "rows in the b-matrix b."
        )

    return MRIToolkitFit(
        dt=Volume(dt.astype(np.float32), affine.copy()),
        kt=kt,
        dwi=Volume(dwi, affine.copy()),
        gradients=gradients,
        mask=Volume(mask, affine.copy()),
        maps=_maps(f, grid, mask, [n for n, v in MAP_VARIABLES.items() if v in f]),
    )


# The maps a fit file stores, by Larmor's name, and the variable each comes from.
MAP_VARIABLES = {
    "FA": "FA",
    "MD": "eigval",
    "AD": "eigval",
    "RD": "eigval",
    "eigenvalues": "eigval",
    "FE": "FE",
    "b0": "DWIB0",
}


def read_mritoolkit_map(source: MatlabSource, name: str) -> Volume:
    """Reads one map an MRIToolkit fit stores, as `MRIToolkitFit.maps` holds it.

    Args:
        source: The fit's `.mat` file, or the file opened with `read_matlab`.
        name: The map's name, one of "FA", "MD", "AD", "RD", "eigenvalues", "FE"
            and "b0".

    Returns:
        The map on the fit's grid, float32, 0 outside the fit's mask (where its
        tensor is not finite) and where the file has NaN.

    Raises:
        ValueError: If the name is unknown or the file lacks its variable.
    """
    f = _open(source)
    if name not in MAP_VARIABLES:
        raise ValueError(f"Unknown map {name!r}; one of {', '.join(MAP_VARIABLES)}.")
    if MAP_VARIABLES[name] not in f:
        raise ValueError(
            f"{f.path}: no {MAP_VARIABLES[name]}, which {name} is read from."
        )
    grid = read_mritoolkit_grid(f)
    mask = None
    if "DT" in f:
        mask = np.isfinite(_read_image(f, "DT", grid)).all(axis=-1)
    return _maps(f, grid, mask, [name])[name]


def _maps(
    f: MatlabFile, grid: Grid, mask: np.ndarray | None, names: list[str]
) -> dict[str, Volume]:
    """Returns the maps `names` of the fit file `f`, as in `MRIToolkitFit.maps`.

    Each map is float32 and 0 outside `mask` and where the file is not finite.
    """
    stored: dict[str, np.ndarray] = {}
    for variable in {MAP_VARIABLES[n] for n in names}:
        data = _read_image(f, variable, grid).astype(np.float64)
        inside = np.isfinite(data)
        if mask is not None:
            inside &= mask.reshape(mask.shape + (1,) * (data.ndim - 3))
        stored[variable] = np.where(inside, data, 0.0)
    maps: dict[str, np.ndarray] = {}
    for name in names:
        data = stored[MAP_VARIABLES[name]]
        if name == "FA":
            data = data / np.sqrt(3.0)  # the file's FA is sqrt(3) times the usual one
        elif name == "MD":
            data = data.mean(axis=-1)
        elif name == "AD":
            data = data[..., 0]
        elif name == "RD":
            data = data[..., 1:].mean(axis=-1)
        elif name == "FE":
            data = map_directions(data, _gradient_frame(grid))
        maps[name] = data.astype(np.float32)
    return {k: Volume(v, grid.affine.copy()) for k, v in maps.items()}


def read_mritoolkit_peaks(
    source: MatlabSource, variable: str, *, reference: Reference | None = None
) -> Volume:
    """Reads a peaks image in MRIToolkit's layout and frame, as world peaks.

    The image's last axis holds 3 values per peak, in MRIToolkit's frame (see
    `read_mritoolkit_frame`).

    Args:
        source: The `.mat` file, or the file opened with `read_matlab`.
        variable: The variable holding the peaks image, as `path` of
            `read_mritoolkit_volume`.
        reference: The grid the image lies on, as for `read_mritoolkit_volume`.
            Defaults to None, the file's own `hdr`.

    Returns:
        The peaks on the reference grid, with 3 values per peak (x, y, z in world
        coordinates).

    Raises:
        ValueError: If the image does not hold 3 values per peak.
    """
    image = read_mritoolkit_volume(source, variable, reference=reference)
    data = image.data
    if data.ndim != 4 or data.shape[-1] % 3 != 0:
        raise ValueError(
            f"{variable}: {data.shape[3:]} values per voxel, not 3 per peak."
        )
    grid = Grid(data.shape[:3], image.affine)
    triplets = data.astype(np.float32).reshape(*data.shape[:3], -1, 3)
    world = map_directions(triplets, _gradient_frame(grid).astype(np.float32))
    return Volume(world.reshape(data.shape), image.affine)


def read_mritoolkit_kind(source: MatlabSource) -> str | None:
    """Reads what kind of MRIToolkit result a `.mat` file holds, from its variables.

    Args:
        source: The `.mat` file, or the file opened with `read_matlab`.

    Returns:
        "fit" for a DTI or DKI fit (`DT` and `hdr`), "fod" for a GRL or mFOD result
        (a struct with the fields `FOD` and `fractions`), "tracts" for
        tractography (`Tracts`), or None for none of these.
    """
    f = _open(source)
    if "DT" in f and "hdr" in f:
        return "fit"
    if _fod_structs(f):
        return "fod"
    if "Tracts" in f:
        return "tracts"
    return None


def _fod_structs(f: MatlabFile) -> list[str]:
    """Returns the top-level structs of `f` that have the fields FOD and fractions."""
    paths = {e.path for e in f.entries()}
    return [n for n in f.names if {f"{n}.FOD", f"{n}.fractions"} <= paths]


def read_mritoolkit_tracts(
    source: MatlabSource, reference: Reference, *, variable: str = "Tracts"
) -> Tractogram:
    """Reads MRIToolkit's tractography (`Tracts` and its per-point values).

    Args:
        source: The tractography `.mat` file, or the file opened with `read_matlab`.
        reference: The grid the tracking ran on, as a Grid, the MRIToolkit fit
            (`.mat`) it ran on, or a NIfTI or MIF file of that image. The file
            stores no grid of its own.
        variable: The cell of (N, 3) point arrays, in mm along the axes of
            MRIToolkit's layout, as `Tracts` holds them. The per-point values are
            read only with `Tracts`. Defaults to "Tracts".

    Returns:
        The streamlines in world coordinates, with the reference grid. Each
        per-point cell (`TractFA`, `TractMD`, `TractFE`, `TractsFOD`, ...) is in
        `point_data` under its name without the prefix (`FA`, `MD`, `FE`, `FOD`),
        as stored, except two. FA is divided by sqrt(3) to the usual FA. FE has
        its first two components swapped, so it is in world order only on a grid
        whose axes are the world's; its signs are lost, as MRIToolkit stores
        absolute values. `TractL`, the number of points of each streamline, is in
        `streamline_data["L"]`.

    Raises:
        ValueError: If the file has no `Tracts`, or the reference cannot be read.
    """
    f = _open(source)
    if variable == "Tracts" and variable not in f:
        raise ValueError(
            f"{f.path}: no tractography (Tracts); read_mritoolkit_fit reads DTI and "
            "DKI fits, larmor.sd.read_mritoolkit_fod FOD fits, and read_matlab any "
            "variable."
        )
    grid = _reference_grid(reference)
    tracts = _cell_of_arrays(f, variable)
    if any(t.ndim != 2 or t.shape[1] != 3 for t in tracts):
        raise ValueError(f"{f.path}: {variable} is not a cell of (N, 3) point arrays.")
    counts = np.array([len(t) for t in tracts], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(counts)])
    if "VDims" in f:
        sizes = f.read_array("VDims").astype(np.float64).ravel()
    else:
        sizes = np.linalg.norm(grid.affine[:3, :3], axis=0)[[1, 0, 2]]
    points = (
        np.concatenate(tracts).astype(np.float64).reshape(-1, 3)
        if tracts
        else np.zeros((0, 3))
    )
    nx, ny, _ = grid.shape
    ijk = np.column_stack(
        [
            nx - points[:, 1] / sizes[1],
            ny - points[:, 0] / sizes[0],
            points[:, 2] / sizes[2] - 1.0,
        ]
    )
    world = ijk @ grid.affine[:3, :3].T + grid.affine[:3, 3]

    point_data = _tract_point_data(f, counts) if variable == "Tracts" else {}
    streamline_data: dict[str, np.ndarray] = {}
    if "TractL" in f and variable == "Tracts":
        lengths = np.array([np.ravel(v)[0] for v in _cell_of_arrays(f, "TractL")])
        if len(lengths) == len(counts):
            streamline_data["L"] = lengths
        else:
            LarmorLogger.warning(
                "%s: TractL has %d entries for %d streamlines; it was left out.",
                f.path,
                len(lengths),
                len(counts),
            )
    return Tractogram(world, offsets, point_data, streamline_data, reference=grid)


def _tract_point_data(f: MatlabFile, counts: np.ndarray) -> dict[str, np.ndarray]:
    """Reads the per-point cells of `f`, by name without the prefix.

    Returns:
        The per-point values by name. FA is divided by sqrt(3), and FE has its
        first two components swapped.
    """
    point_data: dict[str, np.ndarray] = {}
    for name in f.names:
        if not name.startswith("Tract") or name in _NOT_PER_POINT:
            continue
        values = _per_point(f, name, counts)
        if values is not None:
            key = name.removeprefix("Tracts") if name.startswith("Tracts") else name[5:]
            if key == "FA":
                values = values / np.sqrt(3.0)
            elif key == "FE" and values.ndim == 2 and values.shape[1] == 3:
                values = values[:, [1, 0, 2]]
            point_data[key] = values
    return point_data


def _cell_of_arrays(f: MatlabFile, name: str) -> list[np.ndarray]:
    """Reads the cell `name` of numeric arrays of `f` as a list."""
    value = f.read(name)
    if isinstance(value, np.ndarray) and value.size == 0:
        return []
    if not isinstance(value, list) or not all(isinstance(v, np.ndarray) for v in value):
        raise ValueError(f"{f.path}: {name} is not a cell of numeric arrays.")
    return [v for v in value if isinstance(v, np.ndarray)]


def _per_point(f: MatlabFile, name: str, counts: np.ndarray) -> np.ndarray | None:
    """Returns the per-point cell `name` joined, or None if it does not fit `counts`."""
    try:
        cells = _cell_of_arrays(f, name)
    except ValueError:
        cells = []
    if len(cells) != len(counts) or any(
        len(c) != n for c, n in zip(cells, counts, strict=True)
    ):
        LarmorLogger.warning(
            "%s: %s does not hold one row per point of Tracts; it was left out.",
            f.path,
            name,
        )
        return None
    if not cells:
        return np.zeros(0)
    joined = np.concatenate([c.reshape(len(c), -1) for c in cells])
    return joined[:, 0] if joined.shape[1] == 1 else joined
