"""Gradient tables (FSL bvals/bvecs, MRtrix, b-matrix files) and MRtrix responses."""

import re
import warnings
from pathlib import Path
from typing import Literal

import numpy as np

from larmor._log import LarmorLogger
from larmor.io._frames import map_directions, voxel_axes
from larmor.io.load_save import load_header
from larmor.types import GradientTable, Grid

# The b-matrix trace, in s/mm^2, below which a row is read as b = 0.
_ZERO_B = 1e-6

# Vector components below this count as zero when choosing the sign.
_ZERO_COMPONENT = 1e-6


def _try_load(p: str | Path) -> np.ndarray:
    """Loads a text file of numbers.

    Returns:
        The numbers, as read by numpy.loadtxt.

    Raises:
        ValueError: If the file does not exist, is empty, or is not a table of
            numbers; the message says which.
    """
    pt = Path(p)
    if not pt.is_file():
        raise ValueError(f"{pt} does not exist.")
    try:
        with warnings.catch_warnings():
            # An empty file makes numpy warn; it is reported as an error below.
            warnings.simplefilter("ignore", UserWarning)
            data = np.loadtxt(pt, dtype=np.float64)
    except (OSError, ValueError) as exc:
        raise ValueError(f"{pt} cannot be read as a table of numbers: {exc}") from exc
    if data.size == 0:
        raise ValueError(f"{pt} is empty.")
    return data


def _fsl_to_world(g: Grid) -> np.ndarray:
    """Returns the matrix that maps FSL b-vectors on grid `g` to world directions.

    Under shear it is not a rotation, so the mapped vectors must be rescaled to
    their input lengths, as `map_directions` does.
    """
    U = voxel_axes(g)
    F = np.eye(3)
    if np.linalg.det(U) > 0:
        F[0, 0] = -1
    result: np.ndarray = U @ F
    return result


def _check_inputs(
    bvals: np.ndarray, bvecs: np.ndarray, bvals_path: Path, bvecs_path: Path
) -> np.ndarray:
    """Returns the b-vectors `bvecs` as (N, 3), from a (3, N) or (N, 3) table.

    A 3x3 table, ambiguous, is read with one vector per column, with a warning.

    Raises:
        ValueError: If the b-vectors are neither (3, N) nor (N, 3), for the N
            b-values.
    """
    N = len(bvals)
    # A single vector, written as a row or a column, loads as (3,).
    bvecs = np.atleast_2d(bvecs) if bvecs.shape == (3,) else bvecs

    if bvecs.shape == (3, N):
        if N == 3:
            LarmorLogger.warning(
                "%s holds 3 b-vectors as a 3x3 table, which is ambiguous: read as "
                "FSL, one vector per column.",
                bvecs_path,
            )
        return bvecs.T
    if bvecs.shape == (N, 3):
        return bvecs
    raise ValueError(
        f"{bvecs_path} has shape {bvecs.shape}, but {bvals_path} has {N} b-values: "
        f"expected (3, {N}) or ({N}, 3)."
    )


def _check_volume_count(n_entries: int, shape: tuple[int, ...], ref_path: Path) -> None:
    """Checks that the image `ref_path` of shape `shape` has `n_entries` volumes.

    Raises:
        ValueError: If the image is not 4D, or its number of volumes differs.
    """
    if len(shape) != 4:
        raise ValueError(
            f"{ref_path} has no fourth dimension; a gradient table needs a 4D image."
        )
    if shape[3] != n_entries:
        raise ValueError(
            f"The gradient files have {n_entries} entries, but {ref_path} has "
            f"{shape[3]} volumes."
        )


def _check_frame(frame: str) -> None:
    """Raises a ValueError if `frame` is neither "fsl" nor "world"."""
    if frame not in ("fsl", "world"):
        raise ValueError(f'Unsupported frame "{frame}"; expected "fsl" or "world".')


def read_bvals_bvecs(
    bvals_path: str | Path,
    bvecs_path: str | Path,
    *,
    ref_path: str | Path | None = None,
    frame: Literal["fsl", "world"] = "fsl",
    grid: Grid | None = None,
) -> GradientTable:
    """Reads bvals and bvecs files into a gradient table in world coordinates.

    The bvecs file holds one direction per volume, in one of two frames. In the
    "fsl" frame, the convention of FSL's bvecs files, the directions are along
    the voxel axes of the image, with x negated when the image's affine has a
    positive determinant, so converting them needs the image's grid. In the
    "world" frame they are already in world coordinates (RAS+).

    Args:
        bvals_path: The bvals file.
        bvecs_path: The bvecs file, (3, N) or (N, 3).
        ref_path: The image the files belong to. Required for "fsl" unless `grid`
            is given; when given, its number of volumes is checked. Defaults to
            None.
        frame: The frame of the bvecs file. Defaults to "fsl".
        grid: The grid of an image that does not exist as a file, for "fsl"
            instead of `ref_path`; the number of volumes is not checked. Defaults
            to None.

    Returns:
        GradientTable: The table, with its vectors in world coordinates.

    Raises:
        ValueError: If the files cannot be read or do not match each other or the
            image, or "fsl" is given without an image.
    """
    _check_frame(frame)
    if frame == "fsl" and ref_path is None and grid is None:
        raise ValueError("Cannot convert b-vectors to RAS+ without a reference.")
    bvals_path, bvecs_path = Path(bvals_path), Path(bvecs_path)

    bvals = _try_load(bvals_path)
    if bvals.ndim == 2 and min(bvals.shape) > 1:
        raise ValueError(
            f"{bvals_path} has shape {bvals.shape}; a bvals file is one row or one "
            "column."
        )
    bvals = bvals.ravel(order="C")
    bvecs = _try_load(bvecs_path)
    bvecs = _check_inputs(bvals, bvecs, bvals_path, bvecs_path)

    if ref_path is None:
        if grid is not None and frame == "fsl":
            bvecs = map_directions(bvecs, _fsl_to_world(grid))
        return GradientTable(bvals, bvecs)

    ref_path = Path(ref_path)
    ref_grid, ref_shape = load_header(ref_path)
    _check_volume_count(len(bvals), ref_shape, ref_path)

    if frame == "fsl":
        bvecs = map_directions(bvecs, _fsl_to_world(ref_grid))

    return GradientTable(bvals, bvecs)


def write_bvals_bvecs(
    gtab: GradientTable,
    bvals_path: str | Path,
    bvecs_path: str | Path,
    *,
    ref_path: str | Path | None = None,
    frame: Literal["fsl", "world"] = "fsl",
) -> None:
    """Writes a gradient table as bvals and bvecs files.

    The bvals file holds the b-values on one row, and the bvecs file the
    directions on three rows (x, y and z), one column per volume. In the "fsl"
    frame the directions are written along the voxel axes of the image, as
    `read_bvals_bvecs` describes, so the image's grid is needed.

    Args:
        gtab: The table to write.
        bvals_path: The bvals file.
        bvecs_path: The bvecs file.
        ref_path: The image written alongside. Required for "fsl"; when given, its
            number of volumes is checked. Defaults to None.
        frame: The frame to write the bvecs in. Defaults to "fsl".

    Raises:
        ValueError: If the table does not match the image, or "fsl" is given
            without an image.
    """
    _check_frame(frame)
    if frame == "fsl" and ref_path is None:
        raise ValueError("Cannot save to fsl without a reference.")

    bvecs = gtab.bvecs
    if ref_path is not None:
        ref_path = Path(ref_path)
        ref_grid, ref_shape = load_header(ref_path)
        _check_volume_count(len(gtab.bvals), ref_shape, ref_path)
        if frame == "fsl":
            bvecs = map_directions(bvecs, np.linalg.inv(_fsl_to_world(ref_grid)))

    np.savetxt(bvals_path, gtab.bvals[None, :], fmt="%.17g")
    np.savetxt(bvecs_path, bvecs.T, fmt="%.17g")


# MRtrix splits the numbers of a row at any run of these.
_MRTRIX_SEPARATORS = re.compile(r"[ ,;\t]+")


def _load_mrtrix_rows(path: Path) -> np.ndarray:
    """Loads an MRtrix gradient file as an (N, 4) table.

    Returns:
        The rows x, y, z, b.

    Raises:
        ValueError: If the file does not exist, holds no row, or a row is not 4
            numbers; the message names the line.
    """
    if not path.is_file():
        raise ValueError(f"{path} does not exist.")
    rows: list[list[float]] = []
    with path.open(encoding="utf-8", errors="replace") as f:
        for number, line in enumerate(f, start=1):
            text = line.partition("#")[0].strip()
            if not text:
                continue
            items = [item for item in _MRTRIX_SEPARATORS.split(text) if item]
            try:
                row = [float(item) for item in items]
            except ValueError as exc:
                raise ValueError(
                    f"{path}, line {number}: {text!r} is not a row of numbers."
                ) from exc
            if len(row) != 4:
                raise ValueError(
                    f"{path}, line {number}: {len(row)} numbers; each row is x y z b."
                )
            rows.append(row)
    if not rows:
        raise ValueError(f"{path} holds no gradient row.")
    return np.array(rows, dtype=np.float64)


def read_mrtrix_gradients(
    path: str | Path, *, ref_path: str | Path | None = None
) -> GradientTable:
    """Reads an MRtrix gradient table file (`.b`, as given to MRtrix's `-grad`).

    The file has one row per volume, `x y z b`, with the direction in world
    coordinates (RAS+) and b in s/mm^2. Directions that are not unit vectors are
    normalised.

    Args:
        path: The gradient file.
        ref_path: The image the table belongs to; when given, its number of
            volumes is checked. Defaults to None.

    Returns:
        GradientTable: The table, with its vectors in world coordinates.

    Raises:
        ValueError: If the file cannot be read, a row is not 4 numbers, or the
            table does not match the image.

    References:
        Follows MRtrix3's gradient table format (`-grad`, `-export_grad_mrtrix`).
        Tournier, J.-D. et al. MRtrix3: A fast, flexible and open software
        framework for medical image processing and visualisation. NeuroImage 202,
        116137 (2019). doi:10.1016/j.neuroimage.2019.116137.
    """
    path = Path(path)
    table = _load_mrtrix_rows(path)
    if ref_path is not None:
        _, shape = load_header(ref_path)
        _check_volume_count(len(table), shape, Path(ref_path))
    return GradientTable(table[:, 3], table[:, :3])


def read_mrtrix_response(path: str | Path) -> np.ndarray:
    """Reads an MRtrix response function file, as written by `dwi2response`.

    The file has one row per shell, in increasing b-value, with the zonal
    spherical harmonic coefficients of the response for l = 0, 2, 4, ..., in the
    units of the signal. The b-values are not in the file; they are those of the
    data the response was estimated from.

    Args:
        path: The response file.

    Returns:
        np.ndarray: The coefficients, shape (n_shells, n_degrees).

    Raises:
        ValueError: If the file does not exist, holds no row, a row is not numbers,
            or the rows have different lengths.

    References:
        Follows MRtrix3's response function format (`dwi2response`, `dwi2fod`).
        Tournier, J.-D. et al. MRtrix3: A fast, flexible and open software
        framework for medical image processing and visualisation. NeuroImage 202,
        116137 (2019). doi:10.1016/j.neuroimage.2019.116137.
    """
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"{path} does not exist.")
    rows: list[list[float]] = []
    with path.open(encoding="utf-8", errors="replace") as f:
        for number, line in enumerate(f, start=1):
            text = line.partition("#")[0].strip()
            if not text:
                continue
            items = [item for item in _MRTRIX_SEPARATORS.split(text) if item]
            try:
                rows.append([float(item) for item in items])
            except ValueError as exc:
                raise ValueError(
                    f"{path}, line {number}: {text!r} is not a row of numbers."
                ) from exc
            if len(rows[-1]) != len(rows[0]):
                raise ValueError(
                    f"{path}, line {number}: {len(rows[-1])} numbers, but the first "
                    f"row has {len(rows[0])}; every shell needs the same degrees."
                )
    if not rows:
        raise ValueError(f"{path} holds no response row.")
    return np.array(rows, dtype=np.float64)


def write_mrtrix_gradients(
    gtab: GradientTable, path: str | Path, *, ref_path: str | Path | None = None
) -> None:
    """Writes a gradient table as an MRtrix gradient file (`.b`).

    The file has one row per volume, `x y z b`, with the unit direction in world
    coordinates (RAS+) and b in s/mm^2, in full precision.

    Args:
        gtab: The table to write.
        path: The gradient file; an existing file is overwritten.
        ref_path: The image written alongside; when given, its number of volumes
            is checked. Defaults to None.

    Raises:
        ValueError: If the table does not match the image.

    References:
        Follows MRtrix3's gradient table format (`-grad`, `-export_grad_mrtrix`).
        Tournier, J.-D. et al. MRtrix3: A fast, flexible and open software
        framework for medical image processing and visualisation. NeuroImage 202,
        116137 (2019). doi:10.1016/j.neuroimage.2019.116137.
    """
    if ref_path is not None:
        _, shape = load_header(ref_path)
        _check_volume_count(len(gtab.bvals), shape, Path(ref_path))
    np.savetxt(path, np.column_stack([gtab.bvecs, gtab.bvals]), fmt="%.17g")


def read_b_matrix(
    path: str | Path, *, ref_path: str | Path | None = None
) -> GradientTable:
    """Reads a b-matrix file written by `write_b_matrix`.

    The file has one row per volume, b * (gx^2, 2 gx gy, 2 gx gz, gy^2, 2 gy gz,
    gz^2), in s/mm^2 and world coordinates. B-matrix files from other programs
    may use another order, units or frame and are read wrongly. A b-matrix does
    not store the sign of a vector, so each vector is returned with its first
    non-zero component positive.

    Args:
        path: The b-matrix text file.
        ref_path: A volume whose number of volumes the table must match. Defaults
            to None, which does not check.

    Returns:
        GradientTable: The b-values (s/mm^2) and unit b-vectors (world
        coordinates, RAS+) of the file's rows.

    Raises:
        ValueError: If the file is missing, empty or not a table of numbers, if it
            does not have shape (N, 6), or if `ref_path` is given and is not 4D or its
            number of volumes differs from N.
    """
    path = Path(path)
    bmat = np.atleast_2d(_try_load(path))
    if bmat.ndim != 2 or bmat.shape[1] != 6:
        raise ValueError(f"{path} has shape {bmat.shape}; expected (N, 6).")

    bvals, bvecs = _bvals_bvecs_from_b_matrix(bmat)

    if ref_path is not None:
        _, shape = load_header(ref_path)
        _check_volume_count(len(bvals), shape, Path(ref_path))
    return GradientTable(bvals, bvecs)


def write_b_matrix(
    gtab: GradientTable, path: str | Path, *, ref_path: str | Path | None = None
) -> None:
    """Writes the table's b-matrix as a Larmor b-matrix file.

    The file has one row per volume, b * (gx^2, 2 gx gy, 2 gx gz, gy^2, 2 gy gz,
    gz^2), in s/mm^2 and world coordinates (RAS+).

    Args:
        gtab: The table to write.
        path: The text file to write.
        ref_path: A volume whose number of volumes the table must match. Defaults
            to None, which does not check.

    Raises:
        ValueError: If `ref_path` is given and is not 4D or its number of volumes
            differs from the table's.
    """
    if ref_path is not None:
        _, shape = load_header(ref_path)
        _check_volume_count(len(gtab.bvals), shape, Path(ref_path))
    np.savetxt(path, gtab.b_matrix, fmt="%.17g")


def _bvals_bvecs_from_b_matrix(b_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns the b-values and unit b-vectors of an (N, 6) b-matrix.

    The b-value is the trace, and the vector the eigenvector of the largest
    eigenvalue, with its first non-zero component made positive.
    """
    xx, xy, xz, yy, yz, zz = b_matrix.T
    bvals = xx + yy + zz
    bvals[np.abs(bvals) < _ZERO_B] = 0.0

    tensors = np.stack(
        [
            np.stack([xx, xy / 2, xz / 2], axis=-1),
            np.stack([xy / 2, yy, yz / 2], axis=-1),
            np.stack([xz / 2, yz / 2, zz], axis=-1),
        ],
        axis=1,
    )
    bvecs = np.linalg.eigh(tensors)[1][:, :, 2]
    bvecs[bvals == 0] = 0.0

    rows = np.arange(len(bvecs))
    first = np.argmax(np.abs(bvecs) > _ZERO_COMPONENT, axis=1)
    bvecs[bvecs[rows, first] < 0] *= -1
    return bvals, bvecs + 0.0  # + 0.0 turns -0.0 into 0.0
