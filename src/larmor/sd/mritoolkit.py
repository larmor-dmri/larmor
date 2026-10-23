"""Readers of MRIToolkit's GRL and mFOD results in MATLAB files, as SH FODs."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from larmor.io import (
    MatlabFile,
    read_matlab,
    read_mritoolkit_frame,
    read_mritoolkit_volume,
)
from larmor.sd.sh import build_basis_matrix, sh_layout
from larmor.sd.sphere import DIRS_300, get_theta_phi
from larmor.types import Grid, Volume

# Voxels per block of the SH fit, to bound its memory.
_BLOCK = 65_536


@dataclass(frozen=True)
class MRIToolkitFOD:
    """Holds the FODs and fractions of an MRIToolkit fit, read by `read_mritoolkit_fod`.

    The FODs are converted so that they can be used like the FODs of an `SDFit`.

    Attributes:
        fods: One FOD per anisotropic response, in the order the responses were
            added, as SH coefficients (X, Y, Z, C), float32, in the MRtrix
            `tournier` basis and world coordinates. They are scaled as Larmor's
            FODs, 300 / (4 pi) times the amplitudes the fit gives on its 300
            directions.
        tissue_fractions: One signal fraction per response, (X, Y, Z, K), in the
            order they were added; the file does not name them.
        rss: The residual sum of squares, (X, Y, Z), or None if not stored.
    """

    fods: list[Volume]
    tissue_fractions: Volume
    rss: Volume | None


def read_mritoolkit_fod(
    source: str | Path | MatlabFile,
    reference: Grid | str | Path | MatlabFile,
    *,
    variable: str | None = None,
    lmax: int = 8,
) -> MRIToolkitFOD:
    """Reads the result of an MRIToolkit GRL or mFOD fit from a MATLAB file.

    The result is the struct that MRTTrack's PerformDeconv returns, with the fields
    `FOD`, `fractions` and `RSS`, saved in a `.mat` file. It stores each FOD as 300
    amplitudes on a fixed set of directions, which are fitted with SH up to `lmax`
    in world coordinates. The FODs are scaled as Larmor's (300 / (4 pi) times the
    amplitudes), so they compare with Larmor's own fits of the same data.
    MRIToolkit's exports multiply by its kernel's scale instead, which the file does
    not store, so the two differ by one factor.

    Args:
        source: The `.mat` file the struct was saved in, or that file already
            opened with `read_matlab`.
        reference: The grid the fit was made on, as a Grid, the MRIToolkit DTI
            or DKI fit (`.mat`) it came from, or a NIfTI or MIF file of that
            image. The struct stores no grid.
        variable: The struct's name in the file. Defaults to None, the only
            struct variable with a field `FOD`.
        lmax: The highest SH degree of the FODs, even. Defaults to 8.

    Returns:
        MRIToolkitFOD: The FODs, the fractions and the residuals on the reference
        grid.

    Raises:
        ValueError: If the file has no such struct, or its arrays do not fit the
            reference grid.
    """
    f = source if isinstance(source, MatlabFile) else read_matlab(source)
    name = variable if variable is not None else _fod_variable(f)
    fields = {e.path for e in f.entries()}
    if f"{name}.FOD" not in fields or f"{name}.fractions" not in fields:
        raise ValueError(f"{f.path}: {name} has no FOD and fractions fields.")
    amplitudes = read_mritoolkit_volume(f, f"{name}.FOD", reference=reference)
    grid = Grid(amplitudes.data.shape[:3], amplitudes.affine)
    n = len(DIRS_300)
    if amplitudes.data.shape[-1] % n != 0:
        raise ValueError(
            f"{f.path}: {name}.FOD has {amplitudes.data.shape[-1]} values per voxel, "
            f"not a multiple of the {n} directions."
        )
    directions = DIRS_300 @ read_mritoolkit_frame(grid).T
    theta, phi = get_theta_phi(directions)
    # n / (4 pi) turns the weights of n directions into a density on the sphere.
    projection = np.linalg.pinv(build_basis_matrix(theta, phi, lmax)) * (
        n / (4.0 * np.pi)
    )
    fods = [
        Volume(_fit(amplitudes.data[..., s : s + n], projection), grid.affine.copy())
        for s in range(0, amplitudes.data.shape[-1], n)
    ]
    fractions = read_mritoolkit_volume(f, f"{name}.fractions", reference=grid)
    rss = None
    if f"{name}.RSS" in fields:
        rss = read_mritoolkit_volume(f, f"{name}.RSS", reference=grid)
    return MRIToolkitFOD(fods, fractions, rss)


def read_mritoolkit_sh(
    source: str | Path | MatlabFile,
    variable: str,
    reference: Grid | str | Path | MatlabFile | None = None,
    *,
    lmax: int = 8,
) -> Volume:
    """Reads one FOD image of a MATLAB file in MRIToolkit's layout, as world SH.

    The image's first three axes are MRIToolkit's, as `read_mritoolkit_volume` reads
    them. Its directions are in MRIToolkit's frame, the NIfTI voxel axes j, i, k,
    and are turned into world coordinates. The last axis holds either 300 amplitudes
    on MRIToolkit's 300 directions, fitted as by `read_mritoolkit_fod` (with degree
    `lmax` and Larmor's scale), or SH coefficients in the MRtrix `tournier` basis
    (even degrees, or all degrees for an asymmetric FOD), which are rotated exactly.

    Args:
        source: The `.mat` file, or that file already opened with `read_matlab`.
        variable: The image, as `MatlabFile.read` takes it.
        reference: The grid, as for `read_mritoolkit_volume`. Defaults to None,
            which reads the grid from the `hdr` the file holds.
        lmax: The SH degree of FODs given as amplitudes. Defaults to 8.

    Returns:
        Volume: SH coefficients (X, Y, Z, C), float32, in the MRtrix basis and
        world coordinates.

    Raises:
        ValueError: If the last axis holds neither 300 amplitudes nor an SH layout.
    """
    image = read_mritoolkit_volume(source, variable, reference=reference)
    grid = Grid(image.data.shape[:3], image.affine)
    frame = read_mritoolkit_frame(grid)
    n = image.data.shape[-1] if image.data.ndim == 4 else 1
    if n == len(DIRS_300):
        theta, phi = get_theta_phi(DIRS_300 @ frame.T)
        projection = np.linalg.pinv(build_basis_matrix(theta, phi, lmax)) * (
            n / (4.0 * np.pi)
        )
        return Volume(_fit(image.data, projection), grid.affine.copy())
    try:
        order, full = sh_layout(n)
    except ValueError as e:
        raise ValueError(
            f"{variable}: {n} values per voxel are neither {len(DIRS_300)} "
            "amplitudes nor SH coefficients."
        ) from e
    sphere = np.vstack([DIRS_300, -DIRS_300])
    stored = build_basis_matrix(*get_theta_phi(sphere), order, full=full)
    world = build_basis_matrix(*get_theta_phi(sphere @ frame.T), order, full=full)
    projection = stored.T @ np.linalg.pinv(world).T
    data = image.data.reshape(*image.data.shape[:3], n)
    return Volume(_fit(data, projection.T), grid.affine.copy())


def read_mritoolkit_fod_count(
    source: str | Path | MatlabFile, *, variable: str | None = None
) -> int:
    """Returns the number of FODs (anisotropic responses) a GRL or mFOD result holds.

    Args:
        source: The `.mat` file of a GRL or mFOD result, or the file opened with
            `read_matlab`.
        variable: The struct holding the FOD field. Defaults to None, the only
            struct of the file with one.

    Returns:
        The length of the FOD field's last axis divided by 300, at least 1.

    Raises:
        ValueError: If `variable` is None and the file does not hold exactly one struct
            with an FOD field.
    """
    f = source if isinstance(source, MatlabFile) else read_matlab(source)
    name = variable if variable is not None else _fod_variable(f)
    return max(f.entry(f"{name}.FOD").shape[-1] // len(DIRS_300), 1)


def _fod_variable(f: MatlabFile) -> str:
    """Returns the name of the only top-level struct of a file with a field FOD.

    Raises:
        ValueError: If there is not exactly one such struct.
    """
    found = [
        e.path.removesuffix(".FOD")
        for e in f.entries()
        if e.path.endswith(".FOD") and e.path.count(".") == 1
    ]
    if len(found) != 1:
        raise ValueError(
            f"{f.path}: {len(found)} structs with an FOD field ({', '.join(found)}); "
            "name the one to read."
        )
    return found[0]


def _fit(amplitudes: np.ndarray, projection: np.ndarray) -> np.ndarray:
    """Returns the SH coefficients, float32, of amplitudes (X, Y, Z, n), by blocks."""
    flat = amplitudes.reshape(-1, amplitudes.shape[-1])
    out = np.zeros((len(flat), projection.shape[0]), dtype=np.float32)
    for start in range(0, len(flat), _BLOCK):
        block = np.nan_to_num(flat[start : start + _BLOCK].astype(np.float64))
        out[start : start + _BLOCK] = block @ projection.T
    return out.reshape(*amplitudes.shape[:3], -1)
