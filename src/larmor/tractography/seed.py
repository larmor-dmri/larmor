"""Seeds of a tracking, drawn from a mask or read from a file."""

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TextIO

import numpy as np
from numpy.typing import NDArray

from larmor.types import Volume


@dataclass(frozen=True)
class Seeds:
    """The seeds of a tracking, where each streamline starts and optionally how.

    A seed may carry an index, which numbers it among all the seeds of a run and
    keys its random numbers; the tracker numbers seeds without one in the order it
    tracks them. `write_seeds` keeps the indices, so a run can be repeated from the
    seeds it wrote.

    Args:
        positions: Seed positions, shape (N, 3), in mm, world coordinates.
        directions: Direction to start along for each seed, shape (N, 3); vectors
            that are not unit length are normalised. Defaults to None, the tracker
            finding one.
        indices: The global index of each seed, shape (N,), non-negative integers.
            Defaults to None.

    Raises:
        ValueError: If a shape does not match, a value is not finite, a direction is
            zero, or the indices are not non-negative integers.
    """

    positions: NDArray[np.float64]
    directions: NDArray[np.float64] | None = None
    indices: NDArray[np.int64] | None = None

    def __post_init__(self) -> None:
        """Checks the inputs, normalises the directions and stores copies."""
        positions = np.array(self.positions, dtype=np.float64)

        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError(
                f"Expected positions of shape (N, 3), got {positions.shape}."
            )
        if not np.isfinite(positions).all():
            raise ValueError("Found infinite/nan position.")
        object.__setattr__(self, "positions", positions)

        n = positions.shape[0]
        if self.directions is not None:
            directions = _unit_directions(self.directions, n)
            object.__setattr__(self, "directions", directions)

        if self.indices is not None:
            indices = np.asarray(self.indices)
            if indices.shape != (n,):
                raise ValueError(
                    f"Expected indices of shape ({n},) to match the {n} positions, "
                    f"got {indices.shape}."
                )
            if not np.issubdtype(indices.dtype, np.integer):
                raise ValueError(f"Expected integer indices, got {indices.dtype}.")
            if np.any(indices < 0):
                raise ValueError("Found negative indices in indices array.")
            object.__setattr__(self, "indices", indices.astype(np.int64))

    def __len__(self) -> int:
        """Returns the number of seeds."""
        return len(self.positions)


def _unit_directions(values: NDArray[np.float64], n: int) -> NDArray[np.float64]:
    """Returns a checked copy of the n seed directions, normalised to unit length.

    Raises:
        ValueError: If the shape is not (n, 3), a value is not finite, or a
            direction is zero.
    """
    directions = np.array(values, dtype=np.float64)
    if directions.shape != (n, 3):
        raise ValueError(
            f"Expected directions of shape ({n}, 3) to match the {n} "
            f"positions, got {directions.shape}."
        )
    if not np.isfinite(directions).all():
        raise ValueError("Found infinite/nan direction.")
    norm = np.linalg.norm(directions, axis=1, keepdims=True)
    if np.any(norm == 0):
        raise ValueError("Found a zero direction, which cannot be followed.")
    off = np.abs(norm[:, 0] - 1) > 4 * np.finfo(np.float64).eps
    directions[off] /= norm[off]
    return directions


def _in_voxels(
    voxels: NDArray[np.integer], affine: NDArray[np.float64], rng: np.random.Generator
) -> NDArray[np.float64]:
    """Returns a random world position (K, 3) in each of the voxels (K, 3).

    `affine` is the voxel-to-world matrix of the voxels' grid.
    """
    k = voxels.shape[0]
    jitter = rng.uniform(-0.5, 0.5, size=(k, 3))
    ijk = voxels + jitter
    return ijk @ affine[:3, :3].T + affine[:3, 3]


def _mask_voxels(mask: Volume) -> NDArray[np.int64]:
    """Returns the voxel indices (K, 3) whose scaled value in `mask` is above 0.

    Raises:
        ValueError: If the mask is not 3D or has no such voxel.
    """
    if mask.data.ndim != 3:
        raise ValueError(f"Expected 3D mask, got mask of shape {mask.data.shape}")
    voxels = np.argwhere(mask.as_float(dtype=np.float64) > 0)
    if len(voxels) == 0:
        raise ValueError("Expected mask to contain values > 0, found only value <= 0.")
    return voxels


class MaskSeeds:
    """Draws seeds at random positions in a mask, for as long as the tracker asks.

    Each seed lies in a voxel of the mask chosen at random, at a random position
    inside that voxel.
    """

    def __init__(self, mask: Volume) -> None:
        """Finds the voxels of the mask to seed from.

        Args:
            mask: A 3D image; the voxels whose scaled value (data * slope +
                intercept) is above 0 are seeded. A seed whose voxel of
                the tracking mask is below 0.5 does not start a streamline.
        """
        self.voxels = _mask_voxels(mask)
        self.affine = mask.affine

    def draw(self, n: int, rng: np.random.Generator) -> Seeds:
        """Draws n seeds from the mask.

        Args:
            n: The number of seeds.
            rng: The random generator to draw with.

        Returns:
            Seeds: The seeds, positions in world coordinates, each in a voxel of the
            mask drawn at random.
        """
        pos = rng.integers(0, len(self.voxels), n)
        return Seeds(_in_voxels(self.voxels[pos], self.affine, rng))


# Second key of the random stream seeds_per_voxel draws from when given a run
# seed, which keeps its seeds apart from the tracker's batches.
PER_VOXEL_STREAM = 2**32


def seeds_per_voxel(mask: Volume, n: int, rng: np.random.Generator | int) -> Seeds:
    """Draws n seeds per voxel from the given mask.

    Args:
        mask: The voxels to seed: those whose scaled value (data * slope +
            intercept) is above 0.
        n: Seeds per voxel.
        rng: A random generator, or a run seed in [0, 2^64). With a run seed,
            the same run seed gives the same seeds, which differ from the random
            numbers the tracker draws from that seed.

    Returns:
        Seeds: The seeds, n in each voxel of the mask, at random positions in it.

    Raises:
        ValueError: If n is not positive, or the mask is not 3D or has no voxel
            above 0.
    """
    if n <= 0:
        raise ValueError(f"Cannot draw {n} seeds.")
    if isinstance(rng, int):
        rng = np.random.default_rng([rng, PER_VOXEL_STREAM])
    voxels = _mask_voxels(mask)
    repeated = np.repeat(voxels, n, axis=0)
    return Seeds(_in_voxels(repeated, mask.affine, rng))


def write_seeds(
    seeds: Seeds, path: str | Path | TextIO, *, header: bool = True
) -> None:
    """Writes seeds as text, one row per seed, for `read_seeds`.

    The columns are [index] x y z [dx dy dz], with the index and the directions
    only if the seeds have them. The header line, starting with '#', names the
    columns. The values are written exactly, so seeds read back track exactly as
    before.

    Args:
        seeds: The seeds to write.
        path: The text file, whose content is overwritten, or a file open for
            writing, to which the rows are added.
        header: Whether to write the header line first. Defaults to True.
    """
    names = ["x", "y", "z"]
    if seeds.indices is not None:
        names = ["index", *names]
    if seeds.directions is not None:
        names += ["dx", "dy", "dz"]

    # One field per column, so that indices above 2^53 are not rounded.
    rows = np.empty(
        len(seeds), dtype=[(c, np.int64 if c == "index" else np.float64) for c in names]
    )
    if seeds.indices is not None:
        rows["index"] = seeds.indices
    for i, c in enumerate("xyz"):
        rows[c] = seeds.positions[:, i]
        if seeds.directions is not None:
            rows["d" + c] = seeds.directions[:, i]

    fmt = ["%d" if c == "index" else "%.17g" for c in names]
    np.savetxt(path, rows, fmt=fmt, header=" ".join(names) if header else "")


def read_seeds(path: str | Path) -> Seeds:
    """Reads seeds written by `write_seeds`.

    Args:
        path: The text file.

    Returns:
        Seeds: The seeds, in the file's order; empty if the file has only its
        header.

    Raises:
        ValueError: If the header is missing or names other columns, or the rows
            do not have as many columns as the header.
    """
    with open(path, encoding="utf-8") as f:
        header = f.readline().lstrip("#").split()

    valid = [
        ["x", "y", "z"],
        ["index", "x", "y", "z"],
        ["x", "y", "z", "dx", "dy", "dz"],
        ["index", "x", "y", "z", "dx", "dy", "dz"],
    ]
    if header not in valid:
        raise ValueError(
            f"{path} does not start with a seed header ('# [index] x y z [dx dy dz]'),"
            f" got {header}."
        )
    has_index = header[0] == "index"
    has_directions = "dx" in header

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*input contained no data")
        data = np.loadtxt(path, ndmin=2)
        if data.size == 0:
            return Seeds(
                np.empty((0, 3)),
                np.empty((0, 3)) if has_directions else None,
                np.empty(0, np.int64) if has_index else None,
            )
        if data.shape[1] != len(header):
            raise ValueError(
                f"{path}: expected {len(header)} columns from the header, got "
                f"{data.shape[1]}."
            )
        indices = (
            np.loadtxt(path, usecols=0, dtype=np.int64, ndmin=1) if has_index else None
        )

    first = 1 if has_index else 0
    return Seeds(
        positions=data[:, first : first + 3],
        directions=data[:, first + 3 : first + 6] if has_directions else None,
        indices=indices,
    )


class SeedStrategy(Protocol):
    """A source of seeds that the tracker draws from batch by batch.

    Any object with this `draw` method can be passed as the seeds of
    `Tracker.track`; `MaskSeeds` is one.
    """

    def draw(self, n: int, rng: np.random.Generator) -> Seeds:
        """Draws n seeds.

        Args:
            n: The number of seeds.
            rng: The random generator to draw with.

        Returns:
            Seeds: The n seeds.
        """
        ...
