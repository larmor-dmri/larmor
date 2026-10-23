"""The tractogram type."""

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import overload

import numpy as np

from larmor.types._arrays import check_data
from larmor.types.grid import Grid


@dataclass(eq=False)
class Tractogram:
    """A set of streamlines, stored as one flat array of points and their offsets.

    Streamline s is `points[offsets[s]:offsets[s + 1]]`. Index a tractogram with an
    integer to get one streamline's points, or with a slice, a list or array of
    indices, or a boolean mask to get a new tractogram with those streamlines.
    Tractograms compare by identity (`==` is `is`); compare the arrays to compare
    contents.

    Args:
        points: The (P, 3) points of all streamlines, in world coordinates (RAS+)
            in mm, stored as float32.
        offsets: The (S + 1,) index where each streamline starts, with P last,
            stored as int64.
        point_data: Named values per point, each with first axis P, such as an FA
            sampled along the streamlines. Empty by default.
        streamline_data: Named values per streamline, each with first axis S, such
            as a length or a cluster label. Empty by default.
        reference: The voxel grid of the image the tractogram was made on, if
            known. Defaults to None.

    Raises:
        ValueError: If `points` is not (P, 3), `offsets` are not integers that start
            at 0, never decrease and end at P, or a `point_data` or
            `streamline_data` value does not have P or S entries along its first
            axis.
    """

    points: np.ndarray
    offsets: np.ndarray
    point_data: dict[str, np.ndarray] = field(default_factory=dict)
    streamline_data: dict[str, np.ndarray] = field(default_factory=dict)
    reference: Grid | None = None

    def __post_init__(self) -> None:
        """Checks the shapes and the offsets, and converts the arrays to their types."""
        self.points = np.asarray(self.points, dtype=np.float32)
        offsets = np.asarray(self.offsets)
        if offsets.size > 0 and not np.issubdtype(offsets.dtype, np.integer):
            raise ValueError(f"Offsets must be integers, got {offsets.dtype}.")
        self.offsets = offsets.astype(np.int64)

        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError(
                f"Expected points of shape (P, 3), got {self.points.shape}."
            )
        if self.offsets.ndim != 1 or len(self.offsets) == 0:
            raise ValueError(
                f"Expected offsets of shape (S + 1,), got {self.offsets.shape}."
            )

        if self.offsets[0] != 0:
            raise ValueError(f"Offsets must start at 0, got {self.offsets[0]}.")
        if (np.diff(self.offsets) < 0).any():
            raise ValueError("Offsets must never decrease.")
        if self.offsets[-1] != len(self.points):
            raise ValueError(
                f"The last offset must equal the number of points "
                f"({len(self.points)}), got {self.offsets[-1]}."
            )

        self.point_data = check_data(self.point_data, "point_data", self.num_points)
        self.streamline_data = check_data(
            self.streamline_data, "streamline_data", self.num_streamlines
        )

    @property
    def num_streamlines(self) -> int:
        """The number of streamlines, S."""
        return len(self.offsets) - 1

    @property
    def num_points(self) -> int:
        """The number of points of all streamlines together, P."""
        return len(self.points)

    def subsample(self, n: int, *, seed: int | None = None) -> Tractogram:
        """Returns a new tractogram with `n` streamlines chosen at random.

        Args:
            n: The number of streamlines to keep, at most S.
            seed: Seed for the random choice. Defaults to None, a different subset
                each time.

        Returns:
            Tractogram: The selected streamlines with their data and the same
                reference.

        Raises:
            ValueError: If `n` is negative or larger than the number of streamlines.
        """
        return self._take(self.subsample_indices(n, seed=seed))

    def subsample_indices(self, n: int, *, seed: int | None = None) -> np.ndarray:
        """Returns the indices of the streamlines `subsample` keeps for `n` and `seed`.

        Args:
            n: The number of streamlines to keep, at most S.
            seed: Seed for the random choice. Defaults to None, a different subset
                each time.

        Returns:
            np.ndarray: (n,) int64, the indices of the kept streamlines, ascending.

        Raises:
            ValueError: If `n` is negative or larger than the number of streamlines.
        """
        if not 0 <= n <= self.num_streamlines:
            raise ValueError(
                f"Cannot choose {n} streamlines from a tractogram with "
                f"{self.num_streamlines}."
            )
        rng = np.random.default_rng(seed)
        chosen = rng.choice(self.num_streamlines, size=n, replace=False)
        return np.sort(chosen).astype(np.int64)

    @overload
    def __getitem__(self, key: int | np.integer) -> np.ndarray: ...
    @overload
    def __getitem__(
        self,
        key: slice | range | list[int] | list[bool] | tuple[int, ...] | np.ndarray,
    ) -> Tractogram: ...
    def __getitem__(
        self,
        key: int
        | np.integer
        | slice
        | range
        | list[int]
        | list[bool]
        | tuple[int, ...]
        | np.ndarray,
    ) -> np.ndarray | Tractogram:
        """Returns one streamline's points, or a new tractogram with a selection.

        Args:
            key: An integer (negative counts from the end) for one streamline; a
                slice, a range, a list, tuple or 1D array of integers (in any order,
                repeats allowed), or a boolean mask with one entry per streamline
                for a selection.

        Returns:
            np.ndarray | Tractogram: For an integer, the (n, 3) points of that
                streamline, a view into `points`. Otherwise a new tractogram with
                the selected streamlines in the given order, their data and the
                same reference.

        Raises:
            IndexError: If an index is out of range, a mask has the wrong length,
                or the key has an unsupported type.
        """
        # A bool is an int, but t[True] is almost certainly a mistake.
        if isinstance(key, (int, np.integer)) and not isinstance(key, (bool, np.bool)):
            s = int(key)
            if s < 0:
                s += self.num_streamlines
            if not 0 <= s < self.num_streamlines:
                raise IndexError(
                    f"Index {key} is out of range for {self.num_streamlines} "
                    "streamlines."
                )
            points: np.ndarray = self.points[self.offsets[s] : self.offsets[s + 1]]
            return points

        if isinstance(key, slice):
            return self._take(np.arange(*key.indices(self.num_streamlines)))

        if isinstance(key, (list, tuple, range, np.ndarray)):
            return self._take(self._selected_indices(np.asarray(key)))

        raise IndexError(f"Cannot index a tractogram with a {type(key).__name__}.")

    def _selected_indices(self, array: np.ndarray) -> np.ndarray:
        """Returns the streamline indices that a boolean mask or integer array selects.

        Negative indices count from the end.

        Returns:
            np.ndarray: The indices, non-negative, in the given order.

        Raises:
            IndexError: If a mask has the wrong length, an index is out of range,
                or the array is neither boolean nor integer.
        """
        if array.dtype == bool:
            if array.shape != (self.num_streamlines,):
                raise IndexError(
                    "A boolean mask must have one entry per streamline "
                    f"({self.num_streamlines}), got shape {array.shape}."
                )
            return np.flatnonzero(array)

        # An empty list has dtype float, but selects nothing.
        if np.issubdtype(array.dtype, np.integer) or array.size == 0:
            indices = array.astype(np.int64)
            if indices.ndim != 1:
                raise IndexError(
                    f"Expected a 1D list of indices, got shape {indices.shape}."
                )
            n = self.num_streamlines
            if ((indices < -n) | (indices >= n)).any():
                raise IndexError(f"Index out of range for {n} streamlines.")
            indices[indices < 0] += n
            return indices

        raise IndexError(
            f"Cannot index a tractogram with values of type {array.dtype}."
        )

    def _take(self, indices: np.ndarray) -> Tractogram:
        """Returns a new tractogram with the streamlines `indices`, in that order.

        Args:
            indices: The indices of the streamlines to keep.

        Returns:
            The new tractogram.
        """
        starts = self.offsets[indices]
        lengths = np.diff(self.offsets)[indices]
        new_offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
        # The position in `points` of each point of the result.
        gather = np.repeat(starts - new_offsets[:-1], lengths) + np.arange(
            new_offsets[-1]
        )
        return Tractogram(
            points=self.points[gather],
            offsets=new_offsets,
            point_data={
                name: values[gather] for name, values in self.point_data.items()
            },
            streamline_data={
                name: values[indices] for name, values in self.streamline_data.items()
            },
            reference=self.reference,
        )

    def __len__(self) -> int:
        """The number of streamlines, S."""
        return self.num_streamlines

    def __iter__(self) -> Iterator[np.ndarray]:
        """Yields the points of each streamline in turn, each (n, 3) in mm (RAS+)."""
        # Slices `points` directly, without the key checks of self[s] per streamline.
        for start, stop in zip(self.offsets[:-1], self.offsets[1:], strict=True):
            yield self.points[start:stop]
