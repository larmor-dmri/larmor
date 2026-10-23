"""The image volume type."""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from larmor.types._affine import (
    check_affine,
    flip_affine,
    permute_affine,
    reorientation,
)
from larmor.types._arrays import read_only
from larmor.types.grid import Grid


@dataclass(eq=False)
class Volume:
    """An image on a voxel grid, such as a scan, a DWI series or a computed map.

    A volume holds the stored values, the affine that places its voxels in the
    world, and the scaling that turns stored values into real ones.

    `data` and `affine` are read only. The affine is copied but the data is not, so
    the caller must not change the array it passed in. The checks run only on
    construction, so assigning new values to the fields afterwards is not
    supported. Volumes compare by identity (`==` is `is`).

    Args:
        data: The stored values, unscaled, as (X, Y, Z) or (X, Y, Z, N), where N is
            the series, such as the DWI volumes or FOD coefficients.
        affine: The (4, 4) matrix from voxel indices to world coordinates (RAS+)
            in mm, with last row [0, 0, 0, 1].
        slope: Scaling, value = data * slope + intercept. Defaults to 1.0; 0, NaN
            or infinity means no scaling.
        intercept: Scaling offset. Defaults to 0.0.
        xform_codes: The NIfTI sform and qform codes, which name the space the
            affine maps to (1 scanner, 2 aligned to another image, 3 Talairach,
            4 MNI 152, 5 another template). Defaults to (1, 1).

    Raises:
        ValueError: If `data` is not 3D or 4D, `affine` is not a finite 4x4 matrix
            with last row [0, 0, 0, 1] and non-zero axes, or the slope is used but
            the intercept is not finite.
    """

    data: np.ndarray
    affine: np.ndarray
    slope: float = 1.0
    intercept: float = 0.0
    xform_codes: tuple[int, int] = (1, 1)

    def __post_init__(self) -> None:
        """Checks the inputs and makes the arrays read-only."""
        num_dims = len(self.data.shape)
        if num_dims not in range(3, 5):
            raise ValueError(f"Volume only supports 3D or 4D volumes, got {num_dims}D.")

        self.affine = check_affine(self.affine)

        # A slope of 0, NaN or infinity means no scaling, as in nibabel.
        if not math.isfinite(self.slope) or self.slope == 0.0:
            self.slope = 1.0
            self.intercept = 0.0
        elif not math.isfinite(self.intercept):
            raise ValueError(f"The intercept must be finite, got {self.intercept}.")

        self.data = read_only(self.data)
        self.affine = read_only(self.affine)

    @property
    def voxel_sizes(self) -> np.ndarray:
        """The voxel size along each voxel axis, as a (3,) array in mm."""
        sizes: np.ndarray = np.linalg.norm(self.affine[:3, :3], axis=0)
        return sizes

    @property
    def orientation(self) -> tuple[str, str, str]:
        """Closest RAS+ direction of each voxel axis, such as ("R", "A", "S")."""
        return self.grid.orientation

    @property
    def grid(self) -> Grid:
        """The volume's voxel grid, with the spatial shape (X, Y, Z) and the affine."""
        x, y, z = self.data.shape[:3]
        return Grid(shape=(x, y, z), affine=self.affine, xform_codes=self.xform_codes)

    def as_float(self, *, dtype: npt.DTypeLike = np.float32) -> np.ndarray:
        """Returns the scaled values, data * slope + intercept, as a float array.

        When the data already has `dtype` and there is no scaling, this returns
        `data` itself, read only. The scaling can overflow in a small `dtype`.

        Args:
            dtype: A floating-point type. Defaults to float32.

        Returns:
            The scaled values, with the shape of `data`.

        Raises:
            ValueError: If `dtype` is not a floating-point type, or if the data
                is complex.
        """
        dtype = np.dtype(dtype)
        if not np.issubdtype(dtype, np.floating):
            raise ValueError(f"as_float needs a floating-point type, got {dtype}.")
        # A cast would keep only the real part, so complex data is refused.
        if np.iscomplexobj(self.data):
            raise ValueError(
                "The volume holds complex data, which cannot be turned into real "
                "values; pass the magnitude instead (for example np.abs(data))."
            )

        if dtype == self.data.dtype and self.slope == 1.0 and self.intercept == 0.0:
            return self.data

        out = self.data.astype(dtype)
        out *= self.slope
        out += self.intercept

        return out

    def with_data(self, data: np.ndarray) -> Volume:
        """Returns a new volume on the same grid and in the same space, unscaled.

        Args:
            data: The new values, (X, Y, Z) or (X, Y, Z, N), on the same
                spatial grid.

        Returns:
            Volume: The new volume, with slope 1 and intercept 0.

        Raises:
            ValueError: If the spatial shape of `data` differs from this volume's.
        """
        if data.shape[:3] != self.data.shape[:3]:
            raise ValueError(
                f"The new data's spatial shape {data.shape[:3]} differs from the "
                f"volume's {self.data.shape[:3]}."
            )
        return Volume(
            data=data,
            affine=self.affine,
            slope=1,
            intercept=0,
            xform_codes=self.xform_codes,
        )

    def flip_axis(self, axis: int) -> None:
        """Reverses a spatial voxel axis in place, keeping every voxel's world position.

        Args:
            axis: The voxel axis to flip, 0, 1 or 2.

        Raises:
            ValueError: If `axis` is not 0, 1 or 2.
        """
        if axis < 0 or axis > 2:
            raise ValueError(f"Cannot flip axis {axis}.")

        self.affine = read_only(flip_affine(self.affine, axis, self.data.shape[axis]))
        self.data = read_only(np.flip(self.data, axis=axis))

    def permute_axes(self, order: tuple[int, int, int]) -> None:
        """Reorders the spatial voxel axes in place, keeping world positions.

        Args:
            order: A permutation of (0, 1, 2); new axis j is old axis order[j].

        Raises:
            ValueError: If `order` is not a permutation of (0, 1, 2).
        """
        if sorted(list(order)) != [0, 1, 2]:
            raise ValueError(f"Cannot permute axis {order}.")

        # The series axis of a 4D volume, if any, stays last.
        series = tuple(range(3, self.data.ndim))
        self.data = read_only(np.permute_dims(self.data, axes=(*order, *series)))
        self.affine = read_only(permute_affine(self.affine, order))

    def to_orientation(self, codes: str | Sequence[str]) -> None:
        """Reorients the volume in place to the given orientation, such as "RAS".

        Only permutes and flips the axes, without resampling, so every voxel keeps
        its world position.

        Args:
            codes: The orientation to reach, with one of R/L, A/P and S/I each.

        Raises:
            ValueError: If `codes` is not a valid orientation.
        """
        order, flips = reorientation(self.orientation, codes)
        self.permute_axes(order)
        for axis, flip in enumerate(flips):
            if flip:
                self.flip_axis(axis)

    def to_closest_canonical(self) -> None:
        """Reorients the volume in place to its closest RAS orientation."""
        self.to_orientation(codes="RAS")
