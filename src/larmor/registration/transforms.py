"""Spatial transforms between two spaces, as maps of points in world mm."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import Protocol

import numpy as np

from larmor._log import LarmorLogger
from larmor.registration.sampling import read_only, sample_field
from larmor.types import Grid


class Transform(Protocol):
    """A map of points from one space to another, in world coordinates (RAS+) in mm.

    Every transform Larmor reads maps points of the reference space (the grid an
    image is resampled onto) to the input space (where the image's data lives).
    Moving points such as streamlines from the input space into the reference
    space therefore needs the inverse.
    """

    def map_points(self, points: np.ndarray) -> np.ndarray:
        """Maps points to the other space.

        Args:
            points: The points, shape (N, 3), in mm, world coordinates (RAS+).

        Returns:
            The mapped points, shape (N, 3).
        """
        ...

    def jacobians(self, points: np.ndarray) -> np.ndarray:
        """Returns the (N, 3, 3) derivative of `map_points` at each of the points.

        Args:
            points: The points, shape (N, 3), in mm, world coordinates (RAS+).

        Returns:
            The Jacobian matrices, shape (N, 3, 3).
        """
        ...

    def inverse(self) -> Transform:
        """Returns the transform that maps the points back.

        Returns:
            The inverse transform.
        """
        ...


@dataclass(frozen=True, eq=False)
class AffineTransform:
    """An affine transform, which maps p to A p + t.

    Attributes:
        matrix: The (4, 4) float64 matrix holding A and t, read only, with last row
            [0, 0, 0, 1].
    """

    matrix: np.ndarray

    def __post_init__(self) -> None:
        """Checks the matrix and makes it read-only.

        Raises:
            ValueError: If `matrix` is not a finite, invertible 4x4 matrix with last
                row [0, 0, 0, 1].
        """
        matrix = np.array(self.matrix, dtype=np.float64)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError(
                f"An affine transform needs a finite 4x4 matrix, got {matrix.shape}."
            )
        if not np.allclose(matrix[3], [0, 0, 0, 1]):
            raise ValueError(f"The last row must be [0, 0, 0, 1], got {matrix[3]}.")
        if abs(np.linalg.det(matrix[:3, :3])) < 1e-12:
            raise ValueError("The affine transform is singular.")
        object.__setattr__(self, "matrix", read_only(matrix))

    def map_points(self, points: np.ndarray) -> np.ndarray:
        """Maps points to the other space.

        Args:
            points: The points, shape (N, 3), in mm, world coordinates (RAS+).

        Returns:
            The mapped points, shape (N, 3).
        """
        points = np.asarray(points, dtype=np.float64)
        mapped: np.ndarray = points @ self.matrix[:3, :3].T + self.matrix[:3, 3]
        return mapped

    def jacobians(self, points: np.ndarray) -> np.ndarray:
        """Returns the (N, 3, 3) derivative at the points, A everywhere.

        Args:
            points: The points, shape (N, 3), in mm, world coordinates (RAS+).

        Returns:
            The Jacobian matrices, shape (N, 3, 3).
        """
        return np.broadcast_to(self.matrix[:3, :3], (len(points), 3, 3)).copy()

    def inverse(self) -> AffineTransform:
        """Returns the exact inverse.

        Returns:
            The inverse transform.
        """
        return AffineTransform(np.linalg.inv(self.matrix))


@dataclass(frozen=True, eq=False)
class DisplacementField:
    """A non-linear transform, which maps p to p + d(p) with d sampled on a grid.

    d is interpolated trilinearly between the voxel centres. A point up to half a
    voxel beyond the outer centres takes the edge values, and a point further out
    is not moved.

    Attributes:
        grid: The grid the displacements are sampled on.
        displacements: The (X, Y, Z, 3) float64 displacement d at each voxel
            centre, read only, in world coordinates (RAS+) in mm. NaN where the
            field is undefined.
    """

    grid: Grid
    displacements: np.ndarray

    def __post_init__(self) -> None:
        """Checks the shape and makes the displacements read-only.

        Raises:
            ValueError: If `displacements` is not (X, Y, Z, 3) on the grid's shape.
        """
        displacements = np.asarray(self.displacements, dtype=np.float64)
        if displacements.shape != (*self.grid.shape, 3):
            raise ValueError(
                f"Displacements must be {(*self.grid.shape, 3)} for the grid, got "
                f"{displacements.shape}."
            )
        object.__setattr__(self, "displacements", read_only(displacements))

    @cached_property
    def _to_voxels(self) -> np.ndarray:
        """The 4x4 matrix from world coordinates to voxel indices of the grid."""
        return np.linalg.inv(self.grid.affine)

    @cached_property
    def _gradients(self) -> np.ndarray:
        """The (X, Y, Z, 9) derivative of d in world mm at each voxel, row by row."""
        by_voxel = np.stack(
            [
                np.gradient(self.displacements, axis=axis)
                if self.grid.shape[axis] > 1
                else np.zeros_like(self.displacements)
                for axis in range(3)
            ],
            axis=-1,
        )  # (X, Y, Z, 3 components, 3 voxel axes)
        world: np.ndarray = by_voxel @ self._to_voxels[:3, :3]
        return world.reshape(*self.grid.shape, 9)

    def _voxels(self, points: np.ndarray) -> np.ndarray:
        """Returns the voxel coordinates on the grid of world `points` (N, 3)."""
        voxels: np.ndarray = points @ self._to_voxels[:3, :3].T + self._to_voxels[:3, 3]
        return voxels

    def map_points(self, points: np.ndarray) -> np.ndarray:
        """Maps points to the other space.

        Args:
            points: The points, shape (N, 3), in mm, world coordinates (RAS+).

        Returns:
            The mapped points, shape (N, 3).
        """
        points = np.asarray(points, dtype=np.float64)
        moved: np.ndarray = points + sample_field(
            self.displacements, self._voxels(points)
        )
        return moved

    def jacobians(self, points: np.ndarray) -> np.ndarray:
        """Returns the (N, 3, 3) derivative at the points, I + grad d.

        Args:
            points: The points, shape (N, 3), in mm, world coordinates (RAS+).

        Returns:
            The Jacobian matrices, shape (N, 3, 3).
        """
        points = np.asarray(points, dtype=np.float64)
        gradients = sample_field(self._gradients, self._voxels(points))
        result: np.ndarray = np.eye(3) + gradients.reshape(-1, 3, 3)
        return result

    def inverse(self) -> InverseDisplacementField:
        """Returns the inverse, computed point by point when it is used.

        Returns:
            InverseDisplacementField: The inverse field.
        """
        return InverseDisplacementField(self)


@dataclass(frozen=True, eq=False)
class InverseDisplacementField:
    """The inverse of a displacement field, found by iteration at each point.

    The iteration converges where the field is invertible and smooth
    (`|grad d| < 1`), as fields from diffeomorphic registration are. A warning
    reports the points where it does not converge within `max_iterations`.

    Attributes:
        field: The field being inverted.
        tolerance: The largest residual `|x + d(x) - y|` accepted, in mm.
        max_iterations: The most iterations per point.
    """

    field: DisplacementField
    tolerance: float = 1e-4
    max_iterations: int = 200

    def map_points(self, points: np.ndarray) -> np.ndarray:
        """Maps points to the other space.

        Args:
            points: The points, shape (N, 3), in mm, world coordinates (RAS+).

        Returns:
            The mapped points, shape (N, 3).
        """
        target = np.asarray(points, dtype=np.float64)
        x = target.copy()
        active = np.arange(len(target))
        for _ in range(self.max_iterations):
            residual = self.field.map_points(x[active]) - target[active]
            converged = np.linalg.norm(residual, axis=1) < self.tolerance
            # NaN residuals (an undefined field) stop at once, with NaN points.
            undefined = ~np.isfinite(residual).all(axis=1)
            x[active[undefined]] = np.nan
            keep = ~(converged | undefined)
            x[active[keep]] -= residual[keep]
            active = active[keep]
            if len(active) == 0:
                break
        if len(active) > 0:
            residual = self.field.map_points(x[active]) - target[active]
            LarmorLogger.warning(
                f"The inverse of the displacement field did not converge at "
                f"{len(active)} of {len(target)} points within "
                f"{self.max_iterations} iterations (largest residual "
                f"{np.linalg.norm(residual, axis=1).max():.3g} mm)."
            )
        return x

    def jacobians(self, points: np.ndarray) -> np.ndarray:
        """Returns the (N, 3, 3) derivative at the points, the inverse of I + grad d.

        Args:
            points: The points, shape (N, 3), in mm, world coordinates (RAS+).

        Returns:
            The Jacobian matrices, shape (N, 3, 3).
        """
        mapped = self.map_points(points)
        result: np.ndarray = np.linalg.inv(self.field.jacobians(mapped))
        return result

    def inverse(self) -> DisplacementField:
        """Returns the field itself.

        Returns:
            The field being inverted.
        """
        return self.field


@dataclass(frozen=True, eq=False)
class TransformPair:
    """A transform given together with its inverse, such as a tool's inverse warp.

    Attributes:
        forward: The transform `map_points` applies.
        backward: Its inverse, which `inverse` returns as the swapped pair.
    """

    forward: Transform
    backward: Transform

    def map_points(self, points: np.ndarray) -> np.ndarray:
        """Maps points with `forward`.

        Args:
            points: The points, shape (N, 3), in mm, world coordinates (RAS+).

        Returns:
            The mapped points, shape (N, 3).
        """
        return self.forward.map_points(points)

    def jacobians(self, points: np.ndarray) -> np.ndarray:
        """Returns the (N, 3, 3) derivative of `forward` at the points.

        Args:
            points: The points, shape (N, 3), in mm, world coordinates (RAS+).

        Returns:
            The Jacobian matrices, shape (N, 3, 3).
        """
        return self.forward.jacobians(points)

    def inverse(self) -> TransformPair:
        """Returns the pair swapped, so that `backward` becomes the forward map.

        Returns:
            The pair with `backward` and `forward` swapped.
        """
        return TransformPair(self.backward, self.forward)


@dataclass(frozen=True, eq=False)
class TransformChain:
    """Transforms applied one after the other, starting with the first in the list.

    This is the order antsApplyTransforms lists its `-t` options in: the first
    transform takes the reference grid's points, the last gives positions in the
    input image. An ANTs registration written as `1Warp.nii.gz` and
    `0GenericAffine.mat` is the chain [warp, affine].

    Attributes:
        transforms: The transforms, in the order they are applied to points.
    """

    transforms: Sequence[Transform] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Stores the transforms as a tuple."""
        object.__setattr__(self, "transforms", tuple(self.transforms))

    def map_points(self, points: np.ndarray) -> np.ndarray:
        """Maps points through every transform in turn.

        Args:
            points: The points, shape (N, 3), in mm, world coordinates (RAS+).

        Returns:
            The mapped points, shape (N, 3).
        """
        mapped = np.asarray(points, dtype=np.float64)
        for transform in self.transforms:
            mapped = transform.map_points(mapped)
        return mapped

    def jacobians(self, points: np.ndarray) -> np.ndarray:
        """Returns the (N, 3, 3) derivative at the points, by the chain rule.

        Args:
            points: The points, shape (N, 3), in mm, world coordinates (RAS+).

        Returns:
            The Jacobian matrices, shape (N, 3, 3).
        """
        mapped = np.asarray(points, dtype=np.float64)
        result = np.broadcast_to(np.eye(3), (len(mapped), 3, 3)).copy()
        for transform in self.transforms:
            result = transform.jacobians(mapped) @ result
            mapped = transform.map_points(mapped)
        return result

    def inverse(self) -> TransformChain:
        """Returns the chain of the inverses, in the reverse order.

        Returns:
            The inverse chain.
        """
        return TransformChain([t.inverse() for t in reversed(self.transforms)])
