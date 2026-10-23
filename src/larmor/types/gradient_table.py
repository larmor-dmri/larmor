"""The gradient table type."""

from dataclasses import dataclass

import numpy as np

from larmor._log import LarmorLogger

# How many times shell_epsilon a shell may span before a warning.
_WIDE_SHELL_FACTOR = 2.0

# How far from 1 a vector's length may be and still count as a unit vector.
_UNIT_TOLERANCE = 1e-12

# The shortest vector a weighted volume may have.
_MIN_LENGTH = 0.1


@dataclass
class GradientTable:
    """The diffusion weighting of each volume of a DWI, in the order of the volumes.

    On construction each b-value is scaled by the squared length of its vector,
    and each vector is scaled to unit length, so a vector of length 0.5 at
    b = 2800 becomes a unit vector at b = 700. A volume whose scaled b-value falls
    to `b0_threshold` or below then counts as a b0, with a warning. The caller's
    arrays are copied, not changed.

    The checks run only on construction: change a table through `flip` and
    `permute`, not by assigning its fields.

    The b-values must be in s/mm^2, or the b0 threshold and the shells are wrong.

    Args:
        bvals: (N,) b-values in s/mm^2.
        bvecs: (N, 3) gradient directions in world coordinates (RAS+). A row of
            NaN means no direction, like (0, 0, 0).
        b0_threshold: b-values at or below this, in s/mm^2, count as b0. Defaults
            to 50.0.
        shell_epsilon: A gap larger than this between sorted b-values, in s/mm^2,
            starts a new shell. Defaults to 80.0.

    Raises:
        ValueError: If the table is empty, the shapes are not (N,) and (N, 3), a
            b-value is negative, NaN or infinite, a vector has an infinite
            component, or a volume above `b0_threshold` has no direction or a
            vector shorter than 0.1.
    """

    bvals: np.ndarray
    bvecs: np.ndarray
    b0_threshold: float = 50.0
    shell_epsilon: float = 80.0

    def __post_init__(self) -> None:
        """Checks the table, then normalises the b-values and the vectors."""
        self.bvals = np.array(self.bvals, dtype=np.float64)
        self.bvecs = np.array(self.bvecs, dtype=np.float64)

        if self.bvals.ndim != 1:
            raise ValueError(
                f"Expected one-dimensional b-values, got shape {self.bvals.shape}."
            )
        if self.bvecs.shape != (len(self.bvals), 3):
            raise ValueError(
                f"Expected b-vectors of shape ({len(self.bvals)}, 3) for "
                f"{len(self.bvals)} b-values, got {self.bvecs.shape}."
            )
        if len(self.bvals) == 0:
            raise ValueError("A gradient table needs at least one volume, got none.")
        not_finite = ~np.isfinite(self.bvals)
        if not_finite.any():
            raise ValueError(
                f"Volumes {np.flatnonzero(not_finite).tolist()} have a NaN or "
                "infinite b-value."
            )
        negative = self.bvals < 0
        if negative.any():
            raise ValueError(
                f"Volumes {np.flatnonzero(negative).tolist()} have a negative b-value."
            )
        # NaN means no direction and is handled below.
        infinite = np.isinf(self.bvecs).any(axis=1)
        if infinite.any():
            raise ValueError(
                f"Volumes {np.flatnonzero(infinite).tolist()} have an infinite "
                "b-vector component."
            )

        # Before the checks below, so that they also catch NaN rows.
        self.bvecs[np.isnan(self.bvecs).any(axis=1)] = 0.0

        # Checked on the b-values as read, since a zero vector scales b to 0.
        lengths = np.linalg.norm(self.bvecs, axis=1)
        weighted = self.bvals > self.b0_threshold
        no_direction = (lengths == 0) & weighted
        if no_direction.any():
            raise ValueError(
                f"Volumes {np.flatnonzero(no_direction).tolist()} have b above "
                f"{self.b0_threshold} but no direction."
            )

        short = (lengths > 0) & (lengths < _MIN_LENGTH) & weighted
        if short.any():
            raise ValueError(
                f"Volumes {np.flatnonzero(short).tolist()} have b above "
                f"{self.b0_threshold} but a b-vector shorter than {_MIN_LENGTH}, "
                "which would scale their b-value to nearly 0."
            )

        # Unit vectors are left as they are, so normalising twice changes nothing.
        rescale = (lengths > 0) & (np.abs(lengths - 1.0) > _UNIT_TOLERANCE)
        self.bvals[rescale] *= lengths[rescale] ** 2
        self.bvecs[rescale] /= lengths[rescale, None]
        self.bvals[lengths == 0] = 0.0
        now_b0 = weighted & (self.bvals <= self.b0_threshold)
        if now_b0.any():
            LarmorLogger.warning(
                "Volumes %s have b above %s, but their b-vectors are short enough "
                "that the scaled b-value is at or below it: they count as b0 "
                "volumes.",
                np.flatnonzero(now_b0).tolist(),
                self.b0_threshold,
            )

        # Here rather than in `shells`, since the b-values never change afterwards.
        self._warn_about_wide_shells()

    @property
    def b0_mask(self) -> np.ndarray:
        """Marks the b0 volumes, whose b-value is at or below `b0_threshold`.

        Returns:
            np.ndarray: (N,) bool.
        """
        return np.asarray(self.bvals <= self.b0_threshold, dtype=np.bool)

    @property
    def b_matrix(self) -> np.ndarray:
        """The six distinct elements of b * g g^T for each volume.

        The order is (gx^2, 2 gx gy, 2 gx gz, gy^2, 2 gy gz, gz^2) times b, with the
        factor 2 of the off-diagonal elements included.

        Returns:
            np.ndarray: (N, 6) in s/mm^2.
        """
        bvals = self.bvals
        bvecs = self.bvecs
        gx = bvecs[:, 0]
        gy = bvecs[:, 1]
        gz = bvecs[:, 2]

        l_grads = [
            bvals * gx**2,
            2 * bvals * gx * gy,
            2 * bvals * gx * gz,
            bvals * gy**2,
            2 * bvals * gy * gz,
            bvals * gz**2,
        ]

        return np.stack(l_grads, axis=1, dtype=np.float64)

    @property
    def shells(self) -> np.ndarray:
        """The b-value of each shell (the mean of its volumes), ascending.

        Returns:
            np.ndarray: (K,) in s/mm^2; empty when every volume is a b0.
        """
        index = self.shell_index
        return np.array(
            [self.bvals[index == k].mean() for k in range(index.max() + 1)],
            dtype=np.float64,
        )

    def count_shells(self, *, above: float = 0.0) -> int:
        """Returns the number of shells, as `shells` groups them, above a b-value.

        b0s never count.

        Args:
            above: Count only the shells whose b-value is above this, in s/mm^2.
                Defaults to 0, every shell.

        Returns:
            int: The number of shells.
        """
        return int(np.count_nonzero(self.shells > above))

    def _warn_about_wide_shells(self) -> None:
        """Logs a warning for each shell much wider than `shell_epsilon`.

        Such a shell is likely a continuum of b-values, which models would treat as
        a single b-value.
        """
        index = self.shell_index
        for k in range(index.max() + 1):
            members = self.bvals[index == k]
            width = members.max() - members.min()
            if width > _WIDE_SHELL_FACTOR * self.shell_epsilon:
                LarmorLogger.warning(
                    "Shell %d spans %.0f s/mm^2 (%.0f to %.0f), more than %.0f times "
                    "shell_epsilon: its b-values may not form one shell.",
                    k,
                    width,
                    members.min(),
                    members.max(),
                    _WIDE_SHELL_FACTOR,
                )

    @property
    def shell_index(self) -> np.ndarray:
        """The index of each volume in `shells`, or -1 for a b0.

        The b-values of the weighted volumes are sorted, and a gap larger than
        `shell_epsilon` between neighbours starts a new shell.

        Returns:
            np.ndarray: (N,) int.
        """
        weighted = np.flatnonzero(~self.b0_mask)
        order = np.argsort(self.bvals[weighted], kind="stable")
        sorted_b = self.bvals[weighted][order]
        new_shell = np.diff(sorted_b) > self.shell_epsilon
        labels = np.concatenate([[0], np.cumsum(new_shell)])

        index = np.full(len(self.bvals), -1)
        index[weighted[order]] = labels[: len(weighted)]
        return index

    def flip(self, axis: int) -> None:
        """Negates one component of every direction, in place.

        Corrects a table that is wrong for its data, such as a y sign reversed by the
        scanner export. Reorienting a volume never needs it.

        Args:
            axis: The world axis to negate, 0 (x), 1 (y) or 2 (z).

        Raises:
            ValueError: If `axis` is not 0, 1 or 2.
        """
        if axis < 0 or axis > 2:
            raise ValueError(f"Cannot flip gradients for axis {axis}.")

        self.bvecs[:, axis] = -self.bvecs[:, axis]

    def permute(self, order: tuple[int, int, int]) -> None:
        """Reorders the components of every direction, in place.

        New component j is old component order[j]. Like `flip`, it corrects a wrong
        table; reorienting a volume never needs it.

        Args:
            order: A permutation of (0, 1, 2).

        Raises:
            ValueError: If `order` is not a permutation of (0, 1, 2).
        """
        if sorted(order) != [0, 1, 2]:
            raise ValueError(f"Invalid order for permutation: {order}")

        self.bvecs = self.bvecs[:, list(order)]
