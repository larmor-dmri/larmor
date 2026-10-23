"""Selection of diffusion shells by b-value."""

import numpy as np

# Default distance, in s/mm^2, within which a b-value belongs to a shell.
SHELL_TOLERANCE = 100.0


def shell_mask(
    bvals: np.ndarray, shell: float, *, tol: float = SHELL_TOLERANCE
) -> np.ndarray:
    """Selects the volumes whose b-value is within `tol` of `shell`.

    All b-values are in s/mm^2.

    Args:
        bvals: The b-values, shape (N,).
        shell: The b-value of the shell.
        tol: How far a b-value may lie from `shell` to belong to it. Defaults
            to 100.

    Returns:
        A boolean array, shape (N,), True for the volumes of the shell.
    """
    return np.abs(bvals - shell) < tol
