"""Sets of directions on the sphere and their polar angles."""

from importlib.resources import files
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt

# The 300 unit directions FODs are evaluated on, shape (300, 3).
DIRS_300 = np.loadtxt(cast("Path", files("larmor") / "data" / "dir300.txt"))


def get_theta_phi(
    dirs: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Computes the polar angles of the given unit vectors.

    Args:
        dirs: Unit vectors, shape (n, 3).

    Returns:
        theta (from +z) and phi (around z, from +x), each shape (n,), in radians,
        in the frame of dirs.
    """
    theta = np.arccos(dirs[:, 2])
    phi = np.arctan2(dirs[:, 1], dirs[:, 0])
    return theta, phi


def get_dirs(n_dirs: int = 45) -> np.ndarray:
    """Creates n_dirs roughly uniform unit vectors on a Saff-Kuijlaars spiral.

    Returns:
        Unit vectors, shape (n_dirs, 3), flipped to the z >= 0 hemisphere. The first
        and the last are the spiral's two poles, so both are +z.
    """
    phis: list[float] = []
    dirs = []

    for k in range(1, n_dirs + 1):
        h = -1 + 2 * (k - 1) / (n_dirs - 1)
        theta_k = np.arccos(h)
        if k == 1 or k == (n_dirs):
            phi_k = 0
        else:
            phi_k = (phis[-1] + 3.6 / np.sqrt(n_dirs * (1 - h**2))) % (2 * np.pi)

        phis.append(phi_k)
        x = np.cos(phi_k) * np.sin(theta_k)
        y = np.sin(phi_k) * np.sin(theta_k)
        z = np.cos(theta_k)
        dirs.append(
            [
                x * np.sign(z),
                y * np.sign(z),
                z * np.sign(z),
            ]
        )

    return np.asarray(dirs)
