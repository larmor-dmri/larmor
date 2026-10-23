"""Deconvolution kernel built from a set of response functions."""

from collections.abc import Sequence

import numpy as np

from larmor.sd.rf import IsotropicRF, ResponseFunction


def order_rfs(rfs: list[ResponseFunction]) -> list[ResponseFunction]:
    """Returns the response functions with the anisotropic ones first."""
    aniso = [rf for rf in rfs if not isinstance(rf, IsotropicRF)]
    iso = [rf for rf in rfs if isinstance(rf, IsotropicRF)]
    return aniso + iso


class Kernel:
    """The signal of each response function along each direction, as a matrix.

    The kernel `k` has one row per measurement. An anisotropic response function
    adds one column per direction in `dirs`, the response turned to point along
    it, and an isotropic one adds a single column. The columns of each response
    function form a block (`blocks`, a slice per response function); the
    anisotropic blocks come first (`aniso_blocks`) and the isotropic ones last,
    whatever the order of `rfs`. The `w_` matrices have each row multiplied by its
    measurement's weight.
    """

    def __init__(
        self,
        rfs: list[ResponseFunction],
        bvals: np.ndarray,
        bvecs: np.ndarray,
        dirs: np.ndarray | Sequence[np.ndarray] | None = None,
        weights: np.ndarray | None = None,
    ) -> None:
        """Builds the kernel and its weighted copy.

        Args:
            rfs: Response functions; reordered with the anisotropic ones first.
            bvals: b-value of each measurement in s/mm^2, shape (n_meas,).
            bvecs: Gradient direction of each measurement, shape (n_meas, 3).
            dirs: Unit vectors the anisotropic response functions are oriented
                along, shape (n_dirs, 3), in the frame of bvecs: one array for all
                of them, or one per anisotropic response function in the order of
                `rfs`. Required when any response function is anisotropic; may be
                None otherwise.
            weights: Weight of each measurement, shape (n_meas,). Defaults to ones.

        Raises:
            ValueError: If a response function is anisotropic and `dirs` is None,
                or `dirs` is a sequence whose length is not the number of
                anisotropic response functions.
        """
        if dirs is None and any(not isinstance(rf, IsotropicRF) for rf in rfs):
            raise ValueError("dirs is required for an anisotropic response function.")
        if weights is None:
            weights = np.ones(len(bvals))

        self.weights = weights
        self.rfs = order_rfs(rfs)
        self.num_iso = sum(1 for rf in self.rfs if isinstance(rf, IsotropicRF))

        self.k = self._build_kernel(bvals, bvecs, dirs)
        self.w_k = self.k * self.weights[:, np.newaxis]
        self.aniso_blocks = self.blocks[: len(self.rfs) - self.num_iso]

        self.k_aniso, self.k_iso = self._split(self.k)
        self.w_k_aniso, self.w_k_iso = self._split(self.w_k)

    def _split(self, k: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """Splits the columns into the anisotropic ones and the last num_iso."""
        if self.num_iso == 0:
            return k, None
        return k[:, : -self.num_iso], k[:, -self.num_iso :]

    def _build_kernel(
        self,
        bvals: np.ndarray,
        bvecs: np.ndarray,
        dirs: np.ndarray | Sequence[np.ndarray] | None,
    ) -> np.ndarray:
        """Stacks the columns of each response function and records its block."""
        blocks = []
        self.blocks: list[slice] = []
        start = 0

        n_aniso = len(self.rfs) - self.num_iso
        if dirs is None:
            # Every rf is isotropic here, and IsotropicRF ignores its directions.
            dirs_per_rf = [np.empty((0, 3))] * len(self.rfs)
        elif isinstance(dirs, np.ndarray):
            dirs_per_rf = [dirs] * len(self.rfs)
        else:
            if len(dirs) != n_aniso:
                raise ValueError(
                    f"dirs has {len(dirs)} direction sets for {n_aniso} anisotropic "
                    "response functions."
                )
            dirs_per_rf = [*dirs, *([np.empty((0, 3))] * self.num_iso)]
        for rf, rf_dirs in zip(self.rfs, dirs_per_rf, strict=True):
            block = rf.signal(bvals, bvecs, rf_dirs)
            blocks.append(block)
            self.blocks.append(slice(start, start + block.shape[1]))
            start += block.shape[1]

        return np.hstack(blocks)
