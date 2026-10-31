from collections.abc import Callable

import numpy as np
import pytest

import larmor.sd.model
import larmor.sd.msmt_csd
from larmor.dwi import DKIModel, DTIModel, inputs
from larmor.sd import CSDModel, estimate_csd_rf
from larmor.sd.sphere import get_dirs
from larmor.types import GradientTable, Volume

SHAPE = (7, 6, 5)


def _gradients(shells: list[float]) -> GradientTable:
    dirs = get_dirs(30)
    bvals = np.concatenate([np.zeros(4), *[np.full(30, b) for b in shells]])
    bvecs = [np.zeros((4, 3))] + [np.roll(dirs, k, axis=1) for k in range(len(shells))]
    return GradientTable(bvals, np.vstack(bvecs))


def _damaged(gtab: GradientTable, scaled: bool) -> Volume:
    """One fibre per voxel in a random direction, with a few broken voxels."""
    rng = np.random.default_rng(3)
    fibres = rng.standard_normal((*SHAPE, 3))
    fibres /= np.linalg.norm(fibres, axis=-1, keepdims=True)
    cos2 = (fibres @ gtab.bvecs.T) ** 2
    signal = 1000 * np.exp(-gtab.bvals * (0.3e-3 + 1.4e-3 * cos2))
    data = np.abs(signal + 10 * rng.standard_normal(signal.shape)).astype(np.float32)
    data[1, 1, 1, gtab.b0_mask] = 0
    data[2, 2, 2] = 0
    data[3, 1, 2, 20:25] = -5
    if scaled:
        return Volume(
            np.round(data * 4).astype(np.int16), np.eye(4), slope=0.25, intercept=1.0
        )
    data[0, 1, 2, 10] = np.nan
    data[4, 3, 1, gtab.b0_mask] = np.inf
    return Volume(data, np.eye(4))


def _csd(volume: Volume, gtab: GradientTable, mask: np.ndarray) -> np.ndarray:
    rf = estimate_csd_rf(volume, gtab, mask, num_voxels=30).rf
    return CSDModel(rf, lmax=6).fit(volume, gtab, mask=mask).fods[0]


FITS: dict[str, Callable[[Volume, GradientTable, np.ndarray], np.ndarray]] = {
    "dti": lambda volume, gtab, mask: DTIModel().fit(volume, gtab, mask).fa,
    "dki": lambda volume, gtab, mask: DKIModel().fit(volume, gtab, mask).mk,
    "csd": _csd,
}


@pytest.mark.parametrize("scaled", [False, True])
@pytest.mark.parametrize("name", FITS)
def test_fits_do_not_depend_on_the_chunk_size(
    monkeypatch: pytest.MonkeyPatch, name: str, scaled: bool
) -> None:
    # Chunks of a few voxels split the image at many places, and between the broken
    # voxels; only rounding of the batched solves may differ.
    gtab = _gradients([2500.0] if name == "csd" else [1000.0, 2500.0])
    volume = _damaged(gtab, scaled)
    mask = np.ones(SHAPE, dtype=bool)
    whole = FITS[name](volume, gtab, mask)

    monkeypatch.setattr(inputs, "READ_CHUNK", 7)
    monkeypatch.setattr(larmor.sd.model, "FIT_CHUNK", 5)
    monkeypatch.setattr(larmor.sd.msmt_csd, "FIT_CHUNK", 5)
    chunked = FITS[name](volume, gtab, mask)

    np.testing.assert_allclose(chunked, whole, rtol=1e-6, atol=1e-7)
