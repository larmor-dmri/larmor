"""Merging the FODs of a multi-FOD fit into one."""

from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from scipy.io import loadmat

from larmor.sd import merge_fods

MERGE_REF = Path(__file__).parents[1] / "data" / "mfod" / "mfod_merge_ref.mat"

# A grid whose last axis (4) differs from the FODs' coefficients (6), so that a
# fraction broadcast along the wrong axis cannot pass unnoticed.
SHAPE = (2, 3, 4)
C = 6


def _fods() -> list[np.ndarray]:
    rng = np.random.default_rng(0)
    return [rng.normal(size=(*SHAPE, C)) for _ in range(2)]


def _fractions(wm: np.ndarray, gm: np.ndarray) -> np.ndarray:
    """White matter, grey matter and CSF fractions, the CSF taking the rest."""
    return np.stack([wm, gm, 1.0 - wm - gm], axis=-1)


def test_weighted_sum_adds_each_fod_times_its_fraction_in_every_voxel() -> None:
    fods = _fods()
    rng = np.random.default_rng(1)
    wm = rng.uniform(0.0, 0.5, SHAPE)
    gm = rng.uniform(0.0, 0.5, SHAPE)

    merged = merge_fods(fods, _fractions(wm, gm), "ws")

    expected = wm[..., None] * fods[0] + gm[..., None] * fods[1]
    np.testing.assert_allclose(merged, expected)


def test_majority_keeps_the_fod_of_the_largest_anisotropic_fraction() -> None:
    fods = _fods()
    wm = np.full(SHAPE, 0.1)
    gm = np.full(SHAPE, 0.2)
    wm[0] = 0.3  # white matter wins in the first slab, grey matter elsewhere
    # The CSF takes 0.5 to 0.7, the largest share everywhere; it must not count.

    merged = merge_fods(fods, _fractions(wm, gm), "majority")

    assert merged.shape == (*SHAPE, C)
    np.testing.assert_array_equal(merged[0], fods[0][0])
    np.testing.assert_array_equal(merged[1:], fods[1][1:])


@pytest.mark.parametrize("method", ["ws", "majority"])
def test_a_single_fod_with_all_the_anisotropic_fraction_comes_back(
    method: str,
) -> None:
    fod = _fods()[0]
    fractions = np.stack([np.ones(SHAPE), np.zeros(SHAPE)], axis=-1)
    np.testing.assert_allclose(merge_fods([fod], fractions, method), fod)  # type: ignore[arg-type]


def test_an_unknown_method_is_refused() -> None:
    with pytest.raises(ValueError, match="majority"):
        merge_fods(_fods(), _fractions(np.zeros(SHAPE), np.zeros(SHAPE)), "mean")  # type: ignore[arg-type]


def test_fewer_fractions_than_fods_are_refused() -> None:
    with pytest.raises(ValueError):
        merge_fods(_fods(), np.ones((*SHAPE, 1)), "ws")


def test_fods_of_different_shapes_are_refused() -> None:
    fods = _fods()
    fods[1] = fods[1][..., :3]
    with pytest.raises(ValueError):
        merge_fods(fods, _fractions(np.zeros(SHAPE), np.zeros(SHAPE)), "ws")


@pytest.mark.parametrize(
    ("method", "key"), [("ws", "linear"), ("majority", "majority")]
)
def test_merging_matches_mritoolkit(
    method: Literal["ws", "majority"], key: str
) -> None:
    # make_mfod_merge_ref.m: MRIToolkit's two mFOD FODs and fractions, and the FOD
    # its tracking merges from them. White matter wins in some voxels, grey matter
    # in others, and the CSF fraction is the largest in some.
    ref = loadmat(MERGE_REF, simplify_cells=True)["ref"]
    merged = merge_fods([ref["fod1"], ref["fod2"]], ref["fractions"], method)
    # MRIToolkit writes the linear FOD in single precision.
    np.testing.assert_allclose(merged, ref[key], rtol=0, atol=1e-7)
