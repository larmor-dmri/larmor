import numpy as np
import pytest

from larmor.sd import asymmetry, convert_sh_basis, filter_afod
from larmor.sd.asymmetry import _sphere_bases
from larmor.sd.sh import get_sh_indices

LMAX = 8
ODD = get_sh_indices(LMAX, full=True)[0] % 2 == 1
DIRS, B_EVEN, B_FULL = _sphere_bases(LMAX, 300)
# The x axis is flipped, as in HCP: voxel index i grows towards world -x.
AFFINE = np.diag([-1.25, 1.25, 1.25, 1.0])
SHAPE = (8, 7, 6)
# Voxels of different sizes along the axes, one axis flipped and the grid turned, so
# that distances in mm and voxels differ.
TURN = np.array([[0.8, -0.6, 0.0], [0.6, 0.8, 0.0], [0.0, 0.0, 1.0]])
OBLIQUE = np.eye(4)
OBLIQUE[:3, :3] = TURN @ np.diag([-0.8, 1.5, 2.5])


def _slow_filter(
    fod: np.ndarray,
    mask: np.ndarray,
    affine: np.ndarray,
    sigma: float,
    sharpness: float,
) -> np.ndarray:
    """The filter written out voxel by voxel and neighbour by neighbour.

    The weights come from the voxel centres' world positions, not from the module.
    """
    to_full = np.linalg.pinv(B_FULL)
    out = np.zeros((*fod.shape[:3], B_FULL.shape[1]))
    for voxel in zip(*np.nonzero(mask), strict=True):
        here = affine[:3, :3] @ np.array(voxel)
        num = fod[voxel] @ B_EVEN.T
        den = np.ones_like(num)
        for offset in np.ndindex(3, 3, 3):
            near = tuple(int(n) for n in np.array(voxel) + np.array(offset) - 1)
            inside = all(0 <= n < s for n, s in zip(near, mask.shape, strict=True))
            if near == voxel or not inside or not mask[near]:
                continue
            step = affine[:3, :3] @ np.array(near) - here
            distance = np.linalg.norm(step)
            w = (
                np.exp(-(distance**2) / (2 * sigma**2))
                * np.clip(DIRS @ (step / distance), 0, None) ** sharpness
            )
            num += w * (fod[near] @ B_EVEN.T)
            den += w
        out[voxel] = to_full @ (num / den)
    return out


def _fibre(direction: np.ndarray) -> np.ndarray:
    amplitudes = np.exp(-8 * (1 - (DIRS @ direction) ** 2))
    coeffs: np.ndarray = np.linalg.lstsq(B_EVEN, amplitudes, rcond=None)[0]
    return coeffs


def _amplitude(coeffs: np.ndarray, direction: list[float]) -> float:
    return float(B_FULL[np.argmax(DIRS @ direction)] @ coeffs)


# A sparse mask has only voxels with a neighbour outside it; a nearly solid one also
# has voxels whose 26 neighbours are all inside, which take the other path.
@pytest.mark.parametrize("kept", [0.7, 0.97])
def test_matches_the_filter_written_out_voxel_by_voxel(kept: float) -> None:
    rng = np.random.default_rng(0)
    fod = rng.standard_normal((*SHAPE, B_EVEN.shape[1]))
    mask = rng.random(SHAPE) < kept

    out = filter_afod(fod, mask, OBLIQUE, sigma=1.7, sharpness=3.0)

    expected = _slow_filter(fod, mask, OBLIQUE, sigma=1.7, sharpness=3.0)
    np.testing.assert_allclose(out, expected, atol=1e-5)


def test_the_same_fod_everywhere_stays_symmetric_up_to_the_mask_edge() -> None:
    fibre = _fibre(np.array([0.0, 0.6, 0.8]))
    fod = np.broadcast_to(fibre, (*SHAPE, len(fibre))).copy()
    mask = np.zeros(SHAPE, dtype=bool)
    mask[:4] = True

    out = filter_afod(fod, mask, AFFINE)

    assert np.abs(out[mask][:, ODD]).max() < 1e-5
    np.testing.assert_allclose(out[mask][:, ~ODD], fod[mask], atol=1e-5)
    assert np.all(out[~mask] == 0)


def test_the_end_of_a_bundle_points_back_along_it() -> None:
    # The bundle fills i >= 4, which lies towards world -x of its end, i = 4.
    fod = np.zeros((*SHAPE, B_EVEN.shape[1]))
    fod[4:] = _fibre(np.array([1.0, 0.0, 0.0]))

    out = filter_afod(fod, np.ones(SHAPE, dtype=bool), AFFINE)

    end, middle = out[4, 3, 3], out[6, 3, 3]
    assert _amplitude(end, [1, 0, 0]) < 0.5 * _amplitude(end, [-1, 0, 0])
    assert _amplitude(middle, [1, 0, 0]) == pytest.approx(
        _amplitude(middle, [-1, 0, 0]), rel=1e-4
    )


def test_the_basis_of_the_input_is_the_basis_of_the_output() -> None:
    fod = np.zeros((*SHAPE, B_EVEN.shape[1]))
    fod[4:] = _fibre(np.array([1.0, 0.0, 0.0]))
    mask = np.ones(SHAPE, dtype=bool)

    tournier = filter_afod(fod, mask, AFFINE)
    descoteaux = filter_afod(
        convert_sh_basis(fod, "tournier", "descoteaux"),
        mask,
        AFFINE,
        basis="descoteaux",
    )

    np.testing.assert_allclose(
        convert_sh_basis(descoteaux, "descoteaux", "tournier"), tournier, atol=1e-6
    )


def test_without_a_mask_every_nonzero_voxel_is_filtered() -> None:
    fod = np.zeros((*SHAPE, B_EVEN.shape[1]))
    fod[4:] = _fibre(np.array([1.0, 0.0, 0.0]))

    out = filter_afod(fod, None, AFFINE)

    np.testing.assert_array_equal(out, filter_afod(fod, fod[..., 0] != 0, AFFINE))


def test_a_mask_of_another_shape_is_refused() -> None:
    fod = np.zeros((*SHAPE, B_EVEN.shape[1]))

    with pytest.raises(ValueError, match="same shape"):
        filter_afod(fod, np.ones((2, 2, 2), dtype=bool), AFFINE)


def test_every_voxel_is_filtered_when_chunks_are_small(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A solid 12^3 mask has more voxels with all 26 neighbours inside (1000) than
    # voxels at its edge (728), and chunks of 7 split them many times.
    monkeypatch.setattr(asymmetry, "CHUNK_SIZE", 7)
    fibre = _fibre(np.array([0.0, 0.6, 0.8]))
    fod = np.broadcast_to(fibre, (12, 12, 12, len(fibre))).copy()

    out = filter_afod(fod, np.ones((12, 12, 12), dtype=bool), AFFINE)

    np.testing.assert_allclose(out[..., ~ODD], fod, atol=1e-5)


def test_values_outside_the_mask_do_not_reach_it() -> None:
    rng = np.random.default_rng(1)
    fod = rng.standard_normal((*SHAPE, B_EVEN.shape[1]))
    mask = np.zeros(SHAPE, dtype=bool)
    mask[2:6, 2:5, 1:5] = True
    with_nan = fod.copy()
    with_nan[~mask] = np.nan

    np.testing.assert_array_equal(
        filter_afod(with_nan, mask, AFFINE), filter_afod(fod, mask, AFFINE)
    )
