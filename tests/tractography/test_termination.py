import math

import numpy as np
import pytest
from numpy.typing import NDArray
from reference_tracking import inside_mask

from larmor import _core
from larmor.sd.sh import build_basis_matrix, get_sh_indices
from larmor.tractography import EndStatus
from larmor.tractography.termination import (
    _AUTO_CUTOFF_SEED,
    _AUTO_CUTOFF_VOXELS,
    auto_cutoff,
)

LMAX = 8


# EndStatus


def test_end_status_values_are_stable() -> None:
    # Written to TRK files: a renumbering would misread every saved tractogram.
    assert {s.name: s.value for s in EndStatus} == {
        "NONE": 0,
        "MAX_LENGTH": 1,
        "LEFT_VOLUME": 2,
        "LEFT_MASK": 3,
        "LOW_FOD": 4,
        "SHARP_TURN": 5,
        "NO_DIRECTION": 6,
    }


# Mask


def _mask_affine() -> NDArray[np.float64]:
    # Anisotropic, y flipped, offset and oblique: not the FOD's identity grid.
    angle = np.deg2rad(15.0)
    c, s = np.cos(angle), np.sin(angle)
    affine = np.eye(4)
    affine[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0, 0, 1]]) @ np.diag(
        [1.5, -2.0, 2.5]
    )
    affine[:3, 3] = [-10.0, 20.0, 3.0]
    return affine


# A FOD grid of two 10 m voxels around the origin, so a seed starts exactly when it
# is inside the mask.
WIDE_FOD = np.zeros((2, 2, 2, 1), dtype=np.float32)
WIDE_FOD_INV = np.linalg.inv(np.diag([1e4, 1e4, 1e4, 1.0]))


def _inside(
    data: NDArray[np.floating],
    affine: NDArray[np.float64],
    vox: NDArray[np.float64] | list[float],
) -> bool:
    # The compiled start check and the numpy reference's mask test, which must agree.
    mask = np.ascontiguousarray(data, dtype=np.float32)
    inv = np.linalg.inv(affine)
    pos = (affine @ np.append(vox, 1.0))[:3]
    counts, _, _, _ = _core.track_batch(
        WIDE_FOD,
        WIDE_FOD_INV,
        (mask, inv),
        pos[None],
        np.array([[1.0, 0.0, 0.0]]),
        np.zeros(1, dtype=np.int64),
        0,
        1e-3,
        1,
        False,
        _core.TestAlgorithm("straight"),
        np.zeros((1, 2, 3), dtype=np.float32),
        1,
    )
    started = bool(counts[0] > 0)
    assert started is inside_mask(mask, inv, pos)
    return started


def _half_mask() -> NDArray[np.float64]:
    # Ones at x >= 2, zeros below: along x the mask starts halfway between voxel
    # centres 1 and 2.
    data = np.zeros((4, 3, 3))
    data[2:] = 1.0
    return data


@pytest.mark.parametrize(
    ("x", "expected"), [(0.0, False), (1.4, False), (1.6, True), (3.0, True)]
)
def test_mask_is_inside_where_the_nearest_voxel_is(x: float, expected: bool) -> None:
    assert _inside(_half_mask(), _mask_affine(), [x, 1.3, 0.8]) is expected


@pytest.mark.parametrize(
    ("vox", "expected"),
    [([1.45, 1.45, 1.0], True), ([1.0, 1.45, 1.45], True), ([1.55, 1.0, 1.0], False)],
)
def test_one_voxel_mask_holds_its_whole_voxel(vox: list[float], expected: bool) -> None:
    # An interpolated mask would be 0.3 near the corner of a lone voxel, so a thin
    # structure would lose much of its cross-section.
    data = np.zeros((3, 3, 3))
    data[1, 1, 1] = 1.0
    assert _inside(data, _mask_affine(), vox) is expected


@pytest.mark.parametrize(
    ("x", "expected"), [(0.5, True), (float(np.nextafter(0.5, 0.0)), False)]
)
def test_mask_is_inside_at_exactly_half_and_not_just_below(
    x: float, expected: bool
) -> None:
    data = np.array([0.0, 1.0]).reshape(2, 1, 1)
    assert _inside(data, np.eye(4), [x, 0.0, 0.0]) is expected


@pytest.mark.parametrize(
    ("x", "expected"), [(0.4, False), (0.6, True), (0.85, True), (1.6, True)]
)
def test_fraction_map_is_compared_at_its_nearest_voxel(
    x: float, expected: bool
) -> None:
    # Values 0.2, 0.6, 0.9 along x: inside from x = 0.5 on, nearer voxel 1.
    data = np.broadcast_to(np.array([0.2, 0.6, 0.9])[:, None, None], (3, 2, 2))
    assert _inside(data, _mask_affine(), [x, 0.5, 0.5]) is expected


@pytest.mark.parametrize(
    ("vox", "expected"),
    [
        ([3.45, 1.0, 1.0], True),
        ([-0.45, 1.0, 1.0], False),
        ([3.45, -0.45, 2.45], True),
    ],
)
def test_mask_uses_edge_values_in_the_half_voxel_past_its_centres(
    vox: list[float], expected: bool
) -> None:
    assert _inside(_half_mask(), _mask_affine(), vox) is expected


_FACES = [(axis, side) for axis in range(3) for side in (-1, 1)]


@pytest.mark.parametrize("affine", [np.eye(4), _mask_affine()], ids=["eye", "oblique"])
@pytest.mark.parametrize(("axis", "side"), _FACES)
@pytest.mark.parametrize(
    ("beyond", "expected"),
    [(0.45, True), (0.55, False), (10.0, False), (500.0, False)],
)
def test_mask_is_left_off_its_grid_however_far_out(
    affine: NDArray[np.float64], axis: int, side: int, beyond: float, expected: bool
) -> None:
    # All ones, so only the grid check can say no.
    shape = (4, 3, 5)
    vox = np.array([1.3, 0.8, 2.6])
    vox[axis] = -beyond if side < 0 else shape[axis] - 1 + beyond
    assert _inside(np.ones(shape), affine, vox) is expected


# Auto cutoff


def _polar(v: NDArray[np.float64]) -> tuple[float, float]:
    return math.acos(float(np.clip(v[2], -1, 1))), math.atan2(float(v[1]), float(v[0]))


def _fibre(direction: NDArray[np.float64]) -> tuple[NDArray[np.float64], float]:
    # An apodised delta along the direction in the tournier basis: axially
    # symmetric, so its largest peak lies on the direction, with this amplitude.
    ls, _ = get_sh_indices(LMAX)
    theta, phi = _polar(direction / np.linalg.norm(direction))
    b = build_basis_matrix(np.array([theta]), np.array([phi]), LMAX)[0]
    sh: NDArray[np.float64] = b * np.exp(-0.02 * ls * (ls + 1))
    return sh, float(b @ sh)


def _fods(n: int, seed: int = 0) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    # n single-fibre voxels, random directions, scaled 0.5 to 3: the FODs and
    # their largest peak amplitudes.
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(n, 3))
    scales = rng.uniform(0.5, 3.0, n)
    coeffs = np.empty((n, len(get_sh_indices(LMAX)[0])))
    peaks = np.empty(n)
    for i, (d, s) in enumerate(zip(directions, scales, strict=True)):
        sh, amplitude = _fibre(d)
        coeffs[i], peaks[i] = s * sh, s * amplitude
    return coeffs, peaks


# The climb converges to the fibre direction far below this relative error.
RTOL_PEAK = 1e-6


def test_auto_cutoff_is_a_fifth_of_the_mean_largest_peak() -> None:
    coeffs, peaks = _fods(60)
    cutoff = auto_cutoff(coeffs.reshape(3, 4, 5, -1))
    assert cutoff == pytest.approx(0.2 * peaks.mean(), rel=RTOL_PEAK)


@pytest.mark.parametrize("dc", [0.0, -0.01])
def test_auto_cutoff_ignores_voxels_without_positive_dc(dc: float) -> None:
    coeffs, peaks = _fods(20)
    # Large lobes with a non-positive DC term would raise the mean if counted.
    excluded, _ = _fods(5, seed=1)
    excluded = 10 * excluded
    excluded[:, 0] = dc
    image = np.concatenate([coeffs, excluded, np.zeros_like(excluded)])
    cutoff = auto_cutoff(image.reshape(5, 6, -1)[:, :, None])
    assert cutoff == pytest.approx(0.2 * peaks.mean(), rel=RTOL_PEAK)


def test_auto_cutoff_skips_drawn_voxels_without_a_peak() -> None:
    coeffs, peaks = _fods(10)
    no_peak = np.full_like(coeffs[:3], np.nan)
    no_peak[:, 0] = 1.0
    image = np.concatenate([coeffs, no_peak])
    cutoff = auto_cutoff(image.reshape(13, 1, 1, -1))
    assert cutoff == pytest.approx(0.2 * peaks.mean(), rel=RTOL_PEAK)


def test_auto_cutoff_with_many_voxels_uses_a_fixed_subset() -> None:
    n = _AUTO_CUTOFF_VOXELS + 400
    coeffs, peaks = _fods(n, seed=2)
    rows = np.random.default_rng(_AUTO_CUTOFF_SEED).choice(
        n, _AUTO_CUTOFF_VOXELS, replace=False
    )
    # Larger lobes in the voxels left out, so only the expected subset matches.
    left_out = np.setdiff1d(np.arange(n), rows)
    coeffs[left_out] *= 20
    peaks[left_out] *= 20
    image = coeffs.reshape(n, 1, 1, -1)
    first = auto_cutoff(image)
    assert auto_cutoff(image) == first
    assert first == pytest.approx(0.2 * peaks[rows].mean(), rel=RTOL_PEAK)
    assert first != pytest.approx(0.2 * peaks.mean(), rel=1e-3)


def test_auto_cutoff_counts_only_the_voxels_of_the_mask() -> None:
    coeffs, peaks = _fods(20)
    outside, _ = _fods(10, seed=1)
    image = np.concatenate([coeffs, 0.1 * outside]).reshape(30, 1, 1, -1)
    mask = np.zeros((30, 1, 1), dtype=bool)
    mask[:20] = True
    cutoff = auto_cutoff(image, mask=mask)
    assert cutoff == pytest.approx(0.2 * peaks.mean(), rel=RTOL_PEAK)


def test_auto_cutoff_refuses_a_mask_without_a_positive_dc_voxel() -> None:
    coeffs, _ = _fods(4)
    with pytest.raises(ValueError, match="mask"):
        auto_cutoff(coeffs.reshape(2, 2, 1, -1), mask=np.zeros((2, 2, 1), bool))


def test_auto_cutoff_refuses_fod_without_positive_dc() -> None:
    coeffs, _ = _fods(4)
    coeffs[:, 0] = -1.0
    with pytest.raises(ValueError, match="DC"):
        auto_cutoff(coeffs.reshape(2, 2, 1, -1))


def test_auto_cutoff_refuses_fod_without_any_peak() -> None:
    # A positive DC term passes the first check; NaN lobes give no peak.
    coeffs = np.full((2, 2, 1, len(get_sh_indices(LMAX)[0])), np.nan)
    coeffs[..., 0] = 1.0
    with pytest.raises(ValueError, match="peak"):
        auto_cutoff(coeffs)
