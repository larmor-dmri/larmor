from collections.abc import Callable

import numpy as np
import pytest

from larmor.sd import convert_sh_basis
from larmor.sd.sh import build_basis_matrix, get_sh_indices
from larmor.sd.sphere import get_theta_phi
from larmor.types import Volume
from larmor.viewer.render.colour_space import physical_colours
from larmor.viewer.render.glyphs import (
    image_amplitude,
    sh_anisotropy,
    slice_glyphs,
    sphere,
    voxel_glyph,
)

DETAIL = 3
RADIUS = 1.0
# Positions are float32 of size ~RADIUS around centres of a few mm.
TOL = 1e-5


def _lobe(direction: list[float], lmax: int, full: bool = False) -> np.ndarray:
    # The SH of a delta at `direction`, cut at lmax, in the tournier basis.
    u = np.asarray([direction], np.float64)
    theta, phi = get_theta_phi(u / np.linalg.norm(u))
    row: np.ndarray = build_basis_matrix(theta, phi, lmax, "tournier", full=full)[0]
    return row


def _fod() -> np.ndarray:
    # Two crossing fibres on a positive base: a shape with no symmetry to hide in.
    coeffs = _lobe([1, 0.2, 0], 4) + 0.6 * _lobe([0.1, 1, 0.5], 4)
    coeffs[0] += 0.5
    return coeffs


def _volume(coeffs: list[np.ndarray], affine: np.ndarray | None = None) -> Volume:
    # Voxels along x on one axial slice (axis 2, index 0).
    data = np.stack(coeffs)[:, None, None, :].astype(np.float32)
    return Volume(data, np.eye(4) if affine is None else affine)


def _glyphs(
    volume: Volume, basis: str = "tournier", amplitude: float | None = None
) -> np.ndarray:
    g = slice_glyphs(volume, 2, 0, basis, DETAIL, RADIUS, image_amplitude=amplitude)
    return g.positions.reshape(g.count, -1, 3)


@pytest.mark.parametrize("basis", ["descoteaux", "leemans"])
def test_the_same_fod_in_another_basis_gives_the_same_glyph(basis: str) -> None:
    tournier = _fod()
    other = convert_sh_basis(tournier, "tournier", basis)
    expected = _glyphs(_volume([tournier]))
    assert np.allclose(_glyphs(_volume([other]), basis), expected, atol=TOL)


def test_a_symmetric_fod_stored_with_odd_degrees_zero_gives_the_same_glyph() -> None:
    symmetric = _fod()
    ls, ms = get_sh_indices(4)
    full = np.zeros(25)
    full[ls**2 + ls + ms] = symmetric
    expected = _glyphs(_volume([symmetric]))
    assert np.allclose(_glyphs(_volume([full])), expected, atol=TOL)


def test_a_single_lobed_asymmetric_fod_has_one_lobe() -> None:
    # 1 + 3 cos(angle to +x), up to a factor: largest at +x, negative at -x.
    [glyph] = _glyphs(_volume([_lobe([1, 0, 0], 1, full=True)]))
    vertices = sphere(DETAIL).vertices
    towards = np.argmax(vertices[:, 0])
    away = np.argmin(vertices[:, 0])
    assert np.linalg.norm(glyph[towards]) == pytest.approx(RADIUS, rel=1e-5)
    assert np.linalg.norm(glyph[away]) < 1e-6


def test_negative_amplitudes_are_drawn_at_the_centre() -> None:
    # A sharp lobe cut at lmax 8 rings negative away from it.
    coeffs = _lobe([0, 0, 1], 8)
    [glyph] = _glyphs(_volume([coeffs]))
    vertices = sphere(DETAIL).vertices
    raw = build_basis_matrix(*get_theta_phi(vertices.astype(np.float64)), 8, "tournier")
    negative = raw @ coeffs < 0
    assert negative.any()
    # Each vertex stays on its own side of the centre: no glyph is turned inside out.
    outward = np.einsum("ij,ij->i", glyph, vertices)
    assert np.all(outward >= -1e-7)
    assert np.allclose(glyph[negative], 0.0, atol=1e-7)


@pytest.mark.parametrize("whole_image", [False, True])
def test_glyph_size_is_per_voxel_or_relative_to_the_image_amplitude(
    whole_image: bool,
) -> None:
    coeffs = _fod()
    # The second voxel is shifted back to the origin to compare sizes alone.
    volume = _volume([coeffs, 2 * coeffs])
    # The big voxel's largest amplitude: the high amplitude of an image of it alone.
    big_only = image_amplitude(_volume([2 * coeffs]), "tournier", DETAIL)
    small, big = _glyphs(volume, amplitude=big_only if whole_image else None)
    big -= [1, 0, 0]
    reach = np.linalg.norm(big, axis=1).max()
    assert reach == pytest.approx(RADIUS, rel=1e-5)
    expected = big / 2 if whole_image else big
    assert np.allclose(small, expected, atol=TOL)


def test_the_image_amplitude_is_the_90th_percentile_of_the_voxels_largest() -> None:
    coeffs = _fod()
    one = image_amplitude(_volume([coeffs]), "tournier", DETAIL)
    # An all-zero voxel (background) does not count.
    volume = _volume([k * coeffs for k in range(1, 11)] + [0 * coeffs])
    expected = np.percentile(one * np.arange(1, 11), 90)
    assert image_amplitude(volume, "tournier", DETAIL) == pytest.approx(
        expected, rel=1e-5
    )


def test_an_image_amplitude_of_0_gives_no_glyphs() -> None:
    g = slice_glyphs(
        _volume([_fod()]), 2, 0, "tournier", DETAIL, RADIUS, image_amplitude=0
    )
    assert g.count == 0


@pytest.mark.parametrize(("option", "value"), [("threshold", 1.5), ("anisotropy", 0.5)])
def test_voxels_below_the_threshold_or_anisotropy_get_no_glyph(
    option: str, value: float
) -> None:
    # Voxel 0 is weak and round, voxel 1 strong and sharp; 0 must go either way.
    round_ = np.zeros(15)
    round_[0] = 1.0
    sharp = 4 * _lobe([1, 0, 0], 4)
    volume = _volume([round_, sharp])
    limits = {"threshold": 0.0, "anisotropy": 0.0} | {option: value}
    g = slice_glyphs(
        volume,
        2,
        0,
        "tournier",
        DETAIL,
        RADIUS,
        threshold=limits["threshold"],
        anisotropy=limits["anisotropy"],
    )
    assert g.count == 1
    reach = np.linalg.norm(g.positions - [1, 0, 0], axis=1)
    assert reach.max() == pytest.approx(RADIUS, rel=1e-5)


def test_glyphs_are_centred_on_voxel_centres_in_world_mm() -> None:
    # A round FOD's glyph is a sphere, whose vertex mean is its centre.
    affine = np.array(
        [[0, 0, 2.5, -10], [1.5, 0, 0, 4], [0, -2, 0, 7], [0, 0, 0, 1]], float
    )
    data = np.zeros((3, 4, 5, 6), np.float32)
    voxels = [(0, 2, 1), (2, 2, 4), (1, 2, 0)]
    for v in voxels:
        data[v + (0,)] = 1.0
    g = slice_glyphs(Volume(data, affine), 1, 2, "tournier", DETAIL, RADIUS)
    centres = g.positions.reshape(g.count, -1, 3).mean(axis=1)
    expected = np.array([affine[:3, :3] @ v + affine[:3, 3] for v in voxels])
    order = np.lexsort(expected.T)
    assert np.allclose(centres[np.lexsort(centres.T)], expected[order], atol=TOL)


@pytest.mark.parametrize("basis", ["tournier", "descoteaux", "leemans"])
def test_sh_anisotropy_is_zero_when_round_and_the_same_in_every_basis(
    basis: str,
) -> None:
    round_ = np.zeros(15)
    round_[0] = 2.0
    coeffs = convert_sh_basis(np.stack([round_, _fod()]), "tournier", basis)
    anisotropy = sh_anisotropy(coeffs)
    assert anisotropy[0] == 0.0
    assert anisotropy[1] == pytest.approx(sh_anisotropy(_fod()[None])[0], abs=1e-12)
    assert 0.0 < anisotropy[1] < 1.0


def _asymmetric() -> np.ndarray:
    # The crossing with odd degrees added: lobes of unequal size at u and -u.
    ls, ms = get_sh_indices(4)
    full = np.zeros(25)
    full[ls**2 + ls + ms] = _fod()
    return full + 0.5 * _lobe([0, 0.3, 1], 4, full=True)


@pytest.mark.parametrize("make", [_fod, _asymmetric])
def test_a_voxel_glyph_is_its_slice_glyph_centred_at_the_origin(
    make: Callable[[], np.ndarray],
) -> None:
    coeffs = make()
    affine = np.diag([2.0, 3.0, 1.5, 1.0])
    affine[:3, 3] = [-4, 5, 6]
    # A second, larger voxel first: the glyph must be scaled by its own lobe.
    volume = _volume([3 * coeffs, coeffs], affine)
    g = voxel_glyph(volume, (1, 0, 0), "tournier", DETAIL)
    s = slice_glyphs(volume, 2, 0, "tournier", DETAIL, RADIUS)
    centre = affine[:3, :3] @ [1, 0, 0] + affine[:3, 3]
    expected = s.positions.reshape(s.count, -1, 3)[1] - centre
    assert g.count == 1
    assert np.allclose(g.positions, expected, atol=TOL)
    assert np.linalg.norm(g.positions, axis=1).max() == pytest.approx(1.0, rel=1e-5)
    assert np.array_equal(g.faces, sphere(DETAIL).faces)
    # Linear light, as the mesh reads its vertex colours: each direction's colour at
    # full brightness, mixed with some white.
    assert np.allclose(g.colours.max(axis=1), 1.0)
    assert g.colours.min() >= physical_colours(np.full((1, 3), 0.3)).min() - 1e-6
    x = np.argmax(sphere(DETAIL).vertices[:, 0])  # the vertex along x: red
    assert g.colours[x, 0] > g.colours[x, 1] and g.colours[x, 0] > g.colours[x, 2]
    normals = s.normals.reshape(s.count, -1, 3)[1]
    unit = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    g_unit = g.normals / np.linalg.norm(g.normals, axis=1, keepdims=True)
    assert np.allclose(g_unit, unit, atol=1e-5)


def test_a_single_lobed_asymmetric_voxel_glyph_has_one_lobe() -> None:
    volume = _volume([_lobe([1, 0, 0], 1, full=True)])
    g = voxel_glyph(volume, (0, 0, 0), "tournier", DETAIL)
    vertices = sphere(DETAIL).vertices
    reach = np.linalg.norm(g.positions, axis=1)
    assert reach[np.argmax(vertices[:, 0])] == pytest.approx(1.0, rel=1e-5)
    assert reach[np.argmin(vertices[:, 0])] < 1e-6


def test_an_all_zero_voxel_has_no_glyph() -> None:
    volume = _volume([_fod(), np.zeros(15)])
    assert voxel_glyph(volume, (1, 0, 0), "tournier", DETAIL).count == 0
