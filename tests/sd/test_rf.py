import numpy as np
import numpy.typing as npt
import pytest

from larmor.sd import (
    AnisotropicRF,
    IsotropicRF,
    NoddiRF,
    build_response_functions,
)
from larmor.types import GradientTable, Volume

SHAPE = (3, 3, 3)


def _dwi(
    eigen_vals: list[float],
    elsewhere: tuple[npt.NDArray[np.bool_], list[float]] | None = None,
) -> tuple[Volume, GradientTable]:
    """A noise-free DWI whose every voxel has the signal of a diagonal tensor.

    With `elsewhere` = (region, other eigenvalues), the voxels outside the region
    have the other tensor instead.
    """
    rng = np.random.default_rng(0)
    dirs = rng.normal(size=(30, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    bvals = np.concatenate([[0.0], np.full(30, 1000.0)])
    bvecs = np.vstack([np.zeros(3), dirs])

    def signal(vals: list[float]) -> npt.NDArray[np.float64]:
        # numpy types einsum as Any.
        gdg: npt.NDArray[np.float64] = np.einsum(
            "ni,ij,nj->n", bvecs, np.diag(vals), bvecs
        )
        return 1000.0 * np.exp(-bvals * gdg)

    data = np.broadcast_to(signal(eigen_vals), (*SHAPE, len(bvals))).astype(np.float32)
    if elsewhere is not None:
        region, other = elsewhere
        data[~region] = signal(other)

    return Volume(data, np.eye(4)), GradientTable(bvals, bvecs)


@pytest.fixture
def mask() -> npt.NDArray[np.bool_]:
    return np.ones(SHAPE, dtype=bool)


def test_given_eigenvalues_are_used_as_they_are(
    mask: npt.NDArray[np.bool_],
) -> None:
    # Isotropic data has no voxel with FA > 0.7, so estimating would raise: the
    # given eigenvalues must be used without looking at the data.
    dwi = _dwi([0.8e-3, 0.8e-3, 0.8e-3])
    rfs = build_response_functions(
        *dwi, mask, wm_eigen_vals=(1.7e-3, 0.2e-3, 0.2e-3), mean_kurtosis=0.5
    )

    assert rfs == [
        AnisotropicRF(eigen_vals=[1.7e-3, 0.2e-3, 0.2e-3], mean_kurtosis=0.5)
    ]


def test_white_matter_is_estimated_without_eigenvalues(
    mask: npt.NDArray[np.bool_],
) -> None:
    dwi = _dwi([1.7e-3, 0.2e-3, 0.2e-3])
    (wm,) = build_response_functions(*dwi, mask)

    assert isinstance(wm, AnisotropicRF)
    np.testing.assert_allclose(wm.eigen_vals, [1.7e-3, 0.2e-3, 0.2e-3], atol=1e-7)
    assert wm.mean_kurtosis == 0.0  # single shell: no kurtosis is estimated


def test_white_matter_is_estimated_inside_the_mask_only() -> None:
    # Outside the mask, most voxels hold a different high-FA tensor: estimating over
    # every voxel would move the median first eigenvalue to 2.5e-3.
    mask = np.zeros(SHAPE, dtype=bool)
    mask[0] = True
    dwi = _dwi([1.7e-3, 0.2e-3, 0.2e-3], (mask, [2.5e-3, 0.1e-3, 0.1e-3]))
    (wm,) = build_response_functions(*dwi, mask)

    assert isinstance(wm, AnisotropicRF)
    np.testing.assert_allclose(wm.eigen_vals, [1.7e-3, 0.2e-3, 0.2e-3], atol=1e-7)


def test_given_kurtosis_is_kept_for_an_estimated_response(
    mask: npt.NDArray[np.bool_],
) -> None:
    dwi = _dwi([1.7e-3, 0.2e-3, 0.2e-3])
    (wm,) = build_response_functions(*dwi, mask, mean_kurtosis=0.5)

    assert isinstance(wm, AnisotropicRF)
    assert wm.mean_kurtosis == 0.5
    np.testing.assert_allclose(wm.eigen_vals, [1.7e-3, 0.2e-3, 0.2e-3], atol=1e-7)


@pytest.mark.parametrize(
    ("gm", "csf", "isotropic"),
    [
        (None, None, []),
        (0.7e-3, None, [0.7e-3]),
        (None, 3e-3, [3e-3]),
        (0.7e-3, 3e-3, [0.7e-3, 3e-3]),  # GM before CSF
    ],
)
def test_isotropic_responses_follow_white_matter(
    mask: npt.NDArray[np.bool_],
    gm: float | None,
    csf: float | None,
    isotropic: list[float],
) -> None:
    dwi = _dwi([0.8e-3, 0.8e-3, 0.8e-3])
    rfs = build_response_functions(
        *dwi,
        mask,
        wm_eigen_vals=(1.7e-3, 0.2e-3, 0.2e-3),
        gm_diffusivity=gm,
        csf_diffusivity=csf,
    )

    assert isinstance(rfs[0], AnisotropicRF)
    assert rfs[1:] == [IsotropicRF(diffusivity=d) for d in isotropic]


def test_noddi_grey_matter_comes_between_white_matter_and_csf(
    mask: npt.NDArray[np.bool_],
) -> None:
    dwi = _dwi([0.8e-3, 0.8e-3, 0.8e-3])
    rfs = build_response_functions(
        *dwi,
        mask,
        wm_eigen_vals=(1.7e-3, 0.2e-3, 0.2e-3),
        csf_diffusivity=3e-3,
        gm_noddi=True,
    )

    assert isinstance(rfs[0], AnisotropicRF)
    assert rfs[1:] == [NoddiRF(), IsotropicRF(diffusivity=3e-3)]


def test_noddi_grey_matter_with_a_grey_matter_diffusivity_is_refused(
    mask: npt.NDArray[np.bool_],
) -> None:
    dwi = _dwi([0.8e-3, 0.8e-3, 0.8e-3])
    with pytest.raises(ValueError, match="gm_noddi"):
        build_response_functions(
            *dwi,
            mask,
            wm_eigen_vals=(1.7e-3, 0.2e-3, 0.2e-3),
            gm_diffusivity=0.7e-3,
            gm_noddi=True,
        )
