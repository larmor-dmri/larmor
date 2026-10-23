from pathlib import Path

import numpy as np
import pytest
from reference_mppca import denoise

from larmor.dwi import denoise_mppca
from larmor.dwi.denoise import default_patch_size
from larmor.io import load_volume
from larmor.types import Volume

DKI_DIR = Path(__file__).parents[1] / "data" / "dki"


def _block() -> Volume:
    return load_volume(DKI_DIR / "dwi.nii.gz")


def _low_rank_volume(sigma: float, seed: int = 0) -> tuple[Volume, np.ndarray]:
    """A 12x12x12 volume of 60 volumes with a rank-3 signal plus Gaussian noise."""
    rng = np.random.default_rng(seed)
    angles = np.linspace(0, 2 * np.pi, 60)
    profiles = np.stack([np.ones(60), np.cos(angles), np.sin(angles)])
    amplitudes = [1000, 300, 300] * (1 + rng.random((12, 12, 12, 3)))
    truth = amplitudes @ profiles
    noisy = truth + rng.normal(scale=sigma, size=truth.shape)
    return Volume(noisy.astype(np.float32), np.eye(4)), truth


def test_denoising_matches_mrtrix_dwidenoise_on_the_block() -> None:
    ref = np.load(DKI_DIR / "mrtrix_mppca_ref.npz")
    volume = _block()

    result = denoise_mppca(volume, progress=False)
    # Both work in float64 and store float32; measured within 2e-13 before
    # rounding to float32.
    scale = np.abs(ref["denoised"]).max()
    np.testing.assert_allclose(
        result.volume.data, ref["denoised"], rtol=0, atol=1e-6 * scale
    )
    np.testing.assert_allclose(result.noise, ref["noise"], rtol=1e-5)

    # A 5x5x5 patch (fewer voxels than the 133 volumes), given as a NumPy integer,
    # inside a mask.
    box = ref["box"]
    result = denoise_mppca(volume, mask=box, patch_size=np.int64(5), progress=False)
    np.testing.assert_allclose(
        result.volume.data[box], ref["box_denoised"], rtol=0, atol=1e-6 * scale
    )
    np.testing.assert_allclose(result.noise[box], ref["box_noise"], rtol=1e-5)
    assert not result.volume.data[~box].any() and not result.noise[~box].any()


def test_noise_map_recovers_the_noise_level_and_denoising_removes_noise() -> None:
    sigma = 20.0
    volume, truth = _low_rank_volume(sigma)

    result = denoise_mppca(volume, progress=False)

    assert np.median(result.noise) == pytest.approx(sigma, rel=0.03)
    assert np.median(result.num_signal) == 3
    error_before = np.sqrt(np.mean((volume.data - truth) ** 2))
    error_after = np.sqrt(np.mean((result.volume.data - truth) ** 2))
    assert error_after < error_before / 3


@pytest.mark.parametrize(
    "patch", [(3, 3, 3), (3, 5, 1)], ids=["more voxels", "more volumes"]
)
def test_compiled_denoising_matches_the_numpy_reference(
    patch: tuple[int, int, int],
) -> None:
    volume, _ = _low_rank_volume(20.0)
    small = Volume(np.ascontiguousarray(volume.data[:5, :6, :4, :20]), np.eye(4))

    result = denoise_mppca(small, patch_size=patch, n_jobs=2, progress=False)

    denoised, noise, num_signal = denoise(small.data, patch)
    np.testing.assert_allclose(result.volume.data, denoised, rtol=1e-6)
    np.testing.assert_allclose(result.noise, noise, rtol=1e-5)
    np.testing.assert_array_equal(result.num_signal, num_signal)


def test_result_is_the_same_on_any_number_of_threads() -> None:
    volume, _ = _low_rank_volume(20.0)
    one = denoise_mppca(volume, n_jobs=1, progress=False)
    four = denoise_mppca(volume, n_jobs=4, progress=False)
    np.testing.assert_array_equal(one.volume.data, four.volume.data)
    np.testing.assert_array_equal(one.noise, four.noise)


@pytest.mark.parametrize(
    ("n_volumes", "width"), [(1, 1), (27, 3), (28, 5), (125, 5), (198, 7)]
)
def test_default_patch_has_at_least_as_many_voxels_as_volumes(
    n_volumes: int, width: int
) -> None:
    assert default_patch_size(n_volumes) == width


@pytest.mark.parametrize(
    ("data_shape", "patch_size", "message"),
    [((4, 4, 4), None, "4D"), ((4, 4, 4, 5), 4, "odd"), ((4, 4, 4, 5), (3, 3), "odd")],
    ids=["3D", "even", "two sizes"],
)
def test_bad_inputs_are_refused(
    data_shape: tuple[int, ...], patch_size: int | tuple[int, ...] | None, message: str
) -> None:
    volume = Volume(np.ones(data_shape, dtype=np.float32), np.eye(4))
    with pytest.raises(ValueError, match=message):
        denoise_mppca(volume, patch_size=patch_size, progress=False)


def test_complex_volume_is_refused() -> None:
    data = np.ones((6, 6, 6, 10), dtype=np.complex64)
    with pytest.raises(ValueError, match="complex"):
        denoise_mppca(Volume(data, np.eye(4)), progress=False)
