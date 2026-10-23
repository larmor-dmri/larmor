from pathlib import Path

import numpy as np
import pytest

from larmor.dwi import gibbs, remove_gibbs_ringing
from larmor.io import load_volume
from larmor.types import Volume

GIBBS_DIR = Path(__file__).parents[1] / "data" / "gibbs"


def _ringing_disk(size: int = 48, fine: int = 8) -> np.ndarray:
    """A disk of intensity 100 on 0, kept to the frequencies of a size x size grid."""
    n_fine = size * fine
    x, y = np.meshgrid(
        np.arange(n_fine) / fine, np.arange(n_fine) / fine, indexing="ij"
    )
    disk = 100.0 * ((x - size / 2 - 0.3) ** 2 + (y - size / 2 + 0.2) ** 2 <= 15.2**2)
    spectrum = np.fft.fftshift(np.fft.fft2(disk))
    low = slice(n_fine // 2 - size // 2, n_fine // 2 - size // 2 + size)
    return np.real(np.fft.ifft2(np.fft.ifftshift(spectrum[low, low]))) / fine**2


@pytest.mark.parametrize("name", ["dwi", "phantom"])
def test_matches_mrdegibbs(name: str) -> None:
    """MRtrix3's mrdegibbs (3.0.8, defaults) on a CFIN block and a synthetic image."""
    reference = np.load(GIBBS_DIR / "mrdegibbs_ref.npz")[name]
    result = remove_gibbs_ringing(load_volume(GIBBS_DIR / f"{name}.nii.gz")).data

    # Both compute in float64 and write float32; only the rounding differs.
    np.testing.assert_allclose(
        result,
        reference.reshape(result.shape),
        rtol=0,
        atol=1e-5 * np.abs(reference).max(),
    )


def test_disk_loses_its_ringing() -> None:
    image = _ringing_disk()
    volume = Volume(np.stack([image, image.T], axis=-1).astype(np.float32), np.eye(4))
    result = remove_gibbs_ringing(volume).data

    centre = np.array(image.shape) // 2
    radius = np.hypot(
        *np.meshgrid(
            *(np.arange(n) - c for n, c in zip(image.shape, centre, strict=True)),
            indexing="ij",
        )
    )
    inside = radius < 12
    outside = (radius > 18) & (radius < 22)
    for found, original in ((result[..., 0], image), (result[..., 1], image.T)):
        # The input rings by a few per cent on both sides of the edge; a fifth of that
        # is left.
        assert np.abs(original[inside] - 100).max() > 2
        assert (
            np.abs(found[inside] - 100).max()
            < 0.3 * np.abs(original[inside] - 100).max()
        )
        assert np.abs(found[outside]).max() < 0.3 * np.abs(original[outside]).max()
        # The disk keeps its intensity.
        assert found.sum() == pytest.approx(original.sum(), rel=1e-3)


@pytest.mark.parametrize(
    ("slice_axis", "order"), [(0, (2, 0, 1)), (1, (0, 2, 1))], ids=["x", "y"]
)
def test_slice_axis_picks_the_planes(
    slice_axis: int, order: tuple[int, int, int]
) -> None:
    """Moving the slice axis and naming it gives the same planes, rows and columns."""
    data = np.random.default_rng(0).normal(size=(17, 12, 5)).astype(np.float32)
    along_z = remove_gibbs_ringing(Volume(data, np.eye(4)), slice_axis=2).data
    moved = np.ascontiguousarray(data.transpose(order))
    along_moved = remove_gibbs_ringing(Volume(moved, np.eye(4)), slice_axis=slice_axis)

    np.testing.assert_array_equal(along_moved.data, along_z.transpose(order))


def test_a_nan_spreads_over_its_slice_only() -> None:
    data = np.random.default_rng(3).normal(size=(10, 9, 3)).astype(np.float32)
    data[4, 5, 1] = np.nan
    result = remove_gibbs_ringing(Volume(data, np.eye(4))).data

    assert np.isnan(result[:, :, 1]).all()
    assert np.isfinite(result[:, :, [0, 2]]).all()


def test_result_independent_of_threads_and_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = np.random.default_rng(1).normal(size=(9, 10, 4, 5)).astype(np.float32)
    volume = Volume(data, np.eye(4))
    one_thread = remove_gibbs_ringing(volume, n_jobs=1).data

    monkeypatch.setattr(gibbs, "CHUNK_VOXELS", 2 * 9 * 10 * 4)
    np.testing.assert_array_equal(
        remove_gibbs_ringing(volume, n_jobs=3).data, one_thread
    )


def test_scaling_is_applied() -> None:
    data = np.random.default_rng(2).integers(0, 1000, size=(12, 11, 3)).astype(np.int16)
    scaled = remove_gibbs_ringing(Volume(data, np.eye(4), slope=0.5, intercept=10.0))
    plain = remove_gibbs_ringing(Volume(data * 0.5 + 10.0, np.eye(4)))

    assert scaled.data.dtype == np.float32
    np.testing.assert_allclose(scaled.data, plain.data, rtol=1e-5, atol=1e-3)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"slice_axis": 3}, "slice_axis"),
        ({"num_shifts": 0}, "num_shifts"),
        ({"window": (2, 1)}, "window"),
        ({"n_jobs": 0}, "n_jobs"),
    ],
)
def test_invalid_options_are_refused(options: dict[str, object], message: str) -> None:
    volume = Volume(np.zeros((4, 4, 2), dtype=np.float32), np.eye(4))
    with pytest.raises(ValueError, match=message):
        remove_gibbs_ringing(volume, **options)  # type: ignore[arg-type]


def test_complex_volume_is_refused() -> None:
    data = np.ones((8, 8, 2), dtype=np.complex64)
    with pytest.raises(ValueError, match="complex"):
        remove_gibbs_ringing(Volume(data, np.eye(4)))
