"""The numpy reference of MP-PCA denoising, which the compiled version must match."""

import numpy as np


def split_eigenvalues(ascending: np.ndarray, n_larger: int) -> tuple[int, float]:
    """The signal components and noise variance (Veraart et al. 2016, eq. 10-12)."""
    eigenvalues = np.maximum(ascending, 0.0)
    m = len(eigenvalues)
    for p in range(m):
        noise = eigenvalues[: m - p]
        width = (noise[-1] - noise[0]) / (4 * np.sqrt((m - p) / n_larger))
        if noise.mean() >= width:
            return p, float(noise.mean())
    return m, 0.0


def denoise_voxel(
    image: np.ndarray, voxel: tuple[int, int, int], patch: tuple[int, int, int]
) -> tuple[np.ndarray, float, int]:
    """The denoised signal, sigma and number of signal components of one voxel."""
    starts = [
        min(max(position - size // 2, 0), length - size)
        for position, size, length in zip(voxel, patch, image.shape[:3], strict=True)
    ]
    box = tuple(
        slice(start, start + size) for start, size in zip(starts, patch, strict=True)
    )
    x = image[box].reshape(-1, image.shape[3]).T.astype(np.float64)  # (M, N)
    centre = np.ravel_multi_index(
        tuple(position - start for position, start in zip(voxel, starts, strict=True)),
        patch,
    )
    m_volumes, n_voxels = x.shape
    n_larger = max(m_volumes, n_voxels)
    gram = x @ x.T if m_volumes <= n_voxels else x.T @ x
    values, vectors = np.linalg.eigh(gram)
    num_signal, variance = split_eigenvalues(values / n_larger, n_larger)
    signal = vectors[:, len(values) - num_signal :]
    if m_volumes <= n_voxels:
        denoised = signal @ (signal.T @ x[:, centre])
    else:
        denoised = x @ (signal @ signal[centre])
    return denoised, np.sqrt(variance), num_signal


def denoise(image: np.ndarray, patch: tuple[int, int, int]) -> tuple[np.ndarray, ...]:
    """Every voxel of a 4D image denoised: the signals, sigma and signal components."""
    denoised = np.zeros(image.shape)
    noise = np.zeros(image.shape[:3])
    num_signal = np.zeros(image.shape[:3], dtype=np.int64)
    for voxel in np.ndindex(*image.shape[:3]):
        denoised[voxel], noise[voxel], num_signal[voxel] = denoise_voxel(
            image, voxel, patch
        )
    return denoised, noise, num_signal
