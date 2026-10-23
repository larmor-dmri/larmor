"""Denoising, then removing Gibbs ringing
======================================

This example builds a noisy diffusion image with Gibbs ringing, denoises it, removes
the ringing, and measures how much noise and ringing each step leaves.

An MRI image is reconstructed from a finite part of k-space. Cutting the spectrum off
at the highest acquired frequency makes every sharp edge ring: next to the border
between CSF and tissue, for example, the image overshoots and then oscillates with a
period of about two voxels. In diffusion data the ringing biases maps such as the
mean diffusivity (MD) and kurtosis near those borders.

`remove_gibbs_ringing` removes it with the local sub-voxel shift method of Kellner et
al. (2016). Each line of a slice is re-interpolated at the sub-voxel shift that
samples the ringing at its zero crossings, chosen voxel by voxel. Gibbs removal comes
after denoising, because MP-PCA denoising (`denoise_mppca`) relies on noise that is
independent from voxel to voxel, and the re-interpolation changes that. Both steps
come before any motion or distortion correction, which interpolates the image.
"""

import numpy as np

from larmor.dwi import denoise_mppca, remove_gibbs_ringing
from larmor.types import Volume

# %%
# Two tissues with sharp borders
# ------------------------------
# A square of tissue holds a disk of CSF. Each region is drawn on a grid eight
# times finer than the 64 x 64 image, and only the frequencies a 64 x 64
# acquisition records are kept, without a window, so each region's map rings.

size, fine = 64, 8
n_fine = size * fine
x, y = np.meshgrid(np.arange(n_fine) / fine, np.arange(n_fine) / fine, indexing="ij")
csf = (x - 32.3) ** 2 + (y - 31.6) ** 2 <= 12.5**2
tissue = (np.abs(x - 32.2) <= 23.7) & (np.abs(y - 31.7) <= 23.7) & ~csf


def acquired(region: np.ndarray) -> np.ndarray:
    """The region's map as a 64 x 64 acquisition records it."""
    spectrum = np.fft.fftshift(np.fft.fft2(region.astype(float)))
    kept = slice(n_fine // 2 - size // 2, n_fine // 2 + size // 2)
    return np.real(np.fft.ifft2(np.fft.ifftshift(spectrum[kept, kept]))) / fine**2


csf_map, tissue_map = acquired(csf), acquired(tissue)

# %%
# The acquisition has two b0 images and 30 at b = 1000 s/mm^2. CSF has twice the S0
# of tissue and diffuses fast (3e-3 mm^2/s against 0.8e-3), so the border is bright
# in the b0 images and dark in the diffusion-weighted ones. The image has six
# identical slices along z, the axis across the slices, and Gaussian noise of
# sigma = 15.

bvals = np.concatenate([[0.0, 0.0], np.full(30, 1000.0)])
csf_signal = 2000.0 * np.exp(-bvals * 3e-3)
tissue_signal = 1000.0 * np.exp(-bvals * 0.8e-3)
slice_signals = csf_map[..., None] * csf_signal + tissue_map[..., None] * tissue_signal
truth = np.repeat(slice_signals[:, :, None, :], 6, axis=2)
rng = np.random.default_rng(0)
noisy = truth + rng.normal(scale=15.0, size=truth.shape)
volume = Volume(noisy.astype(np.float32), np.eye(4))

# %%
# Denoise first, then remove the ringing
# --------------------------------------
# Real data would be read from a file with `load_volume("dwi.nii.gz")`. `slice_axis`
# is the voxel axis across the acquired slices, z by default. By default the method
# tries shifts in steps of 1/40 of a voxel, up to half a voxel each way, and measures
# the ringing over the neighbours 1 to 3 voxels away.

denoised = denoise_mppca(volume).volume
corrected = remove_gibbs_ringing(denoised, slice_axis=2)

# %%
# What each step leaves
# ---------------------
# The image the scanner would give without ringing or noise has the two signals
# exactly, inside each region. The RMS error against it, over the voxels at least
# 1.5 voxels from a border (where a voxel would hold both regions), measures what
# is left of the noise and of the ringing after each step.

ideal = (
    csf[::fine, ::fine][:, :, None, None] * csf_signal
    + tissue[::fine, ::fine][:, :, None, None] * tissue_signal
)
row, column = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
from_csf = np.abs(np.hypot(row - 32.3, column - 31.6) - 12.5)
from_square = np.minimum(
    np.abs(np.abs(row - 32.2) - 23.7), np.abs(np.abs(column - 31.7) - 23.7)
)
away = (from_csf > 1.5) & (from_square > 1.5)


def error(values: np.ndarray) -> float:
    """RMS error against the ideal image, away from the borders."""
    return float(np.sqrt(np.mean((values - ideal)[away] ** 2)))


for name, values in (
    ("noisy", noisy),
    ("denoised", denoised.data),
    ("denoised, then unringed", corrected.data),
):
    print(f"{name}: RMS error {error(values):.1f}")

assert error(denoised.data) < error(noisy)
assert error(corrected.data) < 0.5 * error(denoised.data)
