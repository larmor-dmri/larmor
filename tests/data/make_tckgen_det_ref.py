"""Generates the deterministic tracking references in tests/data/tracking/.

A synthetic FOD (16x16x5 voxels of 2 mm, x axis flipped, lmax 8, tournier basis,
float32) with fibres in the xy plane, the same in every slice (world mm):

- bundle A along +x, in the band |y + 7| <= 2.5, up to x = 3, except for the
  voxels below y = -8 and x = -10, where it is 40 degrees off x;
- from there a quarter circle of radius 8 around (3, 1), which turns A to +y;
- then A goes straight along +y, in the band |x - 11| <= 2.5, to the image's edge;
  for x > 11 the lobe fades from 1 at y = 1 to 0 at y = 15;
- bundle B along y, in the band |x + 3| <= 2.5, which crosses A (both lobes summed
  where they overlap).

Each fibre is a lobe exp(-8 (1 - (u.a)^2)) projected on the SH basis with a
quadrature, so the fitted lobe keeps its axial symmetry and its peak is the axis
a. Every voxel also has an isotropic background of amplitude 0.05, below the
cutoff: MRtrix treats an all-zero voxel as outside the image (`interp/masked.h`),
Larmor as a low FOD. The tracking mask is the voxels with x >= -11, so streamlines
going -x leave it at x = -11.7. MRtrix looks a mask up in the nearest voxel
(`roi.h`), Larmor interpolates it; for this plane between voxel centres both give
the same boundary. Going -x, streamlines below y = -8 meet the 40 degree kink
first, where the peak they follow vanishes and the next one is beyond the
maximum angle. The seed mask is 4 voxels of A, x in {-6.7, -4.7} (the second in
the crossing), y in {-8.9, -6.9}, in the middle slice, where by symmetry the peak
is +x; 8 seeds per voxel make 32 streamlines. The ends are of every kind: the
image's edge, the mask, a low FOD and a sharp turn.

Run from the repository root, with MRtrix 3.0.8 on the PATH:

    uv run python tests/data/make_tckgen_det_ref.py

which writes fod.nii.gz, mask.nii.gz and seeds.nii.gz, then runs in
tests/data/tracking/, with NAME, MAXLEN = det_ref, 80 and det_ref_maxlen, 10:

    tckgen fod.nii.gz NAME.tck -algorithm SD_Stream
        -seed_grid_per_voxel seeds.nii.gz 2 -seed_direction 1,0,0
        -mask mask.nii.gz -step 0.2 -angle 30 -minlength 5 -maxlength MAXLEN
        -cutoff 0.3 -nthreads 0 -output_seeds NAME_seeds.txt

Grid seeding is finite: tckgen tracks every seed once, so -select and -seeds are
not given (tckgen would override them). -seed_direction applies to any seeder.
-noprecomputed is not used: with SD_Stream, tckgen 3.0.8 crashes with a
segmentation fault as soon as tracking starts (`sd_stream.h` leaves its
`precomputer` pointer uninitialised). tckgen then evaluates the SH through a
lookup table of the Legendre values, interpolated linearly in elevation; in this
FOD the streamlines still match Larmor's to about 1e-5 mm.
"""

import functools
import math
import pathlib
import subprocess

import numpy as np
from numpy.typing import NDArray

from larmor.io import save_volume
from larmor.sd.sh import build_basis_matrix
from larmor.types import Volume

OUT = pathlib.Path("tests/data/tracking")
OUT.mkdir(exist_ok=True)

LMAX = 8
SHAPE = (16, 16, 5)
AFFINE = np.array(
    [
        [-2.0, 0.0, 0.0, 15.3],
        [0.0, 2.0, 0.0, -14.9],
        [0.0, 0.0, 2.0, -3.7],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
HALF_WIDTH = 2.5
A_Y = -7.0
TURN_X = 3.0
RADIUS = 8.0
CENTRE = np.array([TURN_X, A_Y + RADIUS])
B_X = -3.0
BACKGROUND = 0.05
MASK_X = -11.0
FADE = 14.0
KINK_X = -10.0
KINK = np.array([math.cos(math.radians(40)), math.sin(math.radians(40)), 0.0])
X = np.array([1.0, 0.0, 0.0])
Y = np.array([0.0, 1.0, 0.0])


@functools.cache
def fit_matrix() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Quadrature-weighted least squares onto the SH basis, and its directions."""
    nodes, weights = np.polynomial.legendre.leggauss(24)
    phis = np.arange(48) * (2 * math.pi / 48)
    ct, ph = np.meshgrid(nodes, phis, indexing="ij")
    st = np.sqrt(1 - ct**2)
    dirs = np.stack(
        [(st * np.cos(ph)).ravel(), (st * np.sin(ph)).ravel(), ct.ravel()], axis=1
    )
    basis = build_basis_matrix(
        np.arccos(dirs[:, 2]), np.arctan2(dirs[:, 1], dirs[:, 0]), LMAX, "tournier"
    )
    root = np.sqrt(np.repeat(weights, 48))[:, np.newaxis]
    return np.linalg.pinv(root * basis) * root.T, dirs


def lobe(axis: NDArray[np.float64]) -> NDArray[np.float64]:
    """SH coefficients of a sharp lobe along axis."""
    fit, dirs = fit_matrix()
    coeffs: NDArray[np.float64] = fit @ np.exp(-8.0 * (1 - (dirs @ axis) ** 2))
    return coeffs


ijk = np.indices(SHAPE).reshape(3, -1).T.astype(np.float64)
world = ijk @ AFFINE[:3, :3].T + AFFINE[:3, 3]
x, y = world[:, 0], world[:, 1]

coeffs = np.zeros((len(world), (LMAX + 1) * (LMAX + 2) // 2))
coeffs[:, 0] = BACKGROUND * math.sqrt(4 * math.pi)

straight = (x < TURN_X) & (np.abs(y - A_Y) <= HALF_WIDTH)
kink = straight & (x < KINK_X) & (y < A_Y - 1.0)
coeffs[straight & ~kink] += lobe(X)
coeffs[kink] += lobe(KINK)

rel = world[:, :2] - CENTRE
radius = np.linalg.norm(rel, axis=1)
arc = (x >= TURN_X) & (y <= CENTRE[1]) & (np.abs(radius - RADIUS) <= HALF_WIDTH)
for v in np.flatnonzero(arc):
    coeffs[v] += lobe(np.array([-rel[v, 1], rel[v, 0], 0.0]) / radius[v])

up = (y > CENTRE[1]) & (np.abs(x - (TURN_X + RADIUS)) <= HALF_WIDTH)
# The outer half fades along y, so where a streamline there stops depends on the
# cutoff.
fade = np.where(x > TURN_X + RADIUS, np.clip(1 - (y - CENTRE[1]) / FADE, 0, 1), 1)
coeffs[up] += fade[up, np.newaxis] * lobe(Y)

coeffs[np.abs(x - B_X) <= HALF_WIDTH] += lobe(Y)

fod = coeffs.reshape(*SHAPE, -1).astype(np.float32)
save_volume(Volume(fod, AFFINE), OUT / "fod.nii.gz")
mask = (x >= MASK_X).reshape(SHAPE).astype(np.uint8)
save_volume(Volume(mask, AFFINE), OUT / "mask.nii.gz")
# Voxel centres x in {-6.7, -4.7}, y in {-8.9, -6.9}, z = 0.3: far enough from the
# kink that the peak at every seed is +x.
seed_voxels = (np.abs(x + 5.7) < 1.5) & (np.abs(y + 7.9) < 1.5) & (ijk[:, 2] == 2)
seeds = seed_voxels.reshape(SHAPE).astype(np.uint8)
save_volume(Volume(seeds, AFFINE), OUT / "seeds.nii.gz")

for name, max_length in [("det_ref", "80"), ("det_ref_maxlen", "10")]:
    subprocess.run(
        [
            "tckgen",
            "fod.nii.gz",
            f"{name}.tck",
            "-algorithm",
            "SD_Stream",
            "-seed_grid_per_voxel",
            "seeds.nii.gz",
            "2",
            "-seed_direction",
            "1,0,0",
            "-mask",
            "mask.nii.gz",
            "-step",
            "0.2",
            "-angle",
            "30",
            "-minlength",
            "5",
            "-maxlength",
            max_length,
            "-cutoff",
            "0.3",
            "-nthreads",
            "0",
            "-output_seeds",
            f"{name}_seeds.txt",
            "-quiet",
            "-force",
        ],
        cwd=OUT,
        check=True,
    )

version = subprocess.run(["tckgen", "-version"], capture_output=True, text=True)
print("saved", OUT, version.stdout.splitlines()[0])  # noqa: T201 (script output)
