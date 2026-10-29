"""Generates the NODDI phantom in tests/data/noddi/, with known parameters per voxel.

Each voxel holds one NODDI tissue (`noddi_tissue`) with its own v_ic, OD and
v_iso, about a random mean orientation mu. The grid axes are the parameters:
x runs over V_IC, y over OD and z over V_ISO. The signal is the sum of the
compartment attenuations weighted by their volume fractions, without
relaxation, so the signal fractions a fit recovers equal the volume fractions.

The protocol has 6 b = 0 volumes and two shells, b = 1000 s/mm^2 with 30
directions and b = 2000 s/mm^2 with 60. The files are:

- dwi.nii.gz: the noise-free signal, with S0 = 1000 in every voxel.
- dwi_noisy.nii.gz: the same with Rician noise of sigma = S0 / SNR.
- mask.nii.gz: every voxel.
- bvals, bvecs: the gradients, FSL format.
- truth.npz: v_ic, od, kappa and v_iso (X, Y, Z), and mu (X, Y, Z, 3) in world
  coordinates.

The NODDI Matlab Toolbox is run on these files by
tests/matlab_scripts/make_noddi_ref.m.

Run from the repository root:

    uv run python tests/data/make_noddi_phantom.py
"""

import pathlib

import numpy as np

from larmor import _core
from larmor.io import save_volume, write_bvals_bvecs
from larmor.simulation import add_noise, create_gradient_table, noddi_tissue
from larmor.types import GradientTable, Volume

OUT = pathlib.Path("tests/data/noddi")
SEED = 0
S0 = 1000.0
SNR = 30.0
VOXEL_MM = 2.0

# Values between the dictionary's grid points too, so a fit is not tested only
# where it is exact.
V_IC = np.linspace(0.1, 0.9, 9)
OD = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.7])
V_ISO = np.array([0.0, 0.1, 0.3])

# The NODDI compartments do not depend on the pulse timings.
SMALL_DELTA_MS = 20.0
BIG_DELTA_MS = 40.0


def kappa_of(od: np.ndarray) -> np.ndarray:
    """Returns the Watson kappa of an orientation dispersion index."""
    return np.asarray(1 / np.tan(od * np.pi / 2))


def random_directions(count: int, rng: np.random.Generator) -> np.ndarray:
    """Returns unit vectors drawn evenly on the sphere, (count, 3)."""
    v = rng.standard_normal((count, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def voxel_signal(
    v_ic: float, kappa: float, v_iso: float, mu: np.ndarray, gradients: GradientTable
) -> np.ndarray:
    """Returns the b0-normalised NODDI signal of one voxel, one value per volume."""
    tissue = noddi_tissue(intra_fraction=v_ic, free_water_fraction=v_iso, kappa=kappa)
    compartments = tissue.compartments
    attenuation = _core.compartment_attenuation(
        np.array([c.model.code for c in compartments], dtype=np.int64),
        np.array([c.engine_parameters for c in compartments], dtype=np.float64),
        np.ascontiguousarray(gradients.bvals, dtype=np.float64),
        np.ascontiguousarray(gradients.bvecs @ mu, dtype=np.float64),
        SMALL_DELTA_MS,
        BIG_DELTA_MS,
    )
    fractions = np.array([c.fraction for c in compartments])
    return np.asarray(fractions @ attenuation)


rng = np.random.default_rng(SEED)
gradients = create_gradient_table([(1000.0, 30), (2000.0, 60)], num_b0=6, seed=SEED)
shape = (len(V_IC), len(OD), len(V_ISO))
num_volumes = len(gradients.bvals)

v_ic, od, v_iso = np.meshgrid(V_IC, OD, V_ISO, indexing="ij")
kappa = kappa_of(od)
mu = random_directions(v_ic.size, rng).reshape(*shape, 3)

signal = np.empty((*shape, num_volumes))
for index in np.ndindex(shape):
    signal[index] = S0 * voxel_signal(
        float(v_ic[index]),
        float(kappa[index]),
        float(v_iso[index]),
        mu[index],
        gradients,
    )

affine = np.diag([VOXEL_MM, VOXEL_MM, VOXEL_MM, 1.0])
noise_free = Volume(signal.astype(np.float32), affine)
(noisy,) = add_noise([noise_free], S0 / SNR, seed=SEED)

OUT.mkdir(parents=True, exist_ok=True)
save_volume(noise_free, OUT / "dwi.nii.gz")
save_volume(noisy, OUT / "dwi_noisy.nii.gz")
save_volume(Volume(np.ones(shape, dtype=np.uint8), affine), OUT / "mask.nii.gz")
write_bvals_bvecs(gradients, OUT / "bvals", OUT / "bvecs", ref_path=OUT / "dwi.nii.gz")
np.savez_compressed(
    OUT / "truth.npz", v_ic=v_ic, od=od, kappa=kappa, v_iso=v_iso, mu=mu
)
print(f"Wrote {OUT}: {shape} voxels, {num_volumes} volumes.")  # noqa: T201 (a script run by hand)
