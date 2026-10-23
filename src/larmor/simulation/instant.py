"""The instant engine, which sums closed-form compartment signals in each voxel."""

import math
from dataclasses import dataclass, replace

import numpy as np

from larmor import _core
from larmor._threads import n_threads
from larmor.simulation.noise import add_noise
from larmor.simulation.phantom import (
    Phantom,
    Progress,
    VoxelPartition,
    partition_voxels,
    resolve_bundle_seeds,
)
from larmor.simulation.protocol import Protocol, PulseSequence
from larmor.simulation.tissues import Tissue
from larmor.types import GradientTable, Volume

# Voxels per call to the compiled engine.
CHUNK_VOXELS = 16_384


@dataclass(frozen=True, eq=False)
class SimulatedDWI:
    """Holds the images, the ground truth and the inputs that `simulate` returns.

    Attributes:
        echoes: One 4D float32 volume (X, Y, Z, N) per echo time, with noise if
            an SNR was given.
        noise_free: The same without noise.
        gradients: The gradient table of the N volumes.
        echo_times_ms: The echo time of each volume in `echoes`, ms.
        partition: The ground truth, with each voxel's tissue fractions and
            fibres and the streamlines drawn for each bundle.
        sigma: The noise standard deviation, or None without noise.
        phantom: The phantom that was simulated, with a seed on every bundle
            geometry, so that it gives the same streamlines again.
        protocol: The scan that was simulated.
        snr: The signal-to-noise ratio asked for, or None without noise.
        noise_reference: The name of the tissue that defines the noise level.
        seed: The seed of the noise and the bundles, the one given or the one
            drawn when none was given.
    """

    echoes: tuple[Volume, ...]
    noise_free: tuple[Volume, ...]
    gradients: GradientTable
    echo_times_ms: tuple[float, ...]
    partition: VoxelPartition
    sigma: float | None
    phantom: Phantom | None = None
    protocol: Protocol | None = None
    snr: float | None = None
    noise_reference: str = "WM"
    seed: int | None = None


@dataclass(frozen=True, eq=False)
class InstantArrays:
    """A phantom and a protocol as the flat arrays the compiled engine takes.

    The compartments of all tissues are listed in order, C in all, and the voxels
    in the order of the flattened grid, V in all.

    Attributes:
        models: The model code of each compartment, (C,).
        parameters: The diffusivity, perpendicular diffusivity, radius and kappa
            of each compartment, (C, 4).
        tissue: The tissue of each compartment, (C,).
        fraction: Each compartment's fraction within its tissue, (C,).
        relaxation: Each compartment's relaxation weight at each echo, (C, E).
        tissue_fractions: Each voxel's fraction of each tissue, with the
            background added to its tissue, (V, T).
        offsets: Where each voxel's fibres start, (V + 1,).
        fibre_tissue: Each fibre's tissue, (F,).
        fibre_weight: Each fibre's share of its voxel, (F,).
        fibre_direction: Each fibre's unit direction, (F, 3).
    """

    models: np.ndarray
    parameters: np.ndarray
    tissue: np.ndarray
    fraction: np.ndarray
    relaxation: np.ndarray
    tissue_fractions: np.ndarray
    offsets: np.ndarray
    fibre_tissue: np.ndarray
    fibre_weight: np.ndarray
    fibre_direction: np.ndarray


def flatten(
    phantom: Phantom, protocol: Protocol, partition: VoxelPartition
) -> InstantArrays:
    """Returns the compiled engine's arrays for a phantom, protocol and partition."""
    compartments = [
        (index, c) for index, t in enumerate(phantom.tissues) for c in t.compartments
    ]
    decay = [
        c.t2star_ms if protocol.sequence == PulseSequence.GRADIENT_ECHO else c.t2_ms
        for _, c in compartments
    ]
    relaxation = _core.relaxation_weights(
        np.array([c.proton_density for _, c in compartments]),
        np.array([c.t1_ms for _, c in compartments]),
        np.array(decay, dtype=np.float64),
        protocol.tr_ms,
        np.array(protocol.te_ms),
    )
    num_tissues = len(phantom.tissues)
    fractions = partition.tissue_fractions.reshape(-1, num_tissues).copy()
    if phantom.background is not None:
        column = partition.tissue_names.index(phantom.background)
        fractions[:, column] += partition.background.ravel()
    return InstantArrays(
        models=np.array([c.model.code for _, c in compartments], np.int64),
        parameters=np.array(
            [c.engine_parameters for _, c in compartments], dtype=np.float64
        ).reshape(-1, 4),
        tissue=np.array([index for index, _ in compartments], dtype=np.int64),
        fraction=np.array([c.fraction for _, c in compartments], dtype=np.float64),
        relaxation=np.ascontiguousarray(relaxation),
        tissue_fractions=np.ascontiguousarray(fractions),
        offsets=partition.offsets,
        fibre_tissue=partition.fibre_tissue,
        fibre_weight=partition.fibre_weight,
        fibre_direction=partition.fibre_direction,
    )


def relaxation_weights(tissue: Tissue, protocol: Protocol) -> np.ndarray:
    """Returns the relaxation weight of each compartment of the tissue at each echo.

    The weight is PD (1 - exp(-TR / T1)) exp(-TE / T2), with T2* for a gradient
    echo; shape (C, E).
    """
    decay = [
        c.t2star_ms if protocol.sequence == PulseSequence.GRADIENT_ECHO else c.t2_ms
        for c in tissue.compartments
    ]
    weights: np.ndarray = _core.relaxation_weights(
        np.array([c.proton_density for c in tissue.compartments]),
        np.array([c.t1_ms for c in tissue.compartments]),
        np.array(decay, dtype=np.float64),
        protocol.tr_ms,
        np.array(protocol.te_ms),
    )
    return weights


def reference_s0(tissue: Tissue, protocol: Protocol) -> np.ndarray:
    """Returns the b = 0 signal of a voxel full of the tissue at each echo, (E,)."""
    weights = relaxation_weights(tissue, protocol)
    s0: np.ndarray = np.array([c.fraction for c in tissue.compartments]) @ weights
    return s0


def simulate(
    phantom: Phantom,
    protocol: Protocol,
    *,
    snr: float | None = None,
    noise_reference: str = "WM",
    seed: int | None = None,
    n_jobs: int = -1,
    progress: Progress | None = None,
) -> SimulatedDWI:
    """Simulates the diffusion-weighted images of a phantom, voxel by voxel.

    The signal of each compartment in each voxel has a closed form, and the
    voxel's signal is their sum, as if the whole acquisition happened at once:
    there is no k-space, readout, distortion or motion. Each compartment counts
    with its fraction of the voxel and its relaxation weight, with T2* in place
    of T2 for a gradient echo; an oriented compartment is evaluated along each
    fibre in the voxel, weighted by the fibre's share. The result does not
    depend on n_jobs.

    With `snr`, Rician noise is added with sigma = S0_ref / snr, where S0_ref is
    the b = 0 signal, at the first echo time, of a voxel filled with the
    `noise_reference` tissue.

    Args:
        phantom: The object to simulate.
        protocol: The scan.
        snr: The signal-to-noise ratio of S0_ref. Defaults to None, no noise.
        noise_reference: The name of the tissue that defines S0_ref. Defaults to
            "WM".
        seed: The seed of the noise and of the bundle geometries that have no
            seed of their own (see `partition_voxels`); the same seed gives the
            same result. Defaults to None, which draws a seed and returns it in
            the result.
        n_jobs: The number of threads, -1 for all cores. Defaults to -1.
        progress: A function called with (done, total, stage) as the run goes
            on, where stage is "voxelise", "signal" or "noise"; raise an
            exception in it to cancel the run. Defaults to None.

    Returns:
        The images, the noise-free images and the ground truth.

    Raises:
        ValueError: If snr is not positive, the noise reference is not a tissue
            of the phantom, or n_jobs is 0 or below -1.
    """
    threads = n_threads(n_jobs)
    sigma = None
    if snr is not None:
        if not (math.isfinite(snr) and snr > 0):
            raise ValueError(f"snr must be positive, got {snr}.")
        sigma = float(reference_s0(phantom.tissue(noise_reference), protocol)[0]) / snr

    if seed is None:
        seed = int(np.random.SeedSequence().entropy)  # type: ignore[arg-type]
    phantom = replace(phantom, bundles=resolve_bundle_seeds(phantom, seed))
    partition = partition_voxels(phantom, seed=seed, n_jobs=n_jobs, progress=progress)
    arrays = flatten(phantom, protocol, partition)
    gradients = protocol.gradients
    bvals = np.ascontiguousarray(gradients.bvals, dtype=np.float64)
    bvecs = np.ascontiguousarray(gradients.bvecs, dtype=np.float64)
    num_voxels = arrays.tissue_fractions.shape[0]
    out = np.empty((num_voxels, protocol.num_echoes, len(bvals)), dtype=np.float32)
    for start in range(0, num_voxels, CHUNK_VOXELS):
        stop = min(start + CHUNK_VOXELS, num_voxels)
        first, last = arrays.offsets[start], arrays.offsets[stop]
        _core.simulate_instant(
            arrays.models,
            arrays.parameters,
            arrays.tissue,
            arrays.fraction,
            arrays.relaxation,
            bvals,
            bvecs,
            protocol.small_delta_ms,
            protocol.big_delta_ms,
            arrays.tissue_fractions[start:stop],
            arrays.offsets[start : stop + 1] - first,
            arrays.fibre_tissue[first:last],
            arrays.fibre_weight[first:last],
            arrays.fibre_direction[first:last],
            out[start:stop],
            threads,
        )
        if progress is not None:
            progress(stop, num_voxels, "signal")

    grid = phantom.grid
    noise_free = tuple(
        Volume(
            np.ascontiguousarray(out[:, e, :]).reshape(*grid.shape, len(bvals)),
            grid.affine.copy(),
        )
        for e in range(protocol.num_echoes)
    )
    echoes = noise_free
    if sigma is not None:
        echoes = add_noise(noise_free, sigma, seed=seed)
        if progress is not None:
            progress(1, 1, "noise")
    return SimulatedDWI(
        echoes=echoes,
        noise_free=noise_free,
        gradients=gradients,
        echo_times_ms=protocol.te_ms,
        partition=partition,
        sigma=sigma,
        phantom=phantom,
        protocol=protocol,
        snr=snr,
        noise_reference=noise_reference,
        seed=seed,
    )
