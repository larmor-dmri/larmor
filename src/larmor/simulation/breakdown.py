"""The signal of one voxel, factor by factor, as the instant engine computes it."""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.special import i0e, i1e

from larmor import _core
from larmor.simulation.instant import reference_s0
from larmor.simulation.phantom import Phantom, VoxelPartition
from larmor.simulation.protocol import Protocol, PulseSequence
from larmor.simulation.tissues import CompartmentModel, Tissue

# A share of the voxel below this counts as none.
_NO_SHARE = 1e-12

# Each model's water in plain words.
_MODEL_WORDS = {
    CompartmentModel.STICK: "axons",
    CompartmentModel.ZEPPELIN: "outside axons",
    CompartmentModel.POWDER_STICK: "neurites",
    CompartmentModel.SPHERE: "soma",
    CompartmentModel.WATSON_STICK: "neurites",
    CompartmentModel.CYLINDER: "axons",
    CompartmentModel.DOT: "still water",
    CompartmentModel.FREE_WATER: "free water",
}
# The models whose water is in cells other than axons.
_CELLS = {
    CompartmentModel.SPHERE,
    CompartmentModel.POWDER_STICK,
    CompartmentModel.WATSON_STICK,
}


@dataclass(frozen=True)
class CompartmentSignal:
    """One compartment's share of a voxel's signal, factor by factor.

    The compartment's signal is the product of its factors.

    Attributes:
        label: The compartment in plain words, such as "WM axons".
        tissue: The name of its tissue.
        model: Its diffusion model.
        bundle: The index of the bundle whose share of the voxel this is, or
            None for the share the tissue maps and the background give.
        partial_volume: The share of the voxel its tissue fills here (the
            bundle's share, for a bundle).
        fraction: The compartment's fraction of its tissue.
        proton_density: Its proton density.
        t1_factor: 1 - exp(-TR / T1).
        t2_factor: exp(-TE / T2), with T2* for a gradient echo.
        attenuation: The diffusion attenuation at the volume's b-value and
            direction, averaged over the bundle's fibre directions in the voxel,
            each weighted by its share. Oriented compartments occur only in
            bundles.
    """

    label: str
    tissue: str
    model: CompartmentModel
    bundle: int | None
    partial_volume: float
    fraction: float
    proton_density: float
    t1_factor: float
    t2_factor: float
    attenuation: float

    @property
    def no_decay(self) -> float:
        """Returns the water before decay, partial volume x fraction x PD."""
        return self.partial_volume * self.fraction * self.proton_density

    @property
    def after_decay(self) -> float:
        """Returns the water after the T1 and T2 (or T2*) factors."""
        return self.no_decay * self.t1_factor * self.t2_factor

    @property
    def signal(self) -> float:
        """Returns the signal after the diffusion weighting too."""
        return self.after_decay * self.attenuation


@dataclass(frozen=True)
class BundleShare:
    """A bundle's share of the voxel in a `SignalBreakdown`.

    Attributes:
        index: The bundle's index in the phantom.
        name: The bundle's name.
        tissue: The tissue that fills it.
        share: The share of the voxel it fills.
        angle_deg: The angle between its main fibre direction in the voxel and
            the gradient, 0 to 90 degrees; None at b = 0.
    """

    index: int
    name: str
    tissue: str
    share: float
    angle_deg: float | None


@dataclass(frozen=True)
class SignalBreakdown:
    """The noise-free signal of one voxel in one volume, split into its factors.

    Its properties give the signal at three steps: the water in the voxel before
    decay, the signal after relaxation, and the signal after the diffusion
    weighting.

    Attributes:
        voxel: The voxel (i, j, k).
        volume: The volume's index in the protocol's gradient table.
        echo: The echo's index in the protocol's echo times.
        b_value: The volume's b-value, s/mm^2.
        direction: The volume's unit gradient direction (zero at b = 0).
        te_ms: The echo time, ms.
        tr_ms: The repetition time, ms.
        sequence: The pulse sequence.
        tissue_fractions: A (name, share) pair for each tissue of the phantom in
            the voxel, the background tissue included.
        bundles: The bundles in the voxel.
        compartments: The compartments with a share of the voxel, tissue by
            tissue; a tissue that fills bundles is listed once per bundle.
        sigma: The noise standard deviation; None without noise.
    """

    voxel: tuple[int, int, int]
    volume: int
    echo: int
    b_value: float
    direction: tuple[float, float, float]
    te_ms: float
    tr_ms: float
    sequence: PulseSequence
    tissue_fractions: tuple[tuple[str, float], ...]
    bundles: tuple[BundleShare, ...]
    compartments: tuple[CompartmentSignal, ...]
    sigma: float | None

    @property
    def no_decay(self) -> float:
        """Returns the water in the voxel, the sum of the compartments' water."""
        return math.fsum(c.no_decay for c in self.compartments)

    @property
    def after_decay(self) -> float:
        """Returns the signal after relaxation, which is the signal at b = 0."""
        return math.fsum(c.after_decay for c in self.compartments)

    @property
    def signal(self) -> float:
        """Returns the noise-free signal after the diffusion weighting."""
        return math.fsum(c.signal for c in self.compartments)

    @property
    def noise_floor(self) -> float | None:
        """Returns the mean magnitude of pure noise, sigma sqrt(pi / 2), or None."""
        if self.sigma is None:
            return None
        return self.sigma * math.sqrt(math.pi / 2)

    @property
    def expected_magnitude(self) -> float | None:
        """Returns the mean of the noisy (Rician) magnitude, or None without noise."""
        if self.sigma is None:
            return None
        return rician_mean(self.signal, self.sigma)


def rician_mean(signal: float, sigma: float) -> float:
    """Returns the mean of the Rician magnitude of a signal with noise sigma.

    The mean is sigma sqrt(pi / 2) L(-z) with z = signal^2 / (2 sigma^2) and L
    the Laguerre function of order 1/2.

    Returns:
        The mean magnitude; the signal itself when sigma is 0.
    """
    if sigma == 0:
        return abs(signal)
    z = signal * signal / (2 * sigma * sigma)
    laguerre = (1 + z) * i0e(z / 2) + z * i1e(z / 2)
    return float(sigma * math.sqrt(math.pi / 2) * laguerre)


def signal_breakdown(
    phantom: Phantom,
    protocol: Protocol,
    partition: VoxelPartition,
    voxel: Sequence[int],
    volume: int,
    *,
    echo: int = 0,
    snr: float | None = None,
    noise_reference: str = "WM",
    bundle_names: Sequence[str] | None = None,
) -> SignalBreakdown:
    """Splits one voxel's noise-free signal in one volume into its factors.

    The signal is the one `simulate` computes for the voxel, from the same
    compartment models. A tissue that fills bundles is split by bundle, since
    each bundle has its own fibre directions. With `snr`, the noise is that of
    `simulate`, sigma = S0_ref / snr.

    Args:
        phantom: The phantom.
        protocol: The scan.
        partition: The phantom's voxel partition, as `partition_voxels` or
            `simulate` returns it.
        voxel: The voxel (i, j, k).
        volume: The volume's index in the protocol's gradient table.
        echo: The echo's index in the protocol's echo times. Defaults to 0.
        snr: The signal-to-noise ratio of S0_ref, as `simulate` takes it.
            Defaults to None, no noise.
        noise_reference: The name of the tissue that defines S0_ref. Defaults
            to "WM".
        bundle_names: A name for each bundle of the phantom. Defaults to None,
            "bundle 1", "bundle 2" and so on.

    Returns:
        The breakdown.

    Raises:
        ValueError: If the voxel is outside the grid, the volume or echo does
            not exist, the partition is not the phantom's, bundle_names does not
            name every bundle, snr is not positive, or the noise reference is
            not a tissue of the phantom.

    References:
        Gudbjartsson H, Patz S. The Rician distribution of noisy MRI data. Magn
        Reson Med 34 (1995) 910-914.
    """
    index = _check_voxel(voxel, partition)
    gradients = protocol.gradients
    _check_volume_and_echo(protocol, volume, echo)
    names = tuple(t.name for t in phantom.tissues)
    if partition.tissue_names != names or partition.bundle_fractions.shape[-1] != len(
        phantom.bundles
    ):
        raise ValueError("The partition is not the phantom's.")
    bundle_names = _bundle_names(phantom, bundle_names)
    sigma = _noise_sigma(phantom, protocol, snr, noise_reference)

    b_value = float(gradients.bvals[volume])
    direction = np.asarray(gradients.bvecs[volume], dtype=np.float64)
    fractions = np.asarray(partition.tissue_fractions[index], dtype=np.float64).copy()
    if phantom.background is not None:
        fractions[names.index(phantom.background)] += float(partition.background[index])
    shares, bundles = _shares(
        phantom,
        partition,
        index,
        fractions,
        bundle_names,
        direction,
        b_value,
        gradients.b0_threshold,
    )

    timing = (protocol.small_delta_ms, protocol.big_delta_ms)
    attenuations = _attenuations(shares, b_value, direction, timing)
    te = protocol.te_ms[echo]
    compartments = _compartment_signals(shares, attenuations, protocol, te)
    return SignalBreakdown(
        voxel=index,
        volume=volume,
        echo=echo,
        b_value=b_value,
        direction=(float(direction[0]), float(direction[1]), float(direction[2])),
        te_ms=te,
        tr_ms=protocol.tr_ms,
        sequence=protocol.sequence,
        tissue_fractions=tuple(
            (name, float(f)) for name, f in zip(names, fractions, strict=True)
        ),
        bundles=tuple(bundles),
        compartments=tuple(compartments),
        sigma=sigma,
    )


def _check_volume_and_echo(protocol: Protocol, volume: int, echo: int) -> None:
    """Checks that the volume and the echo exist in the protocol.

    Raises:
        ValueError: If either index is out of range.
    """
    gradients = protocol.gradients
    if not 0 <= volume < len(gradients.bvals):
        raise ValueError(
            f"volume must be in [0, {len(gradients.bvals)}), got {volume}."
        )
    if not 0 <= echo < protocol.num_echoes:
        raise ValueError(f"echo must be in [0, {protocol.num_echoes}), got {echo}.")


def _bundle_names(
    phantom: Phantom, bundle_names: Sequence[str] | None
) -> Sequence[str]:
    """Returns the bundle names given, or "bundle 1", "bundle 2" and so on.

    Raises:
        ValueError: If the names given do not name every bundle.
    """
    if bundle_names is None:
        return [f"bundle {k + 1}" for k in range(len(phantom.bundles))]
    if len(bundle_names) != len(phantom.bundles):
        raise ValueError(
            f"bundle_names must name {len(phantom.bundles)} bundles, got "
            f"{len(bundle_names)}."
        )
    return bundle_names


def _noise_sigma(
    phantom: Phantom, protocol: Protocol, snr: float | None, noise_reference: str
) -> float | None:
    """Returns the noise level `simulate` uses for this snr, or None without snr.

    Raises:
        ValueError: If snr is not positive.
    """
    if snr is None:
        return None
    if not (math.isfinite(snr) and snr > 0):
        raise ValueError(f"snr must be positive, got {snr}.")
    return float(reference_s0(phantom.tissue(noise_reference), protocol)[0]) / snr


def _shares(
    phantom: Phantom,
    partition: VoxelPartition,
    index: tuple[int, int, int],
    fractions: np.ndarray,
    bundle_names: Sequence[str],
    direction: np.ndarray,
    b_value: float,
    b0_threshold: float,
) -> tuple[
    list[tuple[Tissue, int | None, float, np.ndarray, np.ndarray]], list[BundleShare]
]:
    """Splits the voxel's tissues into shares, one per bundle a tissue fills.

    Returns:
        Each tissue's shares, one per bundle it fills here, then the rest (maps and
        background), with the fibres (weight, direction) that orient each; and
        the bundles in the voxel.
    """
    flat = int(np.ravel_multi_index(index, partition.grid.shape))
    first, last = int(partition.offsets[flat]), int(partition.offsets[flat + 1])
    fibre_bundle = partition.fibre_bundle[first:last]
    fibre_weight = partition.fibre_weight[first:last]
    fibre_direction = partition.fibre_direction[first:last]
    occupancy = np.asarray(partition.bundle_fractions[index], dtype=np.float64)

    shares: list[tuple[Tissue, int | None, float, np.ndarray, np.ndarray]] = []
    bundles: list[BundleShare] = []
    for t, tissue in enumerate(phantom.tissues):
        bundled = 0.0
        for k, bundle in enumerate(phantom.bundles):
            if bundle.tissue != tissue.name or occupancy[k] <= _NO_SHARE:
                continue
            mine = fibre_bundle == k
            weights = fibre_weight[mine] / occupancy[k]
            directions = fibre_direction[mine]
            shares.append((tissue, k, float(occupancy[k]), weights, directions))
            bundles.append(
                BundleShare(
                    k,
                    str(bundle_names[k]),
                    tissue.name,
                    float(occupancy[k]),
                    _angle(direction, b_value, b0_threshold, weights, directions),
                )
            )
            bundled += float(occupancy[k])
        rest = float(fractions[t]) - bundled
        if rest > _NO_SHARE:
            shares.append((tissue, None, rest, np.ones(0), np.zeros((0, 3))))
    return shares, bundles


def _compartment_signals(
    shares: Sequence[tuple[Tissue, int | None, float, np.ndarray, np.ndarray]],
    attenuations: list[list[float]],
    protocol: Protocol,
    te: float,
) -> list[CompartmentSignal]:
    """Returns the factors of the signal of each compartment of each share."""
    compartments = []
    for (tissue, owner, share, _, _), attenuation in zip(
        shares, attenuations, strict=True
    ):
        labels = _labels(tissue)
        for c, label, a in zip(tissue.compartments, labels, attenuation, strict=True):
            decay = (
                c.t2star_ms
                if protocol.sequence == PulseSequence.GRADIENT_ECHO
                else c.t2_ms
            )
            compartments.append(
                CompartmentSignal(
                    label=label,
                    tissue=tissue.name,
                    model=c.model,
                    bundle=owner,
                    partial_volume=share,
                    fraction=c.fraction,
                    proton_density=c.proton_density,
                    t1_factor=-math.expm1(-protocol.tr_ms / c.t1_ms),
                    t2_factor=math.exp(-te / decay),
                    attenuation=float(a),
                )
            )
    return compartments


def _check_voxel(
    voxel: Sequence[int], partition: VoxelPartition
) -> tuple[int, int, int]:
    """Returns the voxel as three ints; raises ValueError outside the grid."""
    shape = partition.grid.shape
    if len(voxel) != 3 or not all(
        0 <= int(v) < n for v, n in zip(voxel, shape, strict=True)
    ):
        raise ValueError(f"The voxel {tuple(voxel)} is outside the grid {shape}.")
    return (int(voxel[0]), int(voxel[1]), int(voxel[2]))


def _angle(
    gradient: np.ndarray,
    b_value: float,
    b0_threshold: float,
    weights: np.ndarray,
    directions: np.ndarray,
) -> float | None:
    """Returns the angle in degrees between the heaviest fibre and the gradient.

    Returns None at b = 0 or without fibres.
    """
    norm = float(np.linalg.norm(gradient))
    if b_value <= b0_threshold or norm == 0 or len(weights) == 0:
        return None
    main = directions[int(np.argmax(weights))]
    cosine = abs(float(main @ gradient)) / (norm * float(np.linalg.norm(main)))
    return math.degrees(math.acos(min(cosine, 1.0)))


def _attenuations(
    shares: Sequence[tuple[Tissue, int | None, float, np.ndarray, np.ndarray]],
    b_value: float,
    gradient: np.ndarray,
    timing: tuple[float, float],
) -> list[list[float]]:
    """Computes the attenuation of each compartment of each share.

    An oriented compartment is evaluated along each fibre of its bundle and
    averaged by the fibre weights.

    Returns:
        For each share, the attenuation of each compartment of its tissue.
    """
    found = []
    for tissue, _, _, weights, directions in shares:
        values = []
        for c in tissue.compartments:
            if c.is_oriented:
                cosines, use = directions @ gradient, weights
            else:
                cosines, use = np.zeros(1), np.ones(1)
            if len(cosines) == 0:
                # Only a rounding remainder of a bundled tissue has no fibres.
                values.append(0.0)
                continue
            attenuation = _core.compartment_attenuation(
                np.array([c.model.code], dtype=np.int64),
                np.array([c.engine_parameters], dtype=np.float64),
                np.full(len(cosines), b_value, dtype=np.float64),
                np.ascontiguousarray(cosines, dtype=np.float64),
                timing[0],
                timing[1],
            )[0]
            values.append(float(attenuation @ use))
        found.append(values)
    return found


def _labels(tissue: Tissue) -> list[str]:
    """Returns a label for each compartment of a tissue, such as "GM soma"."""
    models = {c.model for c in tissue.compartments}
    labels = []
    for c in tissue.compartments:
        water = c.model in (CompartmentModel.BALL, CompartmentModel.FREE_WATER)
        if water and len(tissue.compartments) == 1:
            labels.append(tissue.name)
        elif c.model == CompartmentModel.ZEPPELIN and (
            CompartmentModel.WATSON_STICK in models
        ):
            labels.append(f"{tissue.name} outside neurites")
        elif c.model != CompartmentModel.BALL:
            labels.append(f"{tissue.name} {_MODEL_WORDS[c.model]}")
        elif models & _CELLS:
            labels.append(f"{tissue.name} outside cells")
        else:
            labels.append(f"{tissue.name} free water")
    # Two compartments of one kind are numbered.
    for label in set(labels):
        if labels.count(label) > 1:
            numbers = iter(range(1, len(labels) + 1))
            labels = [f"{x} {next(numbers)}" if x == label else x for x in labels]
    return labels
