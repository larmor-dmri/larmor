"""The scan: gradients and pulse timings, shared by every simulation engine."""

import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from larmor.types import GradientTable


class PulseSequence(StrEnum):
    """The kind of echo, which decides whether T2 or T2* weights the signal.

    A spin echo refocuses the dephasing from field differences, so its signal
    decays with T2; a gradient echo does not, so its signal decays with T2*.
    """

    SPIN_ECHO = "spin_echo"  # T2
    GRADIENT_ECHO = "gradient_echo"  # T2*

    @classmethod
    def _missing_(cls, value: object) -> PulseSequence:
        """Raises ValueError naming the valid sequences."""
        valid = ", ".join(m.value for m in cls)
        raise ValueError(f"Unknown pulse sequence {value!r}; valid ones are {valid}.")


# A largest b-value outside this range, in s/mm^2, is probably in the wrong unit.
_PLAUSIBLE_B = (100.0, 50_000.0)

# Steps of the electrostatic repulsion; enough for a few hundred directions.
_REPULSION_STEPS = 300


@dataclass(frozen=True, eq=False)
class Protocol:
    """A diffusion-weighted scan: the gradient table and the pulse timings.

    Every simulation engine takes a protocol with the phantom. The diffusion
    weighting is a pair of rectangular gradient pulses (pulsed gradient spin
    echo), so b = gamma^2 G^2 delta^2 (Delta - delta / 3). A gradient echo has
    no refocusing pulse, so it is assumed to use a bipolar pair instead: a lobe
    of +G, then one of -G, each delta long and Delta apart. Its effective
    waveform is the same, and so are its b-value and its restricted signal.

    The protocol keeps its own read-only copy of the gradient table, so change a
    table before building a protocol from it, not after.

    Args:
        gradients: The b-value (s/mm^2) and direction of each volume.
        small_delta_ms: delta, the duration of each gradient pulse, in ms.
        big_delta_ms: Delta, the time between the starts of the two pulses, in ms.
        te_ms: The echo time, or a sequence of echo times for a multi-echo
            acquisition, in ms. Every echo shares the one diffusion weighting,
            which comes before the first echo.
        tr_ms: The repetition time, in ms.
        sequence: SPIN_ECHO (relaxation with T2) or GRADIENT_ECHO (with T2*).
            Defaults to SPIN_ECHO.

    Raises:
        ValueError: If a timing is not positive and finite, delta > Delta,
            Delta + delta >= the first TE, or TR <= the last TE.
    """

    gradients: GradientTable
    small_delta_ms: float
    big_delta_ms: float
    te_ms: tuple[float, ...]
    tr_ms: float
    sequence: PulseSequence = PulseSequence.SPIN_ECHO

    def __init__(
        self,
        gradients: GradientTable,
        small_delta_ms: float,
        big_delta_ms: float,
        te_ms: float | Sequence[float],
        tr_ms: float,
        sequence: PulseSequence = PulseSequence.SPIN_ECHO,
    ) -> None:
        """Checks the timings and copies the gradient table."""
        times = (
            (float(te_ms),)
            if isinstance(te_ms, (int, float, np.number))
            else tuple(float(t) for t in te_ms)
        )
        table = GradientTable(
            np.array(gradients.bvals),
            np.array(gradients.bvecs),
            b0_threshold=gradients.b0_threshold,
            shell_epsilon=gradients.shell_epsilon,
        )
        table.bvals.flags.writeable = False
        table.bvecs.flags.writeable = False
        object.__setattr__(self, "gradients", table)
        object.__setattr__(self, "small_delta_ms", float(small_delta_ms))
        object.__setattr__(self, "big_delta_ms", float(big_delta_ms))
        object.__setattr__(self, "te_ms", times)
        object.__setattr__(self, "tr_ms", float(tr_ms))
        object.__setattr__(self, "sequence", PulseSequence(sequence))
        self._check()

    def _check(self) -> None:
        """Raises ValueError for timings that do not fit together."""
        if len(self.te_ms) == 0:
            raise ValueError("A protocol needs at least one echo time.")
        named = {
            "delta": self.small_delta_ms,
            "Delta": self.big_delta_ms,
            "TR": self.tr_ms,
            **{f"TE {i + 1}": te for i, te in enumerate(self.te_ms)},
        }
        for name, value in named.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite, got {value} ms.")
        if self.small_delta_ms > self.big_delta_ms:
            raise ValueError(
                f"delta ({self.small_delta_ms} ms) must not exceed Delta "
                f"({self.big_delta_ms} ms)."
            )
        first_te = min(self.te_ms)
        # The diffusion block, refocused or bipolar, ends before the readout.
        if self.big_delta_ms + self.small_delta_ms >= first_te:
            raise ValueError(
                f"Delta + delta ({self.big_delta_ms + self.small_delta_ms} ms) must "
                f"be shorter than the echo time ({first_te} ms)."
            )
        if self.tr_ms <= max(self.te_ms):
            raise ValueError(
                f"TR ({self.tr_ms} ms) must be longer than the echo time "
                f"({max(self.te_ms)} ms)."
            )
        largest = float(self.gradients.bvals.max())
        if largest > 0 and not _PLAUSIBLE_B[0] <= largest <= _PLAUSIBLE_B[1]:
            warnings.warn(
                f"The largest b-value is {largest:g}; b-values are in s/mm^2, so "
                f"this may be in the wrong unit (b = 1 ms/um^2 is 1000 s/mm^2).",
                stacklevel=3,
            )

    @property
    def num_echoes(self) -> int:
        """The number of echo times."""
        return len(self.te_ms)

    def to_dict(self) -> dict[str, object]:
        """The protocol as plain values that can be saved as JSON.

        Returns:
            A dict with the b-values, the directions and the timings.
        """
        return {
            "bvals": self.gradients.bvals.tolist(),
            "bvecs": self.gradients.bvecs.tolist(),
            "small_delta_ms": self.small_delta_ms,
            "big_delta_ms": self.big_delta_ms,
            "te_ms": list(self.te_ms),
            "tr_ms": self.tr_ms,
            "sequence": self.sequence.value,
            "b0_threshold": self.gradients.b0_threshold,
            "shell_epsilon": self.gradients.shell_epsilon,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Protocol:
        """Builds a protocol from the values `to_dict` gives.

        Args:
            data: The dict.

        Returns:
            The protocol.

        Raises:
            KeyError: If a value is missing.
            ValueError: If the sequence is unknown or the values do not make a
                valid protocol.
        """
        te = data["te_ms"]
        return cls(
            GradientTable(
                np.asarray(data["bvals"], dtype=np.float64),
                np.asarray(data["bvecs"], dtype=np.float64),
                b0_threshold=_float(data.get("b0_threshold", 50.0)),
                shell_epsilon=_float(data.get("shell_epsilon", 80.0)),
            ),
            small_delta_ms=_float(data["small_delta_ms"]),
            big_delta_ms=_float(data["big_delta_ms"]),
            te_ms=[_float(t) for t in te] if isinstance(te, list) else _float(te),
            tr_ms=_float(data["tr_ms"]),
            sequence=PulseSequence(str(data.get("sequence", PulseSequence.SPIN_ECHO))),
        )


def _float(value: object) -> float:
    """A JSON number as a float; raises ValueError for anything else."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Expected a number, got {value!r}.")
    return float(value)


def create_gradient_table(
    shells: Sequence[tuple[float, int]],
    *,
    num_b0: int = 1,
    seed: int | None = None,
) -> GradientTable:
    """Creates a multi-shell gradient table with evenly spread directions.

    Use it to build the gradient table of a `Protocol`. Each shell's directions
    start at random and are spread by electrostatic repulsion on the half
    sphere: every direction and its opposite carry a charge, and projected
    gradient descent lowers the energy, the sum over pairs of
    1 / norm(x_i - x_j) + 1 / norm(x_i + x_j). The directions are then turned
    to z >= 0. The b = 0 volumes come first, then each shell in the given order.

    Args:
        shells: One pair (b-value in s/mm^2, number of directions) per shell.
        num_b0: The number of b = 0 volumes. Defaults to 1.
        seed: Seed of the starting directions; the same seed gives the same
            table. Defaults to None.

    Returns:
        The gradient table, directions in world coordinates.

    Raises:
        ValueError: If a shell has a b-value that is not positive or fewer than
            one direction, or num_b0 is negative.

    References:
        Jones DK, Horsfield MA, Simmons A. Optimal strategies for measuring
        diffusion in anisotropic systems by magnetic resonance imaging. Magn Reson
        Med 42 (1999) 515-525.
    """
    if num_b0 < 0:
        raise ValueError(f"num_b0 must be at least 0, got {num_b0}.")
    rng = np.random.default_rng(seed)
    bvals = [np.zeros(num_b0)]
    bvecs = [np.zeros((num_b0, 3))]
    for b, count in shells:
        if not b > 0 or count < 1:
            raise ValueError(
                f"A shell needs b > 0 and at least one direction, got ({b}, {count})."
            )
        bvals.append(np.full(count, float(b)))
        bvecs.append(_repelled_directions(int(count), rng))
    return GradientTable(np.concatenate(bvals), np.concatenate(bvecs))


def _repelled_directions(count: int, rng: np.random.Generator) -> np.ndarray:
    """Unit directions spread by antipodal electrostatic repulsion, turned to z >= 0."""
    points = rng.standard_normal((count, 3))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    if count > 1:
        step = 0.1
        energy = _energy(points)
        for _ in range(_REPULSION_STEPS):
            force = _force(points)
            # Only the part of the force along the sphere moves a point.
            force -= np.sum(force * points, axis=1, keepdims=True) * points
            scale = np.abs(force).max()
            if scale == 0:
                break
            moved = points + step * force / scale
            moved /= np.linalg.norm(moved, axis=1, keepdims=True)
            moved_energy = _energy(moved)
            if moved_energy < energy:
                points, energy = moved, moved_energy
                step *= 1.2
            else:
                step *= 0.5
    points[points[:, 2] < 0] *= -1
    return points


def _energy(points: np.ndarray) -> float:
    """Sum over pairs of 1/|x_i - x_j| + 1/|x_i + x_j|."""
    upper = np.triu_indices(len(points), k=1)
    minus = np.linalg.norm(points[:, None] - points[None], axis=2)[upper]
    plus = np.linalg.norm(points[:, None] + points[None], axis=2)[upper]
    return float(np.sum(1 / np.maximum(minus, 1e-12)) + np.sum(1 / plus.clip(1e-12)))


def _force(points: np.ndarray) -> np.ndarray:
    """Minus the gradient of the energy for each point."""
    minus = points[:, None] - points[None]
    plus = points[:, None] + points[None]
    d_minus = np.linalg.norm(minus, axis=2)
    d_plus = np.linalg.norm(plus, axis=2)
    np.fill_diagonal(d_minus, np.inf)
    np.fill_diagonal(d_plus, np.inf)
    d_minus = np.maximum(d_minus, 1e-12)
    d_plus = np.maximum(d_plus, 1e-12)
    force: np.ndarray = np.sum(minus / d_minus[..., None] ** 3, axis=1) + np.sum(
        plus / d_plus[..., None] ** 3, axis=1
    )
    return force
