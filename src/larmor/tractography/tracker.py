"""The tracker, which tracks streamlines through a field of FODs from seeds."""

import contextlib
import json
import secrets
from collections.abc import Callable
from dataclasses import dataclass, fields
from math import asin, ceil, degrees, isfinite, radians, sin
from pathlib import Path
from typing import TextIO

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

import larmor
from larmor import _core
from larmor._log import LarmorLogger
from larmor._threads import n_threads
from larmor.io import BatchWriter, open_tractogram_writer
from larmor.sd import convert_sh_basis, sh_layout
from larmor.tractography.progress import report_tracking
from larmor.tractography.seed import MaskSeeds, Seeds, SeedStrategy, write_seeds
from larmor.tractography.termination import auto_cutoff
from larmor.types import Tractogram, Volume

# The compiled parameters of an algorithm; their type picks the compiled algorithm.
AlgorithmParams = _core.DetParams | _core.ProbParams | _core.TestAlgorithm


@dataclass(frozen=True)
class _Algorithm:
    """A tracking algorithm, with its name, default angle and compiled parameters.

    Attributes:
        name: The algorithm's name, as recorded in the run's parameters.
        default_max_angle: The maximum angle in degrees that a `Tracker` uses when
            it is not given one, or None to derive it from default_min_radius.
        make_params: Builds the compiled parameters, such as `_core.DetParams`,
            from the FOD amplitude cutoff and the maximum angle in degrees.
        default_min_radius: The smallest radius of curvature, in voxels, that the
            default angle allows at the run's step size; used when
            default_max_angle is None.

    References:
        Deterministic ("det"): Tournier J-D, Calamante F, Connelly A. MRtrix:
        diffusion tractography in crossing fiber regions. Int J Imaging Syst
        Technol 22 (2012) 53-66. doi:10.1002/ima.22005

        Probabilistic ("prob"): Tournier J-D, Calamante F, Connelly A. MRtrix:
        diffusion tractography in crossing fiber regions. Int J Imaging Syst
        Technol 22 (2012) 53-66. doi:10.1002/ima.22005
    """

    name: str
    default_max_angle: float | None
    make_params: Callable[[float, float], AlgorithmParams]
    default_min_radius: float | None = None


# Smallest radius of curvature, in voxels, that the deterministic default angle allows.
DET_MIN_RADIUS = 0.65

# The tracking algorithms by name.
ALGORITHMS: dict[str, _Algorithm] = {
    "det": _Algorithm(
        name="det",
        default_max_angle=None,
        make_params=_core.DetParams,
        default_min_radius=DET_MIN_RADIUS,
    ),
    "prob": _Algorithm(name="prob", default_max_angle=15, make_params=_core.ProbParams),
}
# Largest size of the buffer of points of a batch, in bytes.
BUFFER_BYTES = 128 * 2**20
# Largest number of seeds in a batch.
MAX_BATCH = 20_000
# Number of seeds per streamline asked for after which a run stops.
SEEDS_PER_STREAMLINE = 1000
# Number of seeds a batch draws per streamline still missing, with a target.
DRAW_PER_MISSING = 4
# Smallest number of seeds a batch draws, with a target.
MIN_DRAW = 256


@dataclass(frozen=True)
class TractographyRun:
    """The record of one tracking, with what was written, the settings and the counts.

    `Tracker.track` returns it, and `write_json` saves it as a JSON file.

    Attributes:
        path: The tractogram written, or None if it was kept in memory.
        seeds_path: The seeds file written, or None.
        parameters: The values the run used, defaults resolved, as plain values
            that can be saved as JSON.
        num_seeds: The number of seeds tried.
        num_streamlines: The number of streamlines accepted.
        num_rejected: The seeds that gave no streamline, counted by reason:
            "no_start" (no direction to start) and "too_short".
        sampling: Figures about the sampling of the steps (mean samples per step,
            steps per truncation, largest truncation and their counts), or None if
            no step was sampled.
        inputs: What the caller tracked from (files, seed source), as it chose to
            record it so that the JSON can repeat the run; None if not recorded.
        tractogram: The accepted streamlines when the run had no output file,
            else None.
    """

    path: Path | None
    seeds_path: Path | None
    parameters: dict[str, object]
    num_seeds: int
    num_streamlines: int
    num_rejected: dict[str, int]
    sampling: dict[str, float | None] | None = None
    inputs: dict[str, object] | None = None
    tractogram: Tractogram | None = None

    def write_json(self, path: str | Path) -> None:
        """Saves the record as a JSON file, without the in-memory tractogram.

        Args:
            path: The JSON file to write.
        """
        p = Path(path)
        data = {f.name: getattr(self, f.name) for f in fields(self)}
        del data["tractogram"]
        data["path"] = None if self.path is None else str(self.path)
        data["seeds_path"] = None if self.seeds_path is None else str(self.seeds_path)
        p.write_text(json.dumps(data, indent=4), encoding="utf-8")


@dataclass(frozen=True)
class _Prepared:
    """The values and arrays a run works out once before tracking.

    Attributes:
        coeffs: The FOD's SH coefficients in the tournier basis, (X, Y, Z, C).
        full: Whether the FOD has all SH degrees (an asymmetric FOD), else the
            even ones.
        fod_inv: The FOD's world-to-voxel matrix, (4, 4).
        mask: The mask's scaled values, (X, Y, Z) float32, and its world-to-voxel
            matrix; None without a mask.
        step_size: The step size, in mm.
        min_length: The minimum length, in mm.
        max_length: The maximum length, in mm.
        max_angle: The maximum angle per step, in degrees.
        min_radius: The tightest curve max_angle allows at step_size, in mm.
        cutoff: The FOD amplitude cutoff.
        params: The algorithm's compiled parameters.
        max_steps: The most steps of a whole streamline; a buffer row holds
            max_steps + 1 points.
        batch_size: The number of seeds per batch.
    """

    coeffs: NDArray[np.float32]
    full: bool
    fod_inv: NDArray[np.float64]
    mask: tuple[NDArray[np.float32], NDArray[np.float64]] | None
    step_size: float
    min_length: float
    max_length: float
    max_angle: float
    min_radius: float
    cutoff: float
    params: AlgorithmParams
    max_steps: int
    batch_size: int


@dataclass
class _Counts:
    """The running totals of a run.

    The sampling counters are summed over every tracked streamline.
    """

    seeds: int = 0
    accepted: int = 0
    no_start: int = 0
    too_short: int = 0
    sampled_steps: int = 0
    accepted_index_sum: int = 0
    truncations: int = 0
    max_truncation: float = 0.0
    calibration_directions: int = 0

    def add_sampling(self, stats: dict[str, float]) -> None:
        """Adds a batch's sampling counters."""
        self.sampled_steps += int(stats["sampled_steps"])
        self.accepted_index_sum += int(stats["accepted_index_sum"])
        self.truncations += int(stats["truncations"])
        self.max_truncation = max(self.max_truncation, float(stats["max_truncation"]))
        self.calibration_directions = int(stats["calibration_directions"])

    def sampling(self) -> dict[str, float | None] | None:
        """Returns the figures about the sampling of the steps, or None if none was.

        The mean samples per step is the calibration directions plus the
        mean index of the accepted candidate, over all sampled steps. Without any
        truncation, both truncation figures are None.

        Returns:
            The figures, or None.
        """
        if self.sampled_steps == 0:
            return None
        return {
            "mean_samples_per_step": self.calibration_directions
            + self.accepted_index_sum / self.sampled_steps,
            "steps_per_truncation": self.sampled_steps / self.truncations
            if self.truncations
            else None,
            "max_truncation": self.max_truncation if self.truncations else None,
            "truncations": self.truncations,
            "sampled_steps": self.sampled_steps,
            "calibration_directions": self.calibration_directions,
        }


class Tracker:
    """Tracks streamlines through a field of FODs.

    The settings are fixed here and `track` runs one tracking, from seeds to a
    tractogram. The algorithm is "det" (deterministic) or "prob" (probabilistic).
    Lengths are in mm.

    The maximum angle limits the turn between two consecutive steps, so the
    tightest curve it allows has a radius of step / (2 sin(angle / 2)), and the
    same angle means a different curvature at a different step size. Without a
    given angle, "det" sets it from the step size so that this radius is 0.65 times
    the voxel size (about 8.8 degrees at the default step, and no limit for a step
    above 1.3 voxels); "prob" uses 15 degrees. The run records the radius as
    "min_radius".

    References:
        Deterministic ("det"): Tournier J-D, Calamante F, Connelly A. MRtrix:
        diffusion tractography in crossing fiber regions. Int J Imaging Syst
        Technol 22 (2012) 53-66. doi:10.1002/ima.22005

        Probabilistic ("prob"): Tournier J-D, Calamante F, Connelly A. MRtrix:
        diffusion tractography in crossing fiber regions. Int J Imaging Syst
        Technol 22 (2012) 53-66. doi:10.1002/ima.22005
    """

    def __init__(
        self,
        algorithm: str | _Algorithm,
        *,
        step_size: float | None = None,
        max_angle: float | None = None,
        cutoff: float | str = "auto",
        min_length: float | None = None,
        max_length: float | None = None,
        bidirectional: bool = True,
        progress: bool = False,
    ) -> None:
        """Sets the parameters of the tracking algorithm.

        Args:
            algorithm: The name of the algorithm, "det" or "prob".
            step_size: Step size in mm. Defaults to 0.1 x the voxel size.
            max_angle: Max angle between two consecutive steps in degrees, in
                (0, 180]. The smallest radius of curvature it allows is
                step_size / (2 sin(max_angle / 2)). Defaults to the algorithm's
                default: for "det", the angle that keeps that radius at 0.65 times
                the voxel size; for "prob", 15 degrees.
            cutoff: FOD amplitude under which a streamline stops, or "auto" for 0.2
                times the mean largest peak of up to 5000 voxels with a positive DC
                term, inside the tracking mask when `track` is given one. 0 never
                stops for a low amplitude. Defaults to "auto".
            min_length: Min length in mm for a streamline to be kept. Defaults to
                5 x the voxel size.
            max_length: Length in mm at which a streamline stops. Defaults to 100 x
                the voxel size.
            bidirectional: Whether to track both ways from each seed. Defaults to
                True.
            progress: Whether to show a progress bar. Defaults to False.

        Raises:
            ValueError: If the algorithm is unknown, or a value is out of range.
        """
        if isinstance(algorithm, str):
            if algorithm not in ALGORITHMS:
                known = ", ".join(sorted(ALGORITHMS)) or "none yet"
                raise ValueError(
                    f"Unknown algorithm {algorithm!r}; the known ones are: {known}."
                )
            else:
                algo = ALGORITHMS[algorithm]
        else:
            algo = algorithm

        _check_positive("step size", step_size)
        _check_positive("min length", min_length)
        _check_positive("max length", max_length)

        if (
            min_length is not None
            and max_length is not None
            and min_length >= max_length
        ):
            raise ValueError("Cannot track with min length >= max length.")

        if max_angle is not None and not (0 < max_angle <= 180):
            raise ValueError(f"Expected max angle to be in (0, 180], got {max_angle}")

        if isinstance(cutoff, str):
            if cutoff != "auto":
                raise ValueError(
                    f"Expected a number or 'auto' as cutoff, got {cutoff!r}."
                )
        # 0 is allowed: it means never stopping for a low FOD amplitude.
        elif not (isfinite(cutoff) and cutoff >= 0):
            raise ValueError(
                f"The cutoff must be non-negative and finite, got {cutoff}."
            )

        self.algorithm = algo
        self.step_size = step_size
        self.min_length = min_length
        self.max_length = max_length
        self.max_angle = max_angle
        self.cutoff = cutoff
        self.bidirectional = bidirectional
        self.progress = progress

    def track(
        self,
        fod: Volume,
        output: str | Path | None,
        seeds: Seeds | SeedStrategy,
        *,
        num_streamlines: int | None = None,
        mask: Volume | None = None,
        basis: str = "tournier",
        seed: int | None = None,
        output_seeds: str | Path | None = None,
        n_jobs: int = -1,
    ) -> TractographyRun:
        """Tracks streamlines from the seeds, batch by batch.

        Args:
            fod: The FOD as SH coefficients, (X, Y, Z, C). C tells the layout:
                the even degrees of a symmetric FOD, or all degrees of an
                asymmetric one.
            output: The tractogram to write, .tck or .trk, batch by batch. Each
                streamline carries the EndStatus of both ends as `end_status` in
                its streamline data (kept by TRK only). None keeps the streamlines
                in memory and returns them in the run record, which suits small
                runs only.
            seeds: Where streamlines start, as fixed `Seeds`, tracked in order, or
                an endless source such as `MaskSeeds`, drawn from batch by batch.
                A seed without a direction gets one from the algorithm; a seed
                where none is found counts as no_start.
            num_streamlines: Track until this many are accepted. Defaults to None,
                which tracks every seed of a `Seeds`.
            mask: Tracking mask; a streamline stops where it leaves it. A point is
                inside where the scaled value of the mask's nearest voxel is at
                least 0.5. Defaults to None, no mask.
            basis: The FOD's SH basis. Defaults to "tournier".
            seed: The run seed, in [0, 2^64), which fixes the seeds drawn and every
                random number. Defaults to None, which draws one and records it.
            output_seeds: A text file for the seeds of the accepted streamlines, in
                output order, which `read_seeds` reads back. Defaults to None.
            n_jobs: Number of threads, at most the cores; -1 is all cores. Defaults
                to -1. The result does not depend on it.

        Returns:
            TractographyRun: The record of the run.

        Raises:
            ValueError: If an endless seed source has no number of streamlines,
                num_streamlines is not positive, the mask is not 3D, seed or
                n_jobs is out of range, the FOD's lmax is above the supported
                maximum or its coefficient count fits no SH layout, the resolved
                min length is not below the max length, the automatic cutoff
                cannot be measured (no voxel in the mask with a positive DC term or
                with a peak), or no seed lies on the FOD's grid or in the mask.
        """
        if num_streamlines is None and not isinstance(seeds, Seeds):
            raise ValueError(
                "Seeds from an endless source need num_streamlines, or tracking "
                "would never end."
            )
        if num_streamlines is not None and num_streamlines <= 0:
            raise ValueError(f"Cannot seed {num_streamlines} streamlines.")
        if mask is not None and mask.data.ndim != 3:
            raise ValueError(f"Expected 3D mask, got shape {mask.data.shape}")
        if mask is not None:
            # The mask holds its scaled values from here on.
            mask = mask.with_data(mask.as_float(dtype=np.float32))
        # Checked before any output is opened.
        if seed is not None and not 0 <= seed < 2**64:
            raise ValueError(f"The run seed must be in [0, 2^64), got {seed}.")
        threads = n_threads(n_jobs)
        _check_alignment(fod, seeds, mask)

        run_seed = int(seed) if seed is not None else secrets.randbits(64)

        prep = self._prepare(fod, basis, mask)
        counts, tractogram = self._run_batches(
            prep, fod, seeds, num_streamlines, run_seed, threads, output, output_seeds
        )

        parameters: dict[str, object] = {
            "algorithm": self.algorithm.name,
            "step_size": prep.step_size,
            "max_angle": prep.max_angle,
            "min_radius": prep.min_radius,
            "cutoff": prep.cutoff,
            "cutoff_mode": "auto" if self.cutoff == "auto" else "given",
            "min_length": prep.min_length,
            "max_length": prep.max_length,
            "bidirectional": self.bidirectional,
            "max_steps": prep.max_steps,
            "batch_size": prep.batch_size,
            "seed": run_seed,
            "num_threads": threads,
            "num_streamlines_target": num_streamlines,
            "basis": basis,
            "mask": mask is not None,
            "larmor_version": larmor.__version__,
        }
        return TractographyRun(
            path=None if output is None else Path(output),
            seeds_path=None if output_seeds is None else Path(output_seeds),
            parameters=parameters,
            num_seeds=counts.seeds,
            num_streamlines=counts.accepted,
            num_rejected={"no_start": counts.no_start, "too_short": counts.too_short},
            sampling=counts.sampling(),
            tractogram=tractogram,
        )

    def _prepare(self, fod: Volume, basis: str, mask: Volume | None) -> _Prepared:
        """Resolves the defaults and builds the arrays the compiled tracking takes.

        Returns:
            The resolved values and arrays.

        Raises:
            ValueError: If the FOD's lmax is too high, its coefficient count fits no
                SH layout, min_length is not below max_length or the automatic
                cutoff cannot be measured.
        """
        lmax, full = sh_layout(fod.data.shape[-1])
        if lmax > _core.MAX_LMAX:
            raise ValueError(
                f"The FOD's lmax must be at most {_core.MAX_LMAX}, got {lmax}."
            )
        coeffs = np.ascontiguousarray(
            convert_sh_basis(fod.as_float(dtype=np.float32), basis, "tournier"),
            dtype=np.float32,
        )

        voxel = float(np.mean(fod.voxel_sizes))
        step_size = self.step_size if self.step_size is not None else 0.1 * voxel
        min_length = self.min_length if self.min_length is not None else 5 * voxel
        max_length = self.max_length if self.max_length is not None else 100 * voxel
        if min_length >= max_length:
            raise ValueError(
                "Invalid params: "
                f"Got min_length {min_length} and max_length {max_length}."
            )
        max_angle = self._max_angle(step_size, voxel)
        min_radius = step_size / (2 * sin(radians(max_angle) / 2))
        LarmorLogger.info(
            f"The maximum angle of {max_angle:.3g} degrees per {step_size:.3g} mm "
            f"step allows curves down to a radius of {min_radius:.3g} mm."
        )
        cutoff = (
            auto_cutoff(coeffs, mask=None if mask is None else _mask_on_grid(mask, fod))
            if self.cutoff == "auto"
            else float(self.cutoff)
        )
        # Rounded first, so that rounding errors do not add a step.
        max_steps = ceil(round(max_length / step_size, 9))

        mask_arrays = (
            None
            if mask is None
            else (
                np.ascontiguousarray(mask.data, dtype=np.float32),
                np.linalg.inv(mask.affine).astype(np.float64),
            )
        )

        row = max_steps + 1
        return _Prepared(
            coeffs=coeffs,
            full=full,
            fod_inv=np.linalg.inv(fod.affine).astype(np.float64),
            mask=mask_arrays,
            step_size=float(step_size),
            min_length=float(min_length),
            max_length=float(max_length),
            max_angle=float(max_angle),
            min_radius=float(min_radius),
            cutoff=float(cutoff),
            params=self.algorithm.make_params(cutoff, max_angle),
            max_steps=max_steps,
            batch_size=max(1, min(MAX_BATCH, BUFFER_BYTES // (row * 3 * 4))),
        )

    def _max_angle(self, step_size: float, voxel: float) -> float:
        """Returns the given angle, the algorithm's default, or the one from its radius.

        An algorithm with a default radius r (in voxels) gets the angle whose
        tightest curve has radius r times the voxel size, 2 asin(step / (2 r voxel)),
        or 180 degrees for a step longer than 2 r voxel.
        """
        if self.max_angle is not None:
            return self.max_angle
        if self.algorithm.default_max_angle is not None:
            return self.algorithm.default_max_angle
        assert self.algorithm.default_min_radius is not None
        ratio = step_size / (2 * self.algorithm.default_min_radius * voxel)
        return 180.0 if ratio >= 1 else degrees(2 * asin(ratio))

    def _run_batches(
        self,
        prep: _Prepared,
        fod: Volume,
        seeds: Seeds | SeedStrategy,
        num_streamlines: int | None,
        run_seed: int,
        n_threads: int,
        output: str | Path | None,
        output_seeds: str | Path | None,
    ) -> tuple[_Counts, Tractogram | None]:
        """Tracks batch after batch until the target or the end of the seeds.

        Returns:
            The run's totals, and the tractogram when `output` is None.
        """
        row = prep.max_steps + 1
        positions = np.empty((prep.batch_size, row, 3), dtype=np.float32)
        totals = _Counts()
        in_memory: list[Tractogram] = []

        # The first batch is drawn before any file is opened, so a run refused for
        # its seeds leaves existing outputs untouched.
        batch_index = 0
        batch = _next_batch(
            seeds, totals, _draw_size(prep, totals, num_streamlines), run_seed, 0
        )

        writer: contextlib.AbstractContextManager[BatchWriter | None] = (
            open_tractogram_writer(output, reference=fod.grid)
            if output is not None
            else contextlib.nullcontext()
        )
        seeds_file: contextlib.AbstractContextManager[TextIO | None] = (
            open(output_seeds, "w", encoding="utf-8")  # noqa: SIM115
            if output_seeds is not None
            else contextlib.nullcontext()
        )
        stage = (
            "Tracking streamlines" if num_streamlines is not None else "Tracking seeds"
        )
        target = num_streamlines if num_streamlines is not None else _count(seeds)
        report_tracking(0, target or 0, stage)
        with (
            writer as tck_out,
            seeds_file as seeds_out,
            tqdm(
                total=target,
                unit="streamlines" if num_streamlines is not None else "seeds",
                disable=not self.progress,
            ) as bar,
        ):
            while len(batch) > 0:
                assert batch.indices is not None  # _next_batch numbers every seed
                counts, status, followed, stats = _core.track_batch(
                    prep.coeffs,
                    prep.fod_inv,
                    prep.mask,
                    batch.positions,
                    batch.directions,
                    batch.indices,
                    run_seed,
                    prep.step_size,
                    prep.max_steps,
                    self.bidirectional,
                    prep.params,
                    positions,
                    n_threads,
                    full=prep.full,
                )

                totals.add_sampling(stats)
                remaining = (
                    None
                    if num_streamlines is None
                    else num_streamlines - totals.accepted
                )
                keep = _keep(counts, prep, remaining, totals)

                tractogram = _batch_tractogram(positions, counts, status, keep, fod)
                if tck_out is not None:
                    tck_out.write(tractogram)
                else:
                    in_memory.append(tractogram)
                if seeds_out is not None:
                    kept = Seeds(
                        batch.positions[keep], followed[keep], batch.indices[keep]
                    )
                    write_seeds(kept, seeds_out, header=batch_index == 0)

                bar.update(len(keep) if num_streamlines is not None else len(batch))
                bar.set_postfix(
                    seeds=totals.seeds,
                    no_start=totals.no_start,
                    too_short=totals.too_short,
                )
                report_tracking(
                    totals.accepted if num_streamlines is not None else totals.seeds,
                    target or 0,
                    stage,
                )
                if _should_stop(totals, num_streamlines):
                    break
                batch_index += 1
                batch = _next_batch(
                    seeds,
                    totals,
                    _draw_size(prep, totals, num_streamlines),
                    run_seed,
                    batch_index,
                )

        if num_streamlines is not None and totals.accepted < num_streamlines:
            if isinstance(seeds, Seeds) and totals.seeds >= len(seeds):
                LarmorLogger.warning(
                    f"The seeds ran out with {totals.accepted} of {num_streamlines} "
                    "streamlines."
                )
        result = None if output is not None else _concatenate(in_memory, fod)
        return totals, result


def _draw_size(prep: _Prepared, totals: _Counts, num_streamlines: int | None) -> int:
    """Returns how many seeds the next batch takes, a full batch or fewer for a target.

    With a target, a batch draws DRAW_PER_MISSING seeds per streamline still
    missing (at least MIN_DRAW), and a run tries at most SEEDS_PER_STREAMLINE seeds
    per streamline asked for.
    """
    if num_streamlines is None:
        return prep.batch_size
    left = SEEDS_PER_STREAMLINE * num_streamlines - totals.seeds
    missing = num_streamlines - totals.accepted
    wanted = max(MIN_DRAW, DRAW_PER_MISSING * missing)
    return max(0, min(prep.batch_size, left, wanted))


def _next_batch(
    seeds: Seeds | SeedStrategy,
    totals: _Counts,
    size: int,
    run_seed: int,
    batch_index: int,
) -> Seeds:
    """Returns the next `size` seeds, each with its global index.

    Fixed seeds are sliced in order. An endless source is drawn from with a
    generator that depends only on the run seed and the batch number. Seeds without
    indices are numbered in the order they are tracked.
    """
    num_seeds = totals.seeds
    if isinstance(seeds, Seeds):
        batch = _rows(seeds, num_seeds, num_seeds + size)
    else:
        batch = seeds.draw(size, np.random.default_rng([run_seed, batch_index]))
    if batch.indices is None:
        indices = np.arange(num_seeds, num_seeds + len(batch), dtype=np.int64)
        batch = Seeds(batch.positions, batch.directions, indices)
    return batch


def _keep(
    counts: NDArray[np.int64],
    prep: _Prepared,
    remaining: int | None,
    totals: _Counts,
) -> NDArray[np.intp]:
    """Returns the rows of a batch to keep, in seed order, and adds the batch to totals.

    `counts` holds the number of points of each row. A row is kept if its seed
    started and the streamline is at least the minimum length. With a target, at
    most `remaining` rows are kept, and the seeds after the last one kept count as
    never drawn, so that seeds tried = accepted + rejected.
    """
    n = len(counts)
    started = counts > 0
    short = started & ((counts - 1) * prep.step_size < prep.min_length)
    keep = np.flatnonzero(started & ~short)
    if remaining is not None and len(keep) > remaining:
        keep = keep[:remaining]
        n = int(keep[-1]) + 1 if len(keep) else 0
        started, short = started[:n], short[:n]
    totals.seeds += n
    totals.accepted += len(keep)
    totals.no_start += int((~started).sum())
    totals.too_short += int(short.sum())
    return keep


def _should_stop(totals: _Counts, num_streamlines: int | None) -> bool:
    """Returns whether the target is reached, or too many seeds were tried for it."""
    if num_streamlines is None:
        return False
    if totals.accepted >= num_streamlines:
        return True
    if totals.seeds >= SEEDS_PER_STREAMLINE * num_streamlines:
        LarmorLogger.warning(
            f"Stopped after {totals.seeds} seeds with {totals.accepted} of "
            f"{num_streamlines} streamlines: check the seeds, the mask and the cutoff."
        )
        return True
    return False


def _count(seeds: Seeds | SeedStrategy) -> int | None:
    """Returns the number of fixed seeds, or None for an endless source."""
    return len(seeds) if isinstance(seeds, Seeds) else None


def _rows(seeds: Seeds, start: int, stop: int) -> Seeds:
    """Returns rows start to stop - 1 of fixed seeds, fewer at the end."""
    part = slice(start, stop)
    return Seeds(
        seeds.positions[part],
        None if seeds.directions is None else seeds.directions[part],
        None if seeds.indices is None else seeds.indices[part],
    )


def _batch_tractogram(
    positions: NDArray[np.float32],
    counts: NDArray[np.int64],
    status: NDArray[np.uint8],
    keep: NDArray[np.intp],
    fod: Volume,
) -> Tractogram:
    """Returns the kept streamlines of a batch, copied out of the buffers.

    `positions`, `counts` and `status` are the batch's points, numbers of points
    and end statuses; `keep` the rows to keep.
    """
    lengths = counts[keep]
    if len(keep):
        points = np.concatenate([positions[i, : counts[i]] for i in keep])
    else:
        points = np.empty((0, 3), np.float32)
    return Tractogram(
        points,
        np.concatenate([[0], np.cumsum(lengths)]),
        streamline_data={"end_status": status[keep].copy()},
        reference=fod.grid,
    )


def _concatenate(parts: list[Tractogram], fod: Volume) -> Tractogram:
    """Returns the batches kept in memory as one tractogram, on the FOD's grid."""
    if not parts:
        return Tractogram(
            np.empty((0, 3), np.float32),
            np.zeros(1, np.int64),
            streamline_data={"end_status": np.empty((0, 2), np.uint8)},
            reference=fod.grid,
        )
    lengths = np.concatenate([np.diff(t.offsets) for t in parts])
    return Tractogram(
        np.concatenate([t.points for t in parts]),
        np.concatenate([[0], np.cumsum(lengths)]),
        streamline_data={
            "end_status": np.concatenate(
                [t.streamline_data["end_status"] for t in parts]
            )
        },
        reference=fod.grid,
    )


def _seed_points(seeds: Seeds | SeedStrategy) -> NDArray[np.float64] | None:
    """Returns world positions (N, 3) standing for the seeds, or None if unknown.

    For fixed seeds, the seeds themselves; for a mask, its voxel centres.
    """
    if isinstance(seeds, Seeds):
        return seeds.positions
    if isinstance(seeds, MaskSeeds):
        voxels = seeds.voxels.astype(np.float64)
        affine = np.asarray(seeds.affine, dtype=np.float64)
        return voxels @ affine[:3, :3].T + affine[:3, 3]
    return None


def _check_alignment(
    fod: Volume, seeds: Seeds | SeedStrategy, mask: Volume | None
) -> None:
    """Checks before any work that the seeds lie on the FOD and in the mask.

    Raises:
        ValueError: If no seed lies on the FOD's grid or in the mask.
    """
    points = _seed_points(seeds)
    if points is None or len(points) == 0:
        return
    on_fod = _fraction_on_grid(points, fod.data.shape, np.linalg.inv(fod.affine))
    _report(on_fod, "on the FOD's grid; are the seeds in the FOD's space?")
    if mask is None:
        return
    _report(float(_inside_mask(points, mask).mean()), "in the mask")


def _inside_mask(points: NDArray[np.float64], mask: Volume) -> NDArray[np.bool_]:
    """Returns (N,) whether each world point's nearest voxel of the mask is >= 0.5.

    It follows the rule of the compiled tracking: voxel coordinates round halves
    away from zero, and a coordinate at most -0.5 or at least the size minus 0.5 (or
    NaN) is outside.
    """
    inv = np.linalg.inv(mask.affine)
    coordinates = points @ inv[:3, :3].T + inv[:3, 3]
    shape = np.asarray(mask.data.shape)
    on = np.all((coordinates > -0.5) & (coordinates < shape - 0.5), axis=1)
    lower = np.floor(coordinates[on])
    vox = (lower + (coordinates[on] - lower >= 0.5)).astype(np.int64)
    inside = np.zeros(len(points), dtype=bool)
    inside[on] = mask.data[vox[:, 0], vox[:, 1], vox[:, 2]] >= 0.5
    return inside


def _mask_on_grid(mask: Volume, fod: Volume) -> NDArray[np.bool_]:
    """Returns the FOD voxels whose centre is inside the mask, (X, Y, Z) of the FOD."""
    centres = np.indices(fod.data.shape[:3]).reshape(3, -1).T.astype(np.float64)
    world = centres @ fod.affine[:3, :3].T + fod.affine[:3, 3]
    inside = _inside_mask(world, mask)
    return inside.reshape(fod.data.shape[:3])


def _fraction_on_grid(
    points: NDArray[np.float64], shape: tuple[int, ...], inv_affine: NDArray[np.float64]
) -> float:
    """Returns the share of points within half a voxel of the grid's outer centres."""
    if len(points) == 0:
        return 0.0
    vox = points @ inv_affine[:3, :3].T + inv_affine[:3, 3]
    inside = np.all((vox >= -0.5) & (vox <= np.asarray(shape[:3]) - 0.5), axis=1)
    return float(inside.mean())


def _report(fraction: float, where: str) -> None:
    """Refuses a run where no seed lies `where`, and warns when some do not.

    Raises:
        ValueError: If `fraction` is 0.
    """
    if fraction == 0:
        raise ValueError(f"No seed lies {where}")
    if fraction < 1:
        LarmorLogger.warning(
            f"{1 - fraction:.0%} of the seeds do not lie {where.split(';')[0]}; "
            "they will be rejected."
        )


def _check_positive(name: str, value: float | None) -> None:
    """Refuses a value that is given but not positive and finite.

    Raises:
        ValueError: If `value` is not None and not positive and finite.
    """
    if value is not None and not (isfinite(value) and value > 0):
        raise ValueError(f"The {name} must be positive and finite, got {value}.")
