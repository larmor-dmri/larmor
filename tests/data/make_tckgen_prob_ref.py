"""Generates the probabilistic tracking reference in tests/data/tracking/.

Tracks the synthetic FOD of make_tckgen_det_ref.py (run that first) with tckgen
-algorithm iFOD1: the same tracking mask and 4-voxel seed mask, 4 x 4 x 4 grid
seeds per voxel (256 streamlines) along +x, the peak at every seed, iFOD1's
defaults for the step (0.1 x the voxel size, 0.2 mm), the angle (15 degrees) and
the trials (1000), and exact SH evaluation (-noprecomputed). Run from the
repository root, with MRtrix 3.0.8 on the PATH:

    uv run python tests/data/make_tckgen_prob_ref.py

which runs, in tests/data/tracking/, with MRTRIX_RNG_SEED=1 (tckgen's random
numbers are reproducible with this variable and -nthreads 0):

    tckgen fod.nii.gz prob_ref.tck -algorithm iFOD1
        -seed_grid_per_voxel seeds.nii.gz 4 -seed_direction 1,0,0
        -mask mask.nii.gz -minlength 5 -maxlength 80 -cutoff 0.3
        -noprecomputed -nthreads 0 -output_seeds prob_ref_seeds.txt -info

then the same command with MRTRIX_RNG_SEED = 1 ... 10, keeping only what -info
prints, into prob_ref_info.json: the calibration, the mean number of samples per
step, the mean number of steps between truncations and the largest truncation
(null when none occurred), and the fraction of ends of each termination reason.
Their spread over the ten runs is tckgen's run-to-run noise.
"""

import json
import os
import pathlib
import re
import subprocess
import tempfile

OUT = pathlib.Path("tests/data/tracking")
RUN_SEEDS = range(1, 11)


def tckgen(output: str, rng_seed: int, seeds_file: str | None) -> str:
    """Runs tckgen in OUT and returns what it printed."""
    command = [
        "tckgen",
        "fod.nii.gz",
        output,
        "-algorithm",
        "iFOD1",
        "-seed_grid_per_voxel",
        "seeds.nii.gz",
        "4",
        "-seed_direction",
        "1,0,0",
        "-mask",
        "mask.nii.gz",
        "-minlength",
        "5",
        "-maxlength",
        "80",
        "-cutoff",
        "0.3",
        "-noprecomputed",
        "-nthreads",
        "0",
        "-info",
        "-force",
    ]
    if seeds_file is not None:
        command += ["-output_seeds", seeds_file]
    done = subprocess.run(
        command,
        cwd=OUT,
        env={**os.environ, "MRTRIX_RNG_SEED": str(rng_seed)},
        check=True,
        capture_output=True,
        text=True,
    )
    return done.stderr


def figures(info: str) -> dict[str, object]:
    """What tckgen -info reports about the sampling and the ends."""

    def number(pattern: str) -> float | None:
        found = re.search(pattern + r" = ([0-9.eE+-]+)", info)
        return None if found is None else float(found.group(1))

    calibration = re.search(
        r"will use (\d+) directions with a ratio of ([0-9.]+)", info
    )
    assert calibration is not None
    reasons = dict(re.findall(r"\[INFO\]\s+([A-Z][a-z][\w -]+): ([0-9.]+)%", info))
    return {
        "calibration_directions": int(calibration.group(1)),
        "calibration_ratio": float(calibration.group(2)),
        "mean_samples_per_step": number("mean number of samples per step"),
        "steps_per_truncation": number(
            "mean number of steps between rejection sampling truncations"
        ),
        "max_truncation": number("maximum truncation error"),
        "termination_percent": {k: float(v) for k, v in reasons.items()},
    }


runs = []
for rng_seed in RUN_SEEDS:
    if rng_seed == 1:
        info = tckgen("prob_ref.tck", rng_seed, "prob_ref_seeds.txt")
    else:
        with tempfile.TemporaryDirectory() as tmp:
            info = tckgen(str(pathlib.Path(tmp) / "run.tck"), rng_seed, None)
    runs.append({"rng_seed": rng_seed, **figures(info)})

version = subprocess.run(
    ["tckgen", "-version"], capture_output=True, text=True, check=True
).stdout.splitlines()[0]
record = {"mrtrix_version": version.strip("= "), "runs": runs}
(OUT / "prob_ref_info.json").write_text(json.dumps(record, indent=2) + "\n")
print("saved", OUT, version)  # noqa: T201 (script output)
