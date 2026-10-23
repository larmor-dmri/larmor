"""Generates the MRtrix references for streamline lengths, selections and joining.

Runs tckstats, tckedit and tcksift2 (MRtrix 3.0.8) on tests/data/editing/tracks.tck
and writes, next to this script:

- lengths_ref.npz: the tckstats -dump lengths, and for each tckedit run the indices
  of the input streamlines it kept, found by comparing points;
- weights.txt: random per-streamline weights, the input of the weight selection;
- weights_selected.txt: the weights tckedit wrote for the streamlines it kept;
- sift2_weights.txt: the weights tcksift2 wrote, a file with a comment line.

Run from the repository root, with the folder of the ISMRM data that holds fod.nii.gz:
uv run python tests/data/editing/make_lengths_ref.py <ISMRM Data folder>
"""

import pathlib
import subprocess
import sys
import tempfile

import numpy as np

from larmor.io import load_tractogram
from larmor.types import Tractogram

OUT = pathlib.Path("tests/data/editing").resolve()
TRACKS = OUT / "tracks.tck"
FOD = pathlib.Path(sys.argv[1]).resolve() / "fod.nii.gz"

# (min, max) for -minlength and -maxlength; None leaves the option out. The
# streamlines are made of 1 mm steps, so their lengths are within 1e-5 mm of whole
# numbers, where tckedit's float32 lengths and Larmor's float64 ones may fall on
# different sides of a limit: the limits are halfway between whole numbers. The
# last pair keeps nothing.
LENGTH_LIMITS = [(40.5, 60.5), (50.5, None), (None, 45.5), (100.5, 200.5)]
JOIN_SPLIT = 200
JOIN_LIMITS = (50.5, 70.5)
WEIGHT_LIMITS = (0.5, 1.5)


def run(*args: str | pathlib.Path, cwd: pathlib.Path | None = None) -> None:
    """Runs an MRtrix command quietly."""
    command = [str(a) for a in args] + ["-quiet", "-force"]
    subprocess.run(command, check=True, cwd=cwd, stdout=subprocess.DEVNULL)


def kept_indices(source: Tractogram, selected: Tractogram) -> np.ndarray:
    """The index in `source` of each streamline of `selected`, by its points."""
    index = {s.tobytes(): i for i, s in enumerate(source)}
    assert len(index) == len(source), "the input has repeated streamlines"
    return np.array([index[s.tobytes()] for s in selected], dtype=np.int64)


def length_options(low: float | None, high: float | None) -> list[str]:
    """The tckedit options for a pair of length limits."""
    options = [] if low is None else ["-minlength", str(low)]
    return options + ([] if high is None else ["-maxlength", str(high)])


def main() -> None:
    """Runs MRtrix and writes the references."""
    tracks = load_tractogram(TRACKS)
    rng = np.random.default_rng(258)
    weights = rng.uniform(0.0, 2.0, tracks.num_streamlines).astype(np.float32)
    np.savetxt(OUT / "weights.txt", weights[None, :], fmt="%.9g")

    ref: dict[str, np.ndarray] = {}
    with tempfile.TemporaryDirectory() as scratch:
        tmp = pathlib.Path(scratch)

        run("tckstats", TRACKS, "-dump", tmp / "lengths.txt")
        ref["lengths"] = np.loadtxt(tmp / "lengths.txt", ndmin=1)

        limits = [
            (np.nan if low is None else low, np.nan if high is None else high)
            for low, high in LENGTH_LIMITS
        ]
        ref["length_limits"] = np.array(limits)
        for case, (low, high) in enumerate(LENGTH_LIMITS):
            run("tckedit", TRACKS, *length_options(low, high), tmp / "out.tck")
            selected = load_tractogram(tmp / "out.tck")
            ref[f"length_kept_{case}"] = kept_indices(tracks, selected)

        run("tckedit", TRACKS, "-number", JOIN_SPLIT, tmp / "a.tck")
        run("tckedit", TRACKS, "-skip", JOIN_SPLIT, tmp / "b.tck")
        first, second = load_tractogram(tmp / "a.tck"), load_tractogram(tmp / "b.tck")
        assert np.array_equal(kept_indices(tracks, first), np.arange(JOIN_SPLIT))
        assert np.array_equal(
            kept_indices(tracks, second), np.arange(JOIN_SPLIT, len(tracks))
        )
        run(
            "tckedit",
            tmp / "a.tck",
            tmp / "b.tck",
            *length_options(*JOIN_LIMITS),
            tmp / "joined.tck",
        )
        ref["join_split"] = np.array(JOIN_SPLIT)
        ref["join_limits"] = np.array(JOIN_LIMITS)
        ref["join_kept"] = kept_indices(tracks, load_tractogram(tmp / "joined.tck"))

        run(
            "tckedit",
            TRACKS,
            "-tck_weights_in",
            OUT / "weights.txt",
            "-minweight",
            WEIGHT_LIMITS[0],
            "-maxweight",
            WEIGHT_LIMITS[1],
            "-tck_weights_out",
            OUT / "weights_selected.txt",
            tmp / "out.tck",
        )
        ref["weight_limits"] = np.array(WEIGHT_LIMITS)
        ref["weight_kept"] = kept_indices(tracks, load_tractogram(tmp / "out.tck"))

        # Links in the scratch folder keep local paths out of the command_history
        # comment that tcksift2 writes.
        (tmp / "tracks.tck").symlink_to(TRACKS)
        (tmp / "fod.nii.gz").symlink_to(FOD)
        run("tcksift2", "tracks.tck", "fod.nii.gz", "sift2_weights.txt", cwd=tmp)
        (OUT / "sift2_weights.txt").write_bytes(
            (tmp / "sift2_weights.txt").read_bytes()
        )

    np.savez_compressed(OUT / "lengths_ref.npz", **ref)
    version = subprocess.run(["tckedit", "-version"], capture_output=True, text=True)
    print("saved the references,", version.stdout.splitlines()[0])  # noqa: T201 (script output)


if __name__ == "__main__":
    main()
