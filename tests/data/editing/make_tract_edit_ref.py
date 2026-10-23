"""Generates the references of whole edits (lar tract-edit) in tests/data/editing/.

Runs tckedit on tracks.tck with combinations of options: selection by length,
weight and several regions with and without ends_only, cropping to masks with
-number and -skip, and -inverse with a crop. In the two `skip_differs_` cases
some of the first streamlines are cut into several pieces, which tckedit's -skip
does not count; Larmor counts them like any other streamline, so its result differs
there (by design). The spheres are those of
make_regions_ref.py and the masks its shell_mask.nii.gz and float_mask.nii.gz.
Length limits fall between the streamlines' lengths, away from any of them.

tckedit gets every streamline's index + 1 as its weight (`-tck_weights_in`) and
writes the weights of what it keeps (`-tck_weights_out`), so each output is a list
of input indices: the selected streamlines, or for a crop the input streamline of
each piece. The weight limits act on these index weights. tract_edit_ref.npz
holds, for each case of CASES, the indices under its name and the number of points
of each output streamline under `<name>_lengths`.

One more case joins two inputs, tracks.tck twice, with `-include` and a
`-number` that reaches into the second; its output is tract_edit_join.tck.

Run from the repository root, with MRtrix 3.0.8 on the PATH:

    uv run python tests/data/editing/make_tract_edit_ref.py
"""

import pathlib
import subprocess
import tempfile

import numpy as np

OUT = pathlib.Path("tests/data/editing")
TRACKS = OUT / "tracks.tck"
NUM_STREAMLINES = 500
SHELL = str(OUT / "shell_mask.nii.gz")
FLOAT = str(OUT / "float_mask.nii.gz")

SPHERES = {
    "a": "-124,-76,74,5",
    "b": "-116,-62,73,5",
    "c": "-108,-50,74,5",
    "away": "-100,-60,85,8",
    "start": "-129,-88,77,4",
    "end": "-105,-43,76,4",
}

CASES: dict[str, list[str]] = {
    "includes_exclude_min_length": [
        *("-include", SPHERES["a"], "-include", SPHERES["c"]),
        *("-exclude", SPHERES["away"], "-minlength", "45.5"),
    ],
    "includes_mask_max_length": [
        *("-include", SPHERES["a"], "-include", SHELL, "-maxlength", "55.5"),
    ],
    "ends_includes_length": [
        *("-include", SPHERES["start"], "-include", SPHERES["end"], "-ends_only"),
        *("-minlength", "40.5", "-maxlength", "58.5"),
    ],
    "ends_ordered_exclude": [
        *("-include_ordered", SPHERES["start"], "-include_ordered", SPHERES["end"]),
        *("-exclude", SPHERES["b"], "-ends_only"),
    ],
    "ordered_skip_number": [
        *("-include_ordered", SPHERES["a"], "-include_ordered", SPHERES["c"]),
        *("-skip", "10", "-number", "25"),
    ],
    "weight_include": [
        *("-include", SPHERES["b"], "-minweight", "101", "-maxweight", "400"),
    ],
    "mask_shell_number": ["-mask", SHELL, "-number", "50"],
    "mask_sphere_skip_number": ["-mask", SPHERES["c"], "-skip", "20", "-number", "30"],
    # Some of the first streamlines are cropped into several pieces, which
    # tckedit's -skip passes over: Larmor differs here, as its design states.
    "skip_differs_spheres": [
        *("-mask", SPHERES["a"], "-mask", SPHERES["c"], "-skip", "30"),
        *("-number", "40"),
    ],
    "include_mask_float_number": [
        *("-include", SPHERES["c"], "-mask", FLOAT, "-number", "60"),
    ],
    "inverse_include_crop": [
        *("-include", SPHERES["a"], "-inverse", "-mask", FLOAT, "-number", "100"),
    ],
    "skip_differs_inverse_crop": [
        *("-inverse", "-mask", SHELL, "-skip", "5", "-number", "30"),
    ],
}


def run_case(options: list[str], weights_in: str) -> tuple[np.ndarray, np.ndarray]:
    """Runs tckedit; returns the kept input indices and the output's point counts."""
    with tempfile.TemporaryDirectory() as folder:
        out_tck = f"{folder}/out.tck"
        weights_out = f"{folder}/weights.txt"
        subprocess.run(
            ["tckedit", "-quiet", "-force", str(TRACKS), out_tck, *options]
            + ["-tck_weights_in", weights_in, "-tck_weights_out", weights_out],
            check=True,
        )
        text = pathlib.Path(weights_out).read_text().split()
        indices = np.array([int(float(value)) - 1 for value in text], dtype=np.int64)
        return indices, point_counts(out_tck)


def point_counts(path: str) -> np.ndarray:
    """The number of points of each streamline of a float32 TCK file."""
    data = pathlib.Path(path).read_bytes()
    start = int(data.split(b"file: . ")[1].split(b"\n")[0])
    points = np.frombuffer(data[start:], dtype="<f4").reshape(-1, 3)
    ends = np.flatnonzero(np.isnan(points[:, 0]))
    return np.diff(np.concatenate([[-1], ends])) - 1


def main() -> None:
    """Runs every case and saves the references."""
    results = {}
    with tempfile.TemporaryDirectory() as folder:
        # Index + 1: the weights start at 1 so that none is zero.
        weights_in = f"{folder}/weights.txt"
        np.savetxt(weights_in, np.arange(1, NUM_STREAMLINES + 1), fmt="%d")
        for name, options in CASES.items():
            indices, lengths = run_case(options, weights_in)
            results[name] = indices
            results[f"{name}_lengths"] = lengths
            print(name, len(indices))  # noqa: T201 (script output)
    np.savez_compressed(OUT / "tract_edit_ref.npz", **results)

    subprocess.run(
        ["tckedit", "-quiet", "-force", str(TRACKS), str(TRACKS)]
        + [str(OUT / "tract_edit_join.tck"), "-include", SPHERES["c"]]
        + ["-number", "300"],
        check=True,
    )

    version = subprocess.run(["tckedit", "-version"], capture_output=True, text=True)
    print("saved with", version.stdout.splitlines()[0])  # noqa: T201 (script output)


if __name__ == "__main__":
    main()
