"""Generates the region selection and cropping references in tests/data/editing/.

From tracks.tck (500 streamlines, most of them one bundle running from about
(-128, -88, 76) to (-105, -43, 76) mm), it makes:

- shell_mask.nii.gz: a uint8 mask on an oblique grid (1.5 mm voxels, rotated 20
  degrees about (1, 1, 1), x and z axes flipped), 1 in a spherical shell between 4
  and 8 mm around (-116, -62, 73): streamlines that cross it leave and re-enter;
- float_mask.nii.gz: float32 on the same grid, a ball of 9 mm around (-110, -54,
  74) cut into slabs along world x with the values 0.3, 0.5, 0.7, -0.5, -0.3 and
  0.49, which pins how a mask value is rounded to inside or outside;

and runs tckedit for each case of CASES. tckedit gets every streamline's index as
its weight (`-tck_weights_in`) and writes the weights of what it keeps
(`-tck_weights_out`), so each output is a list of input indices: the selected
streamlines, or for a crop the input streamline of each piece, whose points are in
crop_<name>.tck. regions_ref.npz holds the indices of each case under its name and
each sphere as [x, y, z, radius] under sphere_<name>.

Run from the repository root, with MRtrix 3.0.8 on the PATH:

    uv run python tests/data/editing/make_regions_ref.py
"""

import pathlib
import subprocess
import tempfile

import nibabel as nib
import numpy as np

OUT = pathlib.Path("tests/data/editing")
TRACKS = OUT / "tracks.tck"
NUM_STREAMLINES = 500

SPHERES = {
    "a": (-124.0, -76.0, 74.0, 5.0),
    "b": (-116.0, -62.0, 73.0, 5.0),
    "c": (-108.0, -50.0, 74.0, 5.0),
    # Contains c and reaches towards b: tests that a point goes to the first region.
    "big": (-112.0, -55.0, 74.0, 9.0),
    "away": (-100.0, -60.0, 85.0, 8.0),
    "start": (-129.0, -88.0, 77.0, 4.0),
    "end": (-105.0, -43.0, 76.0, 4.0),
}


def sphere(name: str) -> str:
    """The sphere as tckedit's x,y,z,r argument."""
    return ",".join(str(value) for value in SPHERES[name])


SHELL = str(OUT / "shell_mask.nii.gz")
FLOAT = str(OUT / "float_mask.nii.gz")

# name: (tckedit options, whether it crops)
CASES: dict[str, tuple[list[str], bool]] = {
    "include_sphere": (["-include", sphere("a")], False),
    "include_mask": (["-include", SHELL], False),
    "include_float_mask": (["-include", FLOAT], False),
    "include_two": (["-include", sphere("a"), "-include", sphere("c")], False),
    "ordered_two": (
        ["-include_ordered", sphere("a"), "-include_ordered", sphere("c")],
        False,
    ),
    "ordered_two_reversed": (
        ["-include_ordered", sphere("c"), "-include_ordered", sphere("a")],
        False,
    ),
    "ordered_three": (
        ["-include_ordered", sphere("a")]
        + ["-include_ordered", sphere("b")]
        + ["-include_ordered", sphere("c")],
        False,
    ),
    "ordered_three_swapped": (
        ["-include_ordered", sphere("a")]
        + ["-include_ordered", sphere("c")]
        + ["-include_ordered", sphere("b")],
        False,
    ),
    "ordered_overlap": (
        ["-include_ordered", sphere("b"), "-include_ordered", sphere("big")],
        False,
    ),
    "ordered_overlap_reversed": (
        ["-include_ordered", sphere("big"), "-include_ordered", sphere("b")],
        False,
    ),
    "ordered_mask": (
        ["-include_ordered", sphere("a"), "-include_ordered", SHELL],
        False,
    ),
    "exclude_sphere": (["-exclude", sphere("away")], False),
    "exclude_mask": (["-exclude", SHELL], False),
    "include_exclude": (
        ["-include", sphere("a"), "-exclude", sphere("b")],
        False,
    ),
    "ends_include": (["-include", sphere("start"), "-ends_only"], False),
    "ends_include_two": (
        ["-include", sphere("start"), "-include", sphere("end"), "-ends_only"],
        False,
    ),
    "ends_include_mask": (["-include", SHELL, "-ends_only"], False),
    "ends_exclude": (["-exclude", sphere("end"), "-ends_only"], False),
    "ends_ordered": (
        ["-include_ordered", sphere("start")]
        + ["-include_ordered", sphere("end"), "-ends_only"],
        False,
    ),
    "inverse_include": (["-include", sphere("a"), "-inverse"], False),
    "inverse_exclude": (["-exclude", sphere("away"), "-inverse"], False),
    "crop_shell": (["-mask", SHELL], True),
    "crop_float": (["-mask", FLOAT], True),
    "crop_spheres": (["-mask", sphere("a"), "-mask", sphere("c")], True),
    "crop_shell_inverse": (["-mask", SHELL, "-inverse"], True),
}


def oblique_affine() -> np.ndarray:
    """1.5 mm voxels rotated 20 degrees about (1, 1, 1), x and z flipped."""
    axis = np.ones(3) / np.sqrt(3.0)
    angle = np.deg2rad(20.0)
    cross = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    rotation = (
        np.cos(angle) * np.eye(3)
        + np.sin(angle) * cross
        + (1 - np.cos(angle)) * np.outer(axis, axis)
    )
    linear = rotation @ np.diag([-1.5, 1.5, -1.5])
    centre_voxel = np.array([14.5, 14.5, 14.5])
    affine = np.eye(4)
    affine[:3, :3] = linear
    affine[:3, 3] = np.array([-114.0, -60.0, 74.0]) - linear @ centre_voxel
    return affine


def voxel_centres(affine: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    """(X, Y, Z, 3) world coordinates of every voxel centre."""
    grid = np.stack(np.indices(shape), axis=-1).astype(np.float64)
    return grid @ affine[:3, :3].T + affine[:3, 3]


def save_nifti(data: np.ndarray, affine: np.ndarray, path: str) -> None:
    """Saves with the same sform and qform."""
    image = nib.Nifti1Image(data, affine)
    image.set_sform(affine, code=1)
    image.set_qform(affine, code=1)
    nib.save(image, path)


def make_masks() -> None:
    """Writes the shell mask and the float mask."""
    shape = (30, 30, 30)
    affine = oblique_affine()
    centres = voxel_centres(affine, shape)

    shell_distance = np.linalg.norm(centres - np.array([-116.0, -62.0, 73.0]), axis=-1)
    shell = ((shell_distance >= 4.0) & (shell_distance <= 8.0)).astype(np.uint8)
    save_nifti(shell, affine, SHELL)

    ball_centre = np.array([-110.0, -54.0, 74.0])
    in_ball = np.linalg.norm(centres - ball_centre, axis=-1) <= 9.0
    slab = np.clip(((centres[..., 0] - ball_centre[0] + 9.0) // 3.0), 0, 5)
    values = np.array([0.3, 0.5, 0.7, -0.5, -0.3, 0.49], dtype=np.float32)
    floats = np.where(in_ball, values[slab.astype(int)], 0.0).astype(np.float32)
    save_nifti(floats, affine, FLOAT)


def run_case(options: list[str], crops: bool, name: str, weights_in: str) -> np.ndarray:
    """Runs tckedit with the options and returns the kept input indices."""
    with tempfile.TemporaryDirectory() as folder:
        out_tck = str(OUT / f"{name}.tck") if crops else f"{folder}/out.tck"
        weights_out = f"{folder}/weights.txt"
        subprocess.run(
            ["tckedit", "-quiet", "-force", str(TRACKS), out_tck, *options]
            + ["-tck_weights_in", weights_in, "-tck_weights_out", weights_out],
            check=True,
        )
        text = pathlib.Path(weights_out).read_text().split()
        return np.array([int(float(value)) - 1 for value in text], dtype=np.int64)


def main() -> None:
    """Makes the masks, runs every case and saves the references."""
    make_masks()
    with tempfile.TemporaryDirectory() as folder:
        # Index + 1: the weights start at 1 so that none is zero.
        weights_in = f"{folder}/weights.txt"
        np.savetxt(weights_in, np.arange(1, NUM_STREAMLINES + 1), fmt="%d")
        results = {}
        for name, (options, crops) in CASES.items():
            results[name] = run_case(options, crops, name, weights_in)
            print(name, len(results[name]))  # noqa: T201 (script output)
    spheres = {f"sphere_{name}": np.array(s) for name, s in SPHERES.items()}
    np.savez_compressed(OUT / "regions_ref.npz", **results, **spheres)

    version = subprocess.run(["tckedit", "-version"], capture_output=True, text=True)
    print("saved with", version.stdout.splitlines()[0])  # noqa: T201 (script output)


if __name__ == "__main__":
    main()
