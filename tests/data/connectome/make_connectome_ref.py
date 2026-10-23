"""Generates the track density and connectome references in tests/data/connectome/.

Tracks 200 streamlines with tckgen (iFOD1, 1 mm steps, so the header's step size is
the spacing of the points) on the ISMRM FOD, seeded at random in the brain mask.
The parcellation is the ISMRM `fake_parcellation.nii.gz` (60 regions with
scattered unlabelled voxels) cropped around the streamlines, with its last 10
voxels along x cut off, so that some endpoints fall outside it. The template is an
empty 2 mm grid around the streamlines whose x axis is stored from right to left.
The lookup table holds the names and colours of `fake_parcellation.json`, and
weights.txt one random weight per streamline (numpy seed 364).

Then runs tckmap (default, -vox 0.7, -tck_weights_in, -ends_only) and
tck2connectome (default, -assignment_end_voxels, -assignment_radial_search 2,
-tck_weights_in, -scale_length -stat_edge mean with weights, -symmetric
-zero_diagonal, each with -out_assignments) and saves their outputs in ref.npz:
for each TDI `tdi_<case>` the non-zero voxels as flat indices (C order) and values,
with the grid's `shape` and `affine` (in RAS order, as Larmor reads the file), and
for each connectome `matrix_<case>` and `nodes_<case>`.

Run from the repository root, with MRtrix 3.0.8 on the PATH:

    uv run python tests/data/connectome/make_connectome_ref.py <ISMRM Data folder>
"""

import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

import numpy as np

from larmor.io import load_tractogram, load_volume, save_volume
from larmor.types import Volume

OUT = pathlib.Path("tests/data/connectome")
SEED = 364


def run(*args: str) -> None:
    """Runs an MRtrix command quietly, overwriting its outputs."""
    subprocess.run([*args, "-quiet", "-force"], check=True)


def make_inputs(data: pathlib.Path) -> None:
    """Writes tracks.tck, parcellation.nii.gz, template.nii.gz, lut.txt, weights.txt."""
    tracks = OUT / "tracks.tck"
    env = dict(os.environ, MRTRIX_RNG_SEED="1")
    # Run in a scratch folder with links to the inputs, so that no local path ends
    # up in the header's command_history.
    with tempfile.TemporaryDirectory() as scratch:
        for name in ("fod.nii.gz", "mask.nii.gz"):
            os.symlink(data / name, f"{scratch}/{name}")
        subprocess.run(
            [
                *("tckgen", "fod.nii.gz", "tracks.tck", "-algorithm", "iFOD1"),
                *("-step", "1", "-seed_image", "mask.nii.gz", "-mask", "mask.nii.gz"),
                *("-select", "200", "-minlength", "20", "-maxlength", "150"),
                *("-nthreads", "0", "-quiet", "-force"),
            ],
            check=True,
            env=env,
            cwd=scratch,
        )
        shutil.copyfile(f"{scratch}/tracks.tck", tracks)
    points = load_tractogram(tracks).points.astype(np.float64)

    parcellation = load_volume(data / "fake_parcellation.nii.gz")
    to_voxel = np.linalg.inv(parcellation.affine)
    voxels = points @ to_voxel[:3, :3].T + to_voxel[:3, 3]
    low = np.floor(voxels.min(axis=0)).astype(int) - 2
    high = np.ceil(voxels.max(axis=0)).astype(int) + 3
    high[0] -= 10
    data_crop = parcellation.data[low[0] : high[0], low[1] : high[1], low[2] : high[2]]
    affine = parcellation.affine.copy()
    affine[:3, 3] = parcellation.affine[:3, :3] @ low + parcellation.affine[:3, 3]
    save_volume(Volume(data_crop.astype(np.uint8), affine), OUT / "parcellation.nii.gz")

    corner = np.floor(points.min(axis=0)) - 4
    far = np.ceil(points.max(axis=0)) + 4
    shape = tuple(int(n) for n in np.ceil((far - corner) / 2))
    template = np.diag([-2.0, 2.0, 2.0, 1.0])
    template[:3, 3] = [far[0], corner[1], corner[2]]
    save_volume(Volume(np.zeros(shape, np.uint8), template), OUT / "template.nii.gz")

    table = json.loads((data / "fake_parcellation.json").read_text())
    lines = [
        f"{index} {entry['name']} {' '.join(str(c) for c in entry['color'][:3])} 0"
        for index, entry in sorted(table.items(), key=lambda item: int(item[0]))
    ]
    (OUT / "lut.txt").write_text("\n".join(lines) + "\n")

    count = load_tractogram(tracks).num_streamlines
    weights = np.random.default_rng(SEED).uniform(0.0, 2.0, count).astype(np.float32)
    (OUT / "weights.txt").write_text(" ".join(f"{w:.9g}" for w in weights) + "\n")


def main() -> None:
    """Makes the inputs, runs tckmap and tck2connectome, and saves ref.npz."""
    make_inputs(pathlib.Path(sys.argv[1]).resolve())
    tracks = str(OUT / "tracks.tck")
    template = str(OUT / "template.nii.gz")
    parcellation = str(OUT / "parcellation.nii.gz")
    weights = str(OUT / "weights.txt")
    ref: dict[str, np.ndarray] = {}
    tdi_cases = {
        "default": [],
        "fine": ["-vox", "0.7"],
        "weighted": ["-tck_weights_in", weights],
        "ends": ["-ends_only"],
    }
    connectome_cases = {
        "default": [],
        "end_voxels": ["-assignment_end_voxels"],
        "radius_2": ["-assignment_radial_search", "2"],
        "weighted": ["-tck_weights_in", weights],
        "mean_length": [
            *("-scale_length", "-stat_edge", "mean", "-tck_weights_in", weights)
        ],
        "symmetric": ["-symmetric", "-zero_diagonal"],
    }
    with tempfile.TemporaryDirectory() as scratch:
        for case, options in tdi_cases.items():
            out = f"{scratch}/{case}.nii"
            run("tckmap", tracks, "-template", template, *options, out)
            tdi = load_volume(out)
            tdi.to_closest_canonical()
            flat = tdi.data.ravel()
            nonzero = np.flatnonzero(flat)
            ref[f"tdi_{case}_index"] = nonzero.astype(np.int32)
            ref[f"tdi_{case}_value"] = flat[nonzero]
            ref[f"tdi_{case}_shape"] = np.array(tdi.data.shape)
            ref[f"tdi_{case}_affine"] = tdi.affine
        for case, options in connectome_cases.items():
            out = f"{scratch}/{case}.csv"
            assignments = f"{scratch}/{case}.txt"
            run(
                *("tck2connectome", tracks, parcellation, out),
                *("-out_assignments", assignments, *options),
            )
            ref[f"matrix_{case}"] = np.loadtxt(out, delimiter=",")
            ref[f"nodes_{case}"] = np.loadtxt(assignments, dtype=np.int64)
    version = subprocess.run(
        ["tckmap", "-version"], capture_output=True, text=True, check=True
    ).stdout.splitlines()[0]
    ref["mrtrix_version"] = np.array(version)
    np.savez_compressed(OUT / "ref.npz", **ref)


if __name__ == "__main__":
    main()
