"""Generates the registration fixtures in tests/data/registration/.

Two small images on different oblique grids, transforms written by ANTs, FSL and
MRtrix3, and what each tool gives when it applies them to images and to points.
Needs ANTs (on the PATH, or its bin folder in ANTS_BIN), FSL (FSLDIR, default
~/fsl) and MRtrix3 on the PATH.

Run from the repository root:
uv run python tests/data/registration/make_registration_ref.py
"""

import os
import pathlib
import subprocess

import nibabel as nib
import numpy as np

OUT = pathlib.Path("tests/data/registration")
ANTS = os.environ.get("ANTS_BIN", "")
FSLDIR = pathlib.Path(os.environ.get("FSLDIR", "~/fsl")).expanduser()
ENV = {
    **os.environ,
    "FSLDIR": str(FSLDIR),
    "FSLOUTPUTTYPE": "NIFTI_GZ",
    "PATH": os.pathsep.join(
        p for p in (ANTS, str(FSLDIR / "bin"), os.environ["PATH"]) if p
    ),
}


def run(*args: str) -> str:
    result = subprocess.run(
        list(args), cwd=OUT, env=ENV, check=True, capture_output=True, text=True
    )
    return result.stdout


def rotation(axis: np.ndarray, degrees: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    angle = np.deg2rad(degrees)
    k = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


def grid_affine(
    shape: tuple[int, int, int], matrix: np.ndarray, centre: np.ndarray
) -> np.ndarray:
    """An affine whose grid is centred on `centre` (mm)."""
    affine = np.eye(4)
    affine[:3, :3] = matrix
    middle = (np.array(shape) - 1) / 2
    affine[:3, 3] = centre - matrix @ middle
    return affine


def world_points(shape: tuple[int, int, int], affine: np.ndarray) -> np.ndarray:
    ijk = np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing="ij"), -1)
    return ijk @ affine[:3, :3].T + affine[:3, 3]


def intensity(xyz: np.ndarray) -> np.ndarray:
    """A smooth pattern, so that interpolation differences stay small."""
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    blob = 40 * np.exp(-((x - 3) ** 2 + (y + 2) ** 2 + (z - 1) ** 2) / 60)
    return 100 + 20 * np.sin(x / 6) + 15 * np.cos(y / 7) + 1.2 * z + blob


def labels(xyz: np.ndarray) -> np.ndarray:
    """Four regions with sharp edges, as a parcellation has."""
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    out = np.zeros(xyz.shape[:3], dtype=np.int16)
    out[(x > 0) & (y > 0)] = 1
    out[(x <= 0) & (y > 0)] = 2
    out[(x <= 0) & (y <= 0) & (z > 0)] = 3
    out[(x - 2) ** 2 + (y + 3) ** 2 + z**2 < 25] = 4
    return out


def save(data: np.ndarray, affine: np.ndarray, name: str) -> None:
    image = nib.Nifti1Image(data, affine)
    image.set_qform(affine, code=1)
    image.set_sform(affine, code=1)
    nib.save(image, OUT / name)


def write_tracks(streamlines: list[np.ndarray], name: str) -> None:
    """Writes streamlines with tckconvert, from one text file each."""
    for s, points in enumerate(streamlines):
        np.savetxt(OUT / f"_s-{s}.txt", points, fmt="%.4f")
    run("tckconvert", "-quiet", "-force", "_s-[].txt", name)
    for s in range(len(streamlines)):
        (OUT / f"_s-{s}.txt").unlink()


def lps_csv(points: np.ndarray, name: str) -> None:
    """Writes points (RAS+ mm) as ANTs reads them: LPS, with x, y, z, t columns."""
    lps = points * [-1, -1, 1]
    rows = [",".join(f"{v:.6f}" for v in (*p, 0.0)) for p in lps]
    (OUT / name).write_text("x,y,z,t\n" + "\n".join(rows) + "\n")


def read_lps_csv(name: str) -> np.ndarray:
    values = np.loadtxt(OUT / name, delimiter=",", skiprows=1)[:, :3]
    return values * [-1, -1, 1]


def main() -> None:
    OUT.mkdir(exist_ok=True)

    # The moving image: oblique, voxels 2 x 2.2 x 2.5 mm, x stored backwards
    # (negative determinant). The fixed image: a different oblique grid with a
    # positive determinant, so FSL's coordinates flip x in one and not the other.
    moving_shape = (10, 12, 8)
    moving_affine = grid_affine(
        moving_shape,
        rotation(np.array([1.0, 0.3, 0.2]), 10) @ np.diag([-2.0, 2.2, 2.5]),
        np.array([1.0, -1.0, 0.5]),
    )
    fixed_shape = (12, 10, 9)
    fixed_affine = grid_affine(
        fixed_shape,
        rotation(np.array([0.2, -0.4, 1.0]), -12) @ np.diag([1.8, 2.0, 2.0]),
        np.array([-0.5, 0.5, 0.0]),
    )
    moving_xyz = world_points(moving_shape, moving_affine)
    fixed_xyz = world_points(fixed_shape, fixed_affine)
    save(intensity(moving_xyz).astype(np.float32), moving_affine, "moving.nii.gz")
    save(labels(moving_xyz), moving_affine, "moving_labels.nii.gz")
    save(intensity(fixed_xyz).astype(np.float32), fixed_affine, "fixed.nii.gz")

    # Streamlines in the fixed space, inside its grid.
    rng = np.random.default_rng(367)
    streamlines = []
    for n in (6, 2, 9, 4):
        start = rng.uniform(-4, 4, 3)
        steps = rng.normal(0, 0.8, (n, 3)).cumsum(axis=0)
        streamlines.append(start + steps)
    write_tracks(streamlines, "tracks_fixed.tck")
    fixed_points = np.concatenate(streamlines)
    np.savetxt(OUT / "points_fixed.txt", fixed_points, fmt="%.4f")
    fixed_points = np.loadtxt(OUT / "points_fixed.txt")
    lps_csv(fixed_points, "_points_fixed.csv")

    # ANTs: an affine with a centre, as antsRegistration writes it (binary .mat),
    # and its text form; a smooth displacement field on the fixed grid.
    (OUT / "_affine_hand.txt").write_text(
        "#Insight Transform File V1.0\n#Transform 0\n"
        "Transform: AffineTransform_double_3_3\n"
        "Parameters: 1.02 0.1 -0.05 -0.08 0.97 0.12 0.06 -0.1 1.05 3.5 -2.25 1.5\n"
        "FixedParameters: 2.5 -4 6\n"
    )
    run(
        "antsApplyTransforms", "-d", "3", "-r", "fixed.nii.gz",
        "-t", "_affine_hand.txt", "-o", "Linear[ants_affine.mat]",
    )  # fmt: skip
    run("ConvertTransformFile", "3", "ants_affine.mat", "ants_affine.txt")
    run(
        "SimulateDisplacementField", "3", "BSpline", "fixed.nii.gz",
        "ants_warp.nii.gz", "40", "12", "1", "3", "4",
    )  # fmt: skip

    ants_apply = ["antsApplyTransforms", "-d", "3", "--float", "0"]
    for interp, source, name in [
        ("Linear", "moving.nii.gz", "ants_affine_linear.nii.gz"),
        ("NearestNeighbor", "moving_labels.nii.gz", "ants_affine_nearest.nii.gz"),
        ("BSpline[3]", "moving.nii.gz", "ants_affine_cubic.nii.gz"),
    ]:
        run(
            *ants_apply, "-i", source, "-r", "fixed.nii.gz", "-n", interp,
            "-t", "ants_affine.mat", "-o", name,
        )  # fmt: skip
    run(
        *ants_apply, "-i", "moving.nii.gz", "-r", "fixed.nii.gz", "-n", "Linear",
        "-t", "ants_warp.nii.gz", "-t", "ants_affine.mat",
        "-o", "ants_chain_linear.nii.gz",
    )  # fmt: skip
    run(
        *ants_apply, "-i", "fixed.nii.gz", "-r", "moving.nii.gz", "-n", "Linear",
        "-t", "[ants_affine.mat,1]", "-o", "ants_affine_inverse_linear.nii.gz",
    )  # fmt: skip
    points_apply = ["antsApplyTransformsToPoints", "-d", "3", "-p", "1"]
    run(
        *points_apply, "-i", "_points_fixed.csv", "-o", "_ants_chain_points.csv",
        "-t", "ants_warp.nii.gz", "-t", "ants_affine.mat",
    )  # fmt: skip
    np.savetxt(
        OUT / "ants_chain_points.txt", read_lps_csv("_ants_chain_points.csv"), "%.6f"
    )
    # The same points taken as moving-space points, brought into the fixed space.
    run(
        *points_apply, "-i", "_points_fixed.csv", "-o", "_ants_inverse_points.csv",
        "-t", "[ants_affine.mat,1]",
    )  # fmt: skip
    np.savetxt(
        OUT / "ants_affine_inverse_points.txt",
        read_lps_csv("_ants_inverse_points.csv"),
        "%.6f",
    )

    # FSL: a FLIRT matrix from the moving to the fixed image (scaled voxel
    # coordinates), its inverse by convert_xfm, and a relative warp field on the
    # fixed grid made by convertwarp from a smooth field.
    flirt = np.eye(4)
    flirt[:3, :3] = rotation(np.array([0.3, 1.0, -0.2]), 9) @ np.diag(
        [1.03, 0.96, 1.01]
    )
    flirt[:3, 3] = [2.5, -1.75, 3.0]
    np.savetxt(OUT / "flirt.mat", flirt, fmt="%.10f", delimiter="  ")
    run("convert_xfm", "-omat", "flirt_inverse.mat", "-inverse", "flirt.mat")
    for interp, source, name in [
        ("trilinear", "moving.nii.gz", "fsl_flirt_linear.nii.gz"),
        ("nearestneighbour", "moving_labels.nii.gz", "fsl_flirt_nearest.nii.gz"),
    ]:
        run(
            "flirt", "-in", source, "-ref", "fixed.nii.gz", "-applyxfm",
            "-init", "flirt.mat", "-interp", interp, "-out", name,
        )  # fmt: skip
    ijk = np.stack(np.meshgrid(*[np.arange(n) for n in fixed_shape], indexing="ij"))
    field = np.stack(
        [
            1.5 * np.sin(ijk[1] / 3.0),
            -1.2 * np.cos(ijk[2] / 2.5),
            1.0 * np.sin(ijk[0] / 4.0 + ijk[1] / 5.0),
        ],
        axis=-1,
    ).astype(np.float32)
    save(field, fixed_affine, "_fsl_field.nii.gz")
    run(
        "convertwarp", "--ref=fixed.nii.gz", "--warp1=_fsl_field.nii.gz",
        "--relout", "--out=fsl_warp.nii.gz",
    )  # fmt: skip
    for interp, source, name in [
        ("trilinear", "moving.nii.gz", "fsl_warp_linear.nii.gz"),
        ("nn", "moving_labels.nii.gz", "fsl_warp_nearest.nii.gz"),
    ]:
        run(
            "applywarp", "-i", source, "-r", "fixed.nii.gz", "-w", "fsl_warp.nii.gz",
            "--rel", f"--interp={interp}", "-o", name,
        )  # fmt: skip
    moving_points = fixed_points  # points near the centre, inside both grids
    (OUT / "_points.txt").write_text(
        "\n".join(" ".join(f"{v:.4f}" for v in p) for p in moving_points) + "\n"
    )
    flirt_points = run(
        "img2imgcoord", "-src", "moving.nii.gz", "-dest", "fixed.nii.gz",
        "-xfm", "flirt.mat", "-mm", "_points.txt",
    )  # fmt: skip
    np.savetxt(
        OUT / "fsl_flirt_points.txt",
        np.loadtxt(flirt_points.splitlines()[1:]),
        "%.6f",
    )
    warp_points = run(
        "img2imgcoord", "-src", "moving.nii.gz", "-dest", "fixed.nii.gz",
        "-warp", "fsl_warp.nii.gz", "-mm", "_points.txt",
    )  # fmt: skip
    np.savetxt(
        OUT / "fsl_warp_points.txt",
        np.loadtxt(warp_points.splitlines()[1:]),
        "%.6f",
    )

    # MRtrix3: the ANTs affine and the FLIRT matrix converted by transformconvert,
    # and the ANTs chain as a deformation field (MRtrix's documented recipe:
    # warpinit, antsApplyTransforms on each coordinate image, warpcorrect).
    run(
        "transformconvert", "-quiet", "-force", "ants_affine.txt", "itk_import",
        "mrtrix_affine.txt",
    )  # fmt: skip
    run(
        "transformconvert", "-quiet", "-force", "flirt.mat", "moving.nii.gz",
        "fixed.nii.gz", "flirt_import", "mrtrix_from_flirt.txt",
    )  # fmt: skip
    # The identity on a padded grid, so that every fixed voxel finds its position
    # even where it maps outside the moving image.
    run(
        "mrgrid", "-quiet", "-force", "moving.nii.gz", "pad", "-uniform", "8",
        "_padded.nii",
    )  # fmt: skip
    run("warpinit", "-quiet", "-force", "_padded.nii", "_identity[].nii")
    for i in range(3):
        run(
            *ants_apply, "-e", "0", "-i", f"_identity{i}.nii", "-r", "fixed.nii.gz",
            "-n", "Linear", "-t", "ants_warp.nii.gz", "-t", "ants_affine.mat",
            "--default-value", "2147483647", "-o", f"_deform{i}.nii",
        )  # fmt: skip
    run(
        "warpcorrect", "-quiet", "-force", "_deform[].nii", "mrtrix_warp.mif",
        "-marker", "2147483647",
    )  # fmt: skip
    run(
        "warpinvert", "-quiet", "-force", "mrtrix_warp.mif",
        "mrtrix_warp_inverse.mif",
    )  # fmt: skip
    mrtransform = ["mrtransform", "-quiet", "-force", "moving.nii.gz"]
    run(
        *mrtransform, "-linear", "mrtrix_affine.txt", "-template", "fixed.nii.gz",
        "-interp", "linear", "mrtrix_affine_linear.nii.gz",
    )  # fmt: skip
    run(
        "mrtransform", "-quiet", "-force", "moving_labels.nii.gz",
        "-linear", "mrtrix_affine.txt", "-template", "fixed.nii.gz",
        "-interp", "nearest", "mrtrix_affine_nearest.nii.gz",
    )  # fmt: skip
    run(
        *mrtransform, "-warp", "mrtrix_warp.mif", "-interp", "linear",
        "mrtrix_warp_linear.nii.gz",
    )  # fmt: skip
    run(
        "tcktransform", "-quiet", "-force", "tracks_fixed.tck", "mrtrix_warp.mif",
        "mrtrix_warp_tracks.tck",
    )  # fmt: skip
    run(
        "tcktransform", "-quiet", "-force", "mrtrix_warp_tracks.tck",
        "mrtrix_warp_inverse.mif", "mrtrix_warp_inverse_tracks.tck",
    )  # fmt: skip
    # The linear transform as deformation fields for tcktransform: forward on the
    # fixed grid, and its inverse on the moving grid for moving-space points.
    run(
        "transformcompose", "-quiet", "-force", "mrtrix_affine.txt",
        "-template", "fixed.nii.gz", "_affine_deform.mif",
    )  # fmt: skip
    run(
        "transformcalc", "-quiet", "-force", "mrtrix_affine.txt", "invert",
        "_affine_inverse.txt",
    )  # fmt: skip
    run(
        "transformcompose", "-quiet", "-force", "_affine_inverse.txt",
        "-template", "moving.nii.gz", "_affine_inverse_deform.mif",
    )  # fmt: skip
    run(
        "tcktransform", "-quiet", "-force", "tracks_fixed.tck",
        "_affine_deform.mif", "mrtrix_affine_tracks.tck",
    )  # fmt: skip
    run(
        "tcktransform", "-quiet", "-force", "tracks_fixed.tck",
        "_affine_inverse_deform.mif", "mrtrix_affine_inverse_tracks.tck",
    )  # fmt: skip

    for path in OUT.glob("_*"):
        path.unlink()
    versions = [
        run("mrtransform", "-version").splitlines()[0],
        run("antsRegistration", "--version").splitlines()[0],
        (FSLDIR / "etc" / "fslversion").read_text().strip(),
    ]
    print("saved", OUT, versions)  # noqa: T201 (script output)


if __name__ == "__main__":
    main()
