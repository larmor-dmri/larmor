"""Applying transforms against the tools' own output (tests/data/registration)."""

from pathlib import Path

import numpy as np
import pytest

from larmor.io import load_tractogram, load_volume, save_volume
from larmor.registration import (
    AffineTransform,
    Transform,
    TransformChain,
    load_transform,
    read_ants_transform,
    read_flirt_transform,
    read_fsl_warp,
    read_mrtrix_transform,
    read_mrtrix_warp,
    transform_points,
    transform_volume,
)
from larmor.types import Grid, Volume

DATA = Path(__file__).parents[1] / "data" / "registration"

# Images: float32 outputs of float32 inputs around 100, so 1e-4 is float32
# rounding. Points: the tools print 6 decimals, TCK stores float32.
IMAGE_ATOL = 1e-4
POINT_ATOL = 1e-5


def _volume(name: str) -> Volume:
    return load_volume(DATA / name)


def _on_grid(volume: Volume, grid: Grid) -> np.ndarray:
    """The data of a volume stored with other strides, in the voxel order of grid."""
    to_volume = np.linalg.solve(volume.affine, grid.affine)
    ijk = np.stack(np.meshgrid(*map(np.arange, grid.shape), indexing="ij"), -1)
    index = np.rint(ijk @ to_volume[:3, :3].T + to_volume[:3, 3]).astype(int)
    data = volume.as_float(dtype=np.float64)
    reordered: np.ndarray = data[index[..., 0], index[..., 1], index[..., 2]]
    return reordered


def _points(name: str) -> np.ndarray:
    return np.loadtxt(DATA / name)


def _tracks(name: str) -> np.ndarray:
    return load_tractogram(DATA / name).points.astype(np.float64)


MOVING = _volume("moving.nii.gz")
LABELS = _volume("moving_labels.nii.gz")
FIXED = _volume("fixed.nii.gz")


def _ants_affine(name: str) -> AffineTransform:
    transform = read_ants_transform(DATA / name)
    assert isinstance(transform, AffineTransform)
    return transform


ANTS_AFFINE = _ants_affine("ants_affine.mat")
ANTS_CHAIN = TransformChain(
    [read_ants_transform(DATA / "ants_warp.nii.gz"), ANTS_AFFINE]
)
FLIRT = read_flirt_transform(DATA / "flirt.mat", MOVING.grid, FIXED.grid)
FSL_WARP = read_fsl_warp(DATA / "fsl_warp.nii.gz", MOVING.grid)
MRTRIX_AFFINE = read_mrtrix_transform(DATA / "mrtrix_affine.txt")
MRTRIX_WARP = read_mrtrix_warp(DATA / "mrtrix_warp.mif")


@pytest.mark.parametrize(
    ("image", "transform", "interpolation", "expected"),
    [
        (MOVING, ANTS_AFFINE, "linear", "ants_affine_linear.nii.gz"),
        (LABELS, ANTS_AFFINE, "nearest", "ants_affine_nearest.nii.gz"),
        (MOVING, ANTS_AFFINE, "cubic", "ants_affine_cubic.nii.gz"),
        (MOVING, ANTS_CHAIN, "linear", "ants_chain_linear.nii.gz"),
        (MOVING, MRTRIX_AFFINE, "linear", "mrtrix_affine_linear.nii.gz"),
        (LABELS, MRTRIX_AFFINE, "nearest", "mrtrix_affine_nearest.nii.gz"),
        (MOVING, MRTRIX_WARP, "linear", "mrtrix_warp_linear.nii.gz"),
    ],
)
def test_image_matches_ants_and_mrtrix(
    image: Volume, transform: Transform, interpolation: str, expected: str
) -> None:
    ours = transform_volume(image, transform, FIXED.grid, interpolation=interpolation)  # type: ignore[arg-type]
    reference = _on_grid(_volume(expected), FIXED.grid)
    np.testing.assert_allclose(
        ours.as_float(dtype=np.float64), reference, atol=IMAGE_ATOL
    )
    if interpolation == "nearest":
        assert ours.data.dtype == image.data.dtype


def test_inverse_affine_image_matches_ants() -> None:
    ours = transform_volume(FIXED, ANTS_AFFINE.inverse(), MOVING.grid)
    reference = _on_grid(_volume("ants_affine_inverse_linear.nii.gz"), MOVING.grid)
    np.testing.assert_allclose(ours.data, reference, atol=IMAGE_ATOL)


@pytest.mark.parametrize(
    ("image", "transform", "interpolation", "expected"),
    [
        (MOVING, FLIRT, "linear", "fsl_flirt_linear.nii.gz"),
        (LABELS, FLIRT, "nearest", "fsl_flirt_nearest.nii.gz"),
        (MOVING, FSL_WARP, "linear", "fsl_warp_linear.nii.gz"),
        (LABELS, FSL_WARP, "nearest", "fsl_warp_nearest.nii.gz"),
    ],
)
def test_image_matches_fsl_inside_the_input(
    image: Volume, transform: Transform, interpolation: str, expected: str
) -> None:
    # FSL treats a point beyond the outer voxel centres as outside, and fills it with
    # a background of its own, so only voxels that land inside them are compared.
    ours = transform_volume(image, transform, FIXED.grid, interpolation=interpolation)  # type: ignore[arg-type]
    reference = _on_grid(_volume(expected), FIXED.grid)
    ijk = np.stack(np.meshgrid(*map(np.arange, FIXED.grid.shape), indexing="ij"), -1)
    world = ijk.reshape(-1, 3) @ FIXED.affine[:3, :3].T + FIXED.affine[:3, 3]
    to_voxels = np.linalg.inv(image.affine)
    voxels = transform.map_points(world) @ to_voxels[:3, :3].T + to_voxels[:3, 3]
    upper = np.array(image.data.shape[:3]) - 1
    inside = ((voxels >= 0) & (voxels <= upper)).all(axis=1).reshape(FIXED.grid.shape)
    assert inside.sum() > 0.5 * inside.size
    np.testing.assert_allclose(
        ours.as_float(dtype=np.float64)[inside], reference[inside], atol=IMAGE_ATOL
    )


def test_affine_formats_agree_with_mrtrix_conversions() -> None:
    # transformconvert's itk_import and flirt_import give the same world matrices.
    text = _ants_affine("ants_affine.txt")
    from_flirt = read_mrtrix_transform(DATA / "mrtrix_from_flirt.txt")
    np.testing.assert_allclose(text.matrix, ANTS_AFFINE.matrix, atol=1e-12)
    np.testing.assert_allclose(MRTRIX_AFFINE.matrix, ANTS_AFFINE.matrix, atol=1e-12)
    np.testing.assert_allclose(from_flirt.matrix, FLIRT.matrix, atol=1e-6)


def _load_points(name: str) -> np.ndarray:
    return _tracks(name) if name.endswith(".tck") else _points(name)


@pytest.mark.parametrize(
    ("transform", "inverse", "source", "expected"),
    [
        (ANTS_CHAIN, False, "points_fixed.txt", "ants_chain_points.txt"),
        (ANTS_AFFINE, True, "points_fixed.txt", "ants_affine_inverse_points.txt"),
        (FLIRT, True, "points_fixed.txt", "fsl_flirt_points.txt"),
        (MRTRIX_WARP, False, "tracks_fixed.tck", "mrtrix_warp_tracks.tck"),
        (MRTRIX_AFFINE, False, "tracks_fixed.tck", "mrtrix_affine_tracks.tck"),
        (MRTRIX_AFFINE, True, "tracks_fixed.tck", "mrtrix_affine_inverse_tracks.tck"),
    ],
)
def test_points_match_ants_fsl_and_tcktransform(
    transform: Transform, inverse: bool, source: str, expected: str
) -> None:
    applied = transform.inverse() if inverse else transform
    np.testing.assert_allclose(
        transform_points(_load_points(source), applied),
        _load_points(expected),
        atol=POINT_ATOL,
    )


def test_ants_chain_and_its_mrtrix_deformation_agree() -> None:
    points = _points("points_fixed.txt")
    np.testing.assert_allclose(
        ANTS_CHAIN.map_points(points), MRTRIX_WARP.map_points(points), atol=1e-5
    )


def test_numerical_inverse_matches_the_tools_within_their_accuracy() -> None:
    # warpinvert and img2imgcoord invert on a grid; the numerical inverse solves
    # each point to 1e-4 mm. Each tool's answer is within its own round-trip error
    # of the numerical one, which is the more exact.
    cases = [
        (MRTRIX_WARP, "mrtrix_warp_tracks.tck", "mrtrix_warp_inverse_tracks.tck"),
        (FSL_WARP, "points_fixed.txt", "fsl_warp_points.txt"),
    ]
    for warp, moved_name, theirs_name in cases:
        moved, theirs = _load_points(moved_name), _load_points(theirs_name)
        ours = transform_points(moved, warp.inverse())
        np.testing.assert_allclose(warp.map_points(ours), moved, atol=2e-4)
        their_error = np.abs(warp.map_points(theirs) - moved).max()
        assert np.abs(ours - theirs).max() <= 2 * their_error + 1e-4


def test_tool_inverse_warp_file_moves_points_as_tcktransform() -> None:
    inverse = read_mrtrix_warp(DATA / "mrtrix_warp_inverse.mif")
    np.testing.assert_allclose(
        transform_points(_tracks("mrtrix_warp_tracks.tck"), inverse),
        _tracks("mrtrix_warp_inverse_tracks.tck"),
        atol=POINT_ATOL,
    )


@pytest.mark.parametrize(
    ("name", "kind"),
    [
        ("ants_affine.mat", "AffineTransform"),
        ("ants_affine.txt", "AffineTransform"),
        ("ants_warp.nii.gz", "DisplacementField"),
        ("flirt.mat", "AffineTransform"),
        ("mrtrix_affine.txt", "AffineTransform"),
        ("mrtrix_warp.mif", "DisplacementField"),
    ],
)
def test_load_transform_tells_the_format(name: str, kind: str) -> None:
    transform = load_transform(
        DATA / name, fsl_input_grid=MOVING.grid, fsl_reference_grid=FIXED.grid
    )
    assert type(transform).__name__ == kind


def test_load_transform_needs_the_format_of_a_4d_nifti_warp() -> None:
    with pytest.raises(ValueError, match="FSL or an MRtrix3"):
        load_transform(DATA / "fsl_warp.nii.gz", fsl_input_grid=MOVING.grid)


@pytest.mark.parametrize("name", ["flirt.txt", "flirt.xfm"])
def test_4x4_text_matrix_of_unknown_format_needs_the_format(
    tmp_path: Path, name: str
) -> None:
    path = tmp_path / name
    path.write_bytes((DATA / "flirt.mat").read_bytes())
    with pytest.raises(ValueError, match="FLIRT or an MRtrix3"):
        load_transform(path, fsl_input_grid=MOVING.grid, fsl_reference_grid=FIXED.grid)
    given = load_transform(
        path, format="fsl", fsl_input_grid=MOVING.grid, fsl_reference_grid=FIXED.grid
    )
    points = _points("points_fixed.txt")
    np.testing.assert_allclose(given.map_points(points), FLIRT.map_points(points))


def test_3x4_text_matrix_is_read_as_mrtrix(tmp_path: Path) -> None:
    path = tmp_path / "affine.txt"
    rows = (DATA / "mrtrix_affine.txt").read_text().splitlines()
    path.write_text("\n".join(rows[1:4]) + "\n")
    points = _points("points_fixed.txt")
    np.testing.assert_allclose(
        load_transform(path).map_points(points), MRTRIX_AFFINE.map_points(points)
    )


def _fsl_coordinates(grid: Grid) -> np.ndarray:
    """(X, Y, Z, 3) FSL scaled voxel coordinates of a grid with det > 0."""
    assert np.linalg.det(grid.affine[:3, :3]) > 0
    sizes = np.linalg.norm(grid.affine[:3, :3], axis=0)
    ijk = np.stack(np.meshgrid(*map(np.arange, grid.shape), indexing="ij"), -1)
    ijk = ijk.astype(np.float64)
    ijk[..., 0] = grid.shape[0] - 1 - ijk[..., 0]
    coordinates: np.ndarray = ijk * sizes
    return coordinates


def test_warp_types_are_read_as_declared_and_checked(tmp_path: Path) -> None:
    points = _points("points_fixed.txt")
    # An MRtrix3 displacement field: the deformation minus each voxel's position.
    deformation = load_volume(DATA / "mrtrix_warp.mif")
    ijk = np.stack(
        np.meshgrid(*map(np.arange, deformation.grid.shape), indexing="ij"), -1
    )
    own = ijk @ deformation.affine[:3, :3].T + deformation.affine[:3, 3]
    displacement = tmp_path / "displacement.nii"
    save_volume(
        deformation.with_data(deformation.as_float(dtype=np.float64) - own),
        displacement,
    )
    as_given = load_transform(displacement, format="mrtrix", warp_type="displacement")
    np.testing.assert_allclose(
        as_given.map_points(points), MRTRIX_WARP.map_points(points), atol=1e-4
    )
    with pytest.raises(ValueError, match="look like type .displacement"):
        load_transform(displacement, format="mrtrix")
    with pytest.raises(ValueError, match="look like type .deformation"):
        load_transform(DATA / "mrtrix_warp.mif", warp_type="displacement")

    # An FSL absolute warp: the relative one plus each voxel's FSL coordinates.
    relative = load_volume(DATA / "fsl_warp.nii.gz")
    absolute = tmp_path / "absolute.nii"
    coordinates = _fsl_coordinates(relative.grid)
    save_volume(
        relative.with_data(relative.as_float(dtype=np.float64) + coordinates), absolute
    )
    read = load_transform(
        absolute, format="fsl", fsl_input_grid=MOVING.grid, warp_type="absolute"
    )
    np.testing.assert_allclose(
        read.map_points(points), FSL_WARP.map_points(points), atol=1e-4
    )
    with pytest.raises(ValueError, match="look like type .absolute"):
        load_transform(absolute, format="fsl", fsl_input_grid=MOVING.grid)
    with pytest.raises(ValueError, match="absolute, not"):
        load_transform(
            absolute, format="fsl", fsl_input_grid=MOVING.grid, warp_type="deformation"
        )


@pytest.mark.parametrize(
    "text",
    [
        "Transform: Euler3DTransform_double_3_3\nParameters: 0 0 0 0 0 0\n"
        "FixedParameters: 0 0 0\n",
        "#Transform 0\nTransform: AffineTransform_double_3_3\n"
        "Parameters: 1 0 0 0 1 0 0 0 1 0 0 0\nFixedParameters: 0 0 0\n"
        "#Transform 1\nTransform: AffineTransform_double_3_3\n"
        "Parameters: 1 0 0 0 1 0 0 0 1 0 0 0\nFixedParameters: 0 0 0\n",
    ],
    ids=["euler", "composite"],
)
def test_unsupported_itk_transforms_are_refused(tmp_path: Path, text: str) -> None:
    path = tmp_path / "transform.txt"
    path.write_text("#Insight Transform File V1.0\n" + text)
    with pytest.raises(ValueError, match="Euler3DTransform|2 transforms"):
        read_ants_transform(path)
