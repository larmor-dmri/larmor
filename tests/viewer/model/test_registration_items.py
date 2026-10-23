import queue
import shutil
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from larmor.io import (
    load_tractogram,
    load_volume,
    save_surface,
    save_tractogram,
    save_volume,
)
from larmor.registration import (
    read_fsl_warp,
    read_mrtrix_transform,
    transform_surface,
    transform_volume,
)
from larmor.types import Surface, Tractogram, Volume
from larmor.viewer import Category, Viewer
from larmor.viewer.loading import Loader
from larmor.viewer.model.registration import Registration
from larmor.viewer.model.registry import SUBSETS
from larmor.viewer.model.surface import SurfaceData

DATA = Path(__file__).parents[2] / "data" / "registration"
WAIT = 10.0


def _on_grid(path: Path, grid_path: Path) -> np.ndarray:
    """An oracle image in the voxel order of another file's grid."""
    volume, grid = load_volume(path), load_volume(grid_path).grid
    to_volume = np.linalg.solve(volume.affine, grid.affine)
    ijk = np.stack(np.meshgrid(*map(np.arange, grid.shape), indexing="ij"), -1)
    index = np.rint(ijk @ to_volume[:3, :3].T + to_volume[:3, 3]).astype(int)
    data: np.ndarray = volume.data[index[..., 0], index[..., 1], index[..., 2]]
    return data


def test_overlay_through_an_ants_affine_is_resampled_as_labels() -> None:
    item = Viewer().load(
        DATA / "moving_labels.nii.gz",
        Category.OVERLAYS,
        transforms=[str(DATA / "ants_affine.mat")],
        reference=DATA / "fixed.nii.gz",
    )
    assert isinstance(item.data, Volume)
    expected = load_volume(DATA / "ants_affine_nearest.nii.gz").data
    np.testing.assert_array_equal(item.data.data, expected)
    assert item.registration is not None


def test_tractogram_follows_the_image_and_reopens_from_a_session(
    tmp_path: Path,
) -> None:
    viewer = Viewer()
    viewer.load(
        DATA / "tracks_fixed.tck",
        Category.TRACTOGRAMS,
        transforms=[str(DATA / "mrtrix_warp.mif")],
        inverse=True,
    )
    expected = load_tractogram(DATA / "mrtrix_warp_tracks.tck").points
    session = tmp_path / "view.lar.json"
    viewer.save_session(session)
    reopened = Viewer()
    report = reopened.load_session(session)
    assert report.problems == ()
    for v in (viewer, reopened):
        data = v.scene.items[0].data
        assert isinstance(data, Tractogram)
        np.testing.assert_allclose(data.points, expected, atol=1e-5)


def test_background_loader_brings_files_through_the_registration() -> None:
    posted: queue.Queue[Callable[[], None]] = queue.Queue()
    viewer = Viewer()
    errors: list[str] = []
    loader = Loader(viewer, posted.put, on_error=errors.append)
    registration = Registration.from_files(
        [DATA / "mrtrix_affine.txt"], DATA / "fixed.nii.gz"
    )
    loader.open([DATA / "moving.nii.gz"], Category.VOLUMES, registration=registration)
    while loader.busy:
        posted.get(timeout=WAIT)()
    assert errors == []
    expected = _on_grid(DATA / "mrtrix_affine_linear.nii.gz", DATA / "fixed.nii.gz")
    item = viewer.scene.items[0]
    assert isinstance(item.data, Volume)
    np.testing.assert_allclose(item.data.data, expected, atol=1e-4)
    assert item.registration == registration


def test_fods_are_refused() -> None:
    with pytest.raises(ValueError, match="FODs"):
        Viewer().load(
            DATA / "moving.nii.gz",
            Category.FODS,
            transforms=[str(DATA / "ants_affine.mat")],
            reference=DATA / "fixed.nii.gz",
        )


MOVING = DATA / "moving.nii.gz"
FIXED = DATA / "fixed.nii.gz"


def _tracks(points: np.ndarray, path: Path) -> Path:
    save_tractogram(Tractogram(points, np.array([0, len(points)])), path)
    return path


def test_image_through_an_fsl_warp() -> None:
    item = Viewer().load(
        MOVING,
        Category.VOLUMES,
        transforms=[str(DATA / "fsl_warp.nii.gz")],
        reference=FIXED,
        format="fsl",
    )
    moving = load_volume(MOVING)
    warp = read_fsl_warp(DATA / "fsl_warp.nii.gz", moving.grid)
    expected = transform_volume(moving, warp, load_volume(FIXED).grid)
    assert isinstance(item.data, Volume)
    np.testing.assert_allclose(item.data.data, expected.data)


def test_tractogram_through_flirt_both_ways(tmp_path: Path) -> None:
    # img2imgcoord moved these moving-space points into the fixed space; going
    # the other way brings them back. The tractogram has no grid, so the FLIRT
    # run's reference is given.
    fixed_points = np.loadtxt(DATA / "fsl_flirt_points.txt")
    path = _tracks(fixed_points, tmp_path / "in_fixed.tck")
    item = Viewer().load(
        path,
        Category.TRACTOGRAMS,
        transforms=[str(DATA / "flirt.mat")],
        reference=MOVING,
        inverse=True,
        fsl_reference_path=FIXED,
    )
    assert isinstance(item.data, Tractogram)
    original = np.loadtxt(DATA / "points_fixed.txt")
    np.testing.assert_allclose(item.data.points, original, atol=1e-4)

    # Forward, from the moving space, the run's input image is given.
    moving_path = _tracks(original, tmp_path / "in_moving.tck")
    forward = Viewer().load(
        moving_path,
        Category.TRACTOGRAMS,
        transforms=[str(DATA / "flirt.mat")],
        reference=FIXED,
        fsl_input_path=MOVING,
    )
    assert isinstance(forward.data, Tractogram)
    np.testing.assert_allclose(forward.data.points, fixed_points, atol=1e-4)

    # The FLIRT run's images swapped give other points.
    swapped = Viewer().load(
        moving_path,
        Category.TRACTOGRAMS,
        transforms=[str(DATA / "flirt.mat")],
        reference=FIXED,
        fsl_input_path=FIXED,
        fsl_reference_path=MOVING,
    )
    assert isinstance(swapped.data, Tractogram)
    assert not np.allclose(swapped.data.points, fixed_points, atol=1e-2)


def test_tractogram_without_an_fsl_image_says_so(tmp_path: Path) -> None:
    path = _tracks(np.loadtxt(DATA / "points_fixed.txt"), tmp_path / "t.tck")
    with pytest.raises(ValueError, match="grid of the input image"):
        Viewer().load(
            path,
            Category.TRACTOGRAMS,
            transforms=[str(DATA / "flirt.mat")],
            reference=FIXED,
        )


def test_peaks_are_reoriented(tmp_path: Path) -> None:
    moving = load_volume(MOVING)
    vector = np.array([0.0, 1.2, 0.5], dtype=np.float32)
    data = np.broadcast_to(vector, (*moving.data.shape, 3)).copy()
    path = tmp_path / "peaks.nii"
    save_volume(Volume(data, moving.affine), path)
    item = Viewer().load(
        path,
        Category.PEAKS,
        transforms=[str(DATA / "mrtrix_affine.txt")],
        reference=FIXED,
    )
    affine = read_mrtrix_transform(DATA / "mrtrix_affine.txt")
    expected = transform_volume(
        load_volume(path),
        affine,
        load_volume(FIXED).grid,
        interpolation="nearest",
        reorient="peaks",
    )
    assert isinstance(item.data, Volume)
    np.testing.assert_allclose(item.data.data, expected.data)
    inside = np.isfinite(expected.data[..., 0]) & (expected.data[..., 0] != 0)
    assert not np.allclose(expected.data[inside], vector)


def test_surface_moves_with_the_inverse(tmp_path: Path) -> None:
    surface = Surface(
        np.array([[0.0, 0, 0], [5, 0, 0], [0, 5, 0], [0, 0, 5]]),
        np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]),
    )
    path = tmp_path / "s.gii"
    save_surface(surface, path)
    item = Viewer().load(
        path, Category.SURFACES, transforms=[str(DATA / "mrtrix_affine.txt")]
    )
    affine = read_mrtrix_transform(DATA / "mrtrix_affine.txt")
    expected = transform_surface(surface, affine.inverse())
    assert isinstance(item.data, SurfaceData)
    np.testing.assert_allclose(item.data.surface.vertices, expected.vertices)


def test_read_subset_is_kept() -> None:
    item = Viewer().load(
        DATA / "tracks_fixed.tck",
        Category.TRACTOGRAMS,
        max_streamlines=2,
        transforms=[str(DATA / "mrtrix_affine.txt")],
    )
    assert isinstance(item.data, Tractogram)
    assert len(SUBSETS[item.data].indices) == 2


def test_moved_session_finds_its_transforms(tmp_path: Path) -> None:
    first = tmp_path / "first"
    first.mkdir()
    for name in ("tracks_fixed.tck", "mrtrix_affine.txt", "fixed.nii.gz"):
        shutil.copy(DATA / name, first / name)
    viewer = Viewer()
    viewer.load(
        first / "tracks_fixed.tck",
        Category.TRACTOGRAMS,
        transforms=[str(first / "mrtrix_affine.txt")],
        reference=first / "fixed.nii.gz",
    )
    viewer.save_session(first / "view.lar.json")
    moved = shutil.move(first, tmp_path / "moved")
    reopened = Viewer()
    report = reopened.load_session(Path(moved) / "view.lar.json")
    assert report.problems == ()
    registration = reopened.scene.items[0].registration
    assert registration is not None
    assert registration.transforms[0][0] == Path(moved) / "mrtrix_affine.txt"
    assert registration.reference == Path(moved) / "fixed.nii.gz"


@pytest.mark.parametrize(
    ("data", "error"),
    [
        ({"transforms": [[None, None]]}, TypeError),
        ({"transforms": [["a.txt", None]], "relative_transforms": "a"}, TypeError),
        ({"transforms": "a.txt"}, TypeError),
        ({"transforms": [["a.txt", None]], "kind": "ant"}, ValueError),
    ],
)
def test_malformed_session_registration_is_an_error_to_report(
    data: dict[str, object], error: type[Exception]
) -> None:
    with pytest.raises(error):
        Registration.from_json(data)
