import json
import logging
from pathlib import Path

import numpy as np
import pytest

import larmor.simulation.bundles as bundles
from larmor.io import load_tractogram, load_volume, save_tractogram, save_volume
from larmor.simulation import (
    Bundle,
    BundleGeometry,
    Compartment,
    CompartmentModel,
    Phantom,
    Protocol,
    PulseSequence,
    Tissue,
    TissueMap,
    create_bundle,
    csf_tissue,
    free_water_tissue,
    noddi_tissue,
    partition_voxels,
    read_phantom,
    write_phantom,
)
from larmor.simulation.phantom import RINGS_FULLNESS
from larmor.simulation.voxelise import tube_fill
from larmor.types import GradientTable, Grid, Tractogram, Volume

GRID = Grid((6, 6, 6), np.eye(4))


def gm_map(value: float, voxel: tuple[int, int, int] = (0, 0, 0)) -> TissueMap:
    data = np.zeros(GRID.shape)
    data[voxel] = value
    return TissueMap("GM", Volume(data, np.eye(4)))


def dense_bundle(y: float, **kwargs: object) -> Bundle:
    line = BundleGeometry(np.array([[-1.0, y, 2], [7.0, y, 2]]), radius=0.0, seed=0)
    return Bundle(line, streamline_area=10.0, **kwargs)  # type: ignore[arg-type]


def test_partition_fractions_and_background_sum_to_one_everywhere() -> None:
    maps = (
        gm_map(0.6, (1, 2, 2)),
        TissueMap("CSF", Volume(np.full(GRID.shape, 0.3), np.eye(4))),
    )
    phantom = Phantom(GRID, bundles=(dense_bundle(2.0),), tissue_maps=maps)
    partition = partition_voxels(phantom)
    total = partition.tissue_fractions.sum(axis=-1) + partition.background
    np.testing.assert_allclose(total, 1.0, atol=1e-12)
    # The bundle fills its row, so the maps get nothing there.
    assert partition.tissue_fractions[1, 2, 2].tolist() == [1.0, 0.0, 0.0]


def test_small_map_value_is_not_scaled_up() -> None:
    partition = partition_voxels(Phantom(GRID, tissue_maps=(gm_map(0.05),)))
    assert partition.tissue_fractions[0, 0, 0].tolist() == [0.0, 0.05, 0.0]
    assert partition.background[0, 0, 0] == pytest.approx(0.95)


def test_maps_are_scaled_down_to_what_is_left() -> None:
    csf = TissueMap("CSF", Volume(np.full(GRID.shape, 0.75), np.eye(4)))
    partition = partition_voxels(Phantom(GRID, tissue_maps=(gm_map(0.75), csf)))
    assert partition.tissue_fractions[0, 0, 0].tolist() == [0.0, 0.5, 0.5]
    assert partition.background[0, 0, 0] == 0.0


def test_crossing_bundles_share_a_voxel() -> None:
    across = BundleGeometry(np.array([[2.0, -1, 2], [2.0, 7, 2]]), radius=0.0, seed=0)
    phantom = Phantom(
        GRID, bundles=(dense_bundle(2.0), Bundle(across, streamline_area=10.0))
    )
    partition = partition_voxels(phantom)
    assert partition.bundle_fractions[2, 2, 2].tolist() == [0.5, 0.5]
    assert partition.tissue_fractions[2, 2, 2, 0] == pytest.approx(1.0)


def test_geometry_bundle_core_is_full_with_the_default_area() -> None:
    geometry = BundleGeometry(
        np.array([[-1.0, 10, 10], [21.0, 10, 10]]),
        num_streamlines=20000,
        radius=4.0,
        seed=0,
    )
    grid = Grid((20, 20, 20), np.eye(4))
    partition = partition_voxels(Phantom(grid, bundles=(Bundle(geometry),)))
    core = partition.bundle_fractions[5:15, 9:12, 9:12, 0]
    assert core.min() > 0.9


def test_a_geometry_bundle_fills_its_tube_without_stripes() -> None:
    # A straight tube at 30 degrees to x, radius 3 mm, through 1.5 mm voxels:
    # counting streamline length would leave dips of about 10% inside it.
    voxel, radius, length = 1.5, 3.0, 40.0
    affine = np.diag([voxel, voxel, voxel, 1.0])
    affine[:3, 3] = -19.5 * voxel
    grid = Grid((40, 40, 40), affine)
    axis = np.array([np.cos(np.pi / 6), np.sin(np.pi / 6), 0.0])
    ends = np.array([-axis * length / 2, axis * length / 2])
    geometry = BundleGeometry(ends, num_streamlines=1000, radius=radius, seed=0)
    fill = partition_voxels(Phantom(grid, bundles=(Bundle(geometry),)))
    fill_map = fill.bundle_fractions[..., 0]
    # The voxels whose 8 corners are all inside the tube, away from its ends.
    centres = np.indices(grid.shape).reshape(3, -1).T * voxel + affine[:3, 3]
    corners = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)])
    points = centres[:, None] + corners[None] * voxel / 2
    along = points @ axis
    across = np.linalg.norm(points - along[..., None] * axis, axis=2)
    inside = ((across < radius - 1e-6) & (np.abs(along) < length / 2)).all(axis=1)
    interior = fill_map.ravel()[inside]
    assert inside.sum() > 50
    assert np.allclose(interior, 1.0, atol=1e-6)
    assert interior.std() < 1e-6
    # Partly filled voxels lie on the tube's surface or at its ends only.
    partial = ((fill_map > 0) & (fill_map < 1)).ravel()
    centre_along = centres @ axis
    centre_across = np.linalg.norm(centres - centre_along[:, None] * axis, axis=1)
    half_diagonal = np.sqrt(3) * voxel / 2
    at_side = np.abs(centre_across - radius) <= half_diagonal
    at_end = np.abs(np.abs(centre_along) - length / 2) <= half_diagonal
    assert np.all(at_side[partial] | at_end[partial])
    volume = fill_map.sum() * voxel**3
    assert volume == pytest.approx(np.pi * radius**2 * length, rel=0.02)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_a_tube_ending_at_a_voxel_centre_fills_half_of_that_voxel(axis: int) -> None:
    # The ends at voxel centres 2 and 8; the radius covers the voxels across.
    ends = np.full((2, 3), 5.0)
    ends[:, axis] = [2.0, 8.0]
    fill, _ = tube_fill(ends, 3.0, Grid((11, 11, 11), np.eye(4)))
    line = np.moveaxis(fill.reshape(11, 11, 11), axis, 0)[:, 5, 5]
    np.testing.assert_allclose(line, [0, 0, 0.5, 1, 1, 1, 1, 1, 0.5, 0, 0])


def test_a_tube_follows_a_grid_with_another_affine() -> None:
    # Voxels of 2 by 1 by 1 mm, the axes swapped and flipped: a tube along y
    # still fills its core.
    affine = np.array(
        [
            [0.0, -1.0, 0.0, 10.0],
            [2.0, 0.0, 0.0, -10.0],
            [0.0, 0.0, 1.0, -10.0],
            [0, 0, 0, 1],
        ]
    )
    grid = Grid((10, 20, 20), affine)
    geometry = BundleGeometry(
        np.array([[0.0, -8.0, 0.0], [0.0, 8.0, 0.0]]), radius=2.0, seed=0
    )
    fill = partition_voxels(Phantom(grid, bundles=(Bundle(geometry),)))
    centre = np.linalg.solve(affine, [0.0, 0.0, 0.0, 1.0])[:3]
    i, j, k = np.round(centre).astype(int)
    assert fill.bundle_fractions[i, j, k, 0] == pytest.approx(1.0)
    volume = fill.bundle_fractions.sum() * 2.0
    assert volume == pytest.approx(np.pi * 4 * 16, rel=0.05)


def test_tractogram_bundle_fill_ignores_an_outlier_voxel() -> None:
    # Ten lines along x through 40 voxels (10 mm each), and 30 short lines
    # piled into one voxel: the dense voxels are set by the 95th percentile.
    grid = Grid((40, 6, 6), np.eye(4))
    long_lines = [np.linspace([-1.0, 2, 2], [41.0, 2, 2], 85)] * 10
    short = [np.linspace([20.0, 1.8, 2], [20.0, 2.2, 2], 5)] * 30
    lines = long_lines + short
    points = np.concatenate(lines)
    offsets = np.cumsum([0] + [len(line) for line in lines])
    tractogram = Tractogram(points, offsets)
    for fill in (0.7, 0.25):
        bundle = Bundle(tractogram, densest_fill=fill)
        fractions = partition_voxels(Phantom(grid, bundles=(bundle,))).bundle_fractions
        assert fractions[5, 2, 2, 0] == pytest.approx(fill, rel=1e-2)
        assert fractions[20, 2, 2, 0] > fill
    with pytest.raises(ValueError, match="densest_fill"):
        Bundle(tractogram, densest_fill=0.0)


def test_phantom_with_maps_and_no_bundles_is_valid() -> None:
    partition = partition_voxels(Phantom(GRID, tissue_maps=(gm_map(1.0),)))
    assert partition.bundle_fractions.shape == (*GRID.shape, 0)
    assert len(partition.fibre_weight) == 0


def test_oriented_tissue_cannot_be_a_map_or_the_background() -> None:
    wm = TissueMap("WM", Volume(np.zeros(GRID.shape), np.eye(4)))
    with pytest.raises(ValueError, match="fibre direction"):
        Phantom(GRID, tissue_maps=(wm,))
    with pytest.raises(ValueError, match="fibre direction"):
        Phantom(GRID, background="WM")


def test_unknown_tissue_and_bad_map_values_are_rejected() -> None:
    with pytest.raises(ValueError, match="no tissue"):
        Phantom(GRID, bundles=(dense_bundle(2.0, tissue="muscle"),))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        Phantom(GRID, tissue_maps=(gm_map(1.5),))
    with pytest.raises(ValueError, match="NaN"):
        Phantom(GRID, tissue_maps=(gm_map(np.nan),))


@pytest.mark.parametrize("offset", [0.0, 0.3])
def test_a_thin_tube_fills_its_true_volume(offset: float) -> None:
    grid = Grid((21, 21, 21), np.diag([2.0, 2.0, 2.0, 1.0]))
    ends = np.array([[10.0, 20 + offset, 20.2], [30.0, 20 + offset, 20.0]])
    fill, _ = tube_fill(ends, 0.2, grid)
    length = float(np.linalg.norm(ends[1] - ends[0]))
    assert fill.sum() * 8.0 == pytest.approx(np.pi * 0.2**2 * length, rel=0.1)


def test_a_tube_too_thin_for_the_grid_and_an_empty_tube_are_refused() -> None:
    ends = np.array([[1.0, 3, 3], [5.0, 3, 3]])
    with pytest.raises(ValueError, match="too thin"):
        tube_fill(ends, 0.01, GRID)
    with pytest.raises(ValueError, match="radius 0"):
        Bundle(BundleGeometry(ends, radius=0.0))


def test_every_block_round_trips_through_the_file(tmp_path: Path) -> None:
    gm_path = tmp_path / "gm.nii.gz"
    save_volume(Volume(np.full(GRID.shape, 0.25, np.float32), np.eye(4)), gm_path)
    tck_path = tmp_path / "bundle.tck"
    line = BundleGeometry(np.array([[0.0, 1, 1], [5.0, 1, 1]]), radius=1.0, seed=2)
    save_tractogram(create_bundle(line), tck_path)
    phantom = Phantom(
        GRID,
        bundles=(
            Bundle(BundleGeometry(np.array([[0.0, 3, 3], [5.0, 4, 3]]), fan_end=0.5)),
            Bundle(load_tractogram(tck_path), path=tck_path, streamline_area=0.5),
        ),
        tissue_maps=(TissueMap("GM", load_volume(gm_path), gm_path),),
        background="CSF",
    )
    gtab = GradientTable(np.array([0.0, 1000.0]), np.array([[0, 0, 0], [0, 0, 1.0]]))
    protocol = Protocol(gtab, 15.0, 30.0, [80.0, 100.0], 3000.0)
    path = tmp_path / "scene.phantom.json"
    write_phantom(phantom, path, protocol=protocol, viewer={"colours": [1, 2]})
    read = read_phantom(path)
    write_phantom(
        read.phantom,
        tmp_path / "again.phantom.json",
        protocol=read.protocol,
        viewer=read.viewer,
    )

    first = json.loads(path.read_text())
    second = json.loads((tmp_path / "again.phantom.json").read_text())
    assert first == second
    # The writer drew a seed for the bundle without one, so reading is repeatable.
    assert isinstance(first["bundles"][0]["geometry"]["seed"], int)
    assert read.protocol is not None and read.protocol.te_ms == (80.0, 100.0)
    assert read.viewer == {"colours": [1, 2]}
    assert read.phantom.background == "CSF"
    np.testing.assert_allclose(read.phantom.tissue_maps[0].volume.data, 0.25)


def test_densest_fill_round_trips_and_old_files_get_the_default(
    tmp_path: Path,
) -> None:
    tck_path = tmp_path / "bundle.tck"
    line = BundleGeometry(np.array([[0.0, 1, 1], [5.0, 1, 1]]), radius=1.0, seed=2)
    save_tractogram(create_bundle(line), tck_path)
    bundle = Bundle(load_tractogram(tck_path), path=tck_path, densest_fill=0.4)
    path = tmp_path / "scene.phantom.json"
    write_phantom(Phantom(GRID, bundles=(bundle,)), path)
    assert read_phantom(path).phantom.bundles[0].densest_fill == 0.4

    data = json.loads(path.read_text())
    del data["bundles"][0]["densest_fill"]
    path.write_text(json.dumps(data))
    assert read_phantom(path).phantom.bundles[0].densest_fill == Bundle.densest_fill


def test_newer_version_and_unknown_model_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "scene.phantom.json"
    write_phantom(Phantom(GRID), path)
    data = json.loads(path.read_text())
    path.write_text(json.dumps(data | {"version": 2}))
    with pytest.raises(ValueError, match="Update Larmor"):
        read_phantom(path)
    data["tissues"][2]["compartments"][0]["model"] = "bingham"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Unknown compartment model"):
        read_phantom(path)


def test_new_compartments_round_trip_and_old_files_without_kappa_load(
    tmp_path: Path,
) -> None:
    relaxation = {"t1_ms": 800.0, "t2_ms": 70.0, "t2star_ms": 40.0}
    restricted = Tissue(
        "R",
        (
            Compartment(
                CompartmentModel.CYLINDER, 0.5, 2e-3, radius_um=2.0, **relaxation
            ),
            Compartment(CompartmentModel.DOT, 0.5, 0.0, **relaxation),
        ),
    )
    phantom = Phantom(GRID, tissues=(noddi_tissue(), restricted, free_water_tissue()))
    path = tmp_path / "new.phantom.json"
    write_phantom(phantom, path)
    assert read_phantom(path).phantom.tissues == phantom.tissues

    data = json.loads(path.read_text())
    for tissue in data["tissues"]:
        for compartment in tissue["compartments"]:
            if compartment["model"] != "watson_stick":
                del compartment["kappa"]
    path.write_text(json.dumps(data))
    assert read_phantom(path).phantom.tissues == phantom.tissues


@pytest.mark.parametrize(
    ("model", "radius", "allowed"),
    [
        (CompartmentModel.DOT, 0.0, True),
        (CompartmentModel.FREE_WATER, 0.0, True),
        (CompartmentModel.WATSON_STICK, 0.0, False),
        (CompartmentModel.CYLINDER, 2.0, False),
    ],
)
def test_only_unoriented_compartments_fill_the_background(
    model: CompartmentModel, radius: float, allowed: bool
) -> None:
    compartment = Compartment(
        model, 1.0, 3e-3, radius_um=radius, t1_ms=800, t2_ms=70, t2star_ms=40
    )
    tissues = (Tissue("T", (compartment,)),)
    if allowed:
        assert Phantom(GRID, tissues=tissues, background="T").background == "T"
    else:
        with pytest.raises(ValueError, match="needs a fibre direction"):
            Phantom(GRID, tissues=tissues, background="T")


def test_tractogram_bundle_without_a_path_cannot_be_written(tmp_path: Path) -> None:
    line = BundleGeometry(np.array([[0.0, 1, 1], [5.0, 1, 1]]), seed=2)
    phantom = Phantom(GRID, bundles=(Bundle(create_bundle(line)),))
    with pytest.raises(ValueError, match="without a path"):
        write_phantom(phantom, tmp_path / "x.phantom.json")


def test_tissue_map_without_a_path_cannot_be_written(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="has no path"):
        write_phantom(Phantom(GRID, tissue_maps=(gm_map(0.5),)), tmp_path / "x.json")


def test_file_with_another_format_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "session.json"
    path.write_text(json.dumps({"format": "larmor-viewer-session", "version": 1}))
    with pytest.raises(ValueError, match="not a Larmor phantom file"):
        read_phantom(path)


def test_custom_tissues_and_a_gradient_echo_scan_round_trip(tmp_path: Path) -> None:
    soma = Compartment(
        CompartmentModel.SPHERE,
        0.4,
        2.5e-3,
        radius_um=9.0,
        t1_ms=1500.0,
        t2_ms=90.0,
        t2star_ms=40.0,
        proton_density=0.75,
    )
    zeppelin = Compartment(
        CompartmentModel.ZEPPELIN,
        0.6,
        2e-3,
        perpendicular_diffusivity=0.3e-3,
        t1_ms=900.0,
        t2_ms=60.0,
        t2star_ms=35.0,
    )
    tissues = (Tissue("custom", (soma, zeppelin)), csf_tissue(t2star_ms=300.0))
    gtab = GradientTable(np.array([0.0, 2000.0]), np.array([[0, 0, 0], [0, 1.0, 0]]))
    protocol = Protocol(
        gtab, 10.0, 20.0, [35.0, 50.0], 2500.0, PulseSequence.GRADIENT_ECHO
    )
    path = tmp_path / "custom.phantom.json"
    write_phantom(Phantom(GRID, tissues=tissues), path, protocol=protocol)
    read = read_phantom(path)
    assert read.phantom.tissues == tissues
    assert read.protocol is not None
    assert read.protocol.to_dict() == protocol.to_dict()


def test_moved_folder_is_found_through_the_relative_paths(tmp_path: Path) -> None:
    first = tmp_path / "first"
    first.mkdir()
    save_volume(
        Volume(np.full(GRID.shape, 0.5, np.float32), np.eye(4)), first / "gm.nii"
    )
    phantom = Phantom(
        GRID,
        tissue_maps=(TissueMap("GM", load_volume(first / "gm.nii"), first / "gm.nii"),),
    )
    write_phantom(phantom, first / "scene.phantom.json")
    moved = tmp_path / "moved"
    first.rename(moved)
    read = read_phantom(moved / "scene.phantom.json")
    assert read.phantom.tissue_maps[0].path == moved / "gm.nii"


@pytest.mark.parametrize(
    "affine",
    [
        np.eye(4),
        np.array([[1, -1, 0, 3.0], [1, 1, 0, -2.0], [0, 0, 1.5, 1.0], [0, 0, 0, 1.0]]),
    ],
    ids=["straight", "oblique_tie"],
)
@pytest.mark.parametrize(
    ("flips", "order"),
    [((0,), (0, 1, 2)), ((1, 2), (0, 1, 2)), ((), (1, 0, 2)), ((0,), (2, 1, 0))],
)
def test_a_tissue_map_in_another_orientation_keeps_its_world_positions(
    affine: np.ndarray,
    flips: tuple[int, ...],
    order: tuple[int, int, int],
    caplog: pytest.LogCaptureFixture,
) -> None:
    grid = Grid((6, 6, 4) if order == (1, 0, 2) else (6, 8, 6), affine)
    values = np.random.default_rng(0).random(grid.shape, dtype=np.float32)
    stored = Volume(values, grid.affine.copy())
    for axis in flips:
        stored.flip_axis(axis)
    stored.permute_axes(order)
    expected = Volume(values, grid.affine.copy())

    def fractions(volume: Volume) -> np.ndarray:
        phantom = Phantom(grid, tissue_maps=(TissueMap("GM", volume),))
        result: np.ndarray = partition_voxels(phantom, seed=0).tissue_fractions
        return result

    with caplog.at_level(logging.WARNING, logger="larmor"):
        found = fractions(stored)
        assert "reorienting the map" in caplog.text
        caplog.clear()
        wanted = fractions(expected)
        assert not caplog.text
    np.testing.assert_array_equal(found, wanted)


def test_rings_by_default_stand_for_three_times_their_median_section() -> None:
    grid = Grid((20, 20, 20), np.eye(4))
    ends = np.array([[-1.0, 10, 10], [21.0, 10, 10]])
    rings = np.array([[0.0, 2.0, 2.0, 0.0], [1.0, 6.0, 6.0, 0.0]])
    wide = BundleGeometry(ends, num_streamlines=2000, seed=0, rings=rings)
    area = RINGS_FULLNESS * wide.median_section() / 2000
    drawn, given, other_fill = (
        partition_voxels(Phantom(grid, bundles=(bundle,))).bundle_fractions
        for bundle in (
            Bundle(wide),
            Bundle(create_bundle(wide), streamline_area=area),
            Bundle(wide, densest_fill=0.2),
        )
    )
    np.testing.assert_array_equal(drawn, given)
    np.testing.assert_array_equal(drawn, other_fill)
    # The same streamlines spread over a wider section thin out.
    assert drawn[1].sum() / 2.36**2 > drawn[18].sum() / 5.45**2


@pytest.mark.parametrize("count", [500, 5000])
def test_constant_rings_fill_nearly_what_a_tube_of_that_size_does(count: int) -> None:
    grid = Grid((24, 21, 21), np.eye(4))
    ends = np.array([[-1.0, 10, 10], [25.0, 10, 10]])
    rings = np.array([[0.0, 3.0, 3.0, 0.0], [1.0, 3.0, 3.0, 0.0]])
    tube, ringed = (
        partition_voxels(
            Phantom(
                grid,
                bundles=(
                    Bundle(
                        BundleGeometry(
                            ends, radius=3.0, num_streamlines=count, seed=1, rings=r
                        )
                    ),
                ),
            )
        ).bundle_fractions
        for r in (None, rings)
    )
    assert ringed.sum() == pytest.approx(tube.sum(), rel=0.12)
    assert ringed[tube >= 1].mean() > 0.97


def test_a_fan_fills_like_a_tractogram_of_its_streamlines_by_densest_fill() -> None:
    grid = Grid((20, 20, 20), np.eye(4))
    ends = np.array([[-1.0, 10, 10], [21.0, 10, 10]])
    wide = BundleGeometry(ends, num_streamlines=2000, seed=0, radius=2.0, fan_end=2.0)
    for fill in (0.7, 0.35):
        drawn, given = (
            partition_voxels(Phantom(grid, bundles=(Bundle(s, densest_fill=fill),)))
            for s in (wide, create_bundle(wide))
        )
        np.testing.assert_array_equal(drawn.bundle_fractions, given.bundle_fractions)
        np.testing.assert_array_equal(drawn.fibre_weight, given.fibre_weight)
        reached = drawn.bundle_fractions[drawn.bundle_fractions > 0]
        assert np.percentile(reached, 95) == pytest.approx(fill, rel=1e-6)
        # The same streamlines spread over a wider section thin out.
        fractions = drawn.bundle_fractions[..., 0]
        assert fractions[1, 10, 10] > 2 * fractions[18, 10, 10]


def test_the_fold_guard_changes_a_fill_only_where_it_follows_the_streamlines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    affine[:3, 3] = [-30.0, -40.0, -16.0]
    grid = Grid((35, 35, 17), affine)
    corner = np.array([[-25.0, -15, 0], [15, -15, 0], [15, 25, 0]])
    geometry = BundleGeometry(corner, num_streamlines=2000, radius=10.0, seed=1)
    tube, by_length = Bundle(geometry), Bundle(geometry, streamline_area=0.15)

    def fill(bundle: Bundle) -> np.ndarray:
        partition = partition_voxels(Phantom(grid, bundles=(bundle,)))
        result: np.ndarray = partition.bundle_fractions
        return result

    guarded = fill(tube), fill(by_length)
    monkeypatch.setattr(bundles, "FOLD_LIMIT", 1e6)
    np.testing.assert_array_equal(guarded[0], fill(tube))
    assert not np.array_equal(guarded[1], fill(by_length))


def test_rings_are_saved_and_read_back_and_need_no_radius(tmp_path: Path) -> None:
    rings = np.array(
        [[0.0, 1.0, 1.0, 0.0], [0.5, 2.0, 1.0, 30.0], [1.0, 1.0, 1.0, 90.0]]
    )
    ends = np.array([[-1.0, 3, 3], [7.0, 3, 3]])
    written = Phantom(
        GRID, bundles=(Bundle(BundleGeometry(ends, radius=0.0, seed=0, rings=rings)),)
    )
    path = tmp_path / "rings.phantom.json"
    write_phantom(written, path)
    read = read_phantom(path).phantom
    source = read.bundles[0].source
    assert isinstance(source, BundleGeometry) and source.rings is not None
    np.testing.assert_array_equal(source.rings, rings)
    np.testing.assert_array_equal(
        partition_voxels(read).bundle_fractions,
        partition_voxels(written).bundle_fractions,
    )


def test_a_tangled_bundle_of_one_radius_fills_the_same_tube() -> None:
    grid = Grid((20, 20, 20), np.eye(4))
    ends = np.array([[-1.0, 10, 10], [21.0, 10, 10]])

    def fill(tangle_mm: float) -> np.ndarray:
        geometry = BundleGeometry(
            ends, num_streamlines=500, radius=3.0, seed=0, tangle_mm=tangle_mm
        )
        assert geometry.is_tube
        partition = partition_voxels(Phantom(grid, bundles=(Bundle(geometry),)))
        result: np.ndarray = partition.bundle_fractions
        return result

    np.testing.assert_array_equal(fill(5.0), fill(0.0))


def test_a_tangle_is_saved_and_read_back(tmp_path: Path) -> None:
    ends = np.array([[-1.0, 3, 3], [7.0, 3, 3]])
    geometry = BundleGeometry(ends, radius=1.0, seed=0, tangle_mm=4.0)
    written = Phantom(GRID, bundles=(Bundle(geometry),))
    path = tmp_path / "tangle.phantom.json"
    write_phantom(written, path)
    source = read_phantom(path).phantom.bundles[0].source
    assert isinstance(source, BundleGeometry) and source.tangle_mm == 4.0
    np.testing.assert_array_equal(
        create_bundle(source).points, create_bundle(geometry).points
    )


def test_a_fan_of_radius_0_fills_the_voxels_of_its_centreline() -> None:
    ends = np.array([[-1.0, 3, 3], [7.0, 3, 3]])
    fan = BundleGeometry(ends, radius=0.0, fan_end=1.0, num_streamlines=10, seed=0)
    fractions = partition_voxels(Phantom(GRID, bundles=(Bundle(fan),))).bundle_fractions
    np.testing.assert_allclose(fractions[:, 3, 3, 0], 0.7, rtol=1e-3)
    assert fractions.sum() == pytest.approx(6 * 0.7, rel=1e-3)


def test_a_fans_densest_fill_is_saved_and_read_back(tmp_path: Path) -> None:
    fan = BundleGeometry(np.array([[-1.0, 3, 3], [7.0, 3, 3]]), fan_end=1.0, seed=0)
    path = tmp_path / "fan.phantom.json"
    written = Phantom(GRID, bundles=(Bundle(fan, densest_fill=0.4),))
    write_phantom(written, path)
    read = read_phantom(path).phantom
    assert read.bundles[0].densest_fill == 0.4
    np.testing.assert_array_equal(
        partition_voxels(read).bundle_fractions,
        partition_voxels(written).bundle_fractions,
    )
