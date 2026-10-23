import numpy as np
import pytest

from larmor.simulation import (
    Bundle,
    BundleGeometry,
    Phantom,
    Protocol,
    PulseSequence,
    TissueMap,
    create_gradient_table,
    partition_voxels,
    sandi_tissue,
    simulate,
    standard_model_tissue,
)
from larmor.simulation.instant import flatten
from larmor.types import GradientTable, Grid, Volume

GRID = Grid((5, 5, 5), np.eye(4))
GTAB = create_gradient_table([(1000, 12), (3000, 12)], num_b0=1, seed=0)
PROTOCOL = Protocol(GTAB, 15.0, 30.0, 80.0, 3000.0)


def bundle_along(direction: np.ndarray, tissue: str = "WM") -> Bundle:
    # A dense straight line through the centre voxel.
    centre = np.array([2.0, 2, 2])
    line = BundleGeometry(
        np.array([centre - 3 * direction, centre + 3 * direction]), radius=0.0
    )
    return Bundle(line, tissue=tissue, streamline_area=10.0)


def relaxation(
    pd: float, t1: float, decay: float, te: float = 80.0, tr: float = 3000.0
) -> float:
    return float(pd * (1 - np.exp(-tr / t1)) * np.exp(-te / decay))


def wm_signal(
    direction: np.ndarray, tissue: object, protocol: Protocol = PROTOCOL
) -> np.ndarray:
    b = protocol.gradients.bvals
    cos2 = (protocol.gradients.bvecs @ direction) ** 2
    signal = np.zeros(len(b))
    for c in tissue.compartments:  # type: ignore[attr-defined]
        decay = (
            c.t2star_ms if protocol.sequence == PulseSequence.GRADIENT_ECHO else c.t2_ms
        )
        w = c.fraction * relaxation(
            c.proton_density, c.t1_ms, decay, protocol.te_ms[0], protocol.tr_ms
        )
        if c.model == "stick":
            a = np.exp(-b * c.diffusivity * cos2)
        elif c.model == "zeppelin":
            perp = c.perpendicular_diffusivity
            a = np.exp(-b * (perp + (c.diffusivity - perp) * cos2))
        else:
            a = np.exp(-b * c.diffusivity)
        signal += w * a
    return signal


def test_full_voxel_in_a_straight_bundle_is_exactly_the_tissue_signal() -> None:
    # The tissue keeps its own fractions, the ball included, in a full voxel.
    tissue = standard_model_tissue(free_water_fraction=0.1)
    direction = np.array([1.0, 0.3, -0.2]) / np.linalg.norm([1.0, 0.3, -0.2])
    phantom = Phantom(GRID, tissues=(tissue,), bundles=(bundle_along(direction),))
    signal = simulate(phantom, PROTOCOL).noise_free[0].data[2, 2, 2]
    np.testing.assert_allclose(signal, wm_signal(direction, tissue), rtol=1e-5)


def test_unoriented_tissue_in_a_bundle_fills_the_bundle_core() -> None:
    phantom = Phantom(GRID, bundles=(bundle_along(np.array([1.0, 0, 0]), tissue="GM"),))
    signal = simulate(phantom, PROTOCOL).noise_free[0].data[2, 2, 2]
    full = Phantom(GRID, background="GM")
    np.testing.assert_allclose(
        signal, simulate(full, PROTOCOL).noise_free[0].data[2, 2, 2]
    )
    assert signal[0] > 0.25


def test_voxel_signal_is_the_fraction_weighted_sum_of_tissues() -> None:
    gm = np.zeros(GRID.shape)
    csf = np.zeros(GRID.shape)
    gm[0, 0, 0], csf[0, 0, 0] = 0.3, 0.2
    gm[4, 4, 4] = 0.05
    maps = (
        TissueMap("GM", Volume(gm, np.eye(4))),
        TissueMap("CSF", Volume(csf, np.eye(4))),
    )
    direction = np.array([0.0, 0.0, 1.0])
    phantom = Phantom(GRID, bundles=(bundle_along(direction),), tissue_maps=maps)
    data = simulate(phantom, PROTOCOL).noise_free[0].data
    pure_gm = (
        simulate(Phantom(GRID, background="GM"), PROTOCOL).noise_free[0].data[0, 0, 0]
    )
    pure_csf = (
        simulate(Phantom(GRID, background="CSF"), PROTOCOL).noise_free[0].data[0, 0, 0]
    )
    np.testing.assert_allclose(data[0, 0, 0], 0.3 * pure_gm + 0.2 * pure_csf, rtol=1e-5)
    np.testing.assert_allclose(data[4, 4, 4], 0.05 * pure_gm, rtol=1e-5)
    # With CSF as the background, the rest of that voxel is CSF.
    with_csf = Phantom(GRID, tissue_maps=maps, background="CSF")
    np.testing.assert_allclose(
        simulate(with_csf, PROTOCOL).noise_free[0].data[4, 4, 4],
        0.05 * pure_gm + 0.95 * pure_csf,
        rtol=1e-5,
    )


def line_through_centre(direction: np.ndarray, area: float) -> Bundle:
    # One streamline, so a voxel it crosses holds 1 mm of it and is `area` full.
    centre = np.array([2.0, 2, 2])
    line = BundleGeometry(
        np.array([centre - 3 * direction, centre + 3 * direction]),
        num_streamlines=1,
        radius=0.0,
    )
    return Bundle(line, streamline_area=area)


def test_partly_filled_bundle_voxel_keeps_the_tissue_proportions() -> None:
    # A tissue with a ball shows whether the oriented and unoriented parts get
    # the same occupancy.
    tissue = standard_model_tissue(free_water_fraction=0.1)
    x = np.array([1.0, 0, 0])
    phantom = Phantom(
        GRID,
        tissues=(tissue, sandi_tissue()),
        bundles=(line_through_centre(x, 0.4),),
        background="GM",
    )
    result = simulate(phantom, PROTOCOL)
    signal = result.noise_free[0].data[2, 2, 2]
    # The line holds 1 mm of streamline in the voxel, up to float32 rounding.
    fill = result.partition.bundle_fractions[2, 2, 2, 0]
    assert fill == pytest.approx(0.4, rel=1e-4)
    pure_gm = (
        simulate(
            Phantom(GRID, tissues=(tissue, sandi_tissue()), background="GM"), PROTOCOL
        )
        .noise_free[0]
        .data[2, 2, 2]
    )
    expected = fill * wm_signal(x, tissue) + (1 - fill) * pure_gm
    np.testing.assert_allclose(signal, expected, rtol=1e-5)


def test_crossing_voxel_is_half_of_each_direction() -> None:
    x, y = np.array([1.0, 0, 0]), np.array([0.0, 1, 0])
    bundles = (line_through_centre(x, 0.8), line_through_centre(y, 0.8))
    phantom = Phantom(GRID, bundles=bundles)
    signal = simulate(phantom, PROTOCOL).noise_free[0].data[2, 2, 2]
    tissue = phantom.tissue("WM")
    expected = 0.5 * wm_signal(x, tissue) + 0.5 * wm_signal(y, tissue)
    np.testing.assert_allclose(signal, expected, rtol=1e-5)


def test_seed_fixes_bundles_without_their_own_seed() -> None:
    geometry = BundleGeometry(np.array([[0.0, 1, 2], [4.0, 3, 2]]), radius=1.0)
    phantom = Phantom(GRID, bundles=(Bundle(geometry),))
    first = simulate(phantom, PROTOCOL, seed=5)
    second = simulate(phantom, PROTOCOL, seed=5)
    np.testing.assert_array_equal(first.noise_free[0].data, second.noise_free[0].data)
    (drawn,) = first.partition.streamlines
    np.testing.assert_array_equal(drawn.points, second.partition.streamlines[0].points)
    other = simulate(phantom, PROTOCOL, seed=6).partition.streamlines[0]
    assert not np.array_equal(drawn.points, other.points)


def test_gradient_echo_runs_with_the_predefined_tissues_and_uses_t2star() -> None:
    gradient_echo = Protocol(
        GTAB, 15.0, 30.0, [50.0, 70.0], 3000.0, PulseSequence.GRADIENT_ECHO
    )
    direction = np.array([0.0, 1.0, 0.0])
    phantom = Phantom(GRID, bundles=(bundle_along(direction),), background="CSF")
    result = simulate(phantom, gradient_echo)
    assert len(result.noise_free) == 2
    tissue = phantom.tissue("WM")
    for echo, te in zip(result.noise_free, (50.0, 70.0), strict=True):
        single = Protocol(GTAB, 15.0, 30.0, te, 3000.0, PulseSequence.GRADIENT_ECHO)
        np.testing.assert_allclose(
            echo.data[2, 2, 2], wm_signal(direction, tissue, single), rtol=1e-5
        )


def test_one_thread_and_many_give_the_same_signal() -> None:
    geometry = BundleGeometry(
        np.array([[0.0, 0, 0], [2.0, 4, 3], [4.0, 4, 4]]), radius=1.5, seed=0
    )
    phantom = Phantom(GRID, bundles=(Bundle(geometry),), background="CSF")
    one = simulate(phantom, PROTOCOL, snr=20, seed=3, n_jobs=1)
    many = simulate(phantom, PROTOCOL, snr=20, seed=3, n_jobs=4)
    np.testing.assert_array_equal(one.noise_free[0].data, many.noise_free[0].data)
    np.testing.assert_array_equal(one.echoes[0].data, many.echoes[0].data)


def test_raising_in_the_progress_callback_cancels_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("larmor.simulation.instant.CHUNK_VOXELS", 10)
    stages: list[tuple[int, str]] = []

    def progress(done: int, total: int, stage: str) -> None:
        stages.append((done, stage))
        if stage == "signal":
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        simulate(Phantom(GRID, background="CSF"), PROTOCOL, progress=progress)
    assert stages[-1] == (10, "signal")


def test_flat_arrays_of_a_small_phantom() -> None:
    gtab = GradientTable(np.array([0.0, 1000.0]), np.array([[0, 0, 0], [1.0, 0, 0]]))
    protocol = Protocol(gtab, 15.0, 30.0, 80.0, 3000.0)
    gm = np.zeros(GRID.shape)
    gm[0, 0, 0] = 0.4
    phantom = Phantom(
        GRID,
        tissues=(standard_model_tissue(), sandi_tissue()),
        bundles=(bundle_along(np.array([1.0, 0, 0])),),
        tissue_maps=(TissueMap("GM", Volume(gm, np.eye(4))),),
    )
    arrays = flatten(phantom, protocol, partition_voxels(phantom))
    assert arrays.models.tolist() == [0, 1, 4, 3, 2]
    assert arrays.tissue.tolist() == [0, 0, 1, 1, 1]
    np.testing.assert_allclose(arrays.fraction, [0.7, 0.3, 0.15, 0.5, 0.35])
    np.testing.assert_allclose(arrays.parameters[1], [1.7e-3, 0.5e-3, 0.0, 0.0])
    np.testing.assert_allclose(arrays.parameters[2], [3e-3, 0.0, 6.0, 0.0])
    assert arrays.relaxation[0, 0] == pytest.approx(relaxation(0.7, 832.0, 80.0))
    assert arrays.tissue_fractions[0].tolist() == [0.0, 0.4]
    centre = np.ravel_multi_index((2, 2, 2), GRID.shape)
    assert arrays.tissue_fractions[centre].tolist() == [1.0, 0.0]
    fibres = slice(arrays.offsets[centre], arrays.offsets[centre + 1])
    np.testing.assert_allclose(arrays.fibre_weight[fibres].sum(), 1.0)
    # float32 points may split the line's direction over neighbouring cells.
    np.testing.assert_allclose(
        np.abs(arrays.fibre_direction[fibres][:, 0]), 1.0, atol=1e-9
    )
