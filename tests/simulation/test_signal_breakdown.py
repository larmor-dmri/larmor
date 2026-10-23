import math
from dataclasses import replace

import numpy as np
import pytest

from larmor.simulation import (
    Bundle,
    BundleGeometry,
    Compartment,
    CompartmentModel,
    Phantom,
    Protocol,
    PulseSequence,
    SimulatedDWI,
    Tissue,
    TissueMap,
    create_gradient_table,
    noddi_tissue,
    signal_breakdown,
    simulate,
)
from larmor.types import Grid, Volume

GRID = Grid((16, 16, 16), np.eye(4))
GTAB = create_gradient_table([(1000, 6), (3000, 6)], num_b0=1, seed=0)
# Voxels of the phantom below, by what fills them.
VOXELS = {
    "one tissue": (4, 8, 8),  # inside the first bundle only
    "partial bundle over GM": (4, 11, 8),  # the first bundle's edge, GM below
    "crossing": (8, 8, 8),
    "CSF": (1, 1, 1),  # the background only
}


def protocol(sequence: PulseSequence, te: float = 70.0) -> Protocol:
    return Protocol(GTAB, 15.0, 30.0, [te, te + 20.0], 3000.0, sequence)


@pytest.fixture(scope="module")
def phantom() -> Phantom:
    along_x = BundleGeometry(
        np.array([[1.0, 8, 8], [8, 8, 8], [15, 8, 8]]),
        num_streamlines=300,
        radius=2.8,
        seed=1,
    )
    along_y = BundleGeometry(
        np.array([[8.0, 1, 8], [8, 8, 8], [8, 15, 8]]),
        num_streamlines=300,
        radius=2.8,
        seed=2,
    )
    gm = np.zeros(GRID.shape)
    gm[:, :, 6:11] = 0.4
    return Phantom(
        GRID,
        bundles=(Bundle(along_x), Bundle(along_y)),
        tissue_maps=(TissueMap("GM", Volume(gm, np.eye(4))),),
        background="CSF",
    )


@pytest.fixture(scope="module", params=list(PulseSequence))
def simulated(request: pytest.FixtureRequest, phantom: Phantom) -> SimulatedDWI:
    return simulate(phantom, protocol(request.param), seed=0)


@pytest.mark.parametrize("voxel", VOXELS.values(), ids=VOXELS.keys())
def test_final_signal_is_simulates_noise_free_signal(
    simulated: SimulatedDWI, voxel: tuple[int, int, int]
) -> None:
    assert simulated.protocol is not None and simulated.phantom is not None
    for echo in range(2):
        expected = simulated.noise_free[echo].data[voxel]
        found = [
            signal_breakdown(
                simulated.phantom,
                simulated.protocol,
                simulated.partition,
                voxel,
                n,
                echo=echo,
            ).signal
            for n in range(len(GTAB.bvals))
        ]
        np.testing.assert_allclose(found, expected, rtol=1e-6, atol=1e-9)


def test_new_compartments_are_named_and_give_simulates_signal(
    phantom: Phantom,
) -> None:
    relaxation = {"t1_ms": 900.0, "t2_ms": 80.0, "t2star_ms": 50.0}
    axons = Tissue(
        "AX",
        (
            Compartment(
                CompartmentModel.CYLINDER, 0.7, 2e-3, radius_um=3.0, **relaxation
            ),
            Compartment(CompartmentModel.DOT, 0.3, 0.0, **relaxation),
        ),
    )
    still = Tissue(
        "BG",
        (
            Compartment(CompartmentModel.DOT, 0.2, 0.0, **relaxation),
            Compartment(CompartmentModel.FREE_WATER, 0.8, 3e-3, **relaxation),
        ),
    )
    first, second = phantom.bundles
    mixed = replace(
        phantom,
        tissues=(noddi_tissue(), axons, still, *phantom.tissues[1:]),
        bundles=(first, replace(second, tissue="AX")),
        background="BG",
    )
    scan = protocol(PulseSequence.SPIN_ECHO)
    simulated = simulate(mixed, scan, seed=0)
    labels = set()
    for voxel in (VOXELS["crossing"], VOXELS["CSF"]):
        found = []
        for n in range(len(GTAB.bvals)):
            breakdown = signal_breakdown(mixed, scan, simulated.partition, voxel, n)
            found.append(breakdown.signal)
            labels |= {c.label for c in breakdown.compartments}
        expected = simulated.noise_free[0].data[voxel]
        np.testing.assert_allclose(found, expected, rtol=1e-6, atol=1e-9)
    assert labels == {
        "WM neurites",
        "WM outside neurites",
        "WM free water",
        "AX axons",
        "AX still water",
        "BG still water",
        "BG free water",
    }


def test_voxels_hold_what_their_names_say(simulated: SimulatedDWI) -> None:
    assert simulated.protocol is not None and simulated.phantom is not None

    def shares(voxel: tuple[int, int, int]) -> set[tuple[str, int | None]]:
        breakdown = signal_breakdown(
            simulated.phantom,  # type: ignore[arg-type]
            simulated.protocol,  # type: ignore[arg-type]
            simulated.partition,
            voxel,
            1,
        )
        return {(c.tissue, c.bundle) for c in breakdown.compartments}

    assert shares(VOXELS["one tissue"]) == {("WM", 0)}
    assert {("WM", 0), ("GM", None)} <= shares(VOXELS["partial bundle over GM"])
    assert {("WM", 0), ("WM", 1)} <= shares(VOXELS["crossing"])
    assert shares(VOXELS["CSF"]) == {("CSF", None)}


def test_the_bundles_of_a_tissue_add_up_to_its_share(simulated: SimulatedDWI) -> None:
    assert simulated.protocol is not None and simulated.phantom is not None
    breakdown = signal_breakdown(
        simulated.phantom,
        simulated.protocol,
        simulated.partition,
        VOXELS["crossing"],
        1,
    )
    assert len(breakdown.bundles) == 2
    wm = dict(breakdown.tissue_fractions)["WM"]
    assert sum(b.share for b in breakdown.bundles) == pytest.approx(wm)


def test_each_factor_follows_its_formula(phantom: Phantom) -> None:
    scan = protocol(PulseSequence.GRADIENT_ECHO)
    result = simulate(phantom, scan, seed=0)
    voxel = VOXELS["partial bundle over GM"]
    breakdown = signal_breakdown(phantom, scan, result.partition, voxel, 3, echo=1)
    fractions = dict(breakdown.tissue_fractions)
    gm = phantom.tissue("GM")
    for c, settings in zip(
        [c for c in breakdown.compartments if c.tissue == "GM"],
        gm.compartments,
        strict=True,
    ):
        assert c.partial_volume == pytest.approx(fractions["GM"])
        assert c.fraction == settings.fraction
        assert c.proton_density == settings.proton_density
        assert c.t1_factor == pytest.approx(1 - math.exp(-3000.0 / settings.t1_ms))
        assert c.t2_factor == pytest.approx(math.exp(-90.0 / settings.t2star_ms))
    ball = next(c for c in breakdown.compartments if c.label == "GM outside cells")
    models = [c.model for c in gm.compartments]
    diffusivity = gm.compartments[models.index(CompartmentModel.BALL)]
    assert ball.attenuation == pytest.approx(
        math.exp(-breakdown.b_value * diffusivity.diffusivity)
    )
    stick = next(c for c in breakdown.compartments if c.label == "WM axons")
    weight = result.partition.bundle_fractions[voxel][0]
    assert stick.partial_volume == pytest.approx(weight)
    assert {c.label for c in breakdown.compartments} >= {
        "GM soma",
        "GM neurites",
        "WM outside axons",
    }


def factors(breakdown: object) -> list[dict[str, float]]:
    return [
        {
            name: getattr(c, name)
            for name in (
                "partial_volume",
                "fraction",
                "proton_density",
                "t1_factor",
                "t2_factor",
                "attenuation",
            )
        }
        for c in breakdown.compartments  # type: ignore[attr-defined]
    ]


def changed(a: object, b: object) -> set[str]:
    return {
        name
        for x, y in zip(factors(a), factors(b), strict=True)
        for name in x
        if x[name] != y[name]
    }


def test_the_echo_time_changes_only_the_t2_factor(phantom: Phantom) -> None:
    short, long = (
        protocol(PulseSequence.SPIN_ECHO, 60.0),
        protocol(PulseSequence.SPIN_ECHO, 100.0),
    )
    partition = simulate(phantom, short, seed=0).partition
    voxel = VOXELS["crossing"]
    a = signal_breakdown(phantom, short, partition, voxel, 4)
    b = signal_breakdown(phantom, long, partition, voxel, 4)
    assert changed(a, b) == {"t2_factor"}


def test_the_b_value_changes_only_the_attenuations(phantom: Phantom) -> None:
    scan = protocol(PulseSequence.SPIN_ECHO)
    partition = simulate(phantom, scan, seed=0).partition
    stronger = Protocol(
        replace(GTAB, bvals=GTAB.bvals * 2), 15.0, 30.0, [70.0, 90.0], 3000.0
    )
    voxel = VOXELS["partial bundle over GM"]
    a = signal_breakdown(phantom, scan, partition, voxel, 4)
    b = signal_breakdown(phantom, stronger, partition, voxel, 4)
    assert changed(a, b) == {"attenuation"}


def test_noise_gives_the_floor_and_the_mean_rician_magnitude(
    phantom: Phantom,
) -> None:
    scan = protocol(PulseSequence.SPIN_ECHO)
    result = simulate(phantom, scan, snr=10, seed=0)
    breakdown = signal_breakdown(
        phantom, scan, result.partition, VOXELS["crossing"], 7, snr=10
    )
    assert breakdown.sigma == pytest.approx(result.sigma)
    sigma = breakdown.sigma
    assert sigma is not None
    assert breakdown.noise_floor == pytest.approx(sigma * math.sqrt(math.pi / 2))
    rng = np.random.default_rng(0)
    noise = rng.normal(0, sigma, (2, 400_000))
    sampled = np.hypot(breakdown.signal + noise[0], noise[1]).mean()
    assert breakdown.expected_magnitude == pytest.approx(sampled, rel=2e-3)


def test_a_voxel_outside_the_grid_is_refused(phantom: Phantom) -> None:
    scan = protocol(PulseSequence.SPIN_ECHO)
    partition = simulate(phantom, scan, seed=0).partition
    with pytest.raises(ValueError, match="outside the grid"):
        signal_breakdown(phantom, scan, partition, (16, 0, 0), 0)
