import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import erf

from larmor import _core
from larmor.simulation import (
    Bundle,
    BundleGeometry,
    Compartment,
    CompartmentModel,
    Phantom,
    Protocol,
    Tissue,
    csf_tissue,
    free_water_tissue,
    noddi_tissue,
    sandi_tissue,
    simulate,
    standard_model_tissue,
)
from larmor.types import GradientTable, Grid

PRESETS = [
    standard_model_tissue(),
    standard_model_tissue(free_water_fraction=0.1),
    sandi_tissue(),
    csf_tissue(),
    noddi_tissue(),
    free_water_tissue(),
]


def closed_form_s0(tissue: Tissue, te: float, tr: float) -> float:
    return float(
        sum(
            c.fraction
            * c.proton_density
            * (1 - np.exp(-tr / c.t1_ms))
            * np.exp(-te / c.t2_ms)
            for c in tissue.compartments
        )
    )


@pytest.mark.parametrize("tissue", PRESETS, ids=lambda t: t.name)
def test_preset_fractions_sum_to_one_and_t2star_is_below_t2(tissue: Tissue) -> None:
    assert sum(c.fraction for c in tissue.compartments) == pytest.approx(1.0)
    assert all(c.t2star_ms < c.t2_ms for c in tissue.compartments)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"diffusivity": float("nan")}, "finite"),
        ({"diffusivity": -1e-3}, ">= 0"),
        ({"fraction": 1.5}, "at most 1"),
        ({"proton_density": 1.2}, "at most 1"),
        ({"t1_ms": 0.0}, "positive"),
        ({"t2star_ms": 120.0}, "must not exceed T2"),
        ({"model": CompartmentModel.SPHERE}, "radius"),
    ],
)
def test_compartment_rejects_values_out_of_range(
    changes: dict[str, object], message: str
) -> None:
    values: dict[str, object] = {
        "model": CompartmentModel.BALL,
        "fraction": 1.0,
        "diffusivity": 3e-3,
        "t1_ms": 1000.0,
        "t2_ms": 100.0,
        "t2star_ms": 50.0,
    } | changes
    with pytest.raises(ValueError, match=message):
        Compartment(**values)  # type: ignore[arg-type]


def test_tissue_without_compartments_is_rejected() -> None:
    with pytest.raises(ValueError, match="no compartments"):
        Tissue("empty", ())


def test_standard_model_with_one_t2_gives_each_compartment_the_tissue_t2star() -> None:
    tissue = standard_model_tissue(intra_t2_ms=60.0, extra_t2_ms=60.0, t2star_ms=40.0)
    for compartment in tissue.compartments:
        assert compartment.t2star_ms == pytest.approx(40.0, rel=1e-12)


def test_standard_model_shares_one_t2_prime_between_its_compartments() -> None:
    stick, zeppelin = standard_model_tissue().compartments
    mean_t2 = 0.7 * 80.0 + 0.3 * 45.0
    t2_prime = 1 / (1 / 46.55 - 1 / mean_t2)
    for c in (stick, zeppelin):
        assert 1 / c.t2star_ms - 1 / c.t2_ms == pytest.approx(1 / t2_prime)


@pytest.mark.parametrize(
    "tissue",
    [
        standard_model_tissue(free_water_fraction=0.1),
        noddi_tissue(free_water_fraction=0.1),
    ],
    ids=["standard_model", "noddi"],
)
def test_free_water_shares_the_tissue_t2_prime(tissue: Tissue) -> None:
    rates = [1 / c.t2star_ms - 1 / c.t2_ms for c in tissue.compartments]
    assert tissue.compartments[-1].model == CompartmentModel.FREE_WATER
    np.testing.assert_allclose(rates, rates[0], rtol=1e-12)


@pytest.mark.parametrize("t2star_ms", [0.0, 80.0])
def test_standard_model_rejects_t2star_out_of_range(t2star_ms: float) -> None:
    with pytest.raises(ValueError, match="t2star_ms"):
        standard_model_tissue(t2star_ms=t2star_ms)


def test_fractions_that_do_not_sum_to_one_are_rejected() -> None:
    ball = Compartment(
        CompartmentModel.BALL, 0.6, 3e-3, t1_ms=1000, t2_ms=100, t2star_ms=50
    )
    with pytest.raises(ValueError, match="sum to"):
        Tissue("bad", (ball,))


@pytest.mark.parametrize("tissue", PRESETS, ids=lambda t: t.name)
def test_full_voxel_of_a_preset_gives_the_closed_form_s0(tissue: Tissue) -> None:
    gtab = GradientTable(np.zeros(1), np.zeros((1, 3)))
    protocol = Protocol(gtab, 15.0, 30.0, te_ms=[60.0, 90.0], tr_ms=2000.0)
    grid = Grid((3, 3, 3), np.eye(4))
    if tissue.is_oriented:
        # A straight bundle through the centre voxel, dense enough to fill it.
        line = BundleGeometry(np.array([[-1.0, 1, 1], [3.0, 1, 1]]), radius=0.0)
        bundles = (Bundle(line, tissue=tissue.name, streamline_area=10.0),)
        phantom = Phantom(grid, tissues=(tissue,), bundles=bundles)
    else:
        phantom = Phantom(grid, tissues=(tissue,), background=tissue.name)
    result = simulate(phantom, protocol)
    for echo, te in zip(result.noise_free, (60.0, 90.0), strict=True):
        assert echo.data[1, 1, 1, 0] == pytest.approx(
            closed_form_s0(tissue, te, 2000.0), rel=1e-6
        )


def test_sandi_signal_is_the_sum_of_its_compartments() -> None:
    tissue = sandi_tissue()
    bvals = np.array([0.0, 1000.0, 3000.0])
    bvecs = np.array([[0, 0, 0], [1.0, 0, 0], [0, 1.0, 0]])
    protocol = Protocol(GradientTable(bvals, bvecs), 15.0, 30.0, 80.0, 3000.0)
    phantom = Phantom(
        Grid((1, 1, 1), np.eye(4)), tissues=(tissue,), background=tissue.name
    )
    signal = simulate(phantom, protocol).noise_free[0].data[0, 0, 0]
    soma, neurite, extra = tissue.compartments
    weight = (
        0.8 * (1 - np.exp(-3000 / tissue.compartments[0].t1_ms)) * np.exp(-80 / 88.0)
    )
    x = bvals[1:] * neurite.diffusivity
    powder = np.concatenate(
        [[1.0], np.sqrt(np.pi) / (2 * np.sqrt(x)) * erf(np.sqrt(x))]
    )
    # The soma's reference value at b = 3000 is 0.7727 (R = 6 um, D = 3 um^2/ms).
    sphere = np.array([1.0, 0.9176, 0.7727])
    expected = weight * (
        soma.fraction * sphere
        + neurite.fraction * powder
        + extra.fraction * np.exp(-bvals * extra.diffusivity)
    )
    np.testing.assert_allclose(signal, expected, rtol=2e-4)


@pytest.mark.parametrize("kappa", [0.0, 0.3, 1.0, 1.01, 8.0, 64.0])
def test_noddi_couples_the_extra_neurite_tensor_to_density_and_dispersion(
    kappa: float,
) -> None:
    tissue = noddi_tissue(intra_fraction=0.6, free_water_fraction=0.1, kappa=kappa)
    sticks, zeppelin, water = tissue.compartments
    assert (sticks.model, zeppelin.model, water.model) == (
        CompartmentModel.WATSON_STICK,
        CompartmentModel.ZEPPELIN,
        CompartmentModel.FREE_WATER,
    )
    # Zhang et al. 2012, Eq. 1: (1 - v_iso) v_ic, (1 - v_iso) (1 - v_ic), v_iso.
    assert [c.fraction for c in tissue.compartments] == pytest.approx(
        [0.9 * 0.6, 0.9 * 0.4, 0.1]
    )
    assert sticks.kappa == kappa
    # Eq. 7 by direct integration, then Eqs. 5 and 6.
    top = quad(lambda u: u * u * np.exp(kappa * (u * u - 1)), 0, 1)[0]
    bottom = quad(lambda u: np.exp(kappa * (u * u - 1)), 0, 1)[0]
    tau = top / bottom
    d = 1.7e-3
    assert zeppelin.diffusivity == pytest.approx(d - d * 0.6 * (1 - tau), rel=1e-10)
    assert zeppelin.perpendicular_diffusivity == pytest.approx(
        d - d * 0.6 * (1 + tau) / 2, rel=1e-10
    )


def test_noddi_signal_in_a_bundle_is_the_sum_of_its_compartments() -> None:
    tissue = noddi_tissue(free_water_fraction=0.0, kappa=2.0)
    bvals = np.array([0.0, 1000.0, 2000.0, 3000.0])
    bvecs = np.array([[0, 0, 0], [1.0, 0, 0], [0.6, 0.8, 0], [0, 0, 1.0]])
    protocol = Protocol(GradientTable(bvals, bvecs), 15.0, 30.0, 80.0, 3000.0)
    line = BundleGeometry(np.array([[-1.0, 0, 0], [1.0, 0, 0]]), radius=0.0)
    bundle = Bundle(line, tissue=tissue.name, streamline_area=10.0)
    phantom = Phantom(Grid((1, 1, 1), np.eye(4)), tissues=(tissue,), bundles=(bundle,))
    signal = simulate(phantom, protocol).noise_free[0].data[0, 0, 0]
    cosines = np.ascontiguousarray(bvecs[:, 0])
    expected = sum(
        c.fraction
        * c.proton_density
        * (1 - np.exp(-3000 / c.t1_ms))
        * np.exp(-80 / c.t2_ms)
        * _core.compartment_attenuation(
            np.array([c.model.code]),
            np.array([c.engine_parameters]),
            bvals,
            cosines,
            15.0,
            30.0,
        )[0]
        for c in tissue.compartments
    )
    np.testing.assert_allclose(signal, expected, rtol=1e-6)
