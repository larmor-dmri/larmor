from collections.abc import Callable

import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.special import erf, jnp_zeros, jv, jvp

from larmor import _core
from larmor.sd import AnisotropicRF, IsotropicRF
from larmor.simulation import (
    FREE_WATER_DIFFUSIVITY,
    CompartmentModel,
    create_gradient_table,
    free_water_tissue,
)

BVALS = np.array([0.0, 500.0, 1000.0, 3000.0, 10000.0])
# D = 3 um^2/ms, delta/Delta = 15/30 ms.
DELTA, BIG_DELTA = 15.0, 30.0


def attenuation(
    model: CompartmentModel,
    bvals: np.ndarray,
    cosines: np.ndarray | list[float],
    d: float,
    d_perp: float = 0.0,
    radius: float = 0.0,
    delta: float = DELTA,
    big_delta: float = BIG_DELTA,
    kappa: float = 0.0,
) -> np.ndarray:
    result: np.ndarray = _core.compartment_attenuation(
        np.array([model.code], dtype=np.int64),
        np.array([[d, d_perp, radius, kappa]]),
        np.ascontiguousarray(bvals, dtype=np.float64),
        np.ascontiguousarray(cosines, dtype=np.float64),
        delta,
        big_delta,
    )[0]
    return result


def test_python_model_codes_match_the_compiled_enum() -> None:
    # Free water is a ball to the engine.
    codes = {m.value: m.code for m in CompartmentModel}
    assert codes.pop("free_water") == codes["ball"]
    assert codes == _core.compartment_model_codes()


def test_unknown_model_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown compartment model"):
        CompartmentModel("bingham")


def test_stick_and_zeppelin_match_the_anisotropic_rf_without_kurtosis() -> None:
    rng = np.random.default_rng(0)
    bvecs = rng.standard_normal((5, 3))
    bvecs /= np.linalg.norm(bvecs, axis=1, keepdims=True)
    fibre = np.array([[0.0, 0.6, 0.8]])
    cosines = bvecs @ fibre[0]
    stick = AnisotropicRF([2e-3, 0.0, 0.0], 0.0).signal(BVALS, bvecs, fibre)[:, 0]
    zeppelin = AnisotropicRF([1.7e-3, 0.5e-3, 0.5e-3], 0.0).signal(BVALS, bvecs, fibre)
    np.testing.assert_allclose(
        attenuation(CompartmentModel.STICK, BVALS, cosines, 2e-3), stick, rtol=1e-12
    )
    np.testing.assert_allclose(
        attenuation(CompartmentModel.ZEPPELIN, BVALS, cosines, 1.7e-3, 0.5e-3),
        zeppelin[:, 0],
        rtol=1e-12,
    )


def test_ball_matches_the_isotropic_rf() -> None:
    expected = IsotropicRF(3e-3).signal(BVALS, np.zeros((5, 3)), np.zeros((1, 3)))
    np.testing.assert_allclose(
        attenuation(CompartmentModel.BALL, BVALS, np.zeros(5), 3e-3),
        expected[:, 0],
        rtol=1e-12,
    )


def test_powder_stick_is_its_closed_form_and_the_sphere_average_of_a_stick() -> None:
    d = 2e-3
    x = BVALS[1:] * d
    closed = np.sqrt(np.pi) / (2 * np.sqrt(x)) * erf(np.sqrt(x))
    powder = attenuation(CompartmentModel.POWDER_STICK, BVALS, np.zeros(5), d)
    np.testing.assert_allclose(powder[1:], closed, rtol=1e-12)
    assert powder[0] == 1.0
    # Average over evenly spread fibre directions: cosines uniform in [0, 1].
    cosines = (np.arange(20000) + 0.5) / 20000
    averaged = [
        np.mean(attenuation(CompartmentModel.STICK, np.full(20000, b), cosines, d))
        for b in BVALS
    ]
    np.testing.assert_allclose(powder, averaged, rtol=1e-6)


def test_powder_stick_equals_a_numerical_average_over_repelled_directions() -> None:
    directions = create_gradient_table([(1000, 300)], num_b0=0, seed=1).bvecs
    cosines = directions[:, 2]
    stick = [
        np.mean(
            attenuation(CompartmentModel.STICK, np.full(300, 3000.0), cosines, 2e-3)
        )
    ]
    powder = attenuation(CompartmentModel.POWDER_STICK, np.array([3000.0]), [0.0], 2e-3)
    np.testing.assert_allclose(powder, stick, rtol=5e-3)


def test_sphere_roots_satisfy_the_sum_rule() -> None:
    x = _core.sphere_roots()
    np.testing.assert_allclose(x[:4], [2.0816, 5.9404, 9.2058, 12.4044], atol=1e-4)
    assert np.sum(1 / (x**2 * (x**2 - 2))) == pytest.approx(0.1, rel=1e-6)


def test_closed_form_long_pulse_sum_matches_the_direct_sum() -> None:
    # The sphere sums the 2 delta / (a^2 D) part of its series in closed form.
    x = _core.sphere_roots()
    assert np.sum(1 / (x**4 * (x**2 - 2))) == pytest.approx(4 / 175, rel=1e-12)


# Independent references for the restricted compartments, written from the
# published formulas in SI units, with the roots found by scipy from each
# boundary condition. b = (gamma g delta)^2 (Delta - delta / 3) gives gamma g.
# The timings include pulses long compared with R^2 / D (16 ms for R = 4 um at
# D = 1 um^2/ms) and pulses nearly as long as their separation.
TIMINGS = [(15.0, 30.0), (5.0, 60.0), (40.0, 45.0)]
RADII = [2.0, 4.0, 8.0]
RESTRICTED_BVALS = np.array([1000.0, 3000.0, 10000.0])
Reference = Callable[[np.ndarray, float, float, float, float], np.ndarray]


def _gamma_g_squared(bvals: np.ndarray, delta: float, big_delta: float) -> np.ndarray:
    # (gamma g)^2 in 1/(m^2 s^2) from b in s/mm^2 and the timings in ms.
    small, big = delta * 1e-3, big_delta * 1e-3
    gamma_g_squared: np.ndarray = bvals * 1e6 / (small**2 * (big - small / 3))
    return gamma_g_squared


def _sphere_reference_roots(count: int) -> np.ndarray:
    # The roots of (x)^-1 J_3/2(x) = J_5/2(x), x = alpha_m r_s (Palombo et al.
    # 2020, after Eq. 8), bracketed on a fine grid.
    def equation(x: float) -> float:
        return float(jv(1.5, x) / x - jv(2.5, x))

    grid = np.arange(0.5, 4.0 * count, 0.01)
    values = jv(1.5, grid) / grid - jv(2.5, grid)
    brackets = np.flatnonzero(np.sign(values[:-1]) != np.sign(values[1:]))[:count]
    return np.array([brentq(equation, grid[i], grid[i + 1]) for i in brackets])


def _sphere_reference(
    bvals: np.ndarray, d: float, radius: float, delta: float, big_delta: float
) -> np.ndarray:
    # Palombo et al. 2020, Eq. 8:
    # A = exp{-2 (gamma g)^2 / D sum_m alpha_m^-4 / (alpha_m^2 r^2 - 2)
    #     x [2 delta - (2 + e^(-a D (Delta - delta)) - 2 e^(-a D delta)
    #        - 2 e^(-a D Delta) + e^(-a D (Delta + delta))) / (a D)]}, a = alpha_m^2.
    d_si, r_si = d * 1e-6, radius * 1e-6
    small, big = delta * 1e-3, big_delta * 1e-3
    alpha = _sphere_reference_roots(300) / r_si
    a = alpha**2
    bracket = 2 * small - (
        2
        + np.exp(-a * d_si * (big - small))
        - 2 * np.exp(-a * d_si * small)
        - 2 * np.exp(-a * d_si * big)
        + np.exp(-a * d_si * (big + small))
    ) / (a * d_si)
    series = np.sum(alpha**-4 / (a * r_si**2 - 2) * bracket)
    result: np.ndarray = np.exp(
        -2 * _gamma_g_squared(bvals, delta, big_delta) / d_si * series
    )
    return result


def _cylinder_reference(
    bvals: np.ndarray, d: float, radius: float, delta: float, big_delta: float
) -> np.ndarray:
    # De Santis et al. 2016, Eq. 3, with q = gamma delta g / (2 pi) and alpha_m R
    # the roots of J1'(x) = 0:
    # S_perp = exp(-8 pi^2 q^2 sum_m [2 D a delta - 2 + 2 e^(-D a delta)
    #     + 2 e^(-D a Delta) - e^(-D a (Delta - delta)) - e^(-D a (Delta + delta))]
    #     / [delta^2 D^2 alpha_m^6 (R^2 a - 1)]), a = alpha_m^2.
    d_si, r_si = d * 1e-6, radius * 1e-6
    small, big = delta * 1e-3, big_delta * 1e-3
    alpha = jnp_zeros(1, 300) / r_si
    a = alpha**2
    numerator = (
        2 * d_si * a * small
        - 2
        + 2 * np.exp(-d_si * a * small)
        + 2 * np.exp(-d_si * a * big)
        - np.exp(-d_si * a * (big - small))
        - np.exp(-d_si * a * (big + small))
    )
    series = np.sum(numerator / (small**2 * d_si**2 * alpha**6 * (r_si**2 * a - 1)))
    q_squared = _gamma_g_squared(bvals, delta, big_delta) * small**2 / (2 * np.pi) ** 2
    result: np.ndarray = np.exp(-8 * np.pi**2 * q_squared * series)
    return result


def _compare_with_reference(model: CompartmentModel, reference: Reference) -> None:
    # Every timing and radius at once, so that the test fails if any case does.
    for delta, big_delta in TIMINGS:
        for radius in RADII:
            found = attenuation(
                model,
                RESTRICTED_BVALS,
                np.zeros(len(RESTRICTED_BVALS)),
                1e-3,
                radius=radius,
                delta=delta,
                big_delta=big_delta,
            )
            expected = reference(RESTRICTED_BVALS, 1e-3, radius, delta, big_delta)
            np.testing.assert_allclose(
                found, expected, rtol=1e-6, err_msg=f"{delta=} {big_delta=} {radius=}"
            )


def test_sphere_matches_palombo_2020_eq_8() -> None:
    _compare_with_reference(CompartmentModel.SPHERE, _sphere_reference)


def test_cylinder_across_matches_de_santis_2016_eq_3() -> None:
    _compare_with_reference(CompartmentModel.CYLINDER, _cylinder_reference)


def test_sphere_tends_to_the_narrow_pulse_long_time_limit() -> None:
    # delta -> 0 and Delta -> inf at fixed q = gamma G delta: E -> exp(-q^2 R^2 / 5).
    radius_m, delta, big_delta = 5e-6, 1e-3, 2000.0
    q_squared = 1e10  # 1/m^2
    b = q_squared * (big_delta - delta / 3) * 1e-3 * 1e-6  # s/mm^2
    sphere = attenuation(
        CompartmentModel.SPHERE, np.array([b]), [0.0], 3e-3, 0, 5.0, delta, big_delta
    )
    assert sphere[0] == pytest.approx(np.exp(-q_squared * radius_m**2 / 5), rel=1e-3)


def test_large_sphere_tends_to_free_diffusion() -> None:
    # The walls slow only the spins within about sqrt(D t) of them, a share that
    # falls as 1/R, so ln E approaches -b D slowly.
    bvals = np.array([1000.0, 3000.0])
    gaps = []
    for radius in (200.0, 1000.0):
        sphere = attenuation(
            CompartmentModel.SPHERE, bvals, [0, 0], 3e-3, radius=radius
        )
        gaps.append(np.abs(np.log(sphere) / (-bvals * 3e-3) - 1).max())
    assert gaps[1] < 0.01
    assert gaps[1] < gaps[0] / 3


def fibonacci_sphere(n: int) -> np.ndarray:
    """n nearly even unit vectors, each standing for an equal area."""
    k = np.arange(n) + 0.5
    z = 1 - 2 * k / n
    phi = np.pi * (1 + 5**0.5) * k
    r = np.sqrt(1 - z**2)
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)


@pytest.mark.parametrize("kappa", [0.5, 4.0, 16.0])
def test_watson_stick_is_the_stick_averaged_over_the_watson_density(
    kappa: float,
) -> None:
    sphere = fibonacci_sphere(400_000)
    weights = np.exp(kappa * (sphere[:, 2] ** 2 - 1))
    d = 1.7e-3
    for b in (1000.0, 3000.0):
        for angle in (0.0, 0.6, np.pi / 2):
            g = np.array([np.sin(angle), 0.0, np.cos(angle)])
            stick = np.exp(-b * d * (sphere @ g) ** 2)
            expected = (weights @ stick) / weights.sum()
            watson = attenuation(
                CompartmentModel.WATSON_STICK,
                np.array([b]),
                [np.cos(angle)],
                d,
                kappa=kappa,
            )
            assert watson[0] == pytest.approx(expected, rel=1e-6)


def test_watson_stick_tends_to_the_stick_and_to_the_powder_stick() -> None:
    cosines = np.array([1.0, 0.8, 0.5, 0.2, 0.0])
    bvals = np.array([0.0, 1000.0, 3000.0, 5000.0, 10000.0])
    d = 1.7e-3
    stick = attenuation(CompartmentModel.STICK, bvals, cosines, d)
    sharp = attenuation(CompartmentModel.WATSON_STICK, bvals, cosines, d, kappa=1e6)
    np.testing.assert_allclose(sharp, stick, rtol=1e-4)
    powder = attenuation(CompartmentModel.POWDER_STICK, bvals, cosines, d)
    flat = attenuation(CompartmentModel.WATSON_STICK, bvals, cosines, d, kappa=0.0)
    np.testing.assert_allclose(flat, powder, rtol=1e-9)


def test_cylinder_roots_solve_j1_prime_and_satisfy_the_sum_rule() -> None:
    x = _core.cylinder_roots()
    np.testing.assert_allclose(x[:100], jnp_zeros(1, 100), rtol=1e-12)
    assert np.all(np.abs(jvp(1, x[:1000])) < 1e-10)
    assert np.sum(1 / (x**2 * (x**2 - 1))) == pytest.approx(1 / 8, rel=1e-6)
    assert np.sum(1 / (x**4 * (x**2 - 1))) == pytest.approx(7 / 192, rel=1e-12)


def test_thin_cylinder_is_the_stick() -> None:
    cosines = np.array([1.0, 0.7, 0.3, 0.0, 0.0])
    stick = attenuation(CompartmentModel.STICK, BVALS, cosines, 2e-3)
    cylinder = attenuation(CompartmentModel.CYLINDER, BVALS, cosines, 2e-3, radius=1e-3)
    np.testing.assert_allclose(cylinder, stick, rtol=1e-9)


def test_cylinder_tends_to_the_narrow_pulse_long_time_limit() -> None:
    # delta -> 0 and Delta -> inf at fixed q = gamma G delta: E -> exp(-q^2 R^2 / 4)
    # across the cylinder, and free diffusion along it.
    radius_m, delta, big_delta = 5e-6, 1e-3, 2000.0
    q_squared = 1e10  # 1/m^2
    b = q_squared * (big_delta - delta / 3) * 1e-3 * 1e-6  # s/mm^2
    across = attenuation(
        CompartmentModel.CYLINDER, np.array([b]), [0.0], 2e-3, 0, 5.0, delta, big_delta
    )
    assert across[0] == pytest.approx(np.exp(-q_squared * radius_m**2 / 4), rel=1e-3)
    along = attenuation(
        CompartmentModel.CYLINDER, np.array([b]), [1.0], 2e-3, 0, 5.0, delta, big_delta
    )
    assert along[0] == pytest.approx(np.exp(-b * 2e-3), rel=1e-12)


def test_dot_is_one_at_every_b_value_and_direction() -> None:
    dot = attenuation(CompartmentModel.DOT, BVALS, [1.0, 0.5, 0.0, 0.3, 0.9], 0.0)
    np.testing.assert_array_equal(dot, np.ones(5))


def test_free_water_preset_is_exp_minus_b_d_with_d_of_free_water() -> None:
    (water,) = free_water_tissue().compartments
    assert water.model == CompartmentModel.FREE_WATER
    assert water.diffusivity == FREE_WATER_DIFFUSIVITY == 3e-3
    signal = attenuation(water.model, BVALS, np.zeros(5), water.diffusivity)
    np.testing.assert_allclose(signal, np.exp(-BVALS * 3e-3), rtol=1e-12)
