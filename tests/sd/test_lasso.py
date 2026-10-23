from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.io import loadmat

from larmor import _core
from larmor.sd import Lasso

DATA_DIR = Path(__file__).parents[1] / "data"
# make_lasso_ref.m: six problems of FOD shape, each with three lambdas. Only their
# designs, signals and lambdas are used; MATLAB's solutions fit an intercept.
PROBLEMS: list[dict[str, Any]] = list(
    loadmat(DATA_DIR / "lasso" / "lasso_problems.mat", simplify_cells=True)["problems"]
)


def fit(
    x: NDArray[np.float64],
    signals: NDArray[np.float64],
    reg: float,
    alpha: float = 0.5,
    tol: float = 1e-12,
    max_iter: int = 10**7,
    n_threads: int = 1,
) -> NDArray[np.float64]:
    solver = _core.FodSolver.lasso(np.ascontiguousarray(x), reg, alpha, tol, max_iter)
    return _core.fit_single_tissue(np.atleast_2d(signals), solver, n_threads)


def design(problem: dict[str, Any]) -> NDArray[np.float64]:
    # The references were solved on double(X), the single X cast up.
    x: NDArray[np.float64] = problem["X"].astype(np.float64)
    return x


def kkt_violation(
    x: NDArray[np.float64], y: NDArray[np.float64], b: NDArray[np.float64], reg: float
) -> float:
    """The largest violation of the optimality conditions of the elastic net.

    The problem is (1 / 2m) ||y - X0 c||^2 + reg (alpha ||c||_1 + (1 - alpha) / 2
    ||c||^2) over c, with X0 the columns of x divided by their root mean square and
    no intercept; b is c divided by those scales, the coefficients of x.
    """
    alpha = 0.5
    m = len(y)
    scales = np.sqrt(np.mean(x**2, axis=0))
    x0 = x / scales
    c = b * scales
    g = x0.T @ (y - x0 @ c) / m - reg * (1 - alpha) * c
    on = c != 0
    violation = np.maximum(np.abs(g) - reg * alpha, 0.0)
    violation[on] = np.abs(g[on] - reg * alpha * np.sign(c[on]))
    return float(violation.max())


@pytest.mark.parametrize(
    ("problem", "i"),
    [(p, i) for p in PROBLEMS for i in range(len(p["lambda"]))],
    ids=[f"{p['name']}-{lam:g}" for p in PROBLEMS for lam in p["lambda"]],
)
def test_solution_meets_the_optimality_conditions_without_intercept(
    problem: dict[str, Any], i: int
) -> None:
    x, y, reg = design(problem), problem["y"], float(problem["lambda"][i])
    b = fit(x, y, reg)[0]
    assert kkt_violation(x, y, b, reg) < 1e-6 * reg


def _lambda_max(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    """The smallest penalty at which every coefficient is zero (alpha 0.5)."""
    x0 = x / np.sqrt(np.mean(x**2, axis=0))
    return float(np.abs(x0.T @ y).max() / (len(y) * 0.5))


@pytest.mark.parametrize(
    ("factor", "all_zero"), [(1.0001, True), (0.99, False)], ids=["above", "below"]
)
def test_fod_is_zero_from_lambda_max_on(factor: float, all_zero: bool) -> None:
    problem = PROBLEMS[0]
    x, y = design(problem), problem["y"]
    b = fit(x, y, factor * _lambda_max(x, y))
    assert np.all(b == 0) == all_zero


def test_pure_l1_with_tiny_reg_is_least_squares_without_intercept() -> None:
    rng = np.random.default_rng(0)
    x = rng.uniform(0.0, 1.0, (60, 8))
    y = x @ rng.uniform(-1.0, 1.0, 8) + 0.01 * rng.standard_normal(60)
    expected, *_ = np.linalg.lstsq(x, y, rcond=None)
    b = fit(x, y, reg=1e-10, alpha=1.0, tol=1e-14)[0]
    np.testing.assert_allclose(b, expected, rtol=1e-6, atol=1e-8)


INVALID = [
    pytest.param({"reg": -0.1}, "reg", id="reg-negative"),
    pytest.param({"reg": float("nan")}, "reg", id="reg-nan"),
    pytest.param({"alpha": 0.0}, "alpha", id="alpha-zero"),
    pytest.param({"alpha": 1.5}, "alpha", id="alpha-above-one"),
    pytest.param({"alpha": float("nan")}, "alpha", id="alpha-nan"),
    pytest.param({"tol": 0.0}, "tol", id="tol-zero"),
    pytest.param({"tol": float("nan")}, "tol", id="tol-nan"),
    pytest.param({"max_iter": 0}, "max_iter", id="max-iter-zero"),
]


@pytest.mark.parametrize(("change", "match"), INVALID)
def test_lasso_refuses_invalid_settings(change: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        Lasso(**change)


@pytest.mark.parametrize(("change", "match"), INVALID)
def test_binding_refuses_invalid_settings(change: dict[str, Any], match: str) -> None:
    settings: dict[str, Any] = {
        "reg": 0.1,
        "alpha": 0.5,
        "tol": 1e-8,
        "max_iter": 100,
    } | change
    with pytest.raises(ValueError, match=match):
        _core.FodSolver.lasso(np.ones((4, 3)), **settings)


def test_result_does_not_depend_on_n_threads() -> None:
    rng = np.random.default_rng(1)
    x = rng.uniform(0.0, 1.0, (40, 30))
    # More voxels than one compiled block of 128, so the threads share the work.
    signals = rng.uniform(0.0, 1.0, (300, 30)) @ x.T
    np.testing.assert_array_equal(
        fit(x, signals, 0.01, tol=1e-8, n_threads=1),
        fit(x, signals, 0.01, tol=1e-8, n_threads=3),
    )
