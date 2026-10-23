import logging

import pytest

from larmor.sd import (
    DRL,
    L2NNLS,
    AnisotropicRF,
    IsotropicRF,
    ResponseFunction,
    SDMethod,
    SDModel,
    SDSolver,
)

VALUES = ("GRL", "mFOD")


def _wm_rf() -> ResponseFunction:
    return AnisotropicRF(eigen_vals=[1.7e-3, 0.2e-3, 0.2e-3], mean_kurtosis=0.0)


def _second_rf() -> ResponseFunction:
    return AnisotropicRF(eigen_vals=[1.5e-3, 0.9e-3, 0.1e-3], mean_kurtosis=0.0)


def test_members_are_the_implemented_methods() -> None:
    assert {member.name: member.value for member in SDMethod} == {
        "GRL": "GRL",
        "MFOD": "mFOD",
    }


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("GRL", SDMethod.GRL),
        ("grl", SDMethod.GRL),
        ("Grl", SDMethod.GRL),
        ("mFOD", SDMethod.MFOD),
        ("MFOD", SDMethod.MFOD),
        ("mfod", SDMethod.MFOD),
    ],
)
def test_method_is_found_in_any_case(name: str, expected: SDMethod) -> None:
    assert SDMethod(name) is expected


def _assert_lists_valid_values(error: pytest.ExceptionInfo[ValueError]) -> None:
    message = str(error.value)
    for value in VALUES:
        assert value in message, message


@pytest.mark.parametrize("name", ["csd", "dRL", ""])
def test_unknown_method_lists_the_valid_ones(name: str) -> None:
    with pytest.raises(ValueError) as error:
        SDMethod(name)
    _assert_lists_valid_values(error)


@pytest.mark.parametrize(
    ("method", "expected"), [(SDMethod.GRL, DRL()), (SDMethod.MFOD, L2NNLS())]
)
def test_default_solver_is_the_method_solver_with_its_defaults(
    method: SDMethod, expected: SDSolver
) -> None:
    assert method.default_solver() == expected


@pytest.mark.parametrize(("method", "weight"), [("grl", 0.2), ("mfod", 1.0)])
def test_model_takes_the_method_inner_shell_weight(method: str, weight: float) -> None:
    model = SDModel([_wm_rf()], method=method)
    assert model.inner_shell_weight == weight
    assert model.solver is None


@pytest.mark.parametrize(
    ("method", "rfs", "accepted"),
    [
        (SDMethod.GRL, [IsotropicRF(diffusivity=3e-3)], False),
        (SDMethod.GRL, [_wm_rf(), IsotropicRF(diffusivity=3e-3)], True),
        (SDMethod.GRL, [_wm_rf(), _second_rf()], False),
        (SDMethod.MFOD, [IsotropicRF(diffusivity=3e-3)], False),
        (SDMethod.MFOD, [_wm_rf()], True),
        (SDMethod.MFOD, [_wm_rf(), _second_rf(), IsotropicRF(diffusivity=3e-3)], True),
    ],
    ids=["grl-0", "grl-1", "grl-2", "mfod-0", "mfod-1", "mfod-2"],
)
def test_method_checks_its_anisotropic_response_functions(
    method: SDMethod, rfs: list[ResponseFunction], accepted: bool
) -> None:
    if accepted:
        SDModel(rfs, method=method)
    else:
        with pytest.raises(ValueError, match="anisotropic response function"):
            SDModel(rfs, method=method)


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        (SDMethod.MFOD, SDMethod.MFOD),
        ("grl", SDMethod.GRL),
        ("MFOD", SDMethod.MFOD),
    ],
)
def test_model_method_is_always_a_member(
    method: SDMethod | str, expected: SDMethod
) -> None:
    model = SDModel([_wm_rf()], method=method)
    assert model.method is expected
    assert type(model.method) is SDMethod


def test_model_refuses_an_unknown_method() -> None:
    with pytest.raises(ValueError) as error:
        SDModel([_wm_rf()], method="rl")
    _assert_lists_valid_values(error)


@pytest.mark.parametrize("weight", [0.0, -1.0, 1.5, float("nan")])
def test_model_refuses_an_inner_shell_weight_outside_0_1(weight: float) -> None:
    with pytest.raises(ValueError, match="inner_shell_weight"):
        SDModel([_wm_rf(), IsotropicRF(diffusivity=3e-3)], inner_shell_weight=weight)


@pytest.mark.parametrize("weight", [1.0, 0.5])
def test_model_accepts_an_inner_shell_weight_in_0_1(weight: float) -> None:
    model = SDModel(
        [_wm_rf(), IsotropicRF(diffusivity=3e-3)], inner_shell_weight=weight
    )
    assert model.inner_shell_weight == weight


@pytest.mark.parametrize(
    ("rfs", "weight", "warned"),
    [
        ([_wm_rf()], 0.5, True),
        ([_wm_rf()], None, False),
        ([_wm_rf(), IsotropicRF(diffusivity=3e-3)], 0.5, False),
    ],
    ids=["single-given", "single-default", "multi-given"],
)
def test_inner_shell_weight_on_a_single_tissue_fit_is_warned_about(
    rfs: list[ResponseFunction],
    weight: float | None,
    warned: bool,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="larmor"):
        SDModel(rfs, inner_shell_weight=weight)
    assert ("only applies to a multi-tissue fit" in caplog.text) == warned
