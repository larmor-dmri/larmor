"""Progress reports of the FOD fits inside fit_progress."""

from typing import Any

import numpy as np
import pytest

import larmor.sd.model as sd_module
import larmor.sd.msmt_csd as msmt_module
from larmor.sd import AnisotropicRF, CSDModel, MSMTCSDModel, SDModel, ZonalSHRF
from larmor.sd.progress import RESPONSE_STAGE, fit_progress
from larmor.sd.sphere import get_dirs
from larmor.types import GradientTable, Volume


class StopError(Exception):
    pass


def _small_dwi(grl: dict[str, Any]) -> tuple[Volume, GradientTable, np.ndarray]:
    """A 4x4x4 corner of the reference block with 32 voxels in its mask."""
    block = grl["img"][:4, :4, :4].astype(np.float32)
    gtab = GradientTable(
        grl["bvals"].astype(np.float64), grl["bvecs"].astype(np.float64)
    )
    mask = grl["mask"][:4, :4, :4].astype(bool)
    mask[:2] = False
    return Volume(block, np.eye(4)), gtab, mask


def _sd_model() -> SDModel:
    return SDModel(
        [AnisotropicRF(eigen_vals=[1.7e-3, 0.2e-3, 0.2e-3], mean_kurtosis=0.0)]
    )


def _one_shell() -> tuple[Volume, GradientTable]:
    """Seven voxels of positive signal, with one b0 and 30 directions at b = 1000."""
    gtab = GradientTable(
        np.r_[0.0, [1000.0] * 30], np.r_[np.zeros((1, 3)), get_dirs(30)]
    )
    data = np.random.default_rng(0).uniform(300.0, 1000.0, (7, 1, 1, 31))
    data[..., 0] = 2000.0
    return Volume(data, np.eye(4)), gtab


def test_sd_model_reports_each_chunk_and_fits_the_same(
    grl: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    volume, gtab, mask = _small_dwi(grl)
    sd = _sd_model()
    monkeypatch.setattr(sd_module, "FIT_CHUNK", 5)
    expected = sd.fit(volume, gtab, mask=mask, n_jobs=1).fods[0]

    calls: list[tuple[int, int, str]] = []
    with fit_progress(lambda *call: calls.append(call)):
        fods = sd.fit(volume, gtab, mask=mask, n_jobs=1).fods[0]

    assert [done for done, _, _ in calls] == [0, 5, 10, 15, 20, 25, 30, 32]
    assert {(total, stage) for _, total, stage in calls} == {(32, "Fitting voxels")}
    np.testing.assert_array_equal(fods, expected)


def test_an_exception_from_report_stops_the_fit(grl: dict[str, Any]) -> None:
    volume, gtab, mask = _small_dwi(grl)
    sd = _sd_model()
    calls: list[int] = []

    def report(done: int, total: int, stage: str) -> None:
        calls.append(done)
        if done > 0:
            raise StopError

    with pytest.raises(StopError), fit_progress(report):
        sd.fit(volume, gtab, mask=mask, n_jobs=1)
    assert calls == [0, 32]


def test_msmt_and_csd_report_from_zero_to_all_voxels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    volume, gtab = _one_shell()
    b = np.array([0.0, 1000.0])
    root = np.sqrt(4 * np.pi)
    wm = ZonalSHRF(
        bvals=b, coefficients=np.array([[root * 1000, 0.0], [root * 400, -150.0]])
    )
    csf = ZonalSHRF(bvals=b, coefficients=np.array([[root * 2000], [root * 100]]))
    monkeypatch.setattr(msmt_module, "FIT_CHUNK", 3)

    msmt: list[int] = []
    with fit_progress(lambda done, total, stage: msmt.append(done)):
        MSMTCSDModel([wm, csf], lmax=2).fit(volume, gtab, n_jobs=1)
    csd: list[int] = []
    csd_response = ZonalSHRF(
        bvals=np.array([1000.0]), coefficients=np.array([[1800.0, -700.0, 180.0]])
    )
    with fit_progress(lambda done, total, stage: csd.append(done)):
        CSDModel(csd_response, lmax=4).fit(volume, gtab, n_jobs=1)

    assert msmt == [0, 3, 6, 7]
    assert csd == [0, 7]


def test_fits_outside_the_block_report_nothing(grl: dict[str, Any]) -> None:
    volume, gtab, mask = _small_dwi(grl)
    sd = _sd_model()
    calls: list[int] = []
    with fit_progress(lambda done, total, stage: calls.append(done)):
        pass
    sd.fit(volume, gtab, mask=mask, n_jobs=1)
    assert calls == []


@pytest.mark.parametrize("model", [CSDModel(lmax=4), MSMTCSDModel(lmax=4)])
def test_a_cancel_at_the_response_stage_stops_before_estimating(
    model: CSDModel | MSMTCSDModel,
) -> None:
    volume, gtab = _one_shell()
    stages: list[str] = []

    def report(done: int, total: int, stage: str) -> None:
        stages.append(stage)
        raise StopError

    with pytest.raises(StopError), fit_progress(report):
        model.fit(volume, gtab, mask=np.ones((7, 1, 1), bool), n_jobs=1)
    assert stages == [RESPONSE_STAGE]
