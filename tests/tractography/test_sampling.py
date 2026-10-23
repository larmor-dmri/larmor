from pathlib import Path

import numpy as np
import pytest

from larmor.io import load_tractogram, load_volume, read_tsf
from larmor.tractography import sample_volume
from larmor.types import Tractogram, Volume

DATA = Path(__file__).parents[1] / "data"
TSF = DATA / "tsf"


def test_sampled_b0_matches_tcksample() -> None:
    tracks = load_tractogram(DATA / "editing" / "tracks.tck")
    values = sample_volume(tracks, load_volume(TSF / "bzero_crop.nii.gz"))
    expected, _ = read_tsf(TSF / "bzero.tsf", tractogram=tracks)
    assert values.dtype == np.float32
    # tcksample stores float32 and interpolates in its own order: a few ulps.
    np.testing.assert_allclose(values, expected, rtol=1e-5)


def test_points_at_and_beyond_the_edges_match_tcksample() -> None:
    tracks = load_tractogram(TSF / "edges.tck")
    values = sample_volume(tracks, load_volume(TSF / "edges.nii"))
    expected, _ = read_tsf(TSF / "edges.tsf", tractogram=tracks)
    # Integer-valued voxels and short fractions: float32 rounding only.
    np.testing.assert_allclose(values, expected, rtol=1e-6)
    assert (expected == 0).any() and (expected != 0).any()


def _gradient(volumes: int) -> Volume:
    # Value 10 * x + y + 0.1 * z: trilinear interpolation is exact for it.
    x, y, z = np.meshgrid(np.arange(4), np.arange(5), np.arange(6), indexing="ij")
    data = (10.0 * x + y + 0.1 * z).astype(np.float32)
    return Volume(np.repeat(data[..., None], volumes, axis=3), np.eye(4))


def test_a_4d_image_with_one_volume_is_sampled_as_3d() -> None:
    point = np.array([[1.5, 2.25, 3.5]], np.float32)
    values = sample_volume(Tractogram(point, np.array([0, 1])), _gradient(1))
    assert values[0] == pytest.approx(15 + 2.25 + 0.35, rel=1e-6)


def test_a_4d_image_with_several_volumes_is_refused() -> None:
    point = np.zeros((1, 3), np.float32)
    with pytest.raises(ValueError, match="one volume"):
        sample_volume(Tractogram(point, np.array([0, 1])), _gradient(2))


def test_values_are_scaled_by_the_slope_and_intercept_but_zero_outside() -> None:
    gradient = _gradient(1)
    scaled = Volume(gradient.data, gradient.affine, slope=2.0, intercept=1.0)
    points = np.array([[1.5, 2.25, 3.5], [-5.0, 0.0, 0.0]], np.float32)
    values = sample_volume(Tractogram(points, np.array([0, 2])), scaled)
    assert values[0] == pytest.approx(2 * (15 + 2.25 + 0.35) + 1, rel=1e-6)
    assert values[1] == 0.0
