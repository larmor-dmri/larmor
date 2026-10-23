import logging
from pathlib import Path

import numpy as np
import pytest

from larmor.tractography import tract_profile
from larmor.types import Tractogram, Volume

# A map of 0.2 + 0.005 * x on a 1 mm grid with x from 0 to 99 mm: trilinear
# interpolation reproduces it exactly inside the grid. An inner node holds the
# points within half a node spacing on either side, whose mean is the node's x;
# the first and last nodes hold only the points on their inner side, so their
# values lie inward, by up to half a node spacing.
AFFINE = np.eye(4)
AFFINE[:3, 3] = [0.0, -20.0, -20.0]
X = np.arange(100, dtype=np.float32)[:, None, None]
GRADIENT = Volume(
    np.broadcast_to(0.2 + 0.005 * X, (100, 40, 40)).astype(np.float32).copy(), AFFINE
)


def _tractogram(lines: list[np.ndarray]) -> Tractogram:
    offsets = np.concatenate([[0], np.cumsum([len(line) for line in lines])])
    return Tractogram(np.concatenate(lines), offsets)


def _straight_bundle(count: int = 40) -> tuple[Tractogram, np.ndarray]:
    # Lines from x = 10 to x = 90 mm with different point counts and offsets in y
    # and z; every odd one is stored from x = 90 to x = 10.
    rng = np.random.default_rng(0)
    lines, backwards = [], []
    for s in range(count):
        xs = np.linspace(10.0, 90.0, int(rng.integers(20, 60)))
        y, z = rng.normal(scale=2.0, size=2)
        line = np.stack([xs, np.full_like(xs, y), np.full_like(xs, z)], axis=1)
        backwards.append(s % 2 == 1)
        lines.append(line[::-1] if s % 2 else line)
    return _tractogram(lines), np.array(backwards)


def test_a_bundle_along_a_gradient_recovers_the_gradient() -> None:
    bundle, backwards = _straight_bundle()
    profile = tract_profile(bundle, {"fa": GRADIENT}, num_points=17)
    expected = 0.2 + 0.005 * np.linspace(10.0, 90.0, 17)
    np.testing.assert_allclose(profile.values["fa"][1:-1], expected[1:-1], atol=1e-4)
    inward = (profile.values["fa"][[0, -1]] - expected[[0, -1]]) / 0.005 * [1, -1]
    assert np.all((inward > 0) & (inward < 2.5))
    np.testing.assert_array_equal(profile.reversed, backwards)
    np.testing.assert_allclose(profile.core[[0, -1], 0], [10.0, 90.0], atol=1e-4)


def test_streamlines_of_unequal_extent_are_aligned_to_the_centreline() -> None:
    # 20 lines from x = 0 to 99 mm and 20 from x = 50 to 99 mm, on a map equal to x:
    # each node's value is the x of its position, which a profile of fractions of
    # each streamline's own length would not give.
    x = Volume(np.broadcast_to(X, (100, 40, 40)).astype(np.float32).copy(), AFFINE)
    rng = np.random.default_rng(1)
    lines = []
    for s in range(40):
        xs = np.linspace(0.0 if s < 20 else 50.0, 99.0, 60)
        y, z = rng.normal(scale=1.0, size=2)
        lines.append(np.stack([xs, np.full_like(xs, y), np.full_like(xs, z)], 1))
    profile = tract_profile(_tractogram(lines), {"x": x}, num_points=34)
    np.testing.assert_allclose(profile.core[[0, -1], 0], [0.0, 99.0], atol=0.1)
    inner = profile.values["x"][1:-1]
    np.testing.assert_allclose(inner, profile.core[1:-1, 0], atol=0.2)
    # The nodes are 3 mm apart: the end ones lie inward by less than 1.5 mm.
    inward = (profile.values["x"][[0, -1]] - profile.core[[0, -1], 0]) * [1, -1]
    assert np.all((inward > 0) & (inward < 1.5))


def test_start_point_chooses_the_end_where_the_profile_starts() -> None:
    bundle, backwards = _straight_bundle()
    profile = tract_profile(bundle, {"fa": GRADIENT}, num_points=17, start=[95, 0, 0])
    expected = 0.2 + 0.005 * np.linspace(90.0, 10.0, 17)
    np.testing.assert_allclose(profile.values["fa"][1:-1], expected[1:-1], atol=1e-4)
    np.testing.assert_array_equal(profile.reversed, ~backwards)


@pytest.mark.parametrize("direction", [(0.0, 0.0, 1.0), (-0.3, 0.2, 1.0)])
def test_a_bundle_running_up_and_down_starts_at_its_inferior_end(
    direction: tuple[float, float, float],
) -> None:
    rng = np.random.default_rng(3)
    axis = np.array(direction) / np.linalg.norm(direction)
    lines = []
    for s in range(20):
        line = np.linspace(-15.0, 15.0, 25)[:, None] * axis + rng.normal(size=3)
        lines.append(line[::-1] if s % 2 else line)
    profile = tract_profile(_tractogram(lines), {"fa": GRADIENT}, num_points=5)
    np.testing.assert_array_equal(profile.reversed, np.arange(20) % 2 == 1)
    assert profile.core[0, 2] < profile.core[-1, 2]


def test_an_axis_with_two_similar_components_gives_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    line = np.linspace(10.0, 40.0, 20)[:, None] * np.array([1.0, 0.0, 0.9])
    bundle = _tractogram([line, line + [0, 1, 0]])
    with caplog.at_level(logging.WARNING, logger="larmor"):
        tract_profile(bundle, {"fa": GRADIENT}, num_points=5)
    assert "ambiguous" in caplog.text
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="larmor"):
        tract_profile(bundle, {"fa": GRADIENT}, num_points=5, start=[0, 0, 0])
    assert caplog.text == ""


def test_points_outside_a_map_are_left_out_not_counted_as_zero() -> None:
    # Half the lines run at z = 30 mm, beyond the grid's last voxel (z = 19 mm).
    ones = Volume(np.ones((100, 40, 40), np.float32), AFFINE)
    xs = np.linspace(10.0, 90.0, 30)
    lines = [
        np.stack([xs, np.full(30, 0.1 * s), np.full(30, 30.0 * (s % 2))], 1)
        for s in range(10)
    ]
    profile = tract_profile(
        _tractogram(lines), {"m": ones}, num_points=4, weighting="mean"
    )
    np.testing.assert_allclose(profile.values["m"], 1.0)


def test_gaussian_weights_follow_the_mahalanobis_distance() -> None:
    # Lines along x at y = 0, 0, 0 and 3 mm over a map of 10 + y. At every
    # profile point the y values have mean 0.75 and variance 2.25 (S - 1 = 3), so
    # d^2 is 0.25 for the three lines at 0 and 2.25 for the one at 3; x and z do
    # not vary, and the pseudo-inverse ignores them.
    j = np.arange(40, dtype=np.float32)[None, :, None]
    volume = Volume(np.broadcast_to(10 + j - 20, (100, 40, 40)).copy(), AFFINE)
    xs = np.linspace(10.0, 90.0, 30)
    lines = [np.stack([xs, np.full(30, y), np.zeros(30)], 1) for y in (0, 0, 0, 3)]
    bundle = _tractogram(lines)
    gaussian = tract_profile(bundle, {"m": volume}, num_points=4)
    mean = tract_profile(bundle, {"m": volume}, num_points=4, weighting="mean")
    near, far = 3 * np.exp(-0.125), np.exp(-1.125)
    expected = (10 * near + 13 * far) / (near + far)
    np.testing.assert_allclose(gaussian.values["m"], expected, rtol=1e-6)
    np.testing.assert_allclose(mean.values["m"], 10.75, rtol=1e-6)


def test_non_finite_values_are_left_out_and_a_point_without_any_is_nan() -> None:
    data = np.ones((100, 40, 40), np.float32)
    # The first node, at x = 10 mm, holds the points up to x = 30 mm.
    data[:31] = np.nan
    bundle, _ = _straight_bundle(4)
    profile = tract_profile(bundle, {"m": Volume(data, AFFINE)}, num_points=3)
    assert np.isnan(profile.values["m"][0])
    np.testing.assert_allclose(profile.values["m"][1:], 1.0)


def test_segments_are_the_nearest_centreline_nodes() -> None:
    xs = np.array([0.0, 0.9, 2.2, 3.4, 4.6, 8.0])
    line = np.stack([xs, np.zeros(6), np.zeros(6)], 1)
    bundle = _tractogram([line, line[::-1] + [0, 1, 0]])
    profile = tract_profile(bundle, {"fa": GRADIENT}, num_points=5)
    # The centreline runs from x = 0 to 8 mm, with nodes 2 mm apart.
    np.testing.assert_allclose(profile.core[:, 0], [0, 2, 4, 6, 8], atol=1e-4)
    np.testing.assert_array_equal(
        profile.segments(bundle), [0, 0, 1, 2, 2, 4, 4, 2, 2, 1, 0, 0]
    )


def test_the_csv_has_a_row_per_point_and_a_column_per_map(tmp_path: Path) -> None:
    bundle, _ = _straight_bundle(6)
    profile = tract_profile(bundle, {"fa": GRADIENT, "md": GRADIENT}, num_points=3)
    path = tmp_path / "profile.csv"
    profile.write_csv(path)
    rows = path.read_text().splitlines()
    assert rows[0] == "point,core_x,core_y,core_z,fa,md"
    assert len(rows) == 4
    assert rows[2].split(",")[4] == f"{profile.values['fa'][1]:.8g}"


@pytest.mark.parametrize(
    ("bundle", "kwargs", "message"),
    [
        (Tractogram(np.zeros((0, 3)), np.array([0])), {}, "no streamlines"),
        (Tractogram(np.zeros((1, 3)), np.array([0, 1])), {}, "at least 2 points"),
        (
            Tractogram(np.zeros((2, 3)), np.array([0, 2])),
            {"num_points": 1},
            "num_points",
        ),
        (Tractogram(np.zeros((2, 3)), np.array([0, 2])), {"start": [1, 2]}, "start"),
        (
            Tractogram(np.zeros((2, 3)), np.array([0, 2])),
            {"weighting": "median"},
            "median",
        ),
    ],
)
def test_invalid_input_is_refused(
    bundle: Tractogram, kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        tract_profile(bundle, {"fa": GRADIENT}, **kwargs)  # type: ignore[arg-type]
