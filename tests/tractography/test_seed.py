import warnings
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from larmor.tractography import (
    MaskSeeds,
    Seeds,
    SeedStrategy,
    read_seeds,
    seeds_per_voxel,
    write_seeds,
)
from larmor.tractography.seed import PER_VOXEL_STREAM
from larmor.types import Volume

EPS = np.finfo(np.float64).eps


def _draw(strategy: SeedStrategy, n: int, seed: int = 0) -> Seeds:
    # Typed on the protocol, so mypy checks that MaskSeeds satisfies it.
    return strategy.draw(n, np.random.default_rng(seed))


def _affine() -> NDArray[np.float64]:
    # Anisotropic voxels, a flip of x, an offset and a small rotation about z, so a
    # mix-up of axes, of the voxel offset or of the affine's transpose shows.
    angle = np.deg2rad(7.0)
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    affine = np.eye(4)
    affine[:3, :3] = rotation @ np.diag([-2.0, 1.5, 3.0])
    affine[:3, 3] = [40.0, -12.0, 7.5]
    return affine


def _mask() -> NDArray[np.float64]:
    # Negative and zero voxels must not be seeded; any value > 0 must.
    mask = np.zeros((4, 5, 3))
    mask[0, 0, 0] = 1.0
    mask[1, 3, 2] = 0.2
    mask[3, 4, 1] = 7.0
    mask[2, 1, 0] = 1.0
    mask[1, 1, 1] = -3.0
    return mask


def _volume() -> Volume:
    return Volume(_mask(), _affine())


def _to_voxels(world: NDArray[np.float64]) -> NDArray[np.float64]:
    inverse = np.linalg.inv(_affine())
    out: NDArray[np.float64] = world @ inverse[:3, :3].T + inverse[:3, 3]
    return out


def _voxels(seeds: Seeds) -> NDArray[np.int64]:
    out: NDArray[np.int64] = np.rint(_to_voxels(seeds.positions)).astype(np.int64)
    return out


# Seeds


def test_seeds_accept_lists_and_store_float64_and_int64() -> None:
    seeds = Seeds([[1, 2, 3]], [[0, 0, 2]], [5])  # type: ignore[arg-type]
    assert seeds.positions.dtype == np.float64
    assert seeds.directions is not None and seeds.directions.dtype == np.float64
    assert seeds.indices is not None and seeds.indices.dtype == np.int64
    assert seeds.directions.tolist() == [[0.0, 0.0, 1.0]]
    assert len(seeds) == 1


def test_directions_off_unit_length_are_normalised() -> None:
    seeds = Seeds(np.zeros((2, 3)), np.array([[3.0, 0.0, 4.0], [0.0, -2.0, 0.0]]))
    assert seeds.directions is not None
    np.testing.assert_allclose(seeds.directions, [[0.6, 0.0, 0.8], [0.0, -1.0, 0.0]])


def test_unit_directions_are_kept_bit_for_bit() -> None:
    # A rerun from saved seeds must follow the same first step, so a direction
    # within rounding of unit length is not divided by its norm again.
    unit = np.random.default_rng(3).normal(size=(50, 3))
    unit /= np.linalg.norm(unit, axis=1, keepdims=True)
    unit[0, 0] = np.nextafter(unit[0, 0], 2.0)
    assert np.abs(np.linalg.norm(unit, axis=1) - 1).max() <= 4 * EPS
    seeds = Seeds(np.zeros((50, 3)), unit)
    assert seeds.directions is not None
    np.testing.assert_array_equal(seeds.directions, unit)


def test_directions_do_not_change_the_callers_array() -> None:
    directions = np.array([[0.0, 0.0, 2.0], [1.0, 0.0, 0.0]])
    seeds = Seeds(np.zeros((2, 3)), directions)
    assert directions.tolist() == [[0.0, 0.0, 2.0], [1.0, 0.0, 0.0]]
    assert seeds.directions is not None
    seeds.directions[1, 0] = 5.0
    assert directions[1, 0] == 1.0


def test_empty_seeds() -> None:
    seeds = Seeds(np.empty((0, 3)), np.empty((0, 3)), np.empty(0, np.int64))
    assert len(seeds) == 0


@pytest.mark.parametrize(
    ("positions", "directions", "indices", "match"),
    [
        (np.zeros((2, 2)), None, None, "positions of shape"),
        (np.zeros(3), None, None, "positions of shape"),
        (np.array([[0.0, np.nan, 0.0]]), None, None, "nan position"),
        (np.array([[0.0, np.inf, 0.0]]), None, None, "nan position"),
        (np.zeros((2, 3)), np.ones((3, 3)), None, "directions of shape"),
        (np.zeros((2, 3)), np.ones(3), None, "directions of shape"),
        (np.zeros((1, 3)), np.array([[1.0, np.nan, 0.0]]), None, "nan direction"),
        (np.zeros((1, 3)), np.array([[1.0, np.inf, 0.0]]), None, "nan direction"),
        (np.zeros((2, 3)), np.array([[1.0, 0, 0], [0, 0, 0]]), None, "zero direction"),
        (np.zeros((2, 3)), None, np.array([0, 1, 2]), "indices of shape"),
        (np.zeros((2, 3)), None, np.array([0.0, 1.0]), "integer indices"),
        (np.zeros((2, 3)), None, np.array([True, False]), "integer indices"),
        (np.zeros((2, 3)), None, np.array([0, -1]), "negative"),
    ],
)
def test_invalid_seeds_are_refused(
    positions: NDArray[np.float64],
    directions: NDArray[np.float64] | None,
    indices: NDArray[np.int64] | None,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        Seeds(positions, directions, indices)


# MaskSeeds


def test_mask_seeds_lie_in_mask_voxels_in_world_coordinates() -> None:
    seeds = _draw(MaskSeeds(_volume()), 2000)
    assert len(seeds) == 2000
    assert seeds.directions is None and seeds.indices is None
    ijk = _to_voxels(seeds.positions)
    voxel = np.rint(ijk).astype(int)
    assert (_mask()[tuple(voxel.T)] > 0).all()
    assert np.abs(ijk - voxel).max() <= 0.5 + 1e-9


def test_mask_seeds_are_spread_uniformly_over_voxels() -> None:
    voxel = _voxels(_draw(MaskSeeds(_volume()), 20000))
    _, counts = np.unique(voxel, axis=0, return_counts=True)
    # 4 voxels, 5000 expected each whatever their value; the binomial sd is ~61,
    # so 5% is > 4 sd.
    assert len(counts) == 4
    np.testing.assert_allclose(counts, 5000, rtol=0.05)


def test_mask_seeds_cover_the_whole_voxel() -> None:
    ijk = _to_voxels(_draw(MaskSeeds(_volume()), 8000).positions)
    offset = ijk - np.rint(ijk)
    # U(-0.5, 0.5) has mean 0 and sd 1/sqrt(12) ~ 0.289 on each axis; a narrower,
    # wider or one-sided jitter moves one of them by far more than 4 sd.
    np.testing.assert_allclose(offset.mean(axis=0), 0.0, atol=0.015)
    np.testing.assert_allclose(offset.std(axis=0), 1 / np.sqrt(12), atol=0.01)


def test_mask_seeds_draw_zero_gives_empty_seeds() -> None:
    assert _draw(MaskSeeds(_volume()), 0).positions.shape == (0, 3)


def test_same_generator_state_gives_same_seeds() -> None:
    a = _draw(MaskSeeds(_volume()), 50, seed=11)
    b = _draw(MaskSeeds(_volume()), 50, seed=11)
    c = _draw(MaskSeeds(_volume()), 50, seed=12)
    np.testing.assert_array_equal(a.positions, b.positions)
    assert not np.any(np.all(a.positions == c.positions, axis=1))


# seeds_per_voxel


def test_seeds_per_voxel_are_grouped_per_voxel_in_c_order() -> None:
    seeds = seeds_per_voxel(_volume(), 3, np.random.default_rng(0))
    assert seeds.directions is None and seeds.indices is None
    expected = np.repeat(np.argwhere(_mask() > 0), 3, axis=0)
    np.testing.assert_array_equal(_voxels(seeds), expected)


def test_seeds_per_voxel_differ_within_a_voxel() -> None:
    seeds = seeds_per_voxel(_volume(), 4, np.random.default_rng(0))
    assert len(np.unique(seeds.positions, axis=0)) == 16


def test_seeds_per_voxel_cover_the_whole_voxel() -> None:
    seeds = seeds_per_voxel(_volume(), 2000, np.random.default_rng(1))
    offset = _to_voxels(seeds.positions) - _voxels(seeds)
    np.testing.assert_allclose(offset.mean(axis=0), 0.0, atol=0.015)
    np.testing.assert_allclose(offset.std(axis=0), 1 / np.sqrt(12), atol=0.01)


def test_seeds_per_voxel_follow_the_generator() -> None:
    a = seeds_per_voxel(_volume(), 2, np.random.default_rng(5))
    b = seeds_per_voxel(_volume(), 2, np.random.default_rng(5))
    np.testing.assert_array_equal(a.positions, b.positions)


@pytest.mark.parametrize("seed", [0, 7, 2**64 - 1])
def test_seeds_per_voxel_from_a_run_seed_use_their_own_stream(seed: int) -> None:
    seeds = seeds_per_voxel(_volume(), 3, seed)
    expected = seeds_per_voxel(
        _volume(), 3, np.random.default_rng([seed, PER_VOXEL_STREAM])
    )
    np.testing.assert_array_equal(seeds.positions, expected.positions)


def test_seeds_per_voxel_follow_the_run_seed() -> None:
    a, b, c = (seeds_per_voxel(_volume(), 2, s) for s in (5, 5, 6))
    np.testing.assert_array_equal(a.positions, b.positions)
    assert not np.array_equal(a.positions, c.positions)


@pytest.mark.parametrize("n", [0, -1])
def test_seeds_per_voxel_refuse_no_seeds(n: int) -> None:
    with pytest.raises(ValueError, match="Cannot draw"):
        seeds_per_voxel(_volume(), n, np.random.default_rng(0))


# A 2-D mask is refused by Volume before the seeding sees it.
@pytest.mark.parametrize(
    ("mask", "match"),
    [
        (np.ones((2, 2, 2, 2)), "3D mask"),
        (np.ones((2, 2, 2, 1)), "3D mask"),
        (np.zeros((2, 2, 2)), "<= 0"),
        (-np.ones((2, 2, 2)), "<= 0"),
    ],
)
@pytest.mark.parametrize("per_voxel", [False, True])
def test_invalid_masks_are_refused(
    mask: NDArray[np.float64], match: str, per_voxel: bool
) -> None:
    volume = Volume(mask, np.eye(4))
    with pytest.raises(ValueError, match=match):
        if per_voxel:
            seeds_per_voxel(volume, 1, np.random.default_rng(0))
        else:
            MaskSeeds(volume)


# Seeds files

LAYOUTS = [
    (False, False, "# x y z"),
    (True, False, "# index x y z"),
    (False, True, "# x y z dx dy dz"),
    (True, True, "# index x y z dx dy dz"),
]


def _seeds(n: int, has_index: bool, has_directions: bool, seed: int = 42) -> Seeds:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, 2**62, n)
    indices[0] = 2**53 + 1  # the first integer float64 cannot hold
    return Seeds(
        rng.normal(scale=80.0, size=(n, 3)),
        rng.normal(size=(n, 3)) if has_directions else None,
        indices if has_index else None,
    )


def _assert_same(a: Seeds, b: Seeds) -> None:
    np.testing.assert_array_equal(a.positions, b.positions)
    assert (a.directions is None) == (b.directions is None)
    assert (a.indices is None) == (b.indices is None)
    if a.directions is not None and b.directions is not None:
        np.testing.assert_array_equal(a.directions, b.directions)
    if a.indices is not None and b.indices is not None:
        assert a.indices.dtype == b.indices.dtype == np.int64
        assert a.indices.tolist() == b.indices.tolist()


@pytest.mark.parametrize(("has_index", "has_directions", "header"), LAYOUTS)
@pytest.mark.parametrize("n", [1, 200])
def test_seeds_file_reads_back_exactly(
    tmp_path: Path, has_index: bool, has_directions: bool, header: str, n: int
) -> None:
    path = tmp_path / "seeds.txt"
    seeds = _seeds(n, has_index, has_directions)
    write_seeds(seeds, path)
    lines = path.read_text().splitlines()
    assert lines[0] == header
    assert len(lines) == n + 1
    assert all(len(line.split()) == len(header.split()) - 1 for line in lines[1:])
    _assert_same(read_seeds(path), seeds)


@pytest.mark.parametrize(("has_index", "has_directions", "header"), LAYOUTS)
def test_header_only_seeds_file_gives_empty_seeds(
    tmp_path: Path, has_index: bool, has_directions: bool, header: str
) -> None:
    path = tmp_path / "seeds.txt"
    empty = Seeds(
        np.empty((0, 3)),
        np.empty((0, 3)) if has_directions else None,
        np.empty(0, np.int64) if has_index else None,
    )
    write_seeds(empty, path)
    assert path.read_text() == header + "\n"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        back = read_seeds(path)
    assert len(back) == 0
    assert back.positions.shape == (0, 3)
    assert (back.directions is not None) == has_directions
    assert (back.indices is not None) == has_index


@pytest.mark.parametrize(("has_index", "has_directions", "header"), LAYOUTS)
def test_batches_written_to_an_open_file_are_appended(
    tmp_path: Path, has_index: bool, has_directions: bool, header: str
) -> None:
    path = tmp_path / "seeds.txt"
    first = _seeds(3, has_index, has_directions, seed=1)
    second = _seeds(4, has_index, has_directions, seed=2)
    with open(path, "w", encoding="utf-8") as f:
        write_seeds(first, f)
        write_seeds(second, f, header=False)
    lines = path.read_text().splitlines()
    assert lines[0] == header
    assert sum(line.startswith("#") for line in lines) == 1
    back = read_seeds(path)
    expected = Seeds(
        np.concatenate([first.positions, second.positions]),
        None
        if first.directions is None or second.directions is None
        else np.concatenate([first.directions, second.directions]),
        None
        if first.indices is None or second.indices is None
        else np.concatenate([first.indices, second.indices]),
    )
    _assert_same(back, expected)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "1 2 3\n",
        "# a b c\n1 2 3\n",
        "# x y z index\n1 2 3 0\n",
        "# index x y z dx dy\n0 1 2 3 1 0\n",
    ],
)
def test_seeds_file_without_a_seed_header_is_refused(tmp_path: Path, text: str) -> None:
    path = tmp_path / "seeds.txt"
    path.write_text(text)
    with pytest.raises(ValueError, match="seed header"):
        read_seeds(path)


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("# x y z\n1 2 3 4\n", "expected 3 columns .* got 4"),
        ("# index x y z\n1 2 3\n", "expected 4 columns .* got 3"),
        ("# x y z dx dy dz\n1 2 3 0 0 1 5\n", "expected 6 columns .* got 7"),
        ("# index x y z dx dy dz\n0 1 2 3\n", "expected 7 columns .* got 4"),
    ],
)
def test_seeds_file_rows_must_match_the_header(
    tmp_path: Path, text: str, match: str
) -> None:
    path = tmp_path / "seeds.txt"
    path.write_text(text)
    with pytest.raises(ValueError, match=match):
        read_seeds(path)


def test_mask_seeds_are_drawn_where_the_scaled_values_are_positive() -> None:
    # Stored 1 and 2 with an intercept of -1: the scaled values are 0 and 1, so
    # only the centre voxel is in the mask, although every stored value is positive.
    data = np.ones((3, 3, 3), dtype=np.uint8)
    data[1, 1, 1] = 2
    mask = Volume(data, np.eye(4), intercept=-1.0)
    voxels = MaskSeeds(mask).draw(50, np.random.default_rng(0)).positions
    assert np.all(np.abs(voxels - 1.0) <= 0.5)
