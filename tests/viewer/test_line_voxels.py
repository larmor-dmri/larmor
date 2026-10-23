import numpy as np
import occlusion_oracle as oracle
import pytest

from larmor.viewer.render.geometry import Lines, direction_colours
from larmor.viewer.render.line_voxels import (
    LineVoxels,
    ambient_light,
    grid_for,
    segment_starts,
)

SIZE = 48
GPU_SIZE = 64


def _walks(count: int, length: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    start = rng.uniform(10, 38, (count, 1, 3))
    walk = start + np.cumsum(rng.normal(0, 2.5, (count, length, 3)), axis=1)
    offsets = np.arange(0, count * length + 1, length, dtype=np.int64)
    return walk.reshape(-1, 3), offsets


def _lines(count: int, length: int, seed: int) -> Lines:
    points, offsets = _walks(count, length, seed)
    points = points.astype(np.float32)
    return Lines(points, offsets, direction_colours(points, offsets))


@pytest.fixture
def built(gpu: None) -> tuple[Lines, LineVoxels]:
    lines = _lines(300, 12, seed=3)
    voxels = LineVoxels(GPU_SIZE)
    voxels.build(lines, 0.6)
    return lines, voxels


@pytest.mark.parametrize("radius", [0.2, 1.3])
def test_the_voxelization_misses_no_voxel_a_capsule_reaches(radius: float) -> None:
    # The reference against every voxel of a box around each segment, wider than
    # any capsule.
    points, offsets = _walks(20, 8, seed=int(radius * 10))
    tangent = oracle.tangents(points, offsets)
    expected = np.zeros(SIZE**3)
    for i in segment_starts(offsets, len(points)):
        low = np.floor(np.minimum(points[i], points[i + 1]) - radius - 2).astype(int)
        high = np.ceil(np.maximum(points[i], points[i + 1]) + radius + 2).astype(int)
        axes = [np.arange(a, b) for a, b in zip(low, high, strict=True)]
        box = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
        box = box[np.all((box >= 0) & (box < SIZE), axis=1)]
        ends = [np.repeat(v[None], len(box), 0) for v in (points[i], points[i + 1])]
        normals = [np.repeat(v[None], len(box), 0) for v in tangent[i : i + 2]]
        filled = oracle._capsule_occupancy(
            box + 0.5, ends[0], ends[1], normals[0], normals[1], radius
        )
        np.add.at(expected, np.ravel_multi_index(box.T, (SIZE,) * 3), filled)
    expected = np.minimum(expected, 1).reshape((SIZE,) * 3)
    got = oracle.occupancy(points, offsets, radius, SIZE)
    assert got == pytest.approx(expected, abs=1e-6)
    assert got.sum() > 0


def test_a_straight_capsule_fills_about_its_cross_section_per_voxel() -> None:
    points = np.array([[5.3, 20.1, 30.7], [40.3, 31.4, 25.2]])
    length = np.linalg.norm(points[1] - points[0])
    for radius in (0.2, 1.5):
        filled = oracle.occupancy(points, np.array([0, 2]), radius, SIZE).sum()
        # The occupancy ramps over one voxel across the surface: pi (w^2 + 1/12)
        # for a radius w of at least half a voxel; a thinner one is widened to
        # that and fades by (r / w)^2.
        wide = max(radius, 0.5)
        expected = np.pi * (wide**2 + 1 / 12) * (radius / wide) ** 2
        assert filled / length == pytest.approx(expected, rel=0.05)


def test_the_gpu_grids_match_the_reference(built: tuple[Lines, LineVoxels]) -> None:
    lines, voxels = built
    grid = grid_for(lines.points, GPU_SIZE)
    at = grid.to_voxels(lines.points)
    base, levels, light = oracle.light_grids(
        at, lines.offsets, 0.6 / grid.voxel, GPU_SIZE
    )
    gpu = voxels.grid_values(voxels.occupancy)
    # Summed in fixed point: shares below 1 / 65536 may round to nothing.
    assert gpu == pytest.approx(base, abs=1e-4)
    assert ((gpu > 0) != (base > 0)).sum() <= 1e-3 * (base > 0).sum()
    for k in (1, 3, len(levels) - 1):
        assert voxels.grid_values(voxels.occupancy, k) == pytest.approx(
            levels[k], abs=1e-4
        )
    both = (gpu > 0) & (base > 0)
    gpu_light = voxels.grid_values(voxels.light)
    assert gpu_light[both] == pytest.approx(light[both], abs=1e-4)
    expected = oracle.read(light, base > 0, at)
    assert voxels.light_at_points() == pytest.approx(expected, abs=5e-3)


def test_each_voxel_lists_the_segments_that_touch_it(
    built: tuple[Lines, LineVoxels],
) -> None:
    lines, voxels = built
    grid = grid_for(lines.points, GPU_SIZE)
    at = grid.to_voxels(lines.points)
    starts = segment_starts(lines.offsets, len(at))
    radius = 0.6 / grid.voxel
    segment, cells = oracle._capsule_voxels(at[starts], at[starts + 1], radius)
    inside = np.all((cells >= 0) & (cells < GPU_SIZE), axis=1)
    flat = cells[:, 0] + GPU_SIZE * (cells[:, 1] + GPU_SIZE * cells[:, 2])
    expected = set(zip(segment[inside].tolist(), flat[inside].tolist(), strict=True))
    offsets = voxels.read(voxels.offsets, "u4")
    entries = voxels.read(voxels.lists, "u4")[: voxels.num_entries]
    lengths = np.diff(offsets.astype(np.int64)).astype(np.int64)
    owner = np.repeat(np.arange(GPU_SIZE**3), lengths)
    got = set(zip(entries.tolist(), owner.tolist(), strict=True))
    assert len(got) == voxels.num_entries == offsets[-1]  # each pair once
    # float32 on the GPU: a box's edge may fall either side of a voxel boundary.
    assert len(got ^ expected) <= 1e-3 * len(expected)
    touched = voxels.grid_values(voxels.touched, 0).transpose(2, 1, 0).ravel()
    assert np.array_equal(touched > 0, lengths > 0)


@pytest.mark.usefixtures("gpu")
def test_points_inside_a_bundle_get_less_light_than_a_lone_streamline() -> None:
    rng = np.random.default_rng(0)
    t = np.linspace(-30, 30, 30)
    bundle = [
        np.stack([t, *rng.normal(0, 1.5, 2)[:, None] + 0 * t], -1) for _ in range(400)
    ]
    lone = np.stack([t, 0 * t + 25, 0 * t + 25], -1)
    points = np.concatenate([*bundle, lone]).astype(np.float32)
    offsets = np.arange(0, len(points) + 1, 30, dtype=np.int64)
    light = ambient_light(
        Lines(points, offsets, direction_colours(points, offsets)), 0.25
    )
    assert light.shape == (len(points),) and light.dtype == np.float32
    assert 0 <= light.min() and light.max() <= 1
    radial = np.linalg.norm(points[:-30, 1:], axis=1)
    middle = light[:-30][(radial < 1) & (np.abs(points[:-30, 0]) < 20)]
    assert np.median(light[-30:]) == pytest.approx(1.0, abs=0.05)
    assert np.median(middle) < 0.6
