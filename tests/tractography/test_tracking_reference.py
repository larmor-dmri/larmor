# The compiled tracking loop against the numpy reference in reference_tracking.py, on
# varied seeds, grids, masks and dummy steps. The compiled float32 points must equal the
# reference's float64 points rounded to float32, exactly.
import math
import os
from dataclasses import replace

import numpy as np
import pytest
from numpy.typing import NDArray
from reference_tracking import (
    Scene,
    Streamline,
    assert_same,
    track_core,
    track_reference,
)

from larmor.tractography import EndStatus

Vector = NDArray[np.float64]


def run_reference(
    scene: Scene, seeds: Vector, directions: Vector | None, indices: NDArray[np.int64]
) -> list[Streamline | None]:
    return [
        track_reference(scene, s, None if directions is None else directions[i], int(j))
        for i, (s, j) in enumerate(zip(seeds, indices, strict=True))
    ]


def _rotation(axis: Vector, degrees: float) -> Vector:
    axis = axis / np.linalg.norm(axis)
    r = math.radians(degrees)
    k = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    out: Vector = np.eye(3) + math.sin(r) * k + (1 - math.cos(r)) * (k @ k)
    return out


def _oblique_affine() -> Vector:
    affine = np.eye(4)
    affine[:3, :3] = 2.0 * _rotation(np.array([1.0, 2.0, 2.0]), 35.0)
    affine[:3, 3] = [-12.0, 7.0, 3.5]
    return affine


FOD_SHAPE = (12, 12, 12)
AFFINES = {"identity": np.eye(4), "oblique": _oblique_affine()}
STEPS: dict[str, tuple[str, EndStatus | None]] = {
    "straight": ("straight", None),
    "quarter-turn": ("quarter_turn", None),
    "wander": ("wander", None),
    "low-fod": ("refuse", EndStatus.LOW_FOD),
    "sharp-turn": ("refuse", EndStatus.SHARP_TURN),
    "no-direction": ("refuse", EndStatus.NO_DIRECTION),
}


def _mask(fod_affine: Vector) -> tuple[NDArray[np.float32], Vector]:
    # A ball on its own grid: 1.5 FOD voxels per mask voxel, rotated and offset.
    to_fod = np.eye(4)
    to_fod[:3, :3] = 1.5 * _rotation(np.array([2.0, -1.0, 1.0]), 20.0)
    to_fod[:3, 3] = [1.0, 0.5, 1.5]
    grid = np.stack(np.meshgrid(*[np.arange(6)] * 3, indexing="ij"), axis=-1)
    ball = np.linalg.norm(grid - 2.5, axis=-1) < 2.6
    mask = np.ascontiguousarray(ball, dtype=np.float32)
    return mask, fod_affine @ to_fod


def _seeds(fod_affine: Vector, n: int = 40) -> tuple[Vector, Vector]:
    # Voxel coordinates beyond the grid's edge on every side, so some do not start.
    rng = np.random.default_rng(7)
    vox = rng.uniform(-1.5, 12.5, size=(n, 3))
    seeds = vox @ fod_affine[:3, :3].T + fod_affine[:3, 3]
    d = rng.normal(size=(n, 3))
    return seeds, d / np.linalg.norm(d, axis=1, keepdims=True)


def _scene(
    affine: str,
    step: str,
    masked: bool,
    bidirectional: bool,
    max_steps: int,
    search_refuses_past_x: float = math.inf,
) -> Scene:
    fod_affine = AFFINES[affine]
    mask, mask_affine = _mask(fod_affine)
    name, stop = STEPS[step]
    return Scene(
        fod=np.ones((*FOD_SHAPE, 1), dtype=np.float32),
        fod_affine=fod_affine,
        mask=mask if masked else None,
        mask_affine=mask_affine,
        step_size=0.7 * float(np.linalg.norm(fod_affine[:3, 0])),
        max_steps=max_steps,
        bidirectional=bidirectional,
        step=name,
        stop=stop,
        search_refuses_past_x=search_refuses_past_x,
    )


INDICES = np.arange(100, 140, dtype=np.int64)


@pytest.mark.parametrize("searched", [False, True])
@pytest.mark.parametrize("max_steps", [5, 60])
@pytest.mark.parametrize("bidirectional", [True, False])
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("affine", list(AFFINES))
@pytest.mark.parametrize("step", list(STEPS))
def test_compiled_loop_matches_the_reference(
    step: str,
    affine: str,
    masked: bool,
    bidirectional: bool,
    max_steps: int,
    searched: bool,
) -> None:
    seeds, directions = _seeds(AFFINES[affine])
    # Searched seeds past the median x find no direction, so the search refuses some.
    limit = float(np.median(seeds[:, 0])) if searched else math.inf
    scene = _scene(affine, step, masked, bidirectional, max_steps, limit)
    given = None if searched else directions
    reference = run_reference(scene, seeds, given, INDICES)
    started = [r for r in reference if r is not None]
    assert 0 < len(started) < len(reference)
    assert_same(track_core(scene, seeds, given, INDICES), reference)


def test_the_comparison_scenes_reach_every_end() -> None:
    seen: set[EndStatus] = set()
    for step in STEPS:
        for masked, max_steps in ((False, 5), (True, 60)):
            scene = _scene("oblique", step, masked, True, max_steps)
            seeds, directions = _seeds(scene.fod_affine)
            for r in run_reference(scene, seeds, directions, INDICES):
                if r is not None:
                    seen.update(r.ends)
    assert seen == set(EndStatus) - {EndStatus.NONE}


# Random numbers in a batch


def _random_scene() -> Scene:
    return _scene("identity", "wander", False, True, 60)


def _many_seeds(n: int) -> tuple[Vector, Vector]:
    rng = np.random.default_rng(11)
    seeds = rng.uniform(1.0, 10.0, size=(n, 3))
    d = rng.normal(size=(n, 3))
    return seeds, d / np.linalg.norm(d, axis=1, keepdims=True)


def _same(x: Streamline | None, y: Streamline | None) -> bool:
    if x is None or y is None:
        return x is y
    return (
        x.points.shape == y.points.shape
        and np.array_equal(x.points, y.points)
        and x.ends == y.ends
    )


@pytest.mark.parametrize("searched", [False, True])
def test_batch_does_not_depend_on_the_number_of_threads(searched: bool) -> None:
    most = os.cpu_count() or 1
    if most < 2:
        pytest.skip("Needs more than one core.")
    scene = _random_scene()
    seeds, directions = _many_seeds(400)
    given = None if searched else directions
    indices = np.arange(400, dtype=np.int64)
    one = track_core(scene, seeds, given, indices, n_threads=1)
    many = track_core(scene, seeds, given, indices, n_threads=most)
    assert len([s for s in one if s and len(s.points) > 3]) > 300
    assert all(_same(x, y) for x, y in zip(one, many, strict=True))
    for x, y in zip(one, many, strict=True):
        assert x is not None and y is not None
        np.testing.assert_array_equal(x.direction, y.direction)


def test_a_streamline_does_not_depend_on_its_place_in_the_batch() -> None:
    scene = _random_scene()
    seeds, directions = _many_seeds(30)
    indices = np.arange(50, 80, dtype=np.int64)
    order = np.random.default_rng(1).permutation(30)
    a = track_core(scene, seeds, directions, indices)
    b = track_core(scene, seeds[order], directions[order], indices[order])
    assert all(_same(a[i], y) for i, y in zip(order, b, strict=True))


@pytest.mark.parametrize(("offset", "run_seed"), [(30, 3), (0, 4)])
def test_streamlines_differ_with_the_global_index_or_the_run_seed(
    offset: int, run_seed: int
) -> None:
    scene = _random_scene()
    seeds, directions = _many_seeds(30)
    indices = np.arange(30, dtype=np.int64)
    a = track_core(scene, seeds, directions, indices)
    b = track_core(
        replace(scene, run_seed=run_seed), seeds, directions, indices + offset
    )
    # Two points (one step each way) do not depend on the random numbers.
    walked = [(x, y) for x, y in zip(a, b, strict=True) if x and len(x.points) > 3]
    assert len(walked) > 20
    assert not any(_same(x, y) for x, y in walked)
