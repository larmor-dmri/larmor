import numpy as np
import pytest

from larmor.types import Volume
from larmor.viewer import Category, Colouring, Item, OptionName, Viewer, ViewKind
from larmor.viewer.render import peaks_adapter
from larmor.viewer.render.geometry import peak_segments
from larmor.viewer.render.peaks_adapter import PeaksAdapter
from larmor.viewer.render.view import ViewRenderer

AXIS: dict[ViewKind, int] = {
    ViewKind.SAGITTAL: 0,
    ViewKind.CORONAL: 1,
    ViewKind.AXIAL: 2,
}


@pytest.fixture
def builds(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, int]]:
    # The (voxel axis, index) of every slice built.
    built: list[tuple[int, int]] = []

    def record(
        peaks: np.ndarray,
        affine: np.ndarray,
        axis: int,
        index: int,
        length: float,
        reference: float | None,
        threshold: float,
        slope: float,
        intercept: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        built.append((axis, index))
        return peak_segments(
            peaks, affine, axis, index, length, reference, threshold, slope, intercept
        )

    monkeypatch.setattr(peaks_adapter, "peak_segments", record)
    return built


@pytest.fixture
def item(viewer: Viewer) -> Item:
    # A unit peak along x in every voxel of a 4 mm cube; the cursor starts at 2, 2, 2.
    peaks = np.zeros((4, 4, 4, 3), np.float32)
    peaks[..., 0] = 1.0
    return viewer.add(Volume(peaks, np.eye(4)), Category.PEAKS)


def _adapter(
    viewer: Viewer, views: dict[ViewKind, ViewRenderer], item: Item
) -> PeaksAdapter:
    adapter = PeaksAdapter(item, views, viewer, 0.0)
    viewer.scene.subscribe(adapter.on_event)
    return adapter


def test_each_view_draws_the_peaks_of_its_slice_at_the_cursor(
    viewer: Viewer, views: dict[ViewKind, ViewRenderer], item: Item
) -> None:
    adapter = _adapter(viewer, views, item)
    viewer.move_cursor((1.0, 3.0, 0.0))
    for kind, axis in AXIS.items():
        positions = adapter.lines[kind].geometry.positions.data
        centres = positions.reshape(-1, 2, 3).mean(axis=1)
        assert len(centres) == 16
        np.testing.assert_allclose(centres[:, axis], viewer.scene.view.cursor[axis])


@pytest.mark.parametrize(
    ("cursor", "rebuilt"),
    [
        ((2.3, 2.0, 2.0), []),
        ((3.0, 2.0, 2.0), [(0, 3)]),
        ((3.0, 1.0, 2.0), [(0, 3), (1, 1)]),
    ],
    ids=["same voxel", "x", "x and y"],
)
def test_a_cursor_move_rebuilds_only_the_views_whose_slice_changed(
    cursor: tuple[float, float, float],
    rebuilt: list[tuple[int, int]],
    viewer: Viewer,
    views: dict[ViewKind, ViewRenderer],
    item: Item,
    builds: list[tuple[int, int]],
) -> None:
    _adapter(viewer, views, item)
    builds.clear()
    viewer.move_cursor(cursor)
    assert sorted(builds) == rebuilt


@pytest.mark.parametrize(
    ("name", "value", "rebuilt"),
    [
        (OptionName.SCALE, 2.0, True),
        (OptionName.AMPLITUDE, False, True),
        (OptionName.THRESHOLD, 0.5, True),
        (OptionName.ASYMMETRIC_PEAKS, True, True),
        (OptionName.COLOUR, "#ff0000", False),
        (OptionName.COLOURING, Colouring.SINGLE, False),
        (OptionName.LINE_WIDTH, 4.0, False),
        (OptionName.OPACITY_2D, 0.5, False),
    ],
)
def test_only_options_that_shape_the_segments_rebuild_them(
    name: OptionName,
    value: object,
    rebuilt: bool,
    viewer: Viewer,
    views: dict[ViewKind, ViewRenderer],
    item: Item,
    builds: list[tuple[int, int]],
) -> None:
    _adapter(viewer, views, item)
    builds.clear()
    viewer.set(item.id, name, value)
    assert sorted(builds) == ([(0, 2), (1, 2), (2, 2)] if rebuilt else [])


@pytest.mark.parametrize("hidden", ["before", "after"])
def test_a_hidden_item_builds_nothing_until_shown(
    hidden: str,
    viewer: Viewer,
    views: dict[ViewKind, ViewRenderer],
    item: Item,
    builds: list[tuple[int, int]],
) -> None:
    if hidden == "before":
        viewer.set(item.id, OptionName.VISIBLE, False)
        _adapter(viewer, views, item)
    else:
        _adapter(viewer, views, item)
        builds.clear()
        viewer.set(item.id, OptionName.VISIBLE, False)
    viewer.move_cursor((3.0, 1.0, 0.0))
    viewer.set(item.id, OptionName.SCALE, 2.0)
    assert builds == []
    viewer.set(item.id, OptionName.VISIBLE, True)
    assert sorted(builds) == [(0, 3), (1, 1), (2, 0)]


def test_the_threshold_is_a_share_of_the_largest_amplitude(
    viewer: Viewer, views: dict[ViewKind, ViewRenderer]
) -> None:
    # Amplitude 1 everywhere but one voxel of amplitude 4: at 0.5 only that one stays.
    peaks = np.zeros((4, 4, 4, 3), np.float32)
    peaks[..., 0] = 1.0
    peaks[2, 2, 2, 0] = 4.0
    item = viewer.add(Volume(peaks, np.eye(4)), Category.PEAKS)
    adapter = _adapter(viewer, views, item)
    viewer.set(item.id, OptionName.THRESHOLD, 0.5)
    positions = adapter.lines[ViewKind.AXIAL].geometry.positions.data
    assert len(positions) == 2


@pytest.mark.parametrize(
    ("stored", "offsets"),
    [
        ([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], [[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]),
        ([[0.0, -1.0, 0.0]], [[0.0, -0.5, 0.0]]),
    ],
    ids=["peak and opposite", "one-sided"],
)
def test_asymmetric_peaks_run_from_the_voxel_centre_along_the_signed_direction(
    stored: list[list[float]],
    offsets: list[list[float]],
    viewer: Viewer,
    views: dict[ViewKind, ViewRenderer],
) -> None:
    peaks = np.zeros((4, 4, 4, 3 * len(stored)), np.float32)
    peaks[...] = np.ravel(stored)
    item = viewer.add(Volume(peaks, np.eye(4)), Category.PEAKS)
    adapter = _adapter(viewer, views, item)
    viewer.set(item.id, OptionName.ASYMMETRIC_PEAKS, True)
    pairs = adapter.lines[ViewKind.AXIAL].geometry.positions.data.reshape(-1, 2, 3)
    starts, steps = pairs[:, 0], pairs[:, 1] - pairs[:, 0]
    assert len(pairs) == 16 * len(stored)
    # Identity affine on the axial slice at the cursor: centres are (i, j, 2).
    np.testing.assert_array_equal(starts, np.round(starts))
    np.testing.assert_array_equal(starts[:, 2], 2.0)
    centres, counts = np.unique(starts, axis=0, return_counts=True)
    assert len(centres) == 16
    assert set(counts) == {len(stored)}
    # Length 1 mm for the largest peak, so a half-line is 0.5 mm.
    np.testing.assert_allclose(np.unique(steps, axis=0), offsets, atol=1e-6)


def test_switching_asymmetric_peaks_off_restores_the_full_lines(
    viewer: Viewer, views: dict[ViewKind, ViewRenderer], item: Item
) -> None:
    adapter = _adapter(viewer, views, item)
    full = adapter.lines[ViewKind.AXIAL].geometry.positions.data.copy()
    viewer.set(item.id, OptionName.ASYMMETRIC_PEAKS, True)
    halves = adapter.lines[ViewKind.AXIAL].geometry.positions.data.copy()
    viewer.set(item.id, OptionName.ASYMMETRIC_PEAKS, False)
    restored = adapter.lines[ViewKind.AXIAL].geometry.positions.data
    assert not np.array_equal(halves, full)
    np.testing.assert_array_equal(restored, full)
