from dataclasses import replace

import matplotlib
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from larmor.plot import plot_signal_breakdown
from larmor.simulation import (
    Bundle,
    BundleGeometry,
    Phantom,
    Protocol,
    SignalBreakdown,
    TissueMap,
    create_gradient_table,
    partition_voxels,
    signal_breakdown,
)
from larmor.types import Grid, Volume

GRID = Grid((12, 12, 12), np.eye(4))
PROTOCOL = Protocol(
    create_gradient_table([(3000, 6)], num_b0=1, seed=0), 15.0, 30.0, 80.0, 3000.0
)


@pytest.fixture(scope="module")
def breakdowns() -> tuple[SignalBreakdown, SignalBreakdown]:
    """A voxel of WM over GM at b = 3000 with noise, and the same voxel at b = 0."""
    line = BundleGeometry(
        np.array([[0.0, 6, 6], [6, 6, 6], [11, 6, 6]]), radius=1.3, seed=1
    )
    gm = Volume(np.full(GRID.shape, 0.5), np.eye(4))
    phantom = Phantom(GRID, bundles=(Bundle(line),), tissue_maps=(TissueMap("GM", gm),))
    partition = partition_voxels(phantom)
    voxel = (6, 7, 6)
    after = signal_breakdown(phantom, PROTOCOL, partition, voxel, 3, snr=20)
    before = signal_breakdown(phantom, PROTOCOL, partition, voxel, 0)
    return after, before


def bars(ax: Axes) -> list[tuple[float, float, float]]:
    return [
        (p.get_x() + p.get_width() / 2, p.get_y(), p.get_height())
        for container in ax.containers
        for p in container
    ]


def test_three_bars_stack_the_compartments_at_each_step(
    breakdowns: tuple[SignalBreakdown, SignalBreakdown],
) -> None:
    breakdown, _ = breakdowns
    ax = plot_signal_breakdown(breakdown, Figure().subplots())
    shown = [c for c in breakdown.compartments if c.no_decay > 0]
    found = bars(ax)
    assert len(found) == 3 * len(shown)
    for step, total in enumerate(
        (breakdown.no_decay, breakdown.after_decay, breakdown.signal)
    ):
        heights = [h for x, _, h in found if x == step]
        assert sum(heights) == pytest.approx(total)
    labels = ax.get_legend_handles_labels()[1]
    assert labels == ["Noise floor"] + [c.label for c in shown]
    assert "GM soma" in labels and "WM axons" in labels
    ticks = [t.get_text() for t in ax.get_xticklabels()]
    assert ticks == ["No decay", "T1/T2 decay\nTE 80 ms", "Diffusion\nb 3000"]
    # Only the totals are written on the chart, above their bars.
    totals = [breakdown.no_decay, breakdown.after_decay, breakdown.signal]
    assert [t.get_text() for t in ax.texts] == [f"{t:.3g}" for t in totals]


def test_in_a_narrow_figure_the_legend_and_tick_labels_do_not_overlap(
    breakdowns: tuple[SignalBreakdown, SignalBreakdown],
) -> None:
    figure = Figure(figsize=(3.8, 5.6), dpi=100, layout="constrained")  # 380 px
    ax = plot_signal_breakdown(breakdowns[0], figure.subplots())
    canvas = FigureCanvasAgg(figure)
    canvas.draw()  # type: ignore[no-untyped-call]
    renderer = canvas.get_renderer()  # type: ignore[no-untyped-call]
    legend = ax.get_legend()
    assert legend is not None
    box = legend.get_window_extent(renderer)
    assert not box.overlaps(ax.get_window_extent(renderer))
    ticks = [t.get_window_extent(renderer) for t in ax.get_xticklabels()]
    assert all(not box.overlaps(t) for t in ticks)
    assert all(not a.overlaps(b) for a, b in zip(ticks, ticks[1:], strict=False))
    assert box.x0 >= 0 and box.x1 <= figure.bbox.width  # inside the figure
    # The noise floor is the last entry.
    assert [t.get_text() for t in legend.get_texts() if t.get_text()][-1] == (
        "Noise floor"
    )


def test_two_bundles_fill_one_legend_column_each(
    breakdowns: tuple[SignalBreakdown, SignalBreakdown],
) -> None:
    breakdown, _ = breakdowns
    first = [c for c in breakdown.compartments if c.bundle == 0 and c.no_decay > 0]
    second = [replace(c, bundle=1) for c in first]
    bundles = (breakdown.bundles[0], replace(breakdown.bundles[0], index=1, name="B"))
    crossing = replace(
        breakdown, compartments=(*first, *second), bundles=bundles, sigma=None
    )
    ax = plot_signal_breakdown(crossing, Figure().subplots())
    legend = ax.get_legend()
    assert legend is not None
    texts = [t.get_text() for t in legend.get_texts()]
    half = len(texts) // 2
    assert all(t.endswith(")") and "(B)" not in t for t in texts[:half])
    assert all(t.endswith("(B)") for t in texts[half:])


def test_the_noise_floor_is_a_dashed_line(
    breakdowns: tuple[SignalBreakdown, SignalBreakdown],
) -> None:
    breakdown, _ = breakdowns
    ax = plot_signal_breakdown(breakdown, Figure().subplots())
    (line,) = ax.get_lines()
    assert line.get_linestyle() == "--"
    assert np.asarray(line.get_ydata())[0] == pytest.approx(breakdown.noise_floor)


def test_new_axes_are_made_without_any(
    breakdowns: tuple[SignalBreakdown, SignalBreakdown],
) -> None:
    matplotlib.use("Agg")  # no window in a test
    ax = plot_signal_breakdown(breakdowns[0])
    assert isinstance(ax, Axes)
    assert len(bars(ax)) > 0
