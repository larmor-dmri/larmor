"""Plots the signal breakdown of a voxel as three stacked bars."""

from itertools import zip_longest

import matplotlib as mpl
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.legend import Legend
from matplotlib.patches import Rectangle
from matplotlib.transforms import offset_copy

from larmor.simulation import CompartmentSignal, PulseSequence, SignalBreakdown

# The cool colours of each bundle, one per compartment of its tissue.
BUNDLE_COLOURS = (
    ("#4cc9f0", "#4895ef", "#90e0ef", "#3f37c9"),
    ("#80ed99", "#38b000", "#c7f9cc", "#007f5f"),
    ("#c77dff", "#7b2cbf", "#e0aaff", "#5a189a"),
    ("#2ec4b6", "#118ab2", "#9ef0e6", "#073b4c"),
)
# The warm colours of each tissue outside bundles, by the tissue's place in the
# phantom.
TISSUE_COLOURS = (
    ("#f4845f", "#f79d65", "#f7b267", "#e85d04"),
    ("#b5179e", "#f72585", "#ff9f1c", "#7209b7"),
    ("#ffd166", "#fcbf49", "#ffe8a3", "#e9c46a"),
    ("#ef476f", "#d62246", "#ff8fab", "#a4133c"),
)
# The colour and legend label of the noise floor line.
NOISE_COLOUR = "#ef476f"
NOISE_LABEL = "Noise floor"
# The width of a bar, in steps.
BAR_WIDTH = 0.6
# The share of the tallest bar left free above it, for its total.
HEADROOM = 0.14
# How far the legend's top is below the axes, in points.
LEGEND_DROP = 36


def plot_signal_breakdown(
    breakdown: SignalBreakdown,
    ax: Axes | None = None,
) -> Axes:
    """Draws a voxel's signal breakdown as three stacked bars.

    The bars are the signal of the water in the voxel with no decay, after the
    T1 and T2 (or T2*) decay at the echo time, and after the diffusion weighting.
    Each bar is split into the voxel's compartments and has its total written
    above it. With noise, the noise floor is a dashed line.

    Args:
        breakdown: The breakdown, from `larmor.simulation.signal_breakdown`.
        ax: The axes to draw on. Defaults to None, a new figure and axes.

    Returns:
        The axes drawn on.
    """
    if ax is None:
        import matplotlib.pyplot as plt

        _, ax = plt.subplots(figsize=(6, 4.5), layout="constrained")
    muted = mpl.rcParams["xtick.color"]
    totals = _bars(ax, breakdown)
    top = max(max(totals), 1e-12)
    for step, total in enumerate(totals):
        ax.text(
            step,
            total + 0.02 * top,
            f"{total:.3g}",
            ha="center",
            va="bottom",
            color=muted,
        )
    floor = breakdown.noise_floor
    if floor is not None:
        # Named in the legend, since a label on the line would cross a bar.
        ax.axhline(
            floor,
            color=NOISE_COLOUR,
            linewidth=1,
            linestyle="--",
            label=NOISE_LABEL,
        )
        top = max(top, floor)

    decay = "T2*" if breakdown.sequence == PulseSequence.GRADIENT_ECHO else "T2"
    ax.set_xticks(
        range(3),
        [
            "No decay",
            f"T1/{decay} decay\nTE {breakdown.te_ms:g} ms",
            f"Diffusion\nb {breakdown.b_value:g}",
        ],
    )
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(0, top * (1 + HEADROOM))
    ax.set_ylabel("Signal")
    ax.grid(axis="y", linewidth=0.5, alpha=0.4)
    ax.set_axisbelow(True)
    _legend(ax, breakdown)
    return ax


def _legend(ax: Axes, breakdown: SignalBreakdown) -> None:
    """Puts the legend below the axes, a bundle per column when there are two."""
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    shown = _shown(breakdown)
    # Lines come before bars in the handles, so the noise floor is moved last.
    entries = list(zip(handles, labels, strict=True))
    rest = [e for e in entries if e[1] == NOISE_LABEL]
    bars = [e for e in entries if e[1] != NOISE_LABEL]
    bundles = sorted({c.bundle for c, _ in shown if c.bundle is not None})
    rows: list[tuple[Artist, str] | None] = []
    others = bars
    if len(bundles) == 2:
        columns = [
            [e for e, (c, _) in zip(bars, shown, strict=True) if c.bundle == b]
            for b in bundles
        ]
        for pair in zip_longest(*columns):
            rows.extend(pair)
        others = [e for e, (c, _) in zip(bars, shown, strict=True) if c.bundle is None]
    rows += [*others, *rest]
    if len(rows) <= 3 and len(bundles) != 2:
        _place_legend(ax, [*bars, *rest], 1)
        return
    if len(rows) % 2:
        rows.append(None)
    # matplotlib fills a column at a time, so the rows are dealt into columns.
    legend = _place_legend(ax, rows[0::2] + rows[1::2], 2)
    figure = ax.get_figure(root=True)
    get_renderer = getattr(figure.canvas, "get_renderer", None) if figure else None
    if figure is not None and get_renderer is not None:
        width = legend.get_window_extent(get_renderer()).width
        if width > figure.bbox.width:  # too narrow for two columns
            legend.remove()
            _place_legend(ax, [*bars, *rest], 1)


def _place_legend(
    ax: Axes, entries: list[tuple[Artist, str] | None], ncols: int
) -> Legend:
    """Places a legend of `entries` below the axes; None leaves a blank slot."""
    blank = Rectangle((0, 0), 0, 0, visible=False)
    filled = [(blank, "") if e is None else e for e in entries]
    return ax.legend(
        [h for h, _ in filled],
        [label for _, label in filled],
        fontsize="small",
        frameon=False,
        ncols=ncols,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.0),
        bbox_transform=offset_copy(
            ax.transAxes, fig=ax.get_figure(root=True), y=-LEGEND_DROP, units="points"
        ),
    )


def _bars(ax: Axes, breakdown: SignalBreakdown) -> list[float]:
    """Draws the stacked bars and returns the total of each."""
    shown = _shown(breakdown)
    named = len(breakdown.bundles) > 1
    names = {b.index: b.name for b in breakdown.bundles}
    tissues = [name for name, _ in breakdown.tissue_fractions]
    totals = []
    for step in range(3):
        bottom = 0.0
        for c, place in shown:
            height = _height(c, step)
            label = c.label
            if named and c.bundle is not None:
                label = f"{label} ({names[c.bundle]})"
            ax.bar(
                step,
                height,
                bottom=bottom,
                width=BAR_WIDTH,
                color=_colour(c, place, tissues),
                label=label if step == 0 else None,
            )
            bottom += height
        totals.append(bottom)
    return totals


def _shown(breakdown: SignalBreakdown) -> list[tuple[CompartmentSignal, int]]:
    """Returns the compartments with water, each with its place in its tissue."""
    shown = []
    place: dict[tuple[str, int | None], int] = {}
    for c in breakdown.compartments:
        key = (c.tissue, c.bundle)
        place[key] = place.get(key, -1) + 1
        if c.no_decay > 0:
            shown.append((c, place[key]))
    return shown


def _height(c: CompartmentSignal, step: int) -> float:
    """Returns the signal of compartment `c` at bar `step`."""
    return (c.no_decay, c.after_decay, c.signal)[step]


def _colour(c: CompartmentSignal, place: int, tissues: list[str]) -> str:
    """Returns the colour of compartment `c`, the `place`-th of its tissue.

    A bundle's colours go by its index, a tissue's by its place in `tissues`.
    """
    if c.bundle is not None:
        family = BUNDLE_COLOURS[c.bundle % len(BUNDLE_COLOURS)]
    else:
        family = TISSUE_COLOURS[tissues.index(c.tissue) % len(TISSUE_COLOURS)]
    return family[place % len(family)]
