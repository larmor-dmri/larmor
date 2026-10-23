"""The `lar info` command, which prints a summary of an image or a tractogram."""

from pathlib import Path

import numpy as np

from larmor.cli.options import InfoArg
from larmor.io import (
    VOLUME_EXTENSIONS,
    file_kind,
    has_mif_gradients,
    load_header,
    load_tractogram,
)
from larmor.io.formats import read_mif_gradients
from larmor.types import Grid

type Fields = list[tuple[str, str]]


def _print_fields(fields: Fields) -> None:
    """Prints `label: value` lines with the values aligned.

    An empty label continues the line above.
    """
    width = max(len(label) for label, _ in fields) + 2
    for label, value in fields:
        print(f"{label + ':' if label else '':<{width}}{value}")


def _grid_fields(grid: Grid) -> Fields:
    """Returns the voxel size, orientation and affine of a grid as fields."""
    sizes = np.linalg.norm(grid.affine[:3, :3], axis=0)
    rows = [" ".join(f"{v:10.4f}" for v in row) for row in grid.affine[:3]]
    return [
        ("Voxel size", " x ".join(f"{s:g}" for s in sizes) + " mm"),
        ("Orientation", "".join(grid.orientation)),
        ("Affine", rows[0]),
        ("", rows[1]),
        ("", rows[2]),
    ]


def _data_names(data: dict[str, np.ndarray]) -> str:
    """Returns the names of point or streamline data with their sizes, or "none"."""
    names = [f"{name} ({v.shape[1] if v.ndim == 2 else 1})" for name, v in data.items()]
    return ", ".join(names) or "none"


def _print_volume(p: Path) -> None:
    """Prints a volume's info, from its header only."""
    grid, shape = load_header(p)
    fields = [("File", str(p)), ("Shape", " x ".join(str(n) for n in shape))]
    fields += _grid_fields(grid)
    if p.name.endswith(VOLUME_EXTENSIONS["mif"]) and has_mif_gradients(p):
        fields += _gradient_fields(p)
    _print_fields(fields)


def _gradient_fields(p: Path) -> Fields:
    """Returns the b0 count and each shell's b-value and volumes of a MIF's table.

    A table that cannot be read is reported in its field, so the rest of the header
    still prints.

    Returns:
        The (label, value) pairs to print.
    """
    try:
        gtab = read_mif_gradients(p)
    except ValueError as e:
        return [("Gradients", f"unreadable: {e}")]
    counts = [int((gtab.shell_index == k).sum()) for k in range(len(gtab.shells))]
    shells = ", ".join(f"{b:g} ({n})" for b, n in zip(gtab.shells, counts, strict=True))
    return [("b0 volumes", str(int(gtab.b0_mask.sum()))), ("Shells", shells or "none")]


def _print_tractogram(p: Path) -> None:
    """Prints a tractogram's info; the whole file is read."""
    t = load_tractogram(p)
    lengths = np.diff(t.offsets)
    fields = [
        ("File", str(p)),
        ("Streamlines", f"{t.num_streamlines:,}"),
        ("Points", f"{t.num_points:,}"),
    ]
    if t.num_streamlines > 0:
        per_line = f"{lengths.min()} / {lengths.mean():.1f} / {lengths.max()}"
        fields.append(("Points per line", per_line + " (min / mean / max)"))
    fields += [
        ("Point data", _data_names(t.point_data)),
        ("Streamline data", _data_names(t.streamline_data)),
    ]
    if t.reference is None:
        fields.append(("Reference", "none"))
    else:
        fields.append(("Reference", " x ".join(str(n) for n in t.reference.shape)))
        fields += _grid_fields(t.reference)
    _print_fields(fields)


def info(path: InfoArg) -> None:
    """Print a short summary of an image or a tractogram.

    For an image, the summary comes from the header: its shape, voxel size,
    orientation and affine, and the shells of a MIF gradient table. For a
    tractogram, the whole file is read to count its streamlines and points and
    list the data stored with them.
    """
    if file_kind(path) == "volume":
        _print_volume(path)
    else:
        _print_tractogram(path)
