"""Streamline weights files: one value per streamline, as text."""

import re
from pathlib import Path

import numpy as np

# Values are separated by whitespace, commas or semicolons.
_SEPARATORS = re.compile(r"[\s,;]+")


def read_streamline_weights(path: str | Path) -> np.ndarray:
    """Reads a streamline weights file, such as those of tcksift2 and tckedit.

    The file holds one value per streamline, in the tractogram's order, on one row
    or one per row, separated by whitespace, commas or semicolons. Text after a `#`
    is a comment.

    Args:
        path: The text file.

    Returns:
        np.ndarray: The weights, (S,) float32; empty for a file with no values.

    Raises:
        ValueError: If the file cannot be read, a value is not a number, or the
            values form a table of several rows and columns; the message names
            the file and the line.

    References:
        Reads the weights files of MRtrix3 (tcksift2 output, tckedit
        -tck_weights_in). Tournier, J.-D. et al. MRtrix3: A fast, flexible and
        open software framework for medical image processing and visualisation.
        NeuroImage 202, 116137 (2019). doi:10.1016/j.neuroimage.2019.116137
    """
    p = Path(path)
    try:
        lines = p.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"{p} cannot be read as a weights file: {exc}") from exc

    rows: list[np.ndarray] = []
    for number, line in enumerate(lines, start=1):
        fields = [f for f in _SEPARATORS.split(line.split("#", 1)[0]) if f]
        if not fields:
            continue
        try:
            rows.append(np.array(fields, dtype=np.float64))
        except ValueError as exc:
            raise ValueError(f"{p}, line {number}: {exc}") from exc

    if len(rows) > 1 and any(len(row) > 1 for row in rows):
        raise ValueError(
            f"{p} holds a table of {len(rows)} rows; a weights file is one row or "
            "one column."
        )
    if not rows:
        return np.empty(0, dtype=np.float32)
    return np.concatenate(rows).astype(np.float32)


def write_streamline_weights(
    weights: np.ndarray, path: str | Path, *, append: bool = False
) -> None:
    """Writes streamline weights as text, for `read_streamline_weights` or MRtrix.

    The values are written on one row, as float32 with 9 significant digits, so
    they read back exactly. A file written in batches with `append` reads back as
    one array.

    Args:
        weights: (S,) the weight of each streamline, in the tractogram's order.
        path: The text file.
        append: If True, add the values at the end of the file, which must be one
            this function wrote; otherwise overwrite it.

    Raises:
        ValueError: If `weights` is not one-dimensional.

    References:
        Writes the layout of MRtrix3's tckedit -tck_weights_out. Tournier, J.-D. et
        al. MRtrix3: A fast, flexible and open software framework for medical image
        processing and visualisation. NeuroImage 202, 116137 (2019).
        doi:10.1016/j.neuroimage.2019.116137
    """
    values = np.asarray(weights, dtype=np.float32)
    if values.ndim != 1:
        raise ValueError(
            f"Expected one weight per streamline, shape (S,), got {values.shape}."
        )
    text = "".join(f"{value:.9g} " for value in values.tolist())
    with Path(path).open(
        "a" if append else "w", encoding="utf-8", newline="\n"
    ) as file:
        file.write(text)
