"""Lookup tables: the name and colour of each label of a parcellation."""

from pathlib import Path

from larmor.types import Label


def read_lookup_table(path: str | Path) -> dict[int, Label]:
    """Reads a lookup table in the FreeSurfer or the MRtrix layout.

    Each line holds an index, one or two names and the colour, from 0 to 255, as
    `index name R G B A` or `index short-name long-name R G B A`. The last column is
    ignored and may be missing. The name kept is the last one, MRtrix's long name.
    Lines starting with `#` are skipped; a later line for an index replaces an
    earlier one.

    Args:
        path: The text file, for example FreeSurfer's `FreeSurferColorLUT.txt`.

    Returns:
        The labels by their index (the value in the parcellation image).

    Raises:
        ValueError: If the file cannot be read, a line is not a valid entry, or the
            file holds no entry; the message names the file and the line.
    """
    p = Path(path)
    try:
        lines = p.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"{p} cannot be read as a lookup table: {exc}") from exc
    table: dict[int, Label] = {}
    for number, line in enumerate(lines, start=1):
        fields = line.split()
        if not fields or fields[0].startswith("#"):
            continue
        try:
            if len(fields) < 5:
                raise ValueError("expected index, name, R, G, B")
            index = int(fields[0])
            # With 5 fields there is no last column; otherwise R G B come before it.
            colour = fields[2:5] if len(fields) == 5 else fields[-4:-1]
            r, g, b = (int(v) for v in colour)
            if not all(0 <= c <= 255 for c in (r, g, b)):
                raise ValueError("colours must be between 0 and 255")
        except ValueError as exc:
            raise ValueError(f"{p}, line {number}: {exc}: {line.strip()!r}") from exc
        name = fields[1] if len(fields) == 5 else fields[-5]
        table[index] = Label(name, (r, g, b))
    if not table:
        raise ValueError(f"{p} holds no lookup-table entry.")
    return table
