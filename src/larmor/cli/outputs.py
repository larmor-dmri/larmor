"""Builds the output paths of the commands and checks that they can be written."""

from collections.abc import Sequence
from pathlib import Path

from larmor.io import output_files


def _written_files(path: str | Path) -> list[Path]:
    """Returns the files an output path writes, such as the .dat of a .mih."""
    try:
        return output_files(path)
    except ValueError:  # Not a volume extension.
        return [Path(path)]


def check_outputs(paths: Sequence[str | Path], force: bool) -> None:
    """Checks, before any work, that the outputs can be written.

    Every file an output writes is checked, so a `.mih` also checks its `.dat`.

    Raises:
        ValueError: If two outputs are the same file.
        FileNotFoundError: If the folder of an output does not exist.
        FileExistsError: If an output exists and force is False.
    """
    files = [f for path in paths for f in _written_files(path)]
    seen: dict[Path, Path] = {}
    for p in files:
        resolved = p.resolve()
        if resolved in seen:
            raise ValueError(
                f"Outputs {seen[resolved]} and {p} are the same file; give each "
                "output its own path."
            )
        seen[resolved] = p
    for p in files:
        if not p.parent.is_dir():
            raise FileNotFoundError(f"Output folder {p.parent} does not exist.")
        if p.exists() and not force:
            raise FileExistsError(f"File {p} exists. Use --force to overwrite.")


def output_prefix(output: str) -> Path:
    """Returns the output path without a .nii or .nii.gz extension.

    Commands that write several files add their suffixes to it, so `out`, `out.nii`
    and `out.nii.gz` all give `out_fa.nii.gz` and so on.
    """
    path = Path(output)
    for extension in (".nii.gz", ".nii"):
        if path.name.endswith(extension):
            return path.with_name(path.name.removesuffix(extension))
    return path


def with_suffix(prefix: Path, suffix: str) -> Path:
    """Returns `<prefix><suffix>`, such as `out_fa.nii.gz` from `out`."""
    return prefix.with_name(prefix.name + suffix)
