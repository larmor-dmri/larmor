"""Loads items through a registration computed by ANTs, FSL or MRtrix3."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from larmor.io import load_grid
from larmor.types import Grid, Surface, Tractogram, Volume
from larmor.viewer.model.item import Category
from larmor.viewer.model.registry import SUBSETS
from larmor.viewer.model.surface import SurfaceData

# The default interpolation of each image category.
_INTERPOLATION = {
    Category.VOLUMES: "linear",
    Category.OVERLAYS: "nearest",
    Category.PEAKS: "nearest",
}
# The interpolations, formats and warp types a registration accepts.
INTERPOLATIONS = ("nearest", "linear", "cubic")
FORMATS = ("ants", "fsl", "mrtrix")
WARP_TYPES = ("deformation", "displacement", "relative", "absolute")
# The optional single paths, saved absolute and relative to the session.
_PATHS = ("reference", "fsl_input", "fsl_reference")


@dataclass(frozen=True)
class Registration:
    """A registration an item is brought through into the reference space.

    The transform files are those the tools apply to images, in the order
    antsApplyTransforms takes them. Tractograms and surfaces are moved with the
    inverse, so that they land in the same space as the images. See
    `larmor.registration.load_registration` and `apply_transform`.

    Attributes:
        transforms: The files, each with the inverse warp its tool wrote or None.
        reference: An image of the reference space; images are resampled onto
            its grid, and tractograms and surfaces take it as their reference.
            Needed for images.
        inverse: Whether to go the other way, from the reference space into the
            space the transforms bring images from.
        kind: The format, "ants", "fsl" or "mrtrix"; None guesses it from the
            files.
        interpolation: The interpolation of images, "nearest", "linear" or
            "cubic"; None chooses linear for volumes, nearest for overlays and
            peaks.
        warp: The warp fields' type ("deformation" or "displacement" for
            MRtrix3, "relative" or "absolute" for FSL); None is the tool's
            default.
        fsl_input: The input image of the FSL run. Defaults to the item itself
            when it is an image (the reference with `inverse`); a tractogram or
            surface needs it going forward.
        fsl_reference: The reference image of the FSL run. Defaults to
            `reference` (the item itself with `inverse`); a tractogram or surface
            needs it going the other way.
    """

    transforms: tuple[tuple[Path, Path | None], ...]
    reference: Path | None = None
    inverse: bool = False
    kind: str | None = None
    interpolation: str | None = None
    warp: str | None = None
    fsl_input: Path | None = None
    fsl_reference: Path | None = None

    def __post_init__(self) -> None:
        """Checks the choices; the files are read only when an item is loaded.

        Raises:
            ValueError: If there is no transform, or the format, interpolation or
                warp type is unknown.
        """
        if not self.transforms:
            raise ValueError("A registration needs at least one transform file.")
        for name, value, allowed in (
            ("format", self.kind, FORMATS),
            ("interpolation", self.interpolation, INTERPOLATIONS),
            ("warp type", self.warp, WARP_TYPES),
        ):
            if value is not None and value not in allowed:
                raise ValueError(
                    f"Unknown {name} {value!r}: one of {', '.join(allowed)}."
                )

    @classmethod
    def from_files(
        cls,
        transforms: list[str | Path],
        reference: str | Path | None = None,
        inverse: bool = False,
        kind: str | None = None,
        interpolation: str | None = None,
        warp: str | None = None,
        fsl_input: str | Path | None = None,
        fsl_reference: str | Path | None = None,
    ) -> Registration:
        """Builds a registration from paths.

        Each transform is a file, or 'file,inverse_file' with the inverse warp.

        Returns:
            Registration: The registration.

        Raises:
            ValueError: If a transform is not a file or 'file,inverse_file', or a
                choice is unknown.
        """
        pairs = []
        for spec in transforms:
            parts = str(spec).split(",")
            if len(parts) > 2 or not all(parts):
                raise ValueError(
                    f"Expected a transform file or 'file,inverse_file', got {spec!r}."
                )
            inverse_file = Path(parts[1]).absolute() if len(parts) == 2 else None
            pairs.append((Path(parts[0]).absolute(), inverse_file))
        return cls(
            tuple(pairs),
            _absolute(reference),
            inverse,
            kind,
            interpolation,
            warp,
            _absolute(fsl_input),
            _absolute(fsl_reference),
        )

    def apply(self, data: object, category: Category) -> object:
        """Brings an item's data, as read, into the reference space.

        Returns:
            The data in the reference space.

        Raises:
            ValueError: If an image has no reference, the item is an FOD (FODs
                are not reoriented), FSL files lack an image, or a transform file
                cannot be read.
        """
        # Imported here: resampling needs scipy, which the model avoids.
        from larmor.registration import apply_transform, load_registration

        if category == Category.FODS:
            raise ValueError(
                "FODs cannot be loaded through a registration: reorienting FODs is "
                "not supported."
            )
        reference = _grid(self.reference)
        transform = load_registration(
            [f if b is None else (f, b) for f, b in self.transforms],
            data_grid=data.grid if isinstance(data, Volume) else None,
            reference_grid=reference,
            inverse=self.inverse,
            format=self.kind,
            warp_type=self.warp,
            fsl_input_grid=_grid(self.fsl_input),
            fsl_reference_grid=_grid(self.fsl_reference),
        )
        interpolation = cast(
            Literal["nearest", "linear", "cubic"],
            self.interpolation or _INTERPOLATION.get(category, "linear"),
        )
        if isinstance(data, Volume):
            reorient: Literal["peaks"] | None = (
                "peaks" if category == Category.PEAKS else None
            )
            return apply_transform(
                data,
                transform,
                reference=reference,
                interpolation=interpolation,
                reorient=reorient,
            )
        if isinstance(data, Tractogram):
            moved = apply_transform(data, transform, reference=reference)
            subset = SUBSETS.get(data)
            if subset is not None:
                assert isinstance(moved, Tractogram)
                SUBSETS[moved] = subset
            return moved
        if isinstance(data, SurfaceData):
            surface = apply_transform(data.surface, transform, reference=reference)
            assert isinstance(surface, Surface)
            return SurfaceData(surface, data.cortex)
        raise ValueError(f"Cannot bring {type(data).__name__} through a registration.")

    def to_json(self, folder: Path | None = None) -> dict[str, object]:
        """Returns the registration as JSON data, with paths also relative to folder."""
        data: dict[str, object] = {
            "transforms": [
                [str(f), None if b is None else str(b)] for f, b in self.transforms
            ],
            "inverse": self.inverse,
            "kind": self.kind,
            "interpolation": self.interpolation,
            "warp": self.warp,
        }
        for name in _PATHS:
            path = getattr(self, name)
            data[name] = None if path is None else str(path)
        if folder is not None:
            data["relative_transforms"] = [
                [_relative(f, folder), None if b is None else _relative(b, folder)]
                for f, b in self.transforms
            ]
            for name in _PATHS:
                path = getattr(self, name)
                if path is not None:
                    data[f"relative_{name}"] = _relative(path, folder)
        return data

    @classmethod
    def from_json(cls, data: object, folder: Path | None = None) -> Registration:
        """Reads `to_json`'s data; relative paths win when their files exist.

        Args:
            data: The data `to_json` wrote, as read from JSON.
            folder: The folder the relative paths are relative to. Defaults to
                None: only the absolute paths are used.

        Returns:
            Registration: The registration.

        Raises:
            KeyError: If the data lacks the transforms.
            TypeError: If the data is not shaped as `to_json` writes it.
            ValueError: If a value is invalid.
        """
        if not isinstance(data, dict):
            raise TypeError("the registration is not a JSON object")
        absolute = data["transforms"]
        if not isinstance(absolute, list):
            raise TypeError("the registration's transforms are not a list")
        relative = data.get("relative_transforms")
        if relative is None:
            relative = [[None, None]] * len(absolute)
        if not isinstance(relative, list) or len(relative) != len(absolute):
            raise TypeError("the registration's relative transforms do not match")
        pairs = []
        for pair, relative_pair in zip(absolute, relative, strict=True):
            if not (isinstance(pair, list) and isinstance(relative_pair, list)):
                raise TypeError("a transform entry is not a pair of paths")
            if len(pair) != 2 or len(relative_pair) != 2:
                raise TypeError("a transform entry is not a pair of paths")
            located = _locate(pair[0], relative_pair[0], folder)
            if located is None:
                raise TypeError("a transform entry has no file")
            pairs.append((located, _locate(pair[1], relative_pair[1], folder)))
        paths = {
            name: _locate(data.get(name), data.get(f"relative_{name}"), folder)
            for name in _PATHS
        }
        return cls(
            tuple(pairs),
            inverse=bool(data.get("inverse", False)),
            kind=data.get("kind"),
            interpolation=data.get("interpolation"),
            warp=data.get("warp"),
            **paths,
        )


def _absolute(path: str | Path | None) -> Path | None:
    """Returns the path made absolute, or None."""
    return None if path is None else Path(path).absolute()


def _grid(path: Path | None) -> Grid | None:
    """Reads the grid of an image file, or returns None for no file."""
    return None if path is None else load_grid(path)


def _relative(path: Path, folder: Path) -> str | None:
    """Returns the path from folder, with forward slashes; None across drives."""
    try:
        return Path(os.path.relpath(path, folder)).as_posix()
    except ValueError:  # Windows: another drive
        return None


def _locate(absolute: object, relative: object, folder: Path | None) -> Path | None:
    """Returns the relative path from folder if its file exists, else the absolute."""
    if isinstance(relative, str) and folder is not None:
        candidate = Path(os.path.normpath(folder / relative))
        if candidate.is_file():
            return candidate
    if absolute is None:
        return None
    if not isinstance(absolute, str):
        raise TypeError(f"a path is not text: {absolute!r}")
    return Path(absolute)
