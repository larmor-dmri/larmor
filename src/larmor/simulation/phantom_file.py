"""Reading and writing phantom files, which store a phantom and its scan as JSON."""

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Final

import numpy as np

from larmor.io import load_tractogram, load_volume
from larmor.simulation.bundles import BundleGeometry
from larmor.simulation.phantom import Bundle, Phantom, TissueMap
from larmor.simulation.protocol import Protocol
from larmor.simulation.tissues import Tissue
from larmor.types import Grid

# The format name written to a phantom file.
FORMAT: Final = "larmor-phantom"
# The newest version this code reads, and the one it writes.
VERSION: Final = 1

# Upper limit of the bundle seeds the writer draws.
_SEED_LIMIT = 2**31


@dataclass(frozen=True, eq=False)
class PhantomFile:
    """What a phantom file holds, as `read_phantom` returns it.

    Attributes:
        phantom: The phantom.
        protocol: The scan, if the file has one.
        viewer: The settings of the viewer's Simulate workspace, such as the
            bundles' names, kept as they are; empty if none.
    """

    phantom: Phantom
    protocol: Protocol | None = None
    viewer: Mapping[str, Any] = field(default_factory=dict)

    @property
    def bundle_names(self) -> list[str] | None:
        """Returns the bundles' names from the viewer settings, in order.

        Returns:
            The names, or None unless every bundle has one.
        """
        saved = self.viewer.get("bundles")
        if not isinstance(saved, list) or len(saved) != len(self.phantom.bundles):
            return None
        names = [b.get("name") if isinstance(b, dict) else None for b in saved]
        if not all(isinstance(n, str) and n for n in names):
            return None
        return [str(n) for n in names]


def write_phantom(
    phantom: Phantom,
    path: str | Path,
    *,
    protocol: Protocol | None = None,
    viewer: Mapping[str, Any] | None = None,
) -> None:
    """Writes a phantom file (JSON, suggested suffix .phantom.json).

    Tractograms and tissue maps are stored as paths, both absolute and relative
    to the file's folder. A bundle geometry without a seed gets one drawn at
    random and written, so the file always gives the same streamlines.

    Args:
        phantom: The phantom.
        path: The file to write.
        protocol: The scan to store with it. Defaults to None.
        viewer: The viewer's settings, any JSON-ready values. Defaults to None.

    Raises:
        ValueError: If a tractogram bundle or a tissue map has no path.
    """
    path = Path(path).absolute()
    folder = path.parent
    rng = np.random.default_rng()
    bundles = []
    for number, bundle in enumerate(phantom.bundles, 1):
        entry: dict[str, object] = {
            "tissue": bundle.tissue,
            "streamline_area_mm2": bundle.streamline_area,
            "densest_fill": bundle.densest_fill,
        }
        if isinstance(bundle.source, BundleGeometry):
            geometry = bundle.source
            if geometry.seed is None:
                geometry = replace(geometry, seed=int(rng.integers(_SEED_LIMIT)))
            entry["geometry"] = geometry.to_dict()
        else:
            if bundle.path is None:
                raise ValueError(
                    f"Bundle {number} is a tractogram without a path; save it to a "
                    "file and set Bundle.path first."
                )
            entry["tractogram"] = _paths(bundle.path, folder)
        bundles.append(entry)
    maps = []
    for tissue_map in phantom.tissue_maps:
        if tissue_map.path is None:
            raise ValueError(
                f"The map of {tissue_map.tissue!r} has no path; save it to a file and "
                "set TissueMap.path first."
            )
        maps.append({"tissue": tissue_map.tissue, **_paths(tissue_map.path, folder)})
    data: dict[str, object] = {
        "format": FORMAT,
        "version": VERSION,
        "grid": {
            "shape": list(phantom.grid.shape),
            "affine": phantom.grid.affine.tolist(),
        },
        "tissues": [t.to_dict() for t in phantom.tissues],
        "background": phantom.background,
        "bundles": bundles,
        "tissue_maps": maps,
    }
    if protocol is not None:
        data["scan"] = protocol.to_dict()
    if viewer:
        data["viewer"] = dict(viewer)
    path.write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def read_phantom(path: str | Path) -> PhantomFile:
    """Reads a phantom file written by `write_phantom`.

    The tractograms and tissue maps the file refers to are loaded too, from
    their path relative to the file if it exists, else from their absolute path.

    Args:
        path: The file.

    Returns:
        The phantom, the protocol if the file has one, and the viewer block.

    Raises:
        ValueError: If the file is not a phantom file, was written by a newer
            Larmor, or holds invalid values.
        FileNotFoundError: If a tractogram or tissue map cannot be found.
    """
    path = Path(path).absolute()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ValueError(f"{path} is not a phantom file: {e}.") from e
    if not isinstance(data, dict) or data.get("format") != FORMAT:
        raise ValueError(f"{path} is not a Larmor phantom file.")
    version = data.get("version")
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise ValueError(f"{path} has no valid phantom version: {version!r}.")
    if version > VERSION:
        raise ValueError(
            f"{path} was written by a newer Larmor (phantom version {version}); this "
            f"one reads up to version {VERSION}. Update Larmor to open it."
        )
    folder = path.parent
    try:
        raw_grid = data["grid"]
        grid = Grid(tuple(raw_grid["shape"]), np.asarray(raw_grid["affine"], float))
        tissues = tuple(Tissue.from_dict(t) for t in data["tissues"])
        bundles = tuple(_bundle(b, folder, grid) for b in data.get("bundles", []))
        maps = tuple(
            TissueMap(m["tissue"], load_volume(_locate(m, folder)), _locate(m, folder))
            for m in data.get("tissue_maps", [])
        )
        phantom = Phantom(
            grid,
            tissues=tissues,
            bundles=bundles,
            tissue_maps=maps,
            background=data.get("background"),
        )
        scan = data.get("scan")
        protocol = Protocol.from_dict(scan) if scan is not None else None
    except (KeyError, TypeError) as e:
        raise ValueError(f"{path} has an invalid entry: {e!r}.") from e
    viewer = data.get("viewer", {})
    return PhantomFile(phantom, protocol, viewer if isinstance(viewer, dict) else {})


def _paths(source: Path, folder: Path) -> dict[str, str | None]:
    """Returns the absolute path of source and its path from folder."""
    source = Path(source).absolute()
    try:
        relative: str | None = Path(os.path.relpath(source, folder)).as_posix()
    except ValueError:  # Windows: another drive
        relative = None
    return {"path": str(source), "relative_path": relative}


def _locate(entry: Mapping[str, Any], folder: Path) -> Path:
    """Returns an entry's path relative to folder if it exists, else the absolute."""
    relative = entry.get("relative_path")
    if isinstance(relative, str):
        candidate = Path(os.path.normpath(folder / relative))
        if candidate.is_file():
            return candidate
    absolute = entry.get("path")
    if isinstance(absolute, str) and Path(absolute).is_file():
        return Path(absolute)
    raise FileNotFoundError(f"Cannot find {relative or absolute} from {folder}.")


def _bundle(entry: Mapping[str, Any], folder: Path, grid: Grid) -> Bundle:
    """Returns the bundle of one entry of the file, loading its tractogram."""
    area = entry.get("streamline_area_mm2")
    # Files written before densest_fill was saved have the default.
    fill = entry.get("densest_fill", Bundle.densest_fill)
    if "geometry" in entry:
        return Bundle(
            BundleGeometry.from_dict(entry["geometry"]),
            tissue=entry["tissue"],
            streamline_area=area,
            densest_fill=fill,
        )
    source = _locate(entry["tractogram"], folder)
    return Bundle(
        load_tractogram(source, reference=grid),
        tissue=entry["tissue"],
        path=source,
        streamline_area=area,
        densest_fill=fill,
    )
