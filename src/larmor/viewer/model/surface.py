"""Holds a surface item's mesh and cortex mask, and reads them from files."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from larmor._log import LarmorLogger
from larmor.io import load_surface
from larmor.io.formats import read_freesurfer_label
from larmor.types import Surface

# The names of the values per vertex that are curvatures, shown in grey.
CURVATURES = ("curv", "sulc")
# The colour range of a curvature; sulci (> 0) are dark, gyri light.
CURVATURE_RANGE = (-0.3, 0.3)


@dataclass(frozen=True, eq=False)
class SurfaceData:
    """A surface, and its cortex mask when a label gives one."""

    surface: Surface
    # (V,) bool, True on the cortex, False on the medial wall; None if unknown.
    cortex: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Checks that the mask has one entry per vertex."""
        if self.cortex is not None and self.cortex.shape != (
            self.surface.num_vertices,
        ):
            raise ValueError(
                f"A cortex mask has one entry per vertex "
                f"({self.surface.num_vertices}); got shape {self.cortex.shape}."
            )

    def values(self) -> list[str]:
        """Returns the names of the values per vertex a surface can be coloured by."""
        return [
            name
            for name, array in self.surface.vertex_data.items()
            if array.ndim == 1 and np.issubdtype(array.dtype, np.number)
        ]

    def value_range(self, name: str) -> tuple[float, float]:
        """Returns the finite range of the values `name`; 0 to 1 if none is finite."""
        array = self.surface.vertex_data[name]
        finite = array[np.isfinite(array)]
        if not finite.size:
            return (0.0, 1.0)
        return (float(finite.min()), float(finite.max()))


def read_surface(path: Path) -> SurfaceData:
    """Reads a surface, and a FreeSurfer one's curvature and cortex label.

    For a FreeSurfer surface such as `surf/lh.white`, `surf/lh.curv` is read as
    values "curv" and `label/lh.cortex.label` as the cortex mask, when they exist.

    Returns:
        SurfaceData: The surface and its extras.
    """
    hemisphere, dot, _ = path.name.partition(".")
    if path.name.lower().endswith(".gii") or not dot:
        return SurfaceData(load_surface(path))
    curv = path.parent / f"{hemisphere}.curv"
    surface = None
    if curv.is_file() and curv != path:
        try:
            surface = load_surface(path, values=curv)
        except (OSError, ValueError) as e:
            LarmorLogger.warning(f"{path.name}: {curv} is not read: {e}")
    if surface is None:
        surface = load_surface(path)
    label = path.parent.parent / "label" / f"{hemisphere}.cortex.label"
    cortex = None
    if label.is_file():
        try:
            cortex = _cortex_mask(label, surface.num_vertices)
        except (OSError, ValueError) as e:
            LarmorLogger.warning(f"{path.name}: {label} is not read: {e}")
    return SurfaceData(surface, cortex)


def _cortex_mask(label: Path, num_vertices: int) -> np.ndarray:
    """Reads the vertices a label file lists as a mask over a surface's vertices."""
    indices = read_freesurfer_label(label)
    if indices.size and (indices.min() < 0 or indices.max() >= num_vertices):
        raise ValueError(
            f"it names vertices outside the surface's {num_vertices} vertices."
        )
    mask = np.zeros(num_vertices, dtype=bool)
    mask[indices] = True
    return mask
