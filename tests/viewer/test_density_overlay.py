from pathlib import Path

import numpy as np
from PIL import Image

from larmor.io import load_grid, load_tractogram, save_volume
from larmor.tractography import density_grid, track_density
from larmor.viewer import Category, Colormap, OptionName, Viewer, ViewKind

DATA = Path(__file__).parents[1] / "data" / "connectome"


def _red_pixels(path: Path) -> int:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB")).astype(int)
    return int(((rgb[..., 0] > 150) & (rgb[..., 2] < 80)).sum())


def test_a_track_density_image_opens_as_an_overlay(tmp_path: Path) -> None:
    tracks = load_tractogram(DATA / "tracks.tck")
    density = track_density(
        tracks, density_grid(load_grid(DATA / "template.nii.gz"), 1)
    )
    path = tmp_path / "tdi.nii.gz"
    save_volume(density, path)
    # Through the voxel most streamlines pass.
    densest = np.unravel_index(np.argmax(density.data), density.data.shape)
    centre = density.affine[:3, :3] @ np.array(densest) + density.affine[:3, 3]

    viewer = Viewer()
    item = viewer.load(path, Category.OVERLAYS)
    viewer.set(item.id, OptionName.COLORMAP, Colormap.HOT)
    viewer.move_cursor((float(centre[0]), float(centre[1]), float(centre[2])))
    shown = viewer.screenshot(ViewKind.AXIAL, tmp_path / "shown.png", size=(200, 200))
    viewer.set(item.id, OptionName.OPACITY_2D, 0.0)
    hidden = viewer.screenshot(ViewKind.AXIAL, tmp_path / "hidden.png", size=(200, 200))

    assert _red_pixels(shown) > 50
    assert _red_pixels(hidden) < _red_pixels(shown) // 10
