"""Renders the website's simulation images with BrainViewer, off screen.

Builds a phantom of the brain from the 21 ground-truth bundles of the ISMRM 2015
Tractography Challenge, with grey matter and CSF from its five-tissue-type image,
on the grid of the challenge's diffusion images, and runs it. It saves, into the
current folder, the carousel's three slides (the Simulate workspace, the Signal
breakdown and FOD glyphs, the last with the Website export preset), and the step
page and the panel for the Simulation page. `update_carousel.py` runs it and turns
the screenshots into the site's images.

The bundles and the five-tissue-type image are not in Larmor's example data, so
LARMOR_ISMRM_BUNDLES must name the folder of the bundles' .trk files and
LARMOR_ISMRM_5TT the image; the diffusion images come from the example data, in
the folder LARMOR_DATA_DIR names. The viewer's settings go to a temporary home
folder, so the user's own are untouched.
"""

import os
import shutil
import tempfile
from pathlib import Path

BUNDLES = Path(os.environ["LARMOR_ISMRM_BUNDLES"]).expanduser()
FIVE_TT = Path(os.environ["LARMOR_ISMRM_5TT"]).expanduser()
HOME = tempfile.mkdtemp(prefix="larmor-carousel-")
os.environ["HOME"] = HOME
os.environ["MPLCONFIGDIR"] = str(Path(HOME) / "matplotlib")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402 (after HOME is set)
import PySide6QtAds as QtAds  # noqa: E402, TID251 (moves a dock for a screenshot)
from PIL import Image  # noqa: E402
from scipy.ndimage import map_coordinates  # noqa: E402

from larmor.data import fetch_ismrm2015  # noqa: E402
from larmor.io import (  # noqa: E402
    load_dwi,
    load_tractogram,
    load_volume,
    save_tractogram,
    save_volume,
)
from larmor.sd import SDModel, build_response_functions  # noqa: E402
from larmor.simulation import (  # noqa: E402
    Bundle,
    Phantom,
    TissueMap,
    default_tissues,
    example_phantom,
    write_phantom,
)
from larmor.types import Grid, Volume  # noqa: E402
from larmor.viewer import (  # noqa: E402
    Category,
    GlyphScaling,
    OptionName,
    ScanField,
    Viewer,
    ViewKind,
)
from larmor.viewer.app import application  # noqa: E402
from larmor.viewer.model.phantom import PhantomStep  # noqa: E402
from larmor.viewer.model.settings import AppSettings, Settings, Theme  # noqa: E402
from larmor.viewer.qt.docks import Panel  # noqa: E402
from larmor.viewer.qt.main_window import MainWindow  # noqa: E402
from larmor.viewer.qt.style import apply_style  # noqa: E402

SIZE = (900, 600)
# The slides are drawn in a larger window, then scaled down to the slide, so the
# panels have room.
WINDOW = (1350, 900)
BREAKDOWN_WIDTH = 560  # the Signal breakdown's dock beside the views, px
VIEW = ViewKind.CORONAL
# Where the corpus callosum, the corticospinal tract and the SLF cross, in mm: the
# voxel of the Signal breakdown.
CROSSING = (-56.0, -98.0, 94.0)
# The FOD glyphs: the coronal view zoomed on the centrum semiovale.
FOD_CENTRE = (-80.0, -98.0, 88.0)
FOD_ZOOM = 2.0
GLYPH_SCALE = 2.6
# The b = 0 image's maximum, as a multiple of its 99.5th percentile, which darkens
# it behind the glyphs.
B0_DIMMING = 2.6
STEP = PhantomStep.TISSUE  # the step the editor shows
# The cross-section each streamline stands for, mm^2: a few streamlines fill a
# voxel, so the bundles, dense in their cores, fill their edges too.
STREAMLINE_AREA = 0.5
# Bundles of WM-sparse, the tissue with fewer axons; the others are WM.
SPARSE = ("CA", "Cingulum", "Fornix", "UF")


def settle(rounds: int = 80) -> None:
    """Lets the window lay itself out and draw."""
    for _ in range(rounds):
        application().processEvents()


def main() -> None:
    """Runs the phantom in the editor and saves the screenshots."""
    window, folder = run_phantom()
    save_breakdown(window)
    save_editor(window)
    window.close()
    save_fods(folder)


def brain_phantom(folder: Path) -> tuple[Phantom, list[str]]:
    """Builds the phantom of the brain on the grid of the challenge's images.

    The tissues are those of the crossing example, WM and WM-sparse, and the
    predefined GM and CSF, whose maps are written into the folder with the
    bundles.

    Args:
        folder: Where the bundles and the tissue maps are written.

    Returns:
        tuple[Phantom, list[str]]: The phantom, and its bundles' names.
    """
    dwi = load_volume(fetch_ismrm2015()["dwi"])
    grid = Grid(dwi.data.shape[:3], dwi.affine)
    names, bundles = [], []
    for trk in sorted(BUNDLES.glob("*.trk")):
        # As TCK, which has no grid of its own: the TRK files' grid is the
        # challenge's 1 mm one, not the phantom's.
        path = folder / f"{trk.stem}.tck"
        save_tractogram(load_tractogram(trk), path)
        tissue = "WM-sparse" if trk.stem.startswith(SPARSE) else "WM"
        bundles.append(
            Bundle(
                load_tractogram(path),
                tissue=tissue,
                path=path,
                streamline_area=STREAMLINE_AREA,
            )
        )
        names.append(trk.stem.replace("_", " "))
    # The 5TT image's tissues are cortical grey matter, subcortical grey matter,
    # white matter, CSF and pathology; the grey matters together make GM.
    five = load_volume(FIVE_TT)
    voxels = np.indices(grid.shape).reshape(3, -1)
    world = grid.affine[:3, :3] @ voxels + grid.affine[:3, 3:4]
    to_five = np.linalg.inv(five.affine)
    at = to_five[:3, :3] @ world + to_five[:3, 3:4]
    channels = [
        map_coordinates(five.data[..., c], at, order=1).reshape(grid.shape)
        for c in range(4)
    ]
    maps = []
    for tissue, values in (("GM", channels[0] + channels[1]), ("CSF", channels[3])):
        path = folder / f"{tissue}.nii.gz"
        save_volume(Volume(np.clip(values, 0, 1).astype(np.float32), grid.affine), path)
        maps.append(TissueMap(tissue, load_volume(path), path=path))
    crossing = example_phantom("crossing").phantom
    extra = tuple(t for t in default_tissues() if t.name in ("GM", "CSF"))
    phantom = Phantom(
        grid=grid,
        tissues=crossing.tissues + extra,
        bundles=tuple(bundles),
        tissue_maps=tuple(maps),
    )
    return phantom, names


def run_phantom() -> tuple[MainWindow, Path]:
    """Opens the phantom of the brain in a window and runs it.

    Returns:
        tuple[MainWindow, Path]: The window, and the folder the run wrote.
    """
    phantom, names = brain_phantom(Path(HOME))
    phantom_file = Path(HOME) / "brain.phantom.json"
    write_phantom(
        phantom,
        phantom_file,
        protocol=example_phantom("crossing").protocol,
        viewer={"bundles": [{"name": name} for name in names]},
    )
    folder = Path(HOME) / "brain-sim"

    application()
    values = AppSettings(theme=Theme.DARK)
    apply_style(values.theme, values.accent)
    window = MainWindow(settings=Settings(values))
    window.resize(*WINDOW)
    window.show()
    settle()
    viewer = window.viewer
    viewer.load_phantom(phantom_file)
    viewer.set_scan(ScanField.SNR, 30.0)
    viewer.set_scan(ScanField.SEED, 1)
    viewer.run_phantom(folder)
    settle()
    return window, folder


def save_breakdown(window: MainWindow) -> None:
    """Saves the window with the Signal breakdown of the crossing, and the panel."""
    # The b = 0 image and the bundles in the coronal view, beside the Signal
    # breakdown of the voxel where three bundles cross, at b = 2000 with the
    # gradient along x. No setting changes after the voxel is chosen, so the
    # panel shows no comparison line.
    viewer = window.viewer
    viewer.remove(Category.OVERLAYS)
    viewer.remove(Category.PEAKS)
    viewer.toggle_maximised(VIEW)
    viewer.toggle_overlay()
    window.show_breakdown()
    window.docks.toggle_sides()
    # Beside the views, not below them, so the slide has room for the chart. The
    # move clears the panel, so the voxel is pinned after it.
    dock = window.docks.docks[Panel.SIGNAL_BREAKDOWN]
    window.docks.manager.addDockWidget(QtAds.RightDockWidgetArea, dock)
    settle()
    splitter = dock.dockAreaWidget().parentSplitter()
    index = splitter.indexOf(dock.dockAreaWidget())
    sizes = [0] * splitter.count()
    sizes[index - 1], sizes[index] = WINDOW[0] - BREAKDOWN_WIDTH, BREAKDOWN_WIDTH
    splitter.setSizes(sizes)
    viewer.pin_breakdown(CROSSING)
    gradients = viewer.breakdown_gradients()
    assert gradients is not None
    along_x = np.where(gradients.bvals > 1500, np.abs(gradients.bvecs[:, 0]), -1.0)
    viewer.set_breakdown_volume(int(np.argmax(along_x)))
    settle(150)
    grabbed = Path("simulation_breakdown_window.png")
    window.grab().save(str(grabbed))
    with Image.open(grabbed) as image:
        image.convert("RGB").resize(SIZE, Image.Resampling.LANCZOS).save(
            "simulation_breakdown.png"
        )
    window.docks.breakdown().grab().save("simulation_page_breakdown.png")


def save_editor(window: MainWindow) -> None:
    """Saves the Simulate workspace on its Tissue step, for the slide and the page."""
    window.docks.close(Panel.SIGNAL_BREAKDOWN)
    window.docks.toggle_sides()
    window.viewer.toggle_maximised(VIEW)
    window.show_phantom()
    window.steps.go_to(STEP)
    settle(150)
    grabbed = Path("simulation_editor_window.png")
    window.grab().save(str(grabbed))
    with Image.open(grabbed) as image:
        image.convert("RGB").resize(SIZE, Image.Resampling.LANCZOS).save(
            "simulation_editor.png"
        )
    window.step_page.grab().save("simulation_page_editor.png")


def save_fods(folder: Path) -> None:
    """Saves FOD glyphs fitted on the simulated images, over the b = 0 image.

    The coronal view is zoomed on the centrum semiovale, where the corpus
    callosum crosses the corticospinal tracts; the b = 0 image is darkened so the
    glyphs stand out.
    """
    dwi, gtab = load_dwi(
        folder / "dwi.nii.gz", folder / "dwi.bval", folder / "dwi.bvec"
    )
    mask = load_volume(folder / "mask_bundles.nii.gz").data > 0
    responses = build_response_functions(dwi, gtab, mask)
    fit = SDModel(responses, lmax=8).fit(dwi, gtab, mask=mask)
    b0 = dwi.data[..., gtab.bvals < 50].mean(axis=-1)
    glyphs = Viewer()
    image = glyphs.add(
        Volume(b0.astype(np.float32), dwi.affine), Category.VOLUMES, name="b0"
    )
    glyphs.set(
        image.id,
        OptionName.MAXIMUM,
        float(np.percentile(b0, 99.5)) * B0_DIMMING,
    )
    fods = glyphs.add(dwi.with_data(fit.fods[0]), Category.FODS, name="FOD")
    glyphs.set(fods.id, OptionName.SCALING, GlyphScaling.PER_VOXEL)
    glyphs.set(fods.id, OptionName.SCALE, GLYPH_SCALE)
    glyphs.move_cursor(FOD_CENTRE)
    glyphs.toggle_crosshair()
    glyphs.zoom(VIEW, FOD_ZOOM, about=FOD_CENTRE)
    glyphs.export("Website", "simulation_fods.png", VIEW)


if __name__ == "__main__":
    try:
        main()
    finally:
        shutil.rmtree(HOME, ignore_errors=True)
