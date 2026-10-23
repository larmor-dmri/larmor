"""Renders the website's simulation images with BrainViewer, off screen.

Runs the crossing phantom that ships with Larmor and saves, into the current folder,
the carousel's three slides, each 900 x 600 (the phantom editor, the Signal breakdown
and FOD glyphs), and the editor and the panel for the Simulation page. It needs no
data. `update_carousel.py` runs it and turns the screenshots into the site's images.
The viewer's settings go to a temporary home folder, so the user's own are untouched.
"""

import os
import shutil
import tempfile
from pathlib import Path

HOME = tempfile.mkdtemp(prefix="larmor-carousel-")
os.environ["HOME"] = HOME
os.environ["MPLCONFIGDIR"] = str(Path(HOME) / "matplotlib")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402 (after HOME is set)
from PIL import Image  # noqa: E402

from larmor.io import load_dwi, load_volume  # noqa: E402
from larmor.sd import SDModel, build_response_functions  # noqa: E402
from larmor.simulation import example_phantom, write_phantom  # noqa: E402
from larmor.types import Volume  # noqa: E402
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
from larmor.viewer.qt.phantom_steps import PhantomEditor  # noqa: E402
from larmor.viewer.qt.style import apply_style  # noqa: E402

SIZE = (900, 600)
# The breakdown slide is drawn in a larger window, then scaled down to the slide, so
# the chart has room under the panel's table.
WINDOW = (1125, 750)
PANEL_WIDTH = 520  # the Signal breakdown panel's column in that window
EDITOR_ON_PAGE = (780, 600)  # the editor as the Simulation page shows it
CENTRE = (20.0, 20.0, 20.0)  # where the bundles cross, in mm
STEP = PhantomStep.TISSUE  # the step the editor shows
ZOOM = 3.5  # of the glyphs' view, about the crossing


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


def run_phantom() -> tuple[MainWindow, Path]:
    """Opens the crossing phantom in a window and runs it.

    Returns:
        tuple[MainWindow, Path]: The window, and the folder the run wrote.
    """
    example = example_phantom("crossing")
    phantom_file = Path(HOME) / "crossing.phantom.json"
    write_phantom(
        example.phantom, phantom_file, protocol=example.protocol, viewer=example.viewer
    )
    folder = Path(HOME) / "crossing-sim"

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
    # The b = 0 image and the bundles in the axial view, beside the Signal
    # breakdown of the voxel where they cross, at b = 2000 with the gradient
    # along x. No setting changes after the voxel is chosen, so the panel shows
    # no comparison line.
    viewer = window.viewer
    viewer.remove(Category.OVERLAYS)
    viewer.remove(Category.PEAKS)
    for panel in (Panel.LAYERS, Panel.OPTIONS, Panel.PHANTOM):
        if window.docks.is_open(panel):
            window.docks.close(panel)
    viewer.toggle_maximised(ViewKind.AXIAL)
    viewer.toggle_overlay()
    window.show_breakdown()
    viewer.pin_breakdown(CENTRE)
    gradients = viewer.breakdown_gradients()
    assert gradients is not None
    along_x = np.where(gradients.bvals > 1500, np.abs(gradients.bvecs[:, 0]), -1.0)
    viewer.set_breakdown_volume(int(np.argmax(along_x)))
    window.docks.set_side_widths(0, PANEL_WIDTH)
    settle(150)
    grabbed = Path("simulation_breakdown_window.png")
    window.grab().save(str(grabbed))
    with Image.open(grabbed) as image:
        image.convert("RGB").resize(SIZE, Image.Resampling.LANCZOS).save(
            "simulation_breakdown.png"
        )
    window.docks.widget(Panel.SIGNAL_BREAKDOWN).grab().save(
        "simulation_page_breakdown.png"
    )


def save_editor(window: MainWindow) -> None:
    """Saves the guided phantom editor on its Tissue step, for the slide and page."""
    # It opens as a window of its own, which is sized to the slide, then to the page.
    window.show_phantom()
    editor = window.docks.widget(Panel.PHANTOM)
    assert isinstance(editor, PhantomEditor)
    editor.go_to(STEP)
    shots = (
        (SIZE, "simulation_editor.png"),
        (EDITOR_ON_PAGE, "simulation_page_editor.png"),
    )
    for size, name in shots:
        editor.window().resize(*size)
        settle()
        editor.grab().save(name)


def save_fods(folder: Path) -> None:
    """Saves FOD glyphs fitted on the simulated images, over the b = 0 image.

    The axial view is zoomed on the middle of the crossing.
    """
    dwi, gtab = load_dwi(
        folder / "dwi.nii.gz", folder / "dwi.bval", folder / "dwi.bvec"
    )
    mask = load_volume(folder / "mask_bundles.nii.gz").data > 0
    responses = build_response_functions(dwi, gtab, mask)
    fit = SDModel(responses, lmax=8).fit(dwi, gtab, mask=mask)
    b0 = dwi.data[..., gtab.bvals < 50].mean(axis=-1)
    glyphs = Viewer()
    glyphs.add(Volume(b0.astype(np.float32), dwi.affine), Category.VOLUMES, name="b0")
    fods = glyphs.add(dwi.with_data(fit.fods[0]), Category.FODS, name="FOD")
    glyphs.set(fods.id, OptionName.SCALING, GlyphScaling.PER_VOXEL)
    glyphs.move_cursor(CENTRE)
    glyphs.toggle_crosshair()
    glyphs.zoom(ViewKind.AXIAL, ZOOM, about=CENTRE)
    glyphs.screenshot(ViewKind.AXIAL, "simulation_fods.png", size=SIZE, overlay=False)


if __name__ == "__main__":
    try:
        main()
    finally:
        shutil.rmtree(HOME, ignore_errors=True)
