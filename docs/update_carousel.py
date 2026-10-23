"""Renders the landing page's carousel images with BrainViewer, off screen.

Runs the gallery's examples that draw the simulated brain of the ISMRM 2015
Tractography Challenge, and docs/carousel_simulation.py, which draws the phantom
editor, the Signal breakdown panel and FODs of a simulated crossing, in a temporary
folder. It saves the screenshots the slides use, each 900 x 600, as WebP in
docs/_static/carousel/, and the editor and the panel at their size on screen in
docs/_static/simulation/, for the Simulation page. The examples need the data, so
LARMOR_DATA_DIR must name the folder it is (or will be) downloaded into:

    LARMOR_DATA_DIR=~/larmor-data uv run python docs/update_carousel.py

With --simulation, only the simulation slides are rendered, which needs no data:

    uv run python docs/update_carousel.py --simulation
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).parent.parent
EXAMPLES = ROOT / "examples"
CAROUSEL = Path(__file__).parent / "_static" / "carousel"
SIZE = (900, 600)
QUALITY = 85

# The examples to run, and the slide each screenshot becomes.
SCRIPTS = (
    "models/dti_ismrm.py",
    "sd/fods_ismrm.py",
    "tractography/tracking_ismrm.py",
    "viewer/viewer_real_brain.py",
)
SIMULATION = Path(__file__).parent / "carousel_simulation.py"
SLIDES = {
    "tracking_ismrm_tubes.png": "tubes.webp",
    "fods_ismrm_glyphs.png": "fod-glyphs.webp",
    "fods_ismrm_asymmetric.png": "fod-asymmetric.webp",
    "fods_ismrm_peaks.png": "peaks.webp",
    "dti_ismrm_colours.png": "dti-colours.webp",
    "viewer_real_sagittal.png": "t1-fa.webp",
    "tracking_ismrm_slices.png": "slices-3d.webp",
}
SIMULATION_SLIDES = {
    "simulation_editor.png": "phantom-editor.webp",
    "simulation_breakdown.png": "signal-breakdown.webp",
    "simulation_fods.png": "simulated-fods.webp",
}
# The editor and the panel at their size on screen, for the Simulation page.
PAGE_IMAGES = Path(__file__).parent / "_static" / "simulation"
SIMULATION_PAGE = {
    "simulation_page_editor.png": "phantom-editor.webp",
    "simulation_page_breakdown.png": "signal-breakdown.webp",
}


def main() -> None:
    """Runs the examples and writes the slides."""
    only_simulation = sys.argv[1:] == ["--simulation"]
    if not only_simulation and not os.environ.get("LARMOR_DATA_DIR"):
        sys.exit("Set LARMOR_DATA_DIR to the folder of the example data.")
    scripts = [SIMULATION]
    slides = dict(SIMULATION_SLIDES)
    if not only_simulation:
        scripts += [EXAMPLES / script for script in SCRIPTS]
        slides |= SLIDES
    with tempfile.TemporaryDirectory() as folder:
        for script in scripts:
            subprocess.run([sys.executable, str(script)], cwd=folder, check=True)
        for shot, slide in slides.items():
            with Image.open(Path(folder) / shot) as image:
                if image.size != SIZE:
                    sys.exit(f"{shot} is {image.size}, not {SIZE}")
                image.convert("RGB").save(CAROUSEL / slide, quality=QUALITY, method=6)
        for shot, name in SIMULATION_PAGE.items():
            # Lossless, so the text of the panels stays sharp.
            with Image.open(Path(folder) / shot) as image:
                image.convert("RGB").save(PAGE_IMAGES / name, lossless=True, method=6)


if __name__ == "__main__":
    main()
