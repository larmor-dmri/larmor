"""The gallery's image scraper: the screenshots an example writes, shown under its code.

The viewer examples save their screenshots as image files in the folder they run in
(the example's own folder, when sphinx-gallery runs them). After each code block,
`scrape` moves the files the block wrote into the gallery's image folder, as WebP,
and returns the reStructuredText that shows them, each with its alt text from
`ALT_TEXT`. Under the first images of an example on the ISMRM 2015 data it adds the
data's credit, once per example. `clear_leftovers` runs before each example and
deletes images an earlier run left in the examples' folders, so they never show under
the wrong example.
"""

from pathlib import Path
from typing import Any

from PIL import Image
from sphinx.util import logging

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = (".png", ".webp")
WEBP_QUALITY = 90
ISMRM_FETCHER = "fetch_ismrm2015"
CREDIT = """
.. include:: /_snippets/ismrm-credit.md
   :parser: myst_parser.sphinx_
"""

# The alt text of every screenshot, by the name the example saves it under.
ALT_TEXT = {
    "dti_ismrm_colours.png": (
        "Axial slice of the simulated brain coloured by the principal diffusion "
        "direction and brighter where FA is high. The corpus callosum is red, the "
        "corticospinal tracts are blue, and bundles running front to back are green"
    ),
    "fod_fit_synthetic.png": (
        "Three fitted FOD glyphs side by side, for one fibre, two fibres crossing "
        "at right angles and three crossing fibres"
    ),
    "peaks_synthetic.png": (
        "Peaks of four synthetic FODs drawn as lines. Two lines cross at 90, 60 and"
        " 45 degrees, and a single line is left where the fibres cross at 30 "
        "degrees"
    ),
    "fods_ismrm_glyphs.png": (
        "FOD glyphs on a coronal slice of the simulated brain over the T1-like "
        "image. The corpus callosum arches from left to right in red, and the "
        "corticospinal tracts run up and down in blue"
    ),
    "fods_ismrm_peaks.png": (
        "FOD peaks drawn as short coloured lines on the same coronal slice, over "
        "the T1-like image"
    ),
    "fods_ismrm_asymmetric.png": (
        "Asymmetric FOD glyphs on a coronal slice of the right hemisphere, with "
        "lobes along the corticospinal tract and the corpus callosum. Some glyphs "
        "are one-sided where the fibres turn or end"
    ),
    "msmt_csd_cfin_fods.png": (
        "White matter FOD glyphs from MSMT-CSD on a coronal slice of the CFIN brain, "
        "over its mean b=0 image"
    ),
    "msmt_csd_cfin_wm.png": (
        "White matter fraction from MSMT-CSD on a coronal slice of the CFIN brain, "
        "bright in the white matter"
    ),
    "msmt_csd_cfin_gm.png": (
        "Grey matter fraction from MSMT-CSD on the same coronal slice, bright in the "
        "cortex and deep grey matter"
    ),
    "msmt_csd_cfin_csf.png": (
        "CSF fraction from MSMT-CSD on the same coronal slice, bright in the "
        "ventricles and around the brain"
    ),
    "tracking_synthetic.png": (
        "Streamlines of two synthetic bundles crossing at right angles, drawn as "
        "tubes in 3D. One bundle is red and runs left to right; the other is green "
        "and runs front to back"
    ),
    "tracking_ismrm_tubes.png": (
        "Whole-brain streamlines of the simulated brain drawn as lit tubes in 3D, "
        "seen from the front and slightly from the side, coloured by direction"
    ),
    "tracking_ismrm_slices.png": (
        "3D view of the simulated brain from above and in front. A tenth of the "
        "streamlines of 60 mm or more are drawn as thin lines over an axial slice "
        "of the T1-like image"
    ),
    "simulate_crossing_bundles.png": (
        "Three panels of the plane that holds two simulated bundles crossing at 90 "
        "degrees, one along x and one along y. The first panel is the b = 0 image, "
        "where the horizontal bundle is brighter. The second is a diffusion-"
        "weighted image with the gradient along x, where both bundles are dark and "
        "the horizontal one is darker. The third draws the ground-truth peaks as "
        "lines along each bundle, with two lines in the crossing"
    ),
    "signal_breakdown.png": (
        "Two charts of three stacked bars each, for a voxel of white matter, grey "
        "matter and CSF. The bars show the water with no decay, the signal after "
        "the T1 and T2 decay, and the signal after the diffusion weighting, each "
        "split by compartment. The right chart, at a longer echo time, is lower "
        "after the decay"
    ),
    "pipeline_ground_truth.png": (
        "Two panels. The left panel shows the plane of two simulated bundles "
        "crossing at 90 degrees, coloured by the angle between the true fibre "
        "directions and the fitted FOD peaks; it is dark along both bundles and "
        "brighter at the edges of the crossing. The right panel shows histograms of"
        " that angle, mostly under 2 degrees for the voxels with one bundle and "
        "spread a little wider in the crossing"
    ),
    "tract_profile_3d.png": (
        "A synthetic bundle drawn as tubes in 3D and coloured by its FA profile, "
        "yellow where FA is high and purple at the lesion halfway along"
    ),
    "tract_profile_plot.png": (
        "Line plot of FA against the point along the bundle. The line is flat at "
        "0.7 and dips to about 0.4 in the middle, at the lesion"
    ),
    "batches_3d.png": (
        "Three bundles of synthetic streamlines in 3D, one per batch written, each "
        "in its own colour"
    ),
    "tdi_2mm.png": (
        "Track density on a 2 mm grid, on an axial slice. A fan of streamlines is "
        "brightest in its narrow stem"
    ),
    "tdi_05mm.png": (
        "The same track density on a 0.5 mm grid, showing the single streamlines of"
        " the fan's spread-out end"
    ),
    "viewer_overlay_axial.png": (
        "Axial slice of a synthetic head with a blob overlaid in a hot colormap "
        "above a threshold, with colour bars, the crosshair and its readout"
    ),
    "viewer_overlay_coronal.png": (
        "Coronal slice of the same head and blob, with colour bars"
    ),
    "viewer_labels_builtin.png": (
        "Axial slice of a synthetic head with a parcellation into regions shown as "
        "labels in built-in colours"
    ),
    "viewer_labels_table.png": "The same parcellation coloured from a lookup table",
    "viewer_labels_3d.png": (
        "Four regions of the parcellation drawn as coloured surfaces in the 3D view"
    ),
    "viewer_4d_b0.png": (
        "Axial slice of the first volume of a synthetic diffusion series in a "
        "viridis colormap, with a colour bar"
    ),
    "viewer_4d_gradient_x.png": (
        "A later volume of the same series, with a gradient along the bundle, where"
        " the bundle's signal is darker"
    ),
    "viewer_crosshair_axial.png": (
        "Axial view of a synthetic head with the crosshair and the readout of the "
        "value under it"
    ),
    "viewer_crosshair_coronal.png": "Coronal view of the same head with the crosshair",
    "viewer_crosshair_sagittal.png": (
        "Sagittal view of the same head with the crosshair"
    ),
    "viewer_zoomed.png": (
        "The coronal view zoomed in and panned, drawn at twice the resolution"
    ),
    "viewer_las.png": (
        "The axial view in the radiological convention, with the subject's right on"
        " the screen's left"
    ),
    "viewer_clean.png": "The axial view with the crosshair and the text hidden",
    "viewer_all_views.png": (
        "The four views of the window in a 2 x 2 grid on a transparent background"
    ),
    "viewer_tracts_axial.png": (
        "Axial view of synthetic bundles over a head, where only the streamlines in"
        " a slab around the slice are drawn"
    ),
    "viewer_tracts_coronal.png": "Coronal view of the same bundles in a slab",
    "viewer_tracts_3d.png": (
        "A subset of the same bundles in the 3D view, in one orange colour"
    ),
    "viewer_tubes_lines.png": "Two crossing synthetic bundles drawn as lines in 3D",
    "viewer_tubes_direction.png": (
        "The same bundles drawn as lit tubes, coloured by direction"
    ),
    "viewer_tubes_single.png": "The same tubes in one amber colour",
    "viewer_peaks_amplitude.png": (
        "Synthetic FOD peaks drawn as lines on an axial slice of a head, with "
        "lengths set by amplitude. A band through the middle has two crossing peaks"
        " per voxel"
    ),
    "viewer_peaks_directions.png": (
        "The same peaks drawn with equal lengths, showing directions only"
    ),
    "viewer_surfaces_curvature.png": (
        "A synthetic folded surface in the 3D view, coloured light and dark by a "
        "curvature-like value"
    ),
    "viewer_surfaces_cut_line.png": (
        "Axial slice of a synthetic head with the line where the slice cuts the surface"
    ),
    "viewer_3d_default.png": (
        "3D view of a synthetic head with three slices and a volume rendering, from"
        " the default position"
    ),
    "viewer_3d_inside.png": (
        "The head with its volume rendering dimmed, so that the surface of the ball"
        " inside shows"
    ),
    "viewer_3d_slices.png": (
        "The three slices and the ball's surface alone, seen from the front"
    ),
    "viewer_fod_glyphs.png": (
        "Synthetic FOD glyphs on an axial slice, showing crossing fibres, a single "
        "bundle, one-sided FODs at its end, and a rim of round FODs"
    ),
    "viewer_fod_glyphs_anisotropic.png": (
        "The same glyphs with the round FODs of the rim hidden"
    ),
    "viewer_fod_inspector.png": (
        "One asymmetric FOD drawn large, its single lobe pointing to the subject's "
        "right"
    ),
    "viewer_dti_axial.png": (
        "Axial slice of synthetic DTI colours of the principal direction"
    ),
    "viewer_dti_coronal.png": "Coronal slice of the same DTI colours",
    "viewer_dti_fa_axial.png": (
        "The axial DTI colours modulated by FA, darker where diffusion is isotropic"
    ),
    "viewer_dti_fa_coronal.png": "The coronal DTI colours modulated by FA",
    "viewer_views_axial.png": (
        "Axial view of a synthetic head with a ball inside it, saved on its own"
    ),
    "viewer_views_coronal.png": "Coronal view of the same head, saved on its own",
    "viewer_views_sagittal.png": "Sagittal view of the same head, saved on its own",
    "viewer_rotation.webp": "Animation of the 3D view turning around a synthetic head",
    "viewer_sessions.png": (
        "The four views of a session opened again after its project folder was moved"
    ),
    "viewer_real_axial.png": (
        "Axial slice of the simulated T1-like image with the FA map over it in a "
        "hot colormap, transparent below 0.2"
    ),
    "viewer_real_sagittal.png": (
        "Sagittal slice of the simulated brain, showing the T1-like image with FA "
        "over it, brightest in the corpus callosum and the brainstem"
    ),
}


def scrape(
    block: object, block_vars: dict[str, Any], gallery_conf: dict[str, Any]
) -> str:
    """Moves the images a code block wrote into the gallery and shows them."""
    folder = Path(block_vars["src_file"]).parent
    written = sorted(
        (p for p in folder.iterdir() if p.suffix in IMAGE_SUFFIXES),
        key=lambda p: p.stat().st_mtime_ns,
    )
    if not written:
        return ""
    shown = []
    for source, target in zip(written, block_vars["image_path_iterator"], strict=False):
        target = Path(target).with_suffix(".webp")
        if source.suffix == ".webp":
            source.replace(target)
        else:
            with Image.open(source) as image:
                image.save(target, quality=WEBP_QUALITY, method=6)
            source.unlink()
        alt = ALT_TEXT.get(source.name)
        if alt is None:
            logger.warning(
                "%s writes %s, which has no alt text in docs/_ext/gallery_images.py",
                Path(block_vars["src_file"]).name,
                source.name,
            )
            alt = source.stem.replace("_", " ")
        shown.append((target, alt))
    rst = _images_rst(shown, gallery_conf["src_dir"])
    first = shown[0][0].stem.endswith("_001")  # the example's first images
    if first and ISMRM_FETCHER in Path(block_vars["src_file"]).read_text("utf-8"):
        rst += CREDIT
    return rst


def clear_leftovers(gallery_conf: dict[str, Any], fname: object, when: str) -> None:
    """Deletes the images left in the examples' folders, before an example runs."""
    if when != "before":
        return
    folders = gallery_conf["examples_dirs"]
    for folder in [folders] if isinstance(folders, str) else folders:
        for path in (Path(gallery_conf["src_dir"]) / folder).rglob("*"):
            if path.suffix in IMAGE_SUFFIXES:
                path.unlink()


def _images_rst(shown: list[tuple[Path, str]], src_dir: str) -> str:
    """The images, one under the other, each as wide as the text column allows."""
    parts = []
    for path, alt in shown:
        relative = path.relative_to(src_dir).as_posix()
        parts.append(
            f"\n.. image:: /{relative}\n   :alt: {alt}\n"
            "   :class: sphx-glr-single-img\n"
        )
    return "".join(parts) + "\n"
