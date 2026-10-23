"""Sphinx configuration of the Larmor documentation website.

Build it from the repository root, with the folder of the example data, with
``uv run sphinx-build -b dirhtml -W docs docs/_build/html`` and LARMOR_DATA_DIR set;
or, without running the gallery's examples, with ``-D plot_gallery=0`` added.
"""

import importlib.util
import os
import re
import shutil
import sys
from html import escape as html_escape
from importlib.metadata import version as _installed_version
from pathlib import Path
from typing import Any

from docutils.nodes import document, paragraph, section
from sphinx.addnodes import pending_xref
from sphinx.application import Sphinx
from sphinx.transforms.post_transforms import SphinxPostTransform
from sphinx.util import logging
from sphinx_gallery.sorting import ExplicitOrder

# The site's own extensions, in docs/_ext/.
sys.path.insert(0, str(Path(__file__).parent / "_ext"))

project = "Larmor"
author = "Antonio Di Meglio"
copyright = "2026, Antonio Di Meglio"
release = _installed_version("larmor-dmri")
version = ".".join(release.split(".")[:2])

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_sitemap",
    "sphinxext.opengraph",
    "sphinx_gallery.gen_gallery",
    "carousel",
    "viewer_docs",
    "api_reference",
]

root_doc = "index"
# Markdown for the pages; reStructuredText for the gallery's pages, which
# sphinx-gallery writes from the examples.
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}

# Documents for contributors stay in the repository but out of the site.
exclude_patterns = [
    "_build",
    "_extra",
    "_snippets",
    "README.md",
    "contributing.md",
    "releasing.md",
    "style.md",
]

# The gallery: sphinx-gallery runs each script in examples/ and writes its page, with
# the code, its printed output and the screenshots it saves (see
# docs/_ext/gallery_images.py), into docs/gallery/, which is not committed. Examples
# that download a dataset run only when LARMOR_DATA_DIR names the folder to keep it
# in; without it, their pages show the code alone. `-D plot_gallery=0` builds every
# page without running any example.
EXAMPLES = Path(__file__).parent.parent / "examples"
DATA_MARKER = "# needs: example-data"
_NEEDS_DATA = sorted(
    path.name for path in EXAMPLES.rglob("*.py") if DATA_MARKER in path.read_text()
)
if os.environ.get("LARMOR_DATA_DIR") or not _NEEDS_DATA:
    _RUN_PATTERN = r"\.py$"
else:
    _SKIPPED = "|".join(re.escape(name) for name in _NEEDS_DATA)
    _RUN_PATTERN = rf"^(?!.*[/\\](?:{_SKIPPED})$).*\.py$"

sphinx_gallery_conf = {
    "examples_dirs": "../examples",
    "gallery_dirs": "gallery",
    "subsection_order": ExplicitOrder(
        [
            "../examples/simulation",
            "../examples/preprocessing",
            "../examples/models",
            "../examples/sd",
            "../examples/tractography",
            "../examples/files",
            "../examples/viewer",
        ]
    ),
    "within_subsection_order": "FileNameSortKey",
    "filename_pattern": _RUN_PATTERN,
    "image_scrapers": ("gallery_images.scrape",),
    "reset_modules": ("gallery_images.clear_leftovers",),
    # The symbol in grey on a transparent background, for examples without images.
    "default_thumb_file": str(Path(__file__).parent / "_static" / "gallery-thumb.png"),
    "download_all_examples": False,
    "show_signature": False,
    "show_memory": False,
    "write_computation_times": False,
    "remove_config_comments": True,
    "line_numbers": False,
    "capture_repr": (),  # printed output only, not the value of a last expression
    "min_reported_time": 3600,  # no running time under the examples
}

# Markdown: maths with $...$ and $$...$$, ::: fences for directives, and anchors on
# headings up to level 3, as GitHub makes them, so links like cli.md#lar-track work.
myst_enable_extensions = ["colon_fence", "dollarmath"]
myst_heading_anchors = 3

# Docstrings are in the Google style.
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_rtype = False
autodoc_typehints = "description"

# The API reference (docs/_ext/api_reference.py): autodoc imports every public package,
# the viewer too, whose drawing and window libraries are optional; those not installed
# are stood in for, so the docs group alone builds the site, without a display.
autodoc_mock_imports = [
    name
    for name in ("pygfx", "wgpu", "rendercanvas", "PySide6", "PySide6QtAds", "mcp")
    if importlib.util.find_spec(name) is None
]
autodoc_member_order = "bysource"
autodoc_typehints_description_target = "documented"
autosummary_generate = False  # the entries are on the package pages, not stub pages
# `name` in a docstring is code, linked when it names a documented object.
default_role = "py:obj"

# Links to the documentation of the libraries in the signatures and types. The tests
# build the site offline, with LARMOR_DOCS_OFFLINE set, and so without these links.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "nibabel": ("https://nipy.org/nibabel", None),
}
if os.environ.get("LARMOR_DOCS_OFFLINE"):
    intersphinx_mapping = {}
intersphinx_timeout = 30

# The site is public while the code is not: no links to the sources, nor copies of them.
html_show_sourcelink = False
html_copy_source = False

html_theme = "pydata_sphinx_theme"
html_title = "Larmor"
# The symbol on its tile, with the name beside it in the top bar.
html_logo = "../resources/larmor-symbol.svg"
html_favicon = "../resources/larmor-symbol.svg"
html_static_path = ["_static"]
# robots.txt, llms.txt and favicon.ico, copied to the root of the site.
html_extra_path = ["_extra"]
html_css_files = ["larmor.css"]
html_last_updated_fmt = None
# No footer: no copyright line (the licence is on the About page) and no "Created
# using Sphinx" line.
html_show_copyright = False
html_show_sphinx = False

# The landing page has no sidebars; every other page has the section navigation.
html_sidebars = {"index": []}

# The version switcher reads the list of versions from this address in the browser, so
# the site builds without it. version_match names the version being built: the
# workflow sets LARMOR_DOCS_VERSION to the tag on a release and to dev otherwise.
SWITCHER_VERSION = os.environ.get("LARMOR_DOCS_VERSION", "dev").removeprefix("v")

html_theme_options = {
    "logo": {"text": "Larmor"},
    "navbar_align": "left",
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["search-button", "version-switcher", "theme-switcher"],
    "switcher": {
        "json_url": "https://larmor.org/versions.json",
        "version_match": SWITCHER_VERSION,
    },
    "navbar_persistent": [],
    "header_links_before_dropdown": 8,
    "secondary_sidebar_items": {"index": [], "**": ["page-toc"]},
    "show_prev_next": False,
    "show_toc_level": 2,
    "footer_start": [],
    "footer_center": [],
    "footer_end": [],
}

# No general index: the API reference has its own pages.
html_use_index = False
html_domain_indices = False  # nor a module index

# Each page is a folder (/install/, built with -b dirhtml) under one canonical address.
html_baseurl = "https://larmor.org/"

# The sitemap lists every page but search, the 404 page and the pages marked noindex,
# which _page_metadata adds while the pages are written.
sitemap_url_scheme = "{link}"
sitemap_excludes = ["search/", "genindex/", "404/"]

# Link previews: the card image, and each page's meta description.
ogp_site_url = "https://larmor.org/"
ogp_image = "_static/larmor-card.png"
ogp_image_alt = "Larmor: a Python library and viewer for diffusion MRI"
ogp_social_cards = {"enable": False}  # one card for every page, not one drawn per page
ogp_custom_meta_tags = [
    '<meta property="og:image:width" content="1200" />',
    '<meta property="og:image:height" content="630" />',
    '<meta name="twitter:card" content="summary_large_image" />',
]

# Page titles read "Page - Larmor" (_templates/layout.html); these pages get their own,
# the landing page's in full.
templates_path = ["_templates"]
PAGE_TITLES = {
    "index": "Larmor: Python library and viewer for diffusion MRI",
    "cli": "Command-line reference",
}
# The meta description of the generated command-line reference, which has no front
# matter; every other page has its own in its front matter.
CLI_DESCRIPTION = (
    "Every command and option of lar, the Larmor command line for diffusion MRI: "
    "DTI, DKI, denoising, FODs, peaks, tracking and connectomes."
)

_META_DESCRIPTION = re.compile(r'<meta content="([^"]*)" name="description"')
# A gallery page's description is its first paragraph, cut at a sentence's end.
DESCRIPTION_LENGTH = 300


def _page_link(pagename: str) -> str:
    """The address of a page relative to the site, as the dirhtml builder makes it."""
    if pagename == "index":
        return ""
    return pagename.removesuffix("index").rstrip("/") + "/"


def _page_metadata(
    app: Sphinx,
    pagename: str,
    templatename: str,
    context: dict[str, Any],
    doctree: document | None,
) -> None:
    """Sets a page's title, link-preview description and sitemap entry."""
    if pagename in PAGE_TITLES:
        context["title"] = PAGE_TITLES[pagename]
    if pagename == "cli":
        context["metatags"] += (
            f'<meta content="{CLI_DESCRIPTION}" name="description" />\n'
        )
    elif pagename.startswith("gallery/") and doctree is not None:
        context["metatags"] += (
            f'<meta content="{_first_paragraph(doctree)}" name="description" />\n'
        )
    metatags = context.get("metatags", "")
    description = _META_DESCRIPTION.search(metatags)
    if description:
        fields = dict(context.get("meta") or {})
        fields["og:description"] = description[1]
        context["meta"] = fields
    if 'content="noindex" name="robots"' in metatags:
        app.config.sitemap_excludes.append(_page_link(pagename))


def _first_paragraph(doctree: document) -> str:
    """A page's first paragraph as plain text, short enough for a description."""
    # Directly under a heading: not the note sphinx-gallery puts above the text.
    paragraphs = (
        p for p in doctree.findall(paragraph) if isinstance(p.parent, section)
    )
    first = next(paragraphs, None)
    text = " ".join(first.astext().split()) if first is not None else ""
    if len(text) > DESCRIPTION_LENGTH:
        cut = text.rfind(". ", 0, DESCRIPTION_LENGTH)
        text = text[: cut + 1] if cut > 0 else text[:DESCRIPTION_LENGTH]
    return html_escape(text)


def _copy_404(app: Sphinx, exception: Exception | None) -> None:
    """Puts the 404 page at /404.html too, where hosts look for it."""
    page = Path(app.outdir) / "404" / "index.html"
    if exception is None and page.exists():
        shutil.copyfile(page, Path(app.outdir) / "404.html")


class _ExamplesToGallery(SphinxPostTransform):
    """Points links to examples/README.md, which GitHub follows, at the gallery.

    The gallery's pages are generated and not committed, so a page links to the
    examples' README for GitHub readers; on the site the link goes to the gallery.
    """

    default_priority = 8  # before MyST resolves the links (9)

    def run(self, **kwargs: object) -> None:
        """Rewrites the links' targets."""
        for node in self.document.findall(pending_xref):
            target = str(node.get("reftarget", ""))
            if node.get("reftype") == "myst" and target.endswith("/examples/README"):
                node["reftarget"] = "gallery/index"


def _warn_missing_data(app: Sphinx) -> None:
    """Warns when the gallery runs without its data, so its brain pages lack images."""
    runs = app.config.sphinx_gallery_conf["plot_gallery"]
    if runs and _NEEDS_DATA and not os.environ.get("LARMOR_DATA_DIR"):
        logging.getLogger(__name__).warning(
            "LARMOR_DATA_DIR is not set, so the gallery does not run %s: their "
            "pages show the code without images. Set it to the folder of the "
            "example data, or build with -D plot_gallery=0.",
            ", ".join(_NEEDS_DATA),
        )


def setup(app: Sphinx) -> None:
    """Connects the hooks, before the sitemap and link-preview extensions read pages."""
    app.connect("html-page-context", _page_metadata, priority=400)
    app.connect("build-finished", _copy_404)
    app.add_post_transform(_ExamplesToGallery)
    # After sphinx-gallery has read plot_gallery, from conf.py or -D.
    app.connect("builder-inited", _warn_missing_data, priority=900)
