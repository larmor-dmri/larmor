"""A ``carousel`` directive: images that take turns, each with a one-line caption.

Each line of the directive's content is one slide, ``file | alt text | caption``, with
the file in ``docs/_static/carousel/``. The HTML is Bootstrap's carousel, which the
PyData theme already loads; ``_static/larmor-carousel.js`` advances it.
"""

from html import escape
from typing import Any

from docutils import nodes
from docutils.parsers.rst import Directive
from sphinx.application import Sphinx

STATIC = "LARMOR-STATIC/"  # replaced by the page's path to _static/ when it is written
WIDTH, HEIGHT = 900, 600


class Carousel(Directive):
    """The ``carousel`` directive."""

    has_content = True

    def run(self) -> list[nodes.Node]:
        """Builds the carousel's HTML from the slide lines."""
        slides = []
        for line in self.content:
            if not line.strip():
                continue
            parts = [part.strip() for part in line.split("|")]
            if len(parts) != 3:
                raise self.error(f"a slide is 'file | alt | caption', not {line!r}")
            slides.append(parts)
        if not slides:
            raise self.error("the carousel has no slides")
        return [nodes.raw("", _html(slides), format="html")]


def _html(slides: list[list[str]]) -> str:
    """The carousel's HTML; the first slide shows without JavaScript."""
    indicators, items = [], []
    for i, (file, alt, caption) in enumerate(slides):
        first = i == 0
        indicators.append(
            '<button type="button" data-bs-target="#lar-carousel"'
            f' data-bs-slide-to="{i}"'
            f' aria-label="Slide {i + 1}"'
            + (' class="active" aria-current="true"' if first else "")
            + "></button>"
        )
        loading = "" if first else ' loading="lazy"'
        items.append(
            f'<div class="carousel-item{" active" if first else ""}"'
            f' role="group" aria-roledescription="slide"'
            f' aria-label="{i + 1} of {len(slides)}">'
            f'<img src="{STATIC}carousel/{escape(file)}" alt="{escape(alt)}"'
            f' width="{WIDTH}" height="{HEIGHT}" decoding="async"{loading}>'
            f'<p class="lar-carousel-caption">{escape(caption)}</p></div>'
        )
    controls = "".join(
        f'<button class="carousel-control-{side}" type="button"'
        f' data-bs-target="#lar-carousel" data-bs-slide="{side}">'
        f'<span class="carousel-control-{side}-icon" aria-hidden="true"></span>'
        f'<span class="visually-hidden">{label}</span></button>'
        for side, label in (("prev", "Previous"), ("next", "Next"))
    )
    return (
        '<div id="lar-carousel" class="carousel slide lar-carousel" tabindex="0"'
        ' aria-roledescription="carousel" aria-label="Screenshots from BrainViewer">'
        f'<div class="carousel-inner">{"".join(items)}</div>{controls}'
        f'<div class="carousel-indicators">{"".join(indicators)}</div></div>'
    )


def _static_paths(
    app: Sphinx,
    pagename: str,
    templatename: str,
    context: dict[str, Any],
    doctree: nodes.document | None,
) -> None:
    """Points the carousel's images at _static/ from the page being written."""
    body = context.get("body", "")
    if STATIC in body:
        context["body"] = body.replace(STATIC, context["pathto"]("_static/", 1))


def setup(app: Sphinx) -> dict[str, Any]:
    """Registers the directive, its script and the path fix."""
    app.add_directive("carousel", Carousel)
    app.add_js_file("larmor-carousel.js", loading_method="defer")
    app.connect("html-page-context", _static_paths)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
