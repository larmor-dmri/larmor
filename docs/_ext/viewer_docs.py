"""The viewer's pages: its Help guides, its shortcuts and the tools agents call.

The guides are the Markdown files the viewer's Help panel shows; a page of the site
includes one with MyST's ``include``, so both read the same file. A guide links to
another by its file name (``09-settings.md``), which the Help panel opens; here
``_GuideLinks`` points such a link at the page that includes that guide.

``viewer-shortcuts`` and ``agent-tools`` write their lists from the code: the
shortcut registry, and the viewer's methods marked for agents with their
docstrings, as the MCP server describes them.
"""

import inspect
import re
from pathlib import Path

from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.statemachine import StringList
from sphinx.addnodes import pending_xref
from sphinx.application import Sphinx
from sphinx.ext.napoleon.docstring import GoogleDocstring
from sphinx.transforms.post_transforms import SphinxPostTransform

GUIDES = (
    Path(__file__).parents[2] / "src" / "larmor" / "viewer" / "resources" / "guides"
)
GUIDE_NAME = re.compile(r"\d{2}-.+\.md")
# A name in single backticks, as docstrings write code; rST reads it as a title.
_SINGLE_BACKTICKS = re.compile(r"(?<![`\w])`([^`]+)`(?![`\w])")


class _GuideLinks(SphinxPostTransform):
    """Points a link to a guide's file at the page of the site that includes it."""

    default_priority = 8  # before MyST resolves the links (9)

    def run(self, **kwargs: object) -> None:
        """Rewrites the links' targets."""
        pages = guide_pages(Path(self.env.srcdir), self.env.dependencies)
        for node in self.document.findall(pending_xref):
            if node.get("reftype") != "myst" or node.get("refdomain"):
                continue
            target, _, anchor = str(node.get("reftarget", "")).partition("#")
            name = Path(target).name
            if name in pages:
                node["refdomain"] = "doc"
                node["reftarget"] = pages[name]
                node["reftargetid"] = anchor or None


def guide_pages(srcdir: Path, dependencies: dict[str, set[str]]) -> dict[str, str]:
    """The page that includes each guide, by the guide's file name."""
    pages = {}
    for docname, files in dependencies.items():
        for file in files:
            path = (srcdir / file).resolve()
            if path.parent == GUIDES.resolve() and GUIDE_NAME.fullmatch(path.name):
                pages[path.name] = docname
    return pages


class ViewerShortcuts(Directive):
    """The ``viewer-shortcuts`` directive: the keys, then the mouse, as tables."""

    def run(self) -> list[nodes.Node]:
        """Writes the two tables from the shortcut registry."""
        from larmor.viewer.model.shortcuts import (
            MOUSE_CONTROLS,
            SHORTCUTS,
            views_text,
        )

        keys = []
        for shortcut in SHORTCUTS.values():
            description = shortcut.description
            if not shortcut.active:
                description += " (not available yet)"
            text = ", ".join(f"``{k}``" for k in shortcut.keys)
            keys.append((text, description, views_text(shortcut.views)))
        mouse = [
            (c.gesture, c.description, views_text(c.views)) for c in MOUSE_CONTROLS
        ]
        lines = [
            *_table("Keys", ("Keys", "Action", "Selected view"), keys),
            "",
            *_table("Mouse and trackpad", ("Mouse", "Action", "Views"), mouse),
        ]
        return _parse(self, lines)


class AgentTools(Directive):
    """The ``agent-tools`` directive: every tool, its arguments and description."""

    def run(self) -> list[nodes.Node]:
        """Writes one entry per tool, in alphabetical order, as agents see them."""
        from larmor.viewer import Viewer
        from larmor.viewer.mcp_server import (
            SCENE_DESCRIPTION,
            actions,
            description,
            to_json,
            tool_parameters,
        )

        config = self.state.document.settings.env.config
        tools = {"scene": ("", SCENE_DESCRIPTION)}
        for name, method in actions(Viewer()).items():
            arguments = ", ".join(
                p.name if p.default is p.empty else f"{p.name}={to_json(p.default)!r}"
                for p in tool_parameters(method)
            )
            tools[name] = (arguments, description(method))
        lines = []
        for name in sorted(tools):
            arguments, text = tools[name]
            text = _SINGLE_BACKTICKS.sub(r"``\1``", text)
            body = str(GoogleDocstring(inspect.cleandoc(text), config, what="method"))
            lines += [f".. py:function:: {name}({arguments})", "   :no-index:", ""]
            lines += [f"   {line}" if line else "" for line in body.splitlines()]
            lines.append("")
        return _parse(self, lines)


def _table(
    title: str, header: tuple[str, ...], rows: list[tuple[str, ...]]
) -> list[str]:
    """A list-table in reStructuredText."""
    lines = [f".. list-table:: {title}", "   :header-rows: 1", ""]
    for row in (header, *rows):
        lines.append(f"   * - {row[0]}")
        lines += [f"     - {cell}" for cell in row[1:]]
    return lines


def _parse(directive: Directive, lines: list[str]) -> list[nodes.Node]:
    """The nodes of reStructuredText lines, parsed in the directive's place."""
    container = nodes.container()
    source = directive.state.document.current_source or ""
    directive.state.nested_parse(
        StringList(lines, source), directive.content_offset, container
    )
    return list(container.children)


def setup(app: Sphinx) -> dict[str, bool]:
    """Adds the directives and the link transform."""
    app.add_directive("viewer-shortcuts", ViewerShortcuts)
    app.add_directive("agent-tools", AgentTools)
    app.add_post_transform(_GuideLinks)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
