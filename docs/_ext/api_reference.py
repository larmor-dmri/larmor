"""The API reference: a page per public package, with exactly the names in its __all__.

A page holds the directive ``.. api-package:: larmor.dwi``, which writes the package's
docstring, a summary table of its names and then the entry of each name, from its
docstring. Only the names in ``__all__`` appear, so private ones stay out. The top
package re-exports names from the others: its table links to their entries there.
"""

import importlib
import inspect
import re
import sys
from enum import Enum
from pathlib import Path
from types import ModuleType
from typing import TypeAliasType, get_origin

from docutils.nodes import Node
from sphinx.application import Sphinx
from sphinx.errors import PycodeError
from sphinx.ext.autodoc import mock
from sphinx.ext.autosummary import Autosummary, extract_summary
from sphinx.pycode import ModuleAnalyzer
from sphinx.util.docutils import SphinxDirective

TOP = "larmor"
# The folder of the API pages, each with one `.. api-package::` line.
PAGES = Path(__file__).parent.parent / "reference" / "api"


def documented_packages() -> list[str]:
    """The packages that have a page, read from the pages."""
    found = (
        re.search(r"^\.\. api-package:: (\S+)$", page.read_text(), re.M)
        for page in PAGES.glob("*.rst")
    )
    return sorted(match[1] for match in found if match)


def home_package(name: str) -> str | None:
    """The package whose page documents a name the top package re-exports.

    Of the packages that export the same object, the one with the shortest name.

    Returns:
        The package name, or None if no package exports it.
    """
    obj = getattr(importlib.import_module(TOP), name)
    homes = []
    for package in documented_packages():
        module = importlib.import_module(package)
        if package != TOP and name in module.__all__ and getattr(module, name) is obj:
            homes.append(package)
    return min(homes, key=lambda p: (p.count("."), p)) if homes else None


def _data_entry(module: ModuleType, name: str, options: list[str]) -> list[str]:
    """The autodata directive of a constant or a typing alias; others are an error."""
    obj = getattr(module, name)
    if callable(obj) and get_origin(obj) is None:  # Callable[...] is an alias
        raise TypeError(
            f"{module.__name__}.{name} is a {type(obj).__name__}, which the API "
            "reference does not know how to document; add its kind to _entry."
        )
    return [f".. autodata:: {name}", *options, ""]


def _entry(module: ModuleType, name: str) -> list[str]:
    """The autodoc directive for one name, by what kind of object it is."""
    obj = getattr(module, name)
    if inspect.isclass(obj):
        kind = "exception" if issubclass(obj, BaseException) else "class"
        # An enum's members have no docstrings, but its values are its point.
        undoc = ["   :undoc-members:"] if issubclass(obj, Enum) else []
        return [f".. auto{kind}:: {name}", "   :members:", *undoc, ""]
    if inspect.isfunction(obj) or inspect.isbuiltin(obj):
        return [f".. autofunction:: {name}", ""]
    if isinstance(obj, TypeAliasType):
        # autodoc reads a type alias's docstring only in the module that documents
        # it, here the package, so it is given as the directive's content.
        doc = attribute_doc(module.__name__, name) or []
        return [f".. autotype:: {name}", "", *(f"   {line}" for line in doc), ""]
    return _data_entry(module, name, [])


def package_lines(package: str) -> list[str]:
    """The reStructuredText of a package's page, below its title."""
    module = importlib.import_module(package)
    names = list(module.__all__)
    lines = [f".. automodule:: {package}", "   :no-members:", ""]
    lines += [f".. currentmodule:: {package}", ""]
    if package == TOP:
        homes = {name: home_package(name) for name in names}
        listed = [f"~{homes[n]}.{n}" for n in names if homes[n] is not None]
        # Out of the package, as the items are named in full from other packages.
        lines += [".. currentmodule:: None", ""]
        lines += [".. api-autosummary::", "   :nosignatures:", ""]
        lines += [f"   {item}" for item in listed] + [""]
        lines += [f".. currentmodule:: {package}", ""]
        for name in names:
            if homes[name] is None:
                lines += _data_entry(module, name, ["   :no-value:"])
        return lines
    lines += [".. api-autosummary::", "   :nosignatures:", ""]
    lines += [f"   {name}" for name in names] + [""]
    for name in names:
        lines += _entry(module, name)
    return lines


def attribute_doc(package: str, name: str) -> list[str] | None:
    """The docstring of a constant or type alias, in the module that defines it.

    That is the `#:` comment above it, or the string below it. autodoc looks for it
    only in the package's own __init__, which imports it.

    Returns:
        The docstring lines, or None if there are none.
    """
    obj = getattr(sys.modules[package], name)
    for modname, module in list(sys.modules.items()):
        if modname != package and not modname.startswith(f"{package}."):
            continue
        if getattr(module, name, None) is not obj:
            continue
        try:
            docs = ModuleAnalyzer.for_module(modname).find_attr_docs()
        except PycodeError:
            continue
        if ("", name) in docs:
            return list(docs["", name])
    return None


def _in_larmor(module: str) -> bool:
    """Whether a module is Larmor's own."""
    return module == TOP or module.startswith(f"{TOP}.")


def _constant_docstring(
    app: Sphinx,
    what: str,
    name: str,
    obj: object,
    options: object,
    lines: list[str],
) -> None:
    """Gives a constant its `#:` comment, not the docstring of its type."""
    package, _, attr = name.rpartition(".")
    if what != "data" or not _in_larmor(package):
        return
    doc = attribute_doc(package, attr)
    if doc is not None:
        lines[:] = doc


def _enum_signature(
    app: Sphinx,
    what: str,
    name: str,
    obj: object,
    options: object,
    signature: str | None,
    return_annotation: str | None,
) -> tuple[str, str | None] | None:
    """Leaves out an enum's signature, `(*values)`, which says nothing."""
    if what == "class" and inspect.isclass(obj) and issubclass(obj, Enum):
        return "", return_annotation
    return None


class _Summary(Autosummary):
    """The autosummary table, with the summaries of type aliases too."""

    def get_items(self, names: list[str]) -> list[tuple[str, str | None, str, str]]:
        """The table's rows; a type alias's summary is its docstring's first line."""
        items = []
        for display, signature, summary, real_name in super().get_items(names):
            package, _, attr = real_name.rpartition(".")
            if not summary and _in_larmor(package):
                doc = attribute_doc(package, attr)
                if doc:
                    summary = extract_summary(doc, self.state.document.settings)
            items.append((display, signature, summary, real_name))
        return items


class ApiPackage(SphinxDirective):
    """Writes the reference of the package named as the argument."""

    required_arguments = 1

    def run(self) -> list[Node]:
        """Parses the generated reStructuredText into the page."""
        with mock(self.config.autodoc_mock_imports):
            text = "\n".join(package_lines(self.arguments[0]))
        return self.parse_text_to_nodes(text)


def setup(app: Sphinx) -> dict[str, bool]:
    """Registers the directive."""
    app.add_directive("api-package", ApiPackage)
    app.add_directive("api-autosummary", _Summary)
    app.connect("autodoc-process-docstring", _constant_docstring)
    app.connect("autodoc-process-signature", _enum_signature)
    return {"parallel_read_safe": True}
