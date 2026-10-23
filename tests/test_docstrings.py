import importlib
import inspect
import pkgutil
import re
from collections.abc import Iterator
from enum import Enum

import pytest

import larmor


def packages() -> list[str]:
    """The packages that define __all__."""
    names = ["larmor"]
    for info in pkgutil.walk_packages(larmor.__path__, "larmor."):
        if info.ispkg and hasattr(importlib.import_module(info.name), "__all__"):
            names.append(info.name)
    return names


SECTION = (
    r"^\s*(?:Args|Attributes|Returns|Yields|Raises|References|Example|Examples):\s*$"
)


def public_callables() -> Iterator[tuple[str, object]]:
    """Every function and class in an __all__, and the public methods of the classes."""
    for package in packages():
        module = importlib.import_module(package)
        for name in module.__all__:
            obj = getattr(module, name)
            if inspect.isfunction(obj):
                yield f"{package}.{name}", obj
            elif inspect.isclass(obj) and not issubclass(obj, Enum):
                yield f"{package}.{name}", obj
                for attr, member in vars(obj).items():
                    member = getattr(member, "__func__", member)
                    if not attr.startswith("_") and inspect.isfunction(member):
                        yield f"{package}.{name}.{attr}", member


def documented(doc: str, section: str) -> str:
    """The text of one docstring section, or an empty string."""
    match = re.search(rf"^\s*{section}:\s*$(.*?)(?={SECTION}|\Z)", doc, re.M | re.S)
    return match.group(1) if match else ""


@pytest.mark.parametrize(("label", "obj"), list(public_callables()))
def test_public_docstring(label: str, obj: object) -> None:
    doc = inspect.cleandoc(inspect.getdoc(obj) or "")
    assert doc.split("\n")[0].strip(), f"{label}: no summary"
    if inspect.isclass(obj):
        init = vars(obj).get("__init__")
        doc += "\n" + inspect.cleandoc(getattr(init, "__doc__", None) or "")
        function = init if init is not None else obj
    else:
        function = obj
    try:
        parameters = inspect.signature(function).parameters
    except TypeError, ValueError:
        return
    args = documented(doc, "Args") + documented(doc, "Attributes")
    for name, parameter in parameters.items():
        if name in ("self", "cls") or parameter.kind in (
            parameter.VAR_POSITIONAL,
            parameter.VAR_KEYWORD,
        ):
            continue
        assert re.search(rf"^\s*{name}:", args, re.M), f"{label}: {name} not in Args"
    assert not re.search(r"^\s*\w+ \(.*\):", args, re.M), (
        f"{label}: Args holds a type in brackets; types live in the annotations"
    )
