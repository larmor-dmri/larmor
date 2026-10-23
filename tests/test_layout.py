import ast
from collections.abc import Iterator
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "src" / "larmor"
SCANNED = [ROOT / "src", ROOT / "tests", ROOT / "examples"]


def _module_of(path: Path) -> str:
    if not path.is_relative_to(PACKAGE):
        return ""
    parts = path.relative_to(PACKAGE.parent).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _imports(path: Path) -> Iterator[tuple[int, str, str]]:
    # `import a.b.c` is read as `from a.b import c` so every rule sees both forms.
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            for alias in node.names:
                yield node.lineno, node.module, alias.name
        elif isinstance(node, ast.Import):
            for alias in node.names:
                module, _, name = alias.name.rpartition(".")
                yield node.lineno, module, name


def _join(module: str, name: str) -> str:
    return f"{module}.{name}" if module else name


def _is_module(dotted: str) -> bool:
    base = PACKAGE.parent.joinpath(*dotted.split("."))
    return (
        base.with_suffix(".py").is_file()
        or base.with_suffix(".pyi").is_file()
        or (base / "__init__.py").is_file()
    )


def _is_private(part: str) -> bool:
    return part.startswith("_") and not (part.startswith("__") and part.endswith("__"))


def _inside(module: str, package: str) -> bool:
    return module == package or module.startswith(package + ".")


def _exports() -> dict[str, frozenset[str]]:
    found: dict[str, frozenset[str]] = {}
    for init in PACKAGE.rglob("__init__.py"):
        package = _module_of(init)
        # The top level curates names that live elsewhere: it is no home.
        if package == "larmor":
            continue
        for node in ast.parse(init.read_text(encoding="utf-8")).body:
            if (
                isinstance(node, ast.Assign)
                and any(
                    isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets
                )
                and isinstance(node.value, ast.List | ast.Tuple)
            ):
                found[package] = frozenset(
                    e.value
                    for e in node.value.elts
                    if isinstance(e, ast.Constant) and isinstance(e.value, str)
                )
    return found


def test_no_relative_imports_hide_from_the_layout_checks() -> None:
    relative = [
        f"{path.relative_to(ROOT)}:{node.lineno}"
        for path in sorted(PACKAGE.rglob("*.py"))
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.ImportFrom) and node.level > 0
    ]
    assert relative == []


def test_no_module_imports_a_private_name_from_another_module() -> None:
    offenders = [
        f"{path.relative_to(ROOT)}:{line}: {name} from {module}"
        for path in sorted(PACKAGE.rglob("*.py"))
        for line, module, name in _imports(path)
        if _inside(module, "larmor")
        and _is_private(name)
        # Private modules have their own rule.
        and not _is_module(_join(module, name))
    ]
    assert offenders == [], "\n".join(offenders)


def test_exported_names_are_imported_from_their_package_outside_it() -> None:
    packages = _exports()
    assert packages, "no subpackage with __all__ found"
    offenders = [
        f"{path.relative_to(ROOT)}:{line}: {name} from {module}, home {package}"
        for folder in SCANNED
        for path in sorted(folder.rglob("*.py"))
        for line, module, name in _imports(path)
        for package, names in packages.items()
        if module.startswith(package + ".")
        and name in names
        and not _inside(_module_of(path), package)
    ]
    assert offenders == [], "\n".join(offenders)


def test_private_modules_are_imported_only_from_their_parent_package() -> None:
    offenders = []
    for path in sorted(PACKAGE.rglob("*.py")):
        here = _module_of(path)
        for line, module, name in _imports(path):
            target = _join(module, name) if _is_module(_join(module, name)) else module
            parts = target.split(".")
            private = [i for i, part in enumerate(parts) if _is_private(part)]
            if not _inside(target, "larmor") or not private:
                continue
            if _inside(target, "larmor._core"):
                continue
            parent = ".".join(parts[: private[-1]])
            if not _inside(here, parent):
                offenders.append(f"{path.relative_to(ROOT)}:{line}: {target} in {here}")
    assert offenders == [], "\n".join(offenders)


CORE = [
    f"larmor.{p}"
    for p in (
        "types",
        "io",
        "surfaces",
        "dwi",
        "sd",
        "tractography",
        "registration",
        "simulation",
    )
]


def _all_of(init: Path) -> frozenset[str] | None:
    for node in ast.parse(init.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets
        ):
            return frozenset(ast.literal_eval(node.value))
    return None


def test_cli_imports_only_exported_names_from_the_core() -> None:
    # Unlike _exports, the top level counts: the CLI may use its curated names.
    exported = {
        _module_of(init): names
        for init in PACKAGE.rglob("__init__.py")
        if (names := _all_of(init)) is not None
    }
    offenders = [
        f"{path.relative_to(ROOT)}:{line}: {name} from {module}"
        for path in sorted((PACKAGE / "cli").rglob("*.py"))
        for line, module, name in _imports(path)
        if any(_inside(_join(module, name), core) for core in CORE)
        and name not in exported.get(module, frozenset())
        # `from larmor import sd` names a package, not a module inside one.
        and _join(module, name) not in exported
    ]
    assert offenders == [], "\n".join(offenders)
