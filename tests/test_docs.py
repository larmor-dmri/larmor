"""Examples and task pages run, the CLI reference is current, and the site builds."""

import ast
import importlib.util
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import TextIO

import pytest

# docutils ships no type hints, and its stubs are not a dependency.
from docutils.core import publish_doctree  # type: ignore[import-untyped]

ROOT = Path(__file__).parent.parent
EXAMPLES = sorted((ROOT / "examples").rglob("*.py"))
# Read before the fixtures clear it: the folder of downloaded example datasets.
# Examples that download data run only when it is set, so a plain test run never
# downloads anything; CI sets it to a cached folder.
DATA_DIR = os.environ.get("LARMOR_DATA_DIR")
# The line an example that downloads data carries.
DATA_MARKER = "# needs: example-data"


@pytest.fixture(scope="session")
def matplotlib_cache(tmp_path_factory: pytest.TempPathFactory) -> Path:
    # One font cache for all example runs, not one rebuilt per test home.
    return tmp_path_factory.mktemp("matplotlib")


@pytest.mark.parametrize(
    "example", EXAMPLES, ids=lambda path: path.relative_to(ROOT / "examples").as_posix()
)
def test_example_runs(example: Path, tmp_path: Path, matplotlib_cache: Path) -> None:
    """Each example runs to the end without errors."""
    env = dict(os.environ)
    env["MPLCONFIGDIR"] = str(matplotlib_cache)
    if DATA_MARKER in example.read_text(encoding="utf-8"):
        if not DATA_DIR:
            pytest.skip("downloads example data: set LARMOR_DATA_DIR to run it")
        env["LARMOR_DATA_DIR"] = DATA_DIR
    result = subprocess.run(
        [sys.executable, str(example)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_examples_that_download_data_carry_the_marker() -> None:
    uses_data = re.compile(r"larmor\.data\b|from larmor import .*\bdata\b")
    unmarked = [
        e.name
        for e in EXAMPLES
        if uses_data.search(e.read_text(encoding="utf-8"))
        and DATA_MARKER not in e.read_text(encoding="utf-8")
    ]
    assert unmarked == []


DOCS = ROOT / "docs"
# The pages whose code runs as written, on the data they write themselves.
TASK_PAGES = [
    DOCS / "getting-started.md",
    DOCS / "quick-start.md",
    DOCS / "simulation.md",
    *(
        DOCS / "guides" / f"{name}.md"
        for name in (
            "loading-and-saving",
            "gradient-tables",
            "dti",
            "dki",
            "fods",
            "peaks",
            "tractography",
            "surfaces",
            "simulation",
        )
    ),
]
# A comment on the line before a block: the block is shown but not run, or runs only
# with the example data.
NOT_RUN = "<!-- not run:"
NEEDS_DATA = "<!-- needs: example-data -->"
FENCE = re.compile(r"^```(python|bash)\n(.*?)^```$", re.M | re.S)
INCLUDE = re.compile(r"^```\{include\} (\S+)\n```$", re.M)
TAB = re.compile(r"^:::\{tab-item\} (.+)$")


@dataclass(frozen=True)
class Block:
    """A python or bash block of a page."""

    language: str
    code: str
    marker: str  # the comment on the line before it, or ""
    tab: str | None  # the label of the tab it is in, if any


def _page_blocks(page: Path) -> list[Block]:
    """A page's python and bash blocks in order, its includes' among them."""
    text = INCLUDE.sub(
        lambda m: (page.parent / m[1]).read_text(encoding="utf-8").rstrip("\n"),
        page.read_text(encoding="utf-8"),
    )
    blocks = []
    for match in FENCE.finditer(text):
        before = text[: match.start()].rstrip("\n").split("\n")
        marker = before[-1] if before[-1].startswith("<!--") else ""
        tab = None
        for line in reversed(before):
            if line == ":::":  # a tab, or a dropdown, closed above the block
                break
            if found := TAB.match(line):
                tab = found[1]
                break
        blocks.append(Block(match[1], match[2], marker, tab))
    return blocks


@pytest.mark.parametrize("page", TASK_PAGES, ids=lambda p: p.stem)
def test_task_page_runs_as_written(
    page: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The page's blocks run without errors, in order, for a reader of each tab.

    A reader follows one tab of each tab set (Python, or the command line) and the
    blocks outside the tabs, in a folder of their own.
    """
    blocks = _page_blocks(page)
    assert blocks, page
    # The lar of this environment, whatever the PATH of the test run.
    path = f"{Path(sys.executable).parent}{os.pathsep}{os.environ.get('PATH', '')}"
    monkeypatch.setenv("PATH", path)
    if DATA_DIR:
        monkeypatch.setenv("LARMOR_DATA_DIR", DATA_DIR)
    tabs = sorted({b.tab for b in blocks if b.tab}) or [""]
    for tab in tabs:
        folder = tmp_path / (tab.replace(" ", "-") or "page")
        folder.mkdir()
        monkeypatch.chdir(folder)
        namespace: dict[str, object] = {}
        for block in blocks:
            if block.tab not in (None, tab) or block.marker.startswith(NOT_RUN):
                continue
            if block.marker == NEEDS_DATA and not DATA_DIR:
                continue
            if block.language == "python":
                exec(compile(block.code, str(page), "exec"), namespace)
                continue
            result = subprocess.run(
                ["bash", "-e", "-c", block.code],
                capture_output=True,
                text=True,
                check=False,
            )
            assert result.returncode == 0, f"{block.code}\n{result.stderr}"


def test_task_page_markers_are_known() -> None:
    """A misspelt marker would run a block meant to be skipped."""
    markers = {b.marker for page in TASK_PAGES for b in _page_blocks(page)}
    assert all(m in ("", NEEDS_DATA) or m.startswith(NOT_RUN) for m in markers)


def test_cli_reference_is_current(tmp_path: Path) -> None:
    """docs/cli.md matches the CLI; regenerate it with the command in docs/README.md."""
    generated = tmp_path / "cli.md"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "typer",
            "larmor.main",
            "utils",
            "docs",
            "--name",
            "lar",
            "--output",
            str(generated),
        ],
        check=True,
        capture_output=True,
    )
    assert generated.read_text(encoding="utf-8") == (
        ROOT / "docs" / "cli.md"
    ).read_text(encoding="utf-8"), (
        "docs/cli.md is out of date; regenerate it (see docs/README.md)"
    )


SITE_URL = "https://larmor.org/"


# Tests on the built site share one worker, so the site is built once, and two builds
# never write sphinx-gallery's pages into docs/ at the same time.
@pytest.fixture(scope="module")
def website(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The site, built once as published: a folder per page, warnings as errors.

    The gallery's pages are built without running the examples (plot_gallery=0);
    test_example_runs runs them. Test workers share one build, made by the first
    one: two builds at once would both write the gallery's pages into docs/.
    """
    shared = tmp_path_factory.getbasetemp()
    if os.environ.get("PYTEST_XDIST_WORKER"):
        shared = shared.parent  # this run's folder, common to its workers
    out = shared / "site"
    with open(shared / "site.lock", "w") as lock:
        _lock(lock)
        if not (out / "built").exists():
            _build_website(out)
    return out / "html"


def _lock(file: TextIO) -> None:
    """Waits for an exclusive lock on an open file; closing the file releases it."""
    if sys.platform == "win32":
        import msvcrt

        while True:
            try:
                msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
                return
            except OSError:
                time.sleep(1)
    else:
        import fcntl

        fcntl.flock(file, fcntl.LOCK_EX)


def _build_website(out: Path) -> None:
    """Builds the site into out/html, offline: no inventories for intersphinx."""
    env = {**os.environ, "LARMOR_DOCS_OFFLINE": "1"}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "sphinx",
            "-b",
            "dirhtml",
            "-W",
            "-q",
            "-D",
            "plot_gallery=0",
            "-d",
            str(out / "doctrees"),
            str(ROOT / "docs"),
            str(out / "html"),
        ],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    (out / "built").touch()  # only now: a failed build is built again, and fails


@pytest.mark.xdist_group("website")
def test_website_builds_without_warnings(website: Path) -> None:
    """The Sphinx site builds with -W (broken links, orphan pages fail it)."""
    assert (website / "index.html").exists()


@pytest.mark.xdist_group("website")
def test_every_llms_txt_link_is_a_built_page(website: Path) -> None:
    links = re.findall(
        r"\]\((https://[^)#]+)", (website / "llms.txt").read_text(encoding="utf-8")
    )
    assert links
    missing = [
        link
        for link in links
        if not link.startswith(SITE_URL)
        or not (website / link.removeprefix(SITE_URL) / "index.html").exists()
    ]
    assert missing == []


@pytest.mark.xdist_group("website")
def test_llms_txt_links_no_noindex_page(website: Path) -> None:
    links = re.findall(
        r"\]\((https://[^)#]+)", (website / "llms.txt").read_text(encoding="utf-8")
    )
    noindex = [
        link
        for link in links
        if 'content="noindex" name="robots"'
        in (website / link.removeprefix(SITE_URL) / "index.html").read_text(
            encoding="utf-8"
        )
    ]
    assert noindex == []


IMAGE_NAME = re.compile(r"[a-z0-9_{}]+\.(png|webp)")
VIEWS = ("axial", "coronal", "sagittal")


def _saved_images(example: Path) -> set[str]:
    """The image files an example saves, read from its code without running it."""
    names = set()
    for node in ast.walk(ast.parse(example.read_text(encoding="utf-8"))):
        if isinstance(node, ast.JoinedStr):  # f"name_{view}.png", over the 2D views
            text = "".join(
                str(part.value) if isinstance(part, ast.Constant) else "{}"
                for part in node.values
            )
            if IMAGE_NAME.fullmatch(text):
                names |= {text.format(view) for view in VIEWS}
        elif isinstance(node, ast.Call) and getattr(node.func, "attr", "") == (
            "screenshot_views"
        ):  # one file per 2D view, named after the one given
            first = node.args[0]
            assert isinstance(first, ast.Constant) and isinstance(first.value, str)
            names |= {f"{Path(first.value).stem}_{view}.png" for view in VIEWS}
            first.value = ""  # the name itself is not a file written
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if IMAGE_NAME.fullmatch(node.value):
                names.add(node.value)
    return names


def _load(path: Path, name: str) -> ModuleType:
    """Imports a module of the docs folder by its path."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_screenshot_has_alt_text_and_every_alt_text_a_screenshot() -> None:
    sys.path.insert(0, str(ROOT / "docs" / "_ext"))
    try:
        alt_text = _load(ROOT / "docs" / "_ext" / "gallery_images.py", "g").ALT_TEXT
    finally:
        sys.path.remove(str(ROOT / "docs" / "_ext"))
    saved = set().union(*(_saved_images(example) for example in EXAMPLES))
    assert saved - alt_text.keys() == set()
    assert alt_text.keys() - saved == set()


def test_a_gallery_description_is_the_first_paragraph_under_the_title() -> None:
    first_paragraph = _load(ROOT / "docs" / "conf.py", "docs_conf")._first_paragraph
    note = ".. note::\n\n   Go to the end to download the code.\n\n"
    text = "Fits DTI & prints FA. " + "More words here. " * 30
    doctree = publish_doctree(
        f"{note}Title\n=====\n\n{text}\n\nSecond paragraph.\n",
        settings_overrides={"doctitle_xform": False, "report_level": 5},
    )
    description = first_paragraph(doctree)
    assert description.startswith("Fits DTI &amp; prints FA. More words")
    assert description.endswith("here.")
    assert len(description) <= 300 + len("&amp;") - 1


GUIDES = ROOT / "src" / "larmor" / "viewer" / "resources" / "guides"


def test_every_viewer_guide_is_included_by_one_page_of_the_site() -> None:
    included = [
        (page.parent / target).resolve()
        for page in (ROOT / "docs").rglob("*.md")
        if "_build" not in page.parts
        for target in re.findall(
            r"```\{include\} (\S+)", page.read_text(encoding="utf-8")
        )
    ]
    guides = sorted(GUIDES.glob("[0-9][0-9]-*.md"))
    assert sorted(p for p in included if p.parent == GUIDES) == guides


@pytest.mark.xdist_group("website")
def test_the_tool_and_shortcut_pages_list_every_tool_and_shortcut(
    website: Path,
) -> None:
    from html import unescape

    from larmor.viewer import Viewer
    from larmor.viewer.mcp_server import actions
    from larmor.viewer.model.shortcuts import SHORTCUTS

    tools = (website / "agents" / "tools" / "index.html").read_text(encoding="utf-8")
    names = re.findall(
        r'<span class="pre">(\w+)</span></span><span class="sig-paren">', tools
    )
    assert names == sorted({*actions(Viewer()), "scene"})
    shortcuts = unescape(
        (website / "viewer" / "shortcuts" / "index.html").read_text(encoding="utf-8")
    )
    assert all(s.description in shortcuts for s in SHORTCUTS.values())


@pytest.mark.xdist_group("website")
def test_no_page_links_to_the_sources(website: Path) -> None:
    sources = re.compile(r"github\.com/larmor-dmri/[\w.-]+/(?:blob|tree|raw)/")
    pages = [
        p
        for p in website.rglob("*.html")
        if sources.search(p.read_text(encoding="utf-8"))
    ]
    assert pages == []


API_PAGES = ROOT / "docs" / "reference" / "api"


def _api_pages() -> dict[str, str]:
    """The API reference's pages, by the package each documents."""
    pages = {}
    for page in API_PAGES.glob("*.rst"):
        match = re.search(
            r"^\.\. api-package:: (\S+)$", page.read_text(encoding="utf-8"), re.M
        )
        assert match, page
        pages[match[1]] = page.stem
    return pages


def _has_all(init: Path) -> bool:
    """Whether a package's __init__.py assigns __all__, read without importing it."""
    return any(
        isinstance(node, ast.Assign)
        and any(getattr(target, "id", "") == "__all__" for target in node.targets)
        for node in ast.parse(init.read_text(encoding="utf-8")).body
    )


def test_every_package_with_all_has_an_api_page() -> None:
    src = ROOT / "src"
    packages = {
        ".".join(init.parent.relative_to(src).parts)
        for init in (src / "larmor").rglob("__init__.py")
        if _has_all(init)
    }
    assert set(_api_pages()) == packages


@pytest.mark.xdist_group("website")
def test_api_pages_show_exactly_the_names_in_all(website: Path) -> None:
    """Each package's page has an entry, or for a re-export a link, per name."""
    entry = re.compile(r'<dt class="sig[^"]*" id="([^"]+)"')
    # The link in a row's first cell, the name; not those in the summaries.
    link = re.compile(
        r'<tr[^>]*><td><p><a class="reference internal" href="(.*?)#(.*?)"'
    )
    for package, stem in _api_pages().items():
        page = website / "reference" / "api" / stem
        html = (page / "index.html").read_text(encoding="utf-8")
        ids = entry.findall(html)
        assert all(i.startswith(f"{package}.") for i in ids), package
        names = [i.removeprefix(f"{package}.") for i in ids]
        # A member's first part is its class, which has an entry of its own.
        documented = {name.split(".")[0] for name in names}
        assert documented <= set(names), package
        table = html.split('<table class="autosummary', 1)[1].split("</table>", 1)[0]
        listed = set()
        for href, anchor in link.findall(table):
            target = page / href / "index.html" if href else page / "index.html"
            assert f'id="{anchor}"' in target.read_text(encoding="utf-8"), anchor
            listed.add(anchor.rsplit(".", 1)[1])
        expected = set(importlib.import_module(package).__all__)
        assert listed | documented == expected, package
