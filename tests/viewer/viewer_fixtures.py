"""Fixtures shared by the viewer tests."""

import time
from collections.abc import Iterator
from typing import TYPE_CHECKING

import pytest

from larmor.viewer import Viewer

if TYPE_CHECKING:
    from larmor.viewer import ViewKind
    from larmor.viewer.qt.main_window import MainWindow
    from larmor.viewer.render.scene_renderer import SceneRenderer
    from larmor.viewer.render.view import ViewRenderer


@pytest.fixture(scope="session")
def gpu() -> None:
    import pygfx as gfx

    try:
        gfx.renderers.wgpu.get_shared()
    except Exception as error:
        pytest.skip(f"no GPU adapter for pygfx: {error}")


@pytest.fixture
def view_size() -> int:
    return 64


@pytest.fixture
def views(gpu: None, view_size: int) -> dict[ViewKind, ViewRenderer]:
    from rendercanvas.offscreen import RenderCanvas

    from larmor.viewer.model.item import VIEW_KINDS
    from larmor.viewer.render.view import ViewRenderer

    return {
        k: ViewRenderer(RenderCanvas(size=(view_size, view_size)), k)
        for k in VIEW_KINDS
    }


@pytest.fixture
def draws(
    views: dict[ViewKind, ViewRenderer], monkeypatch: pytest.MonkeyPatch
) -> dict[ViewKind, int]:
    from larmor.viewer.model.item import VIEW_KINDS
    from larmor.viewer.render.view import ViewRenderer

    counts: dict[ViewKind, int] = dict.fromkeys(VIEW_KINDS, 0)
    for kind, view in views.items():

        def count(kind: ViewKind = kind, view: ViewRenderer = view) -> None:
            counts[kind] += 1
            ViewRenderer.request_draw(view)

        monkeypatch.setattr(view, "request_draw", count)
    return counts


@pytest.fixture
def scene_renderer(
    views: dict[ViewKind, ViewRenderer],
) -> Iterator[tuple[Viewer, SceneRenderer]]:
    from larmor.viewer.render.scene_renderer import SceneRenderer

    viewer = Viewer()
    renderer = SceneRenderer(viewer, views)
    # The pixel checks sample at and around the cursor, where the crosshair and the
    # readout would draw.
    viewer.toggle_overlay()
    viewer.toggle_crosshair()
    yield viewer, renderer
    renderer.close()


@pytest.fixture
def fresh_slab_budget() -> Iterator[None]:
    """Starts each test with an empty slab budget, whatever earlier tests left."""
    from larmor.viewer.render.tractogram_adapter import SLAB_BUDGET

    SLAB_BUDGET.used = 0
    yield
    # Closed adapters give their bytes back once their sort ends, on a worker.
    deadline = time.monotonic() + 5
    while SLAB_BUDGET.used != 0 and time.monotonic() < deadline:
        time.sleep(0.01)
    SLAB_BUDGET.used = 0


@pytest.fixture
def viewer() -> Viewer:
    return Viewer()


@pytest.fixture(scope="module")
def _module_window(
    qapp: object, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[MainWindow]:
    # One shown, active window for a module's tests. Built before the per-test
    # scratch home exists, so it gets a home of its own.
    from PySide6.QtTest import QTest

    from larmor.viewer.qt.main_window import MainWindow

    home = tmp_path_factory.mktemp("window_home")
    with pytest.MonkeyPatch.context() as patch:
        for name in ("HOME", "USERPROFILE"):
            patch.setenv(name, str(home))
        patch.setenv("XDG_DATA_HOME", str(home / ".local" / "share"))
        patch.setenv("XDG_CONFIG_HOME", str(home / ".config"))
        patch.setenv("XDG_CACHE_HOME", str(home / ".cache"))
        window = MainWindow()
    window.show()
    assert QTest.qWaitForWindowExposed(window)
    window.activateWindow()
    assert QTest.qWaitForWindowActive(window)
    yield window
    window.close()
    window.deleteLater()


@pytest.fixture
def shared_window(_module_window: MainWindow) -> Iterator[MainWindow]:
    """The module's window, emptied and with the views in their initial state."""
    from larmor.viewer.model.view_state import ViewState

    def reset() -> None:
        viewer = _module_window.viewer
        viewer.clear()
        viewer.scene._next_id = 1
        viewer.set_view_state(ViewState())

    reset()
    yield _module_window
    reset()
