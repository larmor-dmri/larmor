import subprocess
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from PIL import Image, ImageSequence
from rendercanvas import BaseLoop

from larmor.types import Surface, Volume
from larmor.viewer import Category, Convention, OptionName, Viewer, ViewKind
from larmor.viewer.model.events import Event, ViewStateChanged
from larmor.viewer.render.offscreen import OffscreenRenderer, rotation_frames

SIZE = (48, 32)


def _volume() -> Volume:
    # A bright column through the middle of the axial plane, along z.
    data = np.zeros((8, 8, 8), np.float32, order="F")
    data[2:6, 2:6, :] = 10.0
    return Volume(data, np.eye(4))


@pytest.fixture
def viewer(gpu: None) -> Viewer:
    viewer = Viewer()
    viewer.add(_volume(), Category.VOLUMES)
    # Text and crosshair would draw over the pixels the tests read.
    viewer.toggle_overlay()
    viewer.toggle_crosshair()
    return viewer


@pytest.fixture
def renderer(viewer: Viewer) -> Iterator[OffscreenRenderer]:
    with OffscreenRenderer(viewer, SIZE) as renderer:
        yield renderer


def _grey(image: np.ndarray) -> np.ndarray:
    return image.astype(float).mean(axis=-1)


def test_screenshot_writes_an_rgb_png_of_the_size_times_the_scale(
    viewer: Viewer, tmp_path: Path
) -> None:
    path = viewer.screenshot(ViewKind.AXIAL, tmp_path / "a.png", size=SIZE, scale=2)
    assert path == tmp_path / "a.png"
    with Image.open(path) as image:
        assert image.format == "PNG"
        assert image.mode == "RGB"
        assert image.size == (2 * SIZE[0], 2 * SIZE[1])
        pixels = _grey(np.asarray(image))
    # The column is bright in the middle of the view, the background black.
    assert pixels[32, 48] > 200
    assert pixels[0, 0] == 0


def test_each_view_shows_its_own_slice(renderer: OffscreenRenderer) -> None:
    axial = _grey(renderer.render(ViewKind.AXIAL))
    sagittal = _grey(renderer.render(ViewKind.SAGITTAL))
    # The column runs along z: a square in the axial view, a full-height band in
    # the sagittal one.
    assert axial[0, 24] == 0
    assert axial[16, 24] > 200
    assert sagittal[0, 24] > 200
    assert sagittal[31, 24] > 200


def test_scale_keeps_the_layout(viewer: Viewer, renderer: OffscreenRenderer) -> None:
    single = renderer.render(ViewKind.AXIAL).astype(float)
    with OffscreenRenderer(viewer, SIZE, scale=2) as double:
        image = double.render(ViewKind.AXIAL).astype(float)
    assert image.shape == (2 * SIZE[1], 2 * SIZE[0], 3)
    # Edges are sampled at a finer step, so only on average do they agree.
    down = image.reshape(SIZE[1], 2, SIZE[0], 2, 3).mean(axis=(1, 3))
    assert np.abs(down - single).mean() < 2


def test_an_open_renderer_follows_the_scene(
    viewer: Viewer, renderer: OffscreenRenderer
) -> None:
    assert _grey(renderer.render(ViewKind.AXIAL))[16, 24] > 200
    viewer.set(Category.VOLUMES, OptionName.VISIBLE, False)
    assert _grey(renderer.render(ViewKind.AXIAL)).max() == 0


def test_closing_stops_following_the_scene(gpu: None) -> None:
    viewer = Viewer()
    renderer = OffscreenRenderer(viewer, SIZE)
    renderer.close()
    viewer.add(_volume(), Category.VOLUMES)
    assert renderer.scene_renderer.adapters == {}


def test_the_context_manager_closes_the_renderer(gpu: None) -> None:
    viewer = Viewer()
    with OffscreenRenderer(viewer, SIZE) as renderer:
        viewer.add(_volume(), Category.VOLUMES)
        assert len(renderer.scene_renderer.adapters) == 1
    viewer.add(_volume(), Category.VOLUMES)
    assert renderer.scene_renderer.adapters == {}


def test_a_screenshot_imports_no_qt(gpu: None, tmp_path: Path) -> None:
    # A fresh interpreter: this test process may have imported Qt already.
    code = (
        "import sys\n"
        "import numpy as np\n"
        "from larmor.types import Volume\n"
        "from larmor.viewer import Category, Viewer, ViewKind\n"
        "viewer = Viewer()\n"
        "viewer.add(Volume(np.ones((4, 4, 4), np.float32), np.eye(4)), "
        "Category.VOLUMES)\n"
        "path = viewer.screenshot(ViewKind.AXIAL, sys.argv[1], size=(16, 16))\n"
        "qt = ('PySide6', 'shiboken6')\n"
        "print(path.is_file(), [m for m in sys.modules if m.startswith(qt)])\n"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", code, str(tmp_path / "a.png")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "True []"


def test_a_screenshot_starts_from_the_convention_set_before_it(
    viewer: Viewer, tmp_path: Path
) -> None:
    # Only on the left of x, so a mirror shows.
    data = np.zeros((8, 8, 8), np.float32, order="F")
    data[1:3, 2:6, :] = 10.0
    viewer.add(Volume(data, np.eye(4)), Category.OVERLAYS)
    images = {}
    for convention in (Convention.RAS, Convention.LAS):
        viewer.set_convention(convention)
        path = viewer.screenshot(
            ViewKind.AXIAL, tmp_path / f"{convention}.png", size=SIZE
        )
        with Image.open(path) as image:
            images[convention] = _grey(np.asarray(image))
    ras, las = images[Convention.RAS], images[Convention.LAS]
    assert not np.array_equal(ras, np.fliplr(ras))
    np.testing.assert_array_equal(las, np.fliplr(ras))


@pytest.mark.parametrize(
    ("view", "maximised", "scale", "expected"),
    [
        (ViewKind.AXIAL, None, 1, (48, 32)),
        (ViewKind.AXIAL, None, 2, (96, 64)),
        # Two views a row and a column, 2 logical pixels apart.
        (None, None, 1, (98, 66)),
        (None, None, 2, (196, 132)),
        (None, ViewKind.CORONAL, 2, (96, 64)),
    ],
    ids=["one view", "one view at 2", "all views", "all views at 2", "maximised"],
)
def test_a_screenshot_has_the_size_of_its_views(
    view: ViewKind | None,
    maximised: ViewKind | None,
    scale: int,
    expected: tuple[int, int],
    viewer: Viewer,
    tmp_path: Path,
) -> None:
    if maximised is not None:
        viewer.toggle_maximised(maximised)
    path = viewer.screenshot(view, tmp_path / "a.png", size=SIZE, scale=scale)
    with Image.open(path) as image:
        assert image.size == expected


def test_all_views_are_laid_out_as_in_the_window(
    viewer: Viewer, renderer: OffscreenRenderer
) -> None:
    image = renderer.render(None)
    width, height = SIZE
    np.testing.assert_array_equal(
        image[:height, :width], renderer.render(ViewKind.AXIAL)
    )
    np.testing.assert_array_equal(
        image[height + 2 :, :width], renderer.render(ViewKind.SAGITTAL)
    )
    assert (image[height : height + 2] == 128).all()


@pytest.mark.parametrize("transparent", [True, False])
def test_a_transparent_screenshot_has_alpha_only_where_something_is_drawn(
    transparent: bool, viewer: Viewer, tmp_path: Path
) -> None:
    path = viewer.screenshot(
        ViewKind.AXIAL, tmp_path / "a.png", size=SIZE, transparent=transparent
    )
    with Image.open(path) as image:
        assert image.mode == ("RGBA" if transparent else "RGB")
        pixels = np.asarray(image)
    if transparent:
        # The view is wider than the volume: its sides are background.
        assert pixels[16, 0, 3] == 0
        assert pixels[16, 24, 3] == 255


def test_a_half_opaque_slice_keeps_its_colour_and_half_its_alpha(
    viewer: Viewer, renderer: OffscreenRenderer
) -> None:
    opaque = renderer.render(ViewKind.AXIAL).astype(int)
    solid = renderer.render(ViewKind.AXIAL, transparent=True).astype(int)
    np.testing.assert_array_equal(solid[16, 24], [*opaque[16, 24], 255])
    viewer.set(Category.VOLUMES, OptionName.OPACITY_2D, 0.5)
    half = renderer.render(ViewKind.AXIAL, transparent=True).astype(int)
    # Straight colour, as at opacity 1, and the alpha composited once.
    assert abs(half[16, 24, 3] - 128) <= 2
    assert np.abs(half[16, 24, :3] - opaque[16, 24]).max() <= 3


def test_a_transparent_jpeg_is_refused(viewer: Viewer, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="PNG or TIFF"):
        viewer.screenshot(ViewKind.AXIAL, tmp_path / "a.jpg", transparent=True)
    assert not (tmp_path / "a.jpg").exists()


@pytest.mark.parametrize("shown", [True, False])
def test_the_overlay_choice_leaves_the_scene_switch_as_it_is(
    shown: bool, viewer: Viewer, tmp_path: Path
) -> None:
    if shown:
        viewer.toggle_overlay()
    images = {}
    for overlay in (True, False, None):
        path = viewer.screenshot(
            ViewKind.AXIAL, tmp_path / f"{overlay}.png", size=SIZE, overlay=overlay
        )
        with Image.open(path) as image:
            images[overlay] = np.asarray(image)
        assert viewer.scene.view.overlay_visible == shown
    assert not np.array_equal(images[True], images[False])
    np.testing.assert_array_equal(images[None], images[shown])


class _Loop:
    def __init__(self) -> None:
        self.calls: list[tuple[Callable[..., None], tuple[object, ...]]] = []

    def call_soon_threadsafe(self, f: Callable[..., None], *args: object) -> None:
        self.calls.append((f, args))

    def run(self) -> None:
        calls, self.calls = self.calls, []
        for f, args in calls:
            f(*args)


def test_a_save_with_work_in_the_background_waits_for_it(
    viewer: Viewer, tmp_path: Path
) -> None:
    loop = _Loop()
    with OffscreenRenderer(viewer, SIZE, loop=cast(BaseLoop, loop)) as renderer:
        before = renderer.render(ViewKind.AXIAL)
        # Peaks along x in every voxel: their lines are built in the background.
        peaks = np.zeros((8, 8, 8, 3), np.float32)
        peaks[..., 0] = 1.0
        viewer.add(Volume(peaks, np.eye(4)), Category.PEAKS)
        early = renderer.save(ViewKind.AXIAL, tmp_path / "early.png")
        for _ in range(10):
            loop.run()
            renderer.flush()
        settled = renderer.save(ViewKind.AXIAL, tmp_path / "settled.png")
    with Image.open(early) as a, Image.open(settled) as b:
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
        assert not np.array_equal(np.asarray(a), before)


def test_screenshot_views_writes_each_2d_view_to_its_own_file(
    viewer: Viewer, renderer: OffscreenRenderer, tmp_path: Path
) -> None:
    paths = viewer.screenshot_views(tmp_path / "fig.png", size=SIZE)
    kinds = (ViewKind.AXIAL, ViewKind.CORONAL, ViewKind.SAGITTAL)
    assert paths == tuple(tmp_path / f"fig_{kind}.png" for kind in kinds)
    for path, kind in zip(paths, kinds, strict=True):
        with Image.open(path) as image:
            np.testing.assert_array_equal(np.asarray(image), renderer.render(kind))
    with Image.open(paths[0]) as axial, Image.open(paths[2]) as sagittal:
        assert not np.array_equal(np.asarray(axial), np.asarray(sagittal))


def _frames(path: Path) -> list[np.ndarray]:
    with Image.open(path) as image:
        return [
            np.asarray(frame.convert("RGB")).astype(float)
            for frame in ImageSequence.Iterator(image)
        ]


@pytest.mark.parametrize("suffix", [".webp", ".gif"])
def test_a_rotation_video_turns_from_the_3d_view(
    suffix: str, viewer: Viewer, renderer: OffscreenRenderer, tmp_path: Path
) -> None:
    start = renderer.render(ViewKind.THREE_D).astype(float)
    path = viewer.record_rotation(
        tmp_path / f"turn{suffix}", size=SIZE, speed=90, duration=1.5, fps=2
    )
    assert path == tmp_path / f"turn{suffix}"
    frames = _frames(path)
    assert len(frames) == 3  # round(duration * fps)
    assert frames[0].shape == (SIZE[1], SIZE[0], 3)
    # WebP is lossy, GIF is 8-bit colour: the first frame only nearly matches.
    assert np.abs(frames[0] - start).mean() < 2
    assert all(np.abs(frame - start).mean() > 10 for frame in frames[1:])


def test_recording_leaves_the_scene_as_it_was(viewer: Viewer, tmp_path: Path) -> None:
    cameras = viewer.scene.view.cameras
    events: list[Event] = []
    viewer.scene.subscribe(events.append)
    viewer.record_rotation(tmp_path / "a.webp", size=SIZE, duration=1, fps=3)
    assert viewer.scene.view.cameras == cameras
    assert not [e for e in events if isinstance(e, ViewStateChanged)]


def test_a_rotation_stopped_early_puts_the_renderer_camera_back(
    renderer: OffscreenRenderer,
) -> None:
    before = renderer.render(ViewKind.THREE_D)
    frames = renderer.render_rotation(speed=45, duration=3, fps=1)
    next(frames)
    turned = next(frames)  # 45 degrees: the square column shows a corner
    frames.close()
    assert not np.array_equal(turned, before)
    assert np.array_equal(renderer.render(ViewKind.THREE_D), before)


def test_a_cancelled_rotation_keeps_the_existing_file(
    viewer: Viewer, tmp_path: Path
) -> None:
    path = tmp_path / "a.webp"
    path.write_bytes(b"old")
    calls: list[tuple[int, int]] = []

    def progress(done: int, total: int) -> bool:
        calls.append((done, total))
        return done < 2

    assert (
        viewer.record_rotation(path, size=SIZE, duration=1, fps=4, on_progress=progress)
        is None
    )
    assert calls == [(1, 4), (2, 4)]
    assert path.read_bytes() == b"old"
    assert sorted(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("separate", [False, True], ids=["one view", "2D views"])
@pytest.mark.parametrize(
    ("name", "transparent", "message"),
    [("a.jpg", True, "needs PNG or TIFF"), ("a.xyz", False, "Not an image format")],
    ids=["transparent jpeg", "unknown extension"],
)
def test_a_bad_image_path_is_refused_before_the_renderer_is_built(
    name: str,
    transparent: bool,
    message: str,
    separate: bool,
    viewer: Viewer,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def no_renderer(*args: object, **kwargs: object) -> None:
        raise AssertionError("refused only after the renderer was built")

    monkeypatch.setattr(OffscreenRenderer, "__init__", no_renderer)
    with pytest.raises(ValueError, match=message):
        if separate:
            viewer.screenshot_views(tmp_path / name, transparent=transparent)
        else:
            viewer.screenshot(ViewKind.AXIAL, tmp_path / name, transparent=transparent)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("name", "transparent", "fps", "message"),
    [
        ("a.gif", True, 2, "needs WebP"),
        ("a.mp4", False, 2, ".webp or .gif"),
        ("a.gif", False, 60, "at most 50 frames"),
    ],
    ids=["transparent gif", "unsupported extension", "fast gif"],
)
def test_a_rotation_video_in_a_bad_format_is_refused(
    name: str,
    transparent: bool,
    fps: int,
    message: str,
    viewer: Viewer,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def no_renderer(*args: object, **kwargs: object) -> None:
        raise AssertionError("refused only after the renderer was built")

    monkeypatch.setattr(OffscreenRenderer, "__init__", no_renderer)
    with pytest.raises(ValueError, match=message):
        viewer.record_rotation(
            tmp_path / name, size=SIZE, duration=0.1, fps=fps, transparent=transparent
        )
    assert list(tmp_path.iterdir()) == []


def test_a_gif_keeps_the_video_duration_at_any_frame_rate(
    viewer: Viewer, tmp_path: Path
) -> None:
    path = viewer.record_rotation(tmp_path / "a.gif", size=SIZE, duration=1, fps=30)
    assert path is not None
    with Image.open(path) as image:
        times = []
        for frame in ImageSequence.Iterator(image):
            times.append(frame.info["duration"])
    assert len(times) == 30
    assert abs(sum(times) - 1000) <= 10


def test_each_rotation_frame_turns_the_scene_by_speed_over_fps(
    viewer: Viewer, renderer: OffscreenRenderer, tmp_path: Path
) -> None:
    # An off-centre block in one corner: a turn the wrong way, by the wrong step
    # or about the wrong axis draws it elsewhere.
    data = np.zeros((8, 8, 8), np.float32, order="F")
    data[0:3, 0:2, 5:8] = 10.0
    viewer.add(Volume(data, np.eye(4)), Category.OVERLAYS)
    path = viewer.record_rotation(
        tmp_path / "turn.webp", size=SIZE, speed=90, duration=1.5, fps=2
    )
    assert path is not None
    start = viewer.scene.view.cameras.three_d
    renderer.scene_renderer.navigation.place_3d(start.rotated(45, 0))
    expected = renderer.render(ViewKind.THREE_D).astype(float)
    # WebP is lossy: the frame only nearly matches.
    assert np.abs(_frames(path)[1] - expected).mean() < 2


def test_a_transparent_rotation_video_keeps_its_alpha(
    viewer: Viewer, tmp_path: Path
) -> None:
    path = viewer.record_rotation(
        tmp_path / "a.webp", size=SIZE, duration=1, fps=2, transparent=True
    )
    assert path is not None
    with Image.open(path) as image:
        assert image.mode == "RGBA"
        assert np.asarray(image)[0, 0, 3] == 0


@pytest.mark.parametrize(
    ("speed", "duration", "fps", "error"),
    [
        ("30", 1.0, 2, TypeError),
        (30.0, 1.0, 2.0, TypeError),
        (30.0, 0.0, 2, ValueError),
        (30.0, -1.0, 2, ValueError),
        (30.0, 1.0, 0, ValueError),
    ],
    ids=["string speed", "float fps", "zero duration", "negative duration", "fps 0"],
)
def test_bad_rotation_settings_are_refused(
    speed: object, duration: object, fps: object, error: type[Exception]
) -> None:
    with pytest.raises(error):
        rotation_frames(speed, duration, fps)


def test_rotation_settings_may_be_numpy_numbers() -> None:
    assert rotation_frames(np.float32(30), np.float32(1), np.int64(2)) == 2


def test_the_axial_view_draws_a_surface_cut_in_its_colour(
    viewer: Viewer, renderer: OffscreenRenderer
) -> None:
    # An octahedron in the volume, cut by the axial plane through the cursor.
    unit = np.array(
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
        np.float32,
    )
    faces = np.array(
        [[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4], [2, 0, 5], [1, 2, 5]]
        + [[3, 1, 5], [0, 3, 5]]
    )
    surface = Surface(3 * unit + np.float32(3.2), faces)
    item = viewer.add(surface, Category.SURFACES, name="lh.pial")  # pial is red

    def red_pixels() -> int:
        image = renderer.render(ViewKind.AXIAL).astype(int)
        r, g, b = image[..., 0], image[..., 1], image[..., 2]
        return int(((r > 200) & (g < 60) & (b < 60)).sum())

    assert red_pixels() > 20
    viewer.set(item.id, OptionName.CUT_LINES, False)
    assert red_pixels() == 0


def test_export_draws_with_a_presets_size_and_leaves_the_scene_as_it_was(
    viewer: Viewer, tmp_path: Path
) -> None:
    item = viewer.scene.items[0]
    before = (viewer.scene.view, item.options)
    path = viewer.export("Website", tmp_path / "w.png", ViewKind.AXIAL)
    with Image.open(path) as image:
        assert image.size == (1800, 1200)
        assert image.mode == "RGB"
    assert (viewer.scene.view, viewer.scene.item(item.id).options) == before


def test_export_views_and_rotation_take_a_preset_by_name(
    viewer: Viewer, tmp_path: Path
) -> None:
    paths = viewer.export_views("Screen", tmp_path / "v.png")
    assert [p.name for p in paths] == ["v_axial.png", "v_coronal.png", "v_sagittal.png"]
    video = viewer.export_rotation(
        "Screen", tmp_path / "r.gif", speed=90.0, duration=0.2, fps=10
    )
    assert video == tmp_path / "r.gif"
    with pytest.raises(ValueError, match="no export preset"):
        viewer.export("Poster", tmp_path / "p.png")
