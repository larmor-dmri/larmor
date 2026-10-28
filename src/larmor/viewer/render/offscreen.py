"""Draws a viewer's views off screen, with no window and no Qt."""

import math
from collections.abc import Callable, Generator, Iterator, Sequence
from importlib.resources import files
from numbers import Integral, Real
from pathlib import Path
from types import TracebackType
from typing import Self, TypeIs

import numpy as np
import pygfx as gfx
from PIL import Image
from pygfx.renderers.wgpu.engine.effectpasses import OutputPass

# The off-screen canvas stands in for Qt's; it renders into memory.
from rendercanvas import BaseLoop
from rendercanvas.offscreen import RenderCanvas

from larmor._log import LarmorLogger
from larmor.viewer.model.item import VIEW_GAP, VIEW_GRID, VIEW_KINDS, ViewKind
from larmor.viewer.model.view_state import Layout
from larmor.viewer.model.viewer import (
    ROTATION_DURATION,
    ROTATION_FPS,
    ROTATION_SPEED,
    Progress,
    Viewer,
)
from larmor.viewer.render.scene_renderer import SceneRenderer
from larmor.viewer.render.view import ViewRenderer
from larmor.viewer.render.worker import flush

DEFAULT_SIZE = (800, 800)
# The lines between the views of an image of all of them, as RGBA.
GAP_COLOUR = (128, 128, 128, 255)
# Formats with no alpha channel, which cannot hold a transparent background.
OPAQUE_FORMATS = (".jpg", ".jpeg")
# The 2D views, which `save_views` writes each to its own file.
VIEWS_2D: tuple[ViewKind, ...] = (ViewKind.AXIAL, ViewKind.CORONAL, ViewKind.SAGITTAL)
# The animated formats a rotation is written in, by extension, as Pillow's name for
# the format and whether it has an alpha channel.
VIDEO_FORMATS: dict[str, tuple[str, bool]] = {
    ".webp": ("WEBP", True),
    ".gif": ("GIF", False),
}
WEBP_QUALITY = 90
# Highest frame rate of a GIF. A GIF stores frame times in whole hundredths of a
# second, rounded down, and browsers play a frame of 10 ms or less as 100 ms.
GIF_MAX_FPS = 50
# pygfx's last pass multiplies the colour, already premultiplied, by the alpha
# again and squares the alpha; the off-screen views use a copy that writes straight
# colour and the alpha as composited.
_SSAA = (files("pygfx.renderers.wgpu") / "wgsl" / "ssaa.wgsl").read_text(
    encoding="utf-8"
)
_PREMULTIPLIED = "return vec4f(rgb * a, a * a);"
_STRAIGHT = "return vec4f(select(rgb / a, vec3f(0.0), a <= 0.0), a);"


class _StraightAlphaPass(OutputPass):  # type: ignore[misc]
    """Copies a view's drawing with straight colour and the composited alpha."""

    wgsl = _SSAA.replace(_PREMULTIPLIED, _STRAIGHT)


def straight_alpha(renderer: gfx.WgpuRenderer) -> None:
    """Makes a renderer write straight colour and alpha, if pygfx's pass allows."""
    if _PREMULTIPLIED not in _SSAA:
        LarmorLogger.warning(
            "pygfx's output pass has changed: transparent screenshots may be off."
        )
        return
    # The renderer's output pass is private API, set once per renderer.
    old = renderer._output_pass
    new = _StraightAlphaPass(gamma=old.gamma, filter=old.filter)
    renderer._output_pass = new


class _Canvas(RenderCanvas):  # type: ignore[misc]
    """Draws into memory, and can carry a loop for its views' background work."""

    def __init__(
        self,
        *args: object,
        work_loop: BaseLoop | None = None,
        changed: Callable[[], None] | None = None,
        **kwargs: object,
    ) -> None:
        """Creates the canvas; its workers hand results back through work_loop."""
        self.work_loop = work_loop
        self._changed = changed
        super().__init__(*args, **kwargs)

    def request_draw(self, draw_function: Callable[[], None] | None = None) -> None:
        """Remembers the draw function, and says the view has something new."""
        super().request_draw(draw_function)
        if self._changed is not None:
            self._changed()


def image_canvas(size: tuple[int, int], scale: float) -> RenderCanvas:
    """Returns a canvas that draws one picture into memory, its work done at once.

    Args:
        size: The width and height in logical pixels.
        scale: Pixels per logical pixel.

    Raises:
        ValueError: If size or scale is not positive, or the picture is larger
            than the GPU draws.
    """
    width, height = _check_size(size, scale)
    return _Canvas(size=(width, height), pixel_ratio=float(scale))


def draw_view(view: ViewRenderer, transparent: bool) -> np.ndarray:
    """Returns one view's picture as RGBA, its background transparent if asked."""
    view.background.visible = not transparent
    try:
        image = np.asarray(view.canvas.draw()).copy()
    finally:
        view.background.visible = True
    return image


class OffscreenRenderer:
    """Draws a viewer's four views into memory, as its window would show them.

    It follows the viewer's scene like a window does, so actions taken after it
    is created show in the next image.

    Args:
        viewer: The viewer whose scene is drawn.
        size: The width and height of each view in logical pixels, the units of
            the window's views; text and lines keep their size in them.
        scale: Pixels per logical pixel in the images, for example 2 for twice
            the resolution with the same layout.
        loop: An event loop to hand the results of background work (lines,
            glyphs, surfaces) back through; None runs the work at once, so that
            every image is complete.
        changed: Called whenever a view has something new to draw, such as the
            result of background work.

    Raises:
        ValueError: If size is not two positive whole numbers, scale is not a
            positive number, or the images would be larger than the GPU draws.
    """

    def __init__(
        self,
        viewer: Viewer,
        size: tuple[int, int] = DEFAULT_SIZE,
        scale: float = 1.0,
        *,
        loop: BaseLoop | None = None,
        changed: Callable[[], None] | None = None,
    ) -> None:
        """Creates a canvas and a renderer for each view, and draws the scene."""
        width, height = _check_size(size, scale)
        self.viewer = viewer
        self.size = (width, height)
        self.scale = float(scale)
        self.views = {
            kind: ViewRenderer(
                _Canvas(
                    size=(width, height),
                    pixel_ratio=self.scale,
                    work_loop=loop,
                    changed=changed,
                ),
                kind,
            )
            for kind in VIEW_KINDS
        }
        for view in self.views.values():
            straight_alpha(view.renderer)
        self.scene_renderer = SceneRenderer(viewer, self.views)

    def resize(self, size: tuple[int, int], scale: float | None = None) -> None:
        """Sets the size of each view, and its scale if given, for the next images.

        Args:
            size: The width and height of each view in logical pixels.
            scale: Pixels per logical pixel; None keeps the current one.

        Raises:
            ValueError: As for the constructor.
        """
        scale = self.scale if scale is None else scale
        width, height = _check_size(size, scale)
        for view in self.views.values():
            # The ratio first, as setting the logical size then sets the pixels.
            view.canvas.set_pixel_ratio(float(scale))
            view.canvas.set_logical_size(width, height)
        self.size = (width, height)
        self.scale = float(scale)

    def set_colours(self, background: str, guides: str) -> None:
        """Sets the views' background colour and the colour of the crosshair."""
        for view in self.views.values():
            view.set_colours(background, guides)

    def render(
        self,
        view: ViewKind | None,
        *,
        overlay: bool | None = None,
        transparent: bool = False,
    ) -> np.ndarray:
        """Draws one view, or the views as the window lays them out, into an image.

        Args:
            view: The view to draw, or None for the window's layout: the
                maximised view if there is one, else the four views in a 2 x 2
                grid with a thin grey line between them.
            overlay: Whether to draw the text and colour bars over the views;
                None draws them as the scene's information overlay switch says.
                The switch is not changed.
            transparent: Whether the background is transparent rather than the
                views' background colour.

        Returns:
            The image, rows from the top, uint8: (height, width, 3) RGB, or
            (height, width, 4) RGBA if transparent. Each view is its size times
            the scale, rounded.

        Raises:
            TypeError: If view is neither a ViewKind nor None.
        """
        if view is not None and not isinstance(view, ViewKind):
            raise TypeError(
                f"Expected a ViewKind, such as ViewKind.AXIAL, or None; got {view!r}."
            )
        return self.render_rows(
            layout(self.viewer, view), overlay=overlay, transparent=transparent
        )

    def render_rows(
        self,
        rows: Sequence[Sequence[ViewKind]],
        *,
        overlay: bool | None = None,
        transparent: bool = False,
    ) -> np.ndarray:
        """Draws views laid out in rows, a thin line between them, into an image.

        Args:
            rows: The views, row by row from the top, each left to right.
            overlay: Whether to draw the text and colour bars; None follows the
                scene's switch.
            transparent: Whether the background is transparent.

        Returns:
            The image, as `render` gives it.
        """
        tiles = [
            [self._draw(kind, overlay, transparent) for kind in row] for row in rows
        ]
        gap = round(VIEW_GAP * self.scale)
        colour = (0, 0, 0, 0) if transparent else GAP_COLOUR
        image = _compose(tiles, gap, colour)
        return image if transparent else image[..., :3].copy()

    def save(
        self,
        view: ViewKind | None,
        path: str | Path,
        *,
        overlay: bool | None = None,
        transparent: bool = False,
    ) -> Path:
        """Draws one view, or the window's layout, and writes the image to a file.

        Args:
            view: The view to draw, or None for the window's layout (see
                `render`).
            path: The file to write; its extension sets the format (.png, .jpg,
                .tif, ...). An existing file is replaced.
            overlay: Whether to draw the text and colour bars; None follows the
                scene's switch.
            transparent: Whether the background is transparent (RGBA); needs a
                format with an alpha channel, such as PNG or TIFF.

        Returns:
            The path written.

        Raises:
            TypeError: If view is neither a ViewKind nor None.
            ValueError: If the extension is not an image format, or it is JPEG
                and the background is transparent.
            OSError: If the file cannot be written.
        """
        p = Path(path)
        check_image(p, transparent)
        self.flush()
        image = self.render(view, overlay=overlay, transparent=transparent)
        Image.fromarray(image).save(p)
        return p

    def save_views(
        self,
        path: str | Path,
        *,
        overlay: bool | None = None,
        transparent: bool = False,
    ) -> tuple[Path, ...]:
        """Writes each 2D view to its own file, named after path and the view.

        Args:
            path: The name the files are made from: "fig.png" gives
                "fig_axial.png", "fig_coronal.png" and "fig_sagittal.png".
            overlay: Whether to draw the text and colour bars; None follows the
                scene's switch.
            transparent: Whether the background is transparent (RGBA); needs a
                format with an alpha channel, such as PNG or TIFF.

        Returns:
            The paths written: axial, coronal, sagittal.

        Raises:
            ValueError: As for `save`.
            OSError: If a file cannot be written.
        """
        p = Path(path)
        return tuple(
            self.save(
                kind, view_path(p, kind), overlay=overlay, transparent=transparent
            )
            for kind in VIEWS_2D
        )

    def render_rotation(
        self,
        speed: float = ROTATION_SPEED,
        duration: float = ROTATION_DURATION,
        fps: int = ROTATION_FPS,
        *,
        overlay: bool | None = None,
        transparent: bool = False,
    ) -> Generator[np.ndarray]:
        """Draws the 3D view turning about the screen's up, one frame at a time.

        The turn starts from the scene's 3D camera, which is never changed; this
        renderer's camera is put back when the frames end or the iterator is
        closed.

        Args:
            speed: Degrees the scene turns per second, towards the screen's
                right; negative turns it to the left.
            duration: Seconds of video.
            fps: Frames per second.
            overlay: Whether to draw the text and colour bars; None follows the
                scene's switch.
            transparent: Whether the background is transparent (RGBA).

        Returns:
            The frames as `render` gives them, duration times fps of them,
            rounded; a frame is drawn only when asked for.

        Raises:
            TypeError: If speed or duration is not a real number, or fps not a
                whole number.
            ValueError: If speed is not finite, duration not positive, or fps
                not positive.
        """
        count = rotation_frames(speed, duration, fps)
        start = self.viewer.scene.view.cameras.three_d
        navigation = self.scene_renderer.navigation

        def frames() -> Generator[np.ndarray]:
            """Yields each frame, and puts the camera back at the end."""
            try:
                for i in range(count):
                    navigation.place_3d(
                        start.rotated(rotation_angle(speed, fps, i), 0.0)
                    )
                    yield self.render(
                        ViewKind.THREE_D, overlay=overlay, transparent=transparent
                    )
            finally:
                navigation.place_3d()

        return frames()

    def save_rotation(
        self,
        path: str | Path,
        speed: float = ROTATION_SPEED,
        duration: float = ROTATION_DURATION,
        fps: int = ROTATION_FPS,
        *,
        overlay: bool | None = None,
        transparent: bool = False,
        on_progress: Progress | None = None,
    ) -> Path | None:
        """Writes the 3D view turning about the screen's up as an animated image.

        A WebP holds one frame at a time in memory; a GIF keeps each one, as
        8-bit colour, until the end.

        Args:
            path: The file to write: .webp or .gif. An existing file is replaced
                once the video is finished, and kept if it is cancelled.
            speed: Degrees the scene turns per second (see `render_rotation`).
            duration: Seconds of video.
            fps: Frames per second.
            overlay: Whether to draw the text and colour bars; None follows the
                scene's switch.
            transparent: Whether the background is transparent; needs WebP.
            on_progress: Called after each frame with the frames drawn and their
                total; it returns False to cancel.

        Returns:
            The path written, or None if on_progress cancelled it.

        Raises:
            TypeError: If speed, duration or fps is of the wrong type (see
                `render_rotation`).
            ValueError: If the extension is neither .webp nor .gif, it is GIF and
                the background is transparent or fps is above 50, or speed,
                duration or fps has a bad value (see `render_rotation`).
            OSError: If the file cannot be written.
        """
        p = Path(path)
        name, count = check_video(p, speed, duration, fps, transparent)
        self.flush()
        frames = self.render_rotation(
            speed, duration, fps, overlay=overlay, transparent=transparent
        )
        counted = _counted(frames, count, on_progress)
        # Written beside the file and moved over it once complete.
        temporary = p.with_name(f".{p.name}.part")
        try:
            first = Image.fromarray(next(counted))
            rest = _LazyFrames(counted, count - 1, first.mode, first.size)
            first.save(
                temporary,
                format=name,
                save_all=True,
                append_images=[rest] if count > 1 else [],
                duration=frame_times(count, fps) if name == "GIF" else 1000 / fps,
                loop=0,
                quality=WEBP_QUALITY,
            )
        except _CancelledError:
            temporary.unlink(missing_ok=True)
            return None
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        finally:
            frames.close()
        temporary.replace(p)
        return p

    def flush(self) -> None:
        """Finishes the background work, so that the next image is complete."""
        flush(view.canvas for view in self.views.values())

    def close(self) -> None:
        """Stops following the scene and frees the views' drawings."""
        self.scene_renderer.close()

    def _draw(
        self, kind: ViewKind, overlay: bool | None, transparent: bool
    ) -> np.ndarray:
        """Returns one view's image as RGBA, with the overlay and background asked."""
        view = self.views[kind]
        shown = view.overlay_visible
        if overlay is not None:
            view.overlay_visible = overlay
        # With no background object, the canvas is cleared to transparent black.
        view.background.visible = not transparent
        try:
            image = np.asarray(view.canvas.draw()).copy()
        finally:
            view.overlay_visible = shown
            view.background.visible = True
        return image

    def __enter__(self) -> Self:
        """Returns the renderer, which closes at the end of the block."""
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Closes the renderer."""
        self.close()


def view_path(path: Path, view: ViewKind) -> Path:
    """Returns the file of one view of `save_views`, the view's name after the stem."""
    return path.with_name(f"{path.stem}_{view}{path.suffix}")


def video_format(path: Path) -> tuple[str, bool]:
    """Returns Pillow's name for a video file's format, and whether it has alpha.

    Raises:
        ValueError: If the extension is neither .webp nor .gif.
    """
    found = VIDEO_FORMATS.get(path.suffix.lower())
    if found is None:
        raise ValueError(
            f"A rotation video is .webp or .gif, not {path.suffix!r}: {path}"
        )
    return found


def check_image(path: Path, transparent: bool) -> None:
    """Refuses an image file Pillow cannot write, or a transparent JPEG.

    Raises:
        ValueError: If the extension is not an image format Pillow writes, or it
            is JPEG and the background is transparent.
    """
    Image.init()
    if path.suffix.lower() not in Image.registered_extensions():
        raise ValueError(f"Not an image format: {path.suffix!r}: {path}")
    if transparent and path.suffix.lower() in OPAQUE_FORMATS:
        raise ValueError(
            f"A transparent screenshot needs PNG or TIFF, not {path.suffix}: {path}"
        )


def check_video(
    path: Path, speed: float, duration: float, fps: int, transparent: bool
) -> tuple[str, int]:
    """Returns Pillow's format name and the number of frames of a rotation video.

    Raises:
        TypeError: If speed, duration or fps is of the wrong type.
        ValueError: If the extension is neither .webp nor .gif, it is GIF and the
            background is transparent or fps is above 50, or speed, duration or
            fps has a bad value.
    """
    name, alpha = video_format(path)
    if transparent and not alpha:
        raise ValueError(f"A transparent video needs WebP, not {path.suffix}: {path}")
    count = rotation_frames(speed, duration, fps)
    if name == "GIF" and fps > GIF_MAX_FPS:
        raise ValueError(
            f"A GIF plays at most {GIF_MAX_FPS} frames per second, not {fps}; "
            f"use WebP: {path}"
        )
    return name, count


def frame_times(count: int, fps: int) -> list[int]:
    """Returns frame times in ms, in whole hundredths, as a GIF stores them."""
    # Rounding each frame's end, not its length, keeps the running total within
    # 5 ms of i / fps.
    ends = [round((i + 1) * 100 / fps) * 10 for i in range(count)]
    return [end - start for start, end in zip([0, *ends[:-1]], ends, strict=True)]


def rotation_frames(speed: object, duration: object, fps: object) -> int:
    """Returns the number of frames of a rotation video.

    Raises:
        TypeError: If speed or duration is not a real number, or fps not a
            whole number.
        ValueError: If speed is not finite, duration not positive and finite,
            or fps not positive.
    """
    if not _is_real(speed):
        raise TypeError(f"A speed is a number of degrees per second; got {speed!r}.")
    if not _is_real(duration):
        raise TypeError(f"A duration is a number of seconds; got {duration!r}.")
    if isinstance(fps, bool) or not isinstance(fps, Integral):
        raise TypeError(f"Frames per second is a whole number; got {fps!r}.")
    if not math.isfinite(float(speed)):
        raise ValueError(f"A speed is a finite number; got {speed!r}.")
    if not _is_positive(duration):
        raise ValueError(
            f"A duration is a positive number of seconds; got {duration!r}."
        )
    if fps <= 0:
        raise ValueError(f"Frames per second is positive; got {fps!r}.")
    return max(1, round(float(duration) * int(fps)))


def rotation_angle(speed: float, fps: int, frame: int) -> float:
    """Returns the degrees the scene has turned at a frame of a rotation video."""
    return frame * (float(speed) / fps)


class _CancelledError(Exception):
    """Raised through Pillow's writer when the progress callback cancels."""


def _counted(
    frames: Iterator[np.ndarray], total: int, progress: Progress | None
) -> Iterator[np.ndarray]:
    """Yields the frames, telling progress after each and stopping if it says so."""
    for done, frame in enumerate(frames, start=1):
        if progress is not None and not progress(done, total):
            raise _CancelledError
        yield frame


class _LazyFrames(Image.Image):
    """Gives Pillow's writers the frames of an iterator, one per seek, in order."""

    def __init__(
        self, frames: Iterator[np.ndarray], count: int, mode: str, size: tuple[int, int]
    ) -> None:
        """Creates an image of count frames, with the first frame's mode and size."""
        super().__init__()
        self._frames = frames
        self.n_frames = count
        self.is_animated = count > 1
        self._mode = mode
        self._size = size
        self._index = -1

    def seek(self, frame: int) -> None:
        """Makes the next frame current; frames come only in order."""
        if frame == self._index:
            return
        if frame != self._index + 1:
            raise ValueError(f"Frames are read in order: {frame} after {self._index}.")
        try:
            array = next(self._frames)
        except StopIteration:
            raise EOFError("No more frames.") from None
        self.im = Image.fromarray(array).im
        self._index = frame

    def tell(self) -> int:
        """Returns the current frame."""
        return max(self._index, 0)


def layout(viewer: Viewer, view: ViewKind | None) -> tuple[tuple[ViewKind, ...], ...]:
    """Returns the views an image shows, row by row: one, or the window's layout.

    Args:
        viewer: The viewer, whose maximised view and layout the window follows.
        view: The view to draw, or None for the window's views: the maximised
            view if there is one, the selected view in the One layout, else the
            four views in a grid, also for the 1 + 3 layout.

    Returns:
        The rows of views, top to bottom, each left to right.
    """
    if view is None:
        state = viewer.scene.view
        view = state.maximised
        if view is None and state.layout == Layout.ONE:
            view = state.selected_view
        if view is None:
            return VIEW_GRID
    return ((view,),)


def _compose(
    rows: list[list[np.ndarray]], gap: int, colour: tuple[int, int, int, int]
) -> np.ndarray:
    """Returns the images laid out in rows, gap pixels of colour between them."""
    heights = [max(tile.shape[0] for tile in row) for row in rows]
    widths = [sum(tile.shape[1] for tile in row) + gap * (len(row) - 1) for row in rows]
    image = np.empty(
        (sum(heights) + gap * (len(rows) - 1), max(widths), 4), dtype=np.uint8
    )
    image[...] = colour
    top = 0
    for row, height in zip(rows, heights, strict=True):
        left = 0
        for tile in row:
            image[top : top + tile.shape[0], left : left + tile.shape[1]] = tile
            left += tile.shape[1] + gap
        top += height + gap
    return image


def _check_size(size: object, scale: object) -> tuple[int, int]:
    """Returns a view size as two positive whole numbers.

    The scale must be a positive number, and the size times the scale within
    the largest image the GPU draws.

    Returns:
        The width and height.

    Raises:
        ValueError: If the size or scale is not valid.
    """
    values = tuple(size) if isinstance(size, (tuple, list)) else ()
    if len(values) != 2 or not all(
        isinstance(v, (int, np.integer)) and not isinstance(v, bool) and v > 0
        for v in values
    ):
        raise ValueError(
            f"A size is a width and a height, in whole pixels; got {size!r}."
        )
    if not _is_positive(scale):
        raise ValueError(f"A scale is a positive number; got {scale!r}.")
    assert _is_real(scale)
    width, height = int(values[0]), int(values[1])
    limit = gfx.renderers.wgpu.get_shared().device.limits["max-texture-dimension-2d"]
    if max(width, height) * float(scale) > limit:
        raise ValueError(
            f"A view of {width} x {height} at scale {scale:g} is larger than the "
            f"{limit} pixels a side the GPU draws; choose a smaller size or scale."
        )
    return width, height


def _is_real(value: object) -> TypeIs[Real]:
    """Returns whether value is a real number, such as a NumPy float, not a bool."""
    return isinstance(value, Real) and not isinstance(value, bool)


def _is_positive(value: object) -> bool:
    """Returns whether value is a finite real number above 0, not a bool."""
    return _is_real(value) and math.isfinite(float(value)) and float(value) > 0
