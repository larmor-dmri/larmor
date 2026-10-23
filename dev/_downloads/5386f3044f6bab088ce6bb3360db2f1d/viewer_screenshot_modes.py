"""Separate views and rotation videos
==================================

This example builds a synthetic head with a ball inside it, saves the axial, coronal
and sagittal views to three files, and records a short video of the 3D view turning,
with a progress callback that reports when the last frame is drawn.

Besides one image of a view or of the window's layout, the viewer can save the three
2D views as separate images in one go, and record the 3D view turning round as an
animated WebP or GIF, for a slide or a web page.
"""

import numpy as np

from larmor.types import Volume
from larmor.viewer import Category, OptionName, Viewer

# %%
# A synthetic head
# ----------------
# The head is a 2 mm image of an ellipsoid with a bright scalp, dark fluid under it
# and a brain whose intensity rises and falls with the distance from its centre. The
# mask is a ball of radius 20 mm in the right hemisphere.

shape = (90, 108, 90)
affine = np.diag([2.0, 2.0, 2.0, 1.0])
affine[:3, 3] = -np.array(shape)  # half the extent in mm: the centre at 0
i, j, k = np.indices(shape, dtype=np.float32)
x, y, z = (2 * i / shape[0] - 1, 2 * j / shape[1] - 1, 2 * k / shape[2] - 1)
r = np.sqrt((x / 0.8) ** 2 + (y / 0.9) ** 2 + (z / 0.85) ** 2)
t1 = np.select(
    [r < 0.82, r < 0.9, r < 1.0],
    [500 + 250 * np.cos(9 * r), np.full(shape, 80.0), np.full(shape, 900.0)],
    0.0,
).astype(np.float32)
world = np.stack([i, j, k], axis=-1) @ affine[:3, :3].T + affine[:3, 3]
inside = np.linalg.norm(world - [30.0, 10.0, 10.0], axis=-1) < 20.0

viewer = Viewer()
head = viewer.add(Volume(t1, affine), Category.VOLUMES, name="head")
viewer.add(Volume(inside.astype(np.uint8), affine), Category.OVERLAYS, name="ball")
viewer.move_cursor((30.0, 10.0, 10.0))  # the slices through the ball

# %%
# The 2D views as separate images
# -------------------------------
# One call saves each 2D view to its own file, named after the path given:
# "viewer_views.png" gives "viewer_views_axial.png", "viewer_views_coronal.png"
# and "viewer_views_sagittal.png". Size, scale, overlay and transparency work as
# for a single screenshot.

paths = viewer.screenshot_views("viewer_views.png", size=(400, 400))
print("saved", ", ".join(p.name for p in paths))

# %%
# A rotation video
# ----------------
# The 3D view turns about the screen's up axis, starting from its current camera
# position. The speed is in degrees per second, the duration in seconds and the frame
# rate in frames per second. By default the view turns at 30 degrees per second for
# 12 seconds at 30 frames per second, which is one full turn. Here a quarter turn in
# 1.5 seconds at 20 frames per second keeps the example quick. The viewer's camera is
# left as it was. The progress callback is called after each frame, and returning
# False from it would cancel the video.

viewer.set(head.id, OptionName.OPACITY_3D, 0.3)  # the ball shows through


def progress(done: int, total: int) -> bool:
    """Reports the last frame; returns True to go on."""
    if done == total:
        print(f"drew {done} frames")
    return True


video = viewer.record_rotation(
    "viewer_rotation.webp",
    size=(300, 300),
    speed=60.0,
    duration=1.5,
    fps=20,
    on_progress=progress,
)
print("saved", video)
