"""Saving and reopening a viewer session
=====================================

This example writes a synthetic head and a mask to a project folder, shows them in
the viewer, saves a session there, moves the folder, opens the session in a new
viewer and saves a screenshot of all views.

A session saves what the viewer shows in a JSON file: the files loaded, each item's
options, and the state of the views, such as the crosshair, the zoom and the 3D
camera. Opening the session again reads the files and restores the rest. Each file is
stored with its full path and with its path from the session file's folder, so a
project folder that holds the session can be moved and still opens.
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np

from larmor.io import save_volume
from larmor.types import Volume
from larmor.viewer import Category, Colormap, OptionName, Viewer, ViewKind

# %%
# A project folder
# ----------------
# The project holds a 2 mm image of an ellipsoid with a brighter core, centred at the
# world origin (world coordinates are in mm, RAS+), and a mask of the core, both
# written as NIfTI files.

shape = (80, 96, 80)
affine = np.diag([2.0, 2.0, 2.0, 1.0])
affine[:3, 3] = -np.array(shape)  # half the extent in mm: the centre at 0
i, j, k = np.indices(shape, dtype=np.float32)
x, y, z = (2 * i / shape[0] - 1, 2 * j / shape[1] - 1, 2 * k / shape[2] - 1)
r = np.sqrt((x / 0.8) ** 2 + (y / 0.9) ** 2 + (z / 0.85) ** 2)
head = np.where(r < 0.8, 400.0 + 300.0 * (r < 0.4), 0.0).astype(np.float32)
core = (r < 0.4).astype(np.float32)

work = Path(tempfile.mkdtemp())
project = work / "project"
project.mkdir()
save_volume(Volume(head, affine), project / "head.nii.gz")
save_volume(Volume(core, affine), project / "core.nii.gz")

# %%
# Showing and saving
# ------------------
# The files are loaded and shown with a few options changed. The views are moved too:
# the crosshair goes off centre, the axial view is zoomed in and the 3D camera is
# turned.

viewer = Viewer()
head_item = viewer.load(project / "head.nii.gz", Category.VOLUMES)
core_item = viewer.load(project / "core.nii.gz", Category.OVERLAYS)
viewer.set(head_item.id, OptionName.COLORMAP, Colormap.BONE)
viewer.set(core_item.id, OptionName.COLORMAP, Colormap.HOT)
viewer.set(core_item.id, OptionName.MAXIMUM, 2.0)  # the mask's 1 in the middle
viewer.set(core_item.id, OptionName.OPACITY_2D, 0.5)
viewer.move_cursor((10.0, -8.0, 6.0))
viewer.zoom(ViewKind.AXIAL, 1.5)
viewer.rotate_camera(30.0, -10.0)

skipped = viewer.save_session(project / "view.lar.json")
print("Left out of the session:", skipped or "nothing")

# %%
# Reopening after a move
# ----------------------
# After the project folder moves, the files' full paths no longer exist, but their
# paths from the session file's folder still do. A new viewer opens the session and
# reports any file it could not open.

moved = work / "moved"
shutil.move(project, moved)
again = Viewer()
report = again.load_session(moved / "view.lar.json")
print("Opened:", [item.name for item in report.items])
print("Problems:", list(report.problems) or "none")
assert again.scene.view.cursor == viewer.scene.view.cursor
again.screenshot(None, "viewer_sessions.png", size=(400, 400))

shutil.rmtree(work)
