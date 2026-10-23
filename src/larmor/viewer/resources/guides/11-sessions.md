# Sessions

A session is a file that records what the viewer shows, so that you can open the same view again later or on
another computer. It keeps the files loaded and each item's options. It also keeps the state of the views: the
crosshair, the zoom and pan of each view, the 3D camera, the main volume, the selection, the enlarged view, RAS or
LAS, and the overlay and crosshair switches. The settings of the Settings panel are not part of a session.

- **File > Save Session...** (**Ctrl+S**) writes a session file, `name.lar.json`. Items added from Python without
  a file cannot be saved, and a message names them.
- **File > Save As...**, or **Save As...** in an item's right-click menu in Layers, writes the selected item to a
  file. A tractogram is saved as `.tck`, `.trk`, `.vtk` or `.fib`, and a volume, overlay, FOD or peaks image as
  NIfTI or MIF. The item then refers to the new file, so a session saved afterwards keeps it.
- **File > Open Session...** (**Ctrl+Shift+O**) replaces what is shown with the session. The files are read in the
  background, with the progress bar and Cancel, and the options and views are restored once all are read.

The session stores each file's full path and also its path relative to the session file's folder, and tries the
relative path first. If you keep the session file in the project folder, you can therefore move the folder or
copy it to another computer. A file that is missing or cannot be read is reported, and the rest of the session
still opens.

From the command line, `lar bv --session name.lar.json` opens a session. Files given with the other options, such
as `--volumes` (`-v`), `--overlays` (`-o`) or `--tracts`, are added to it. From Python:

```python
viewer.save_item(bundle.id, "project/bundle.tck")
viewer.save_session("project/view.lar.json")
report = viewer.load_session("project/view.lar.json")
print(report.problems)
```
