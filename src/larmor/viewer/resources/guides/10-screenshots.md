# Screenshots

The Screenshots panel saves what the views show as an image or a video, for a paper or a slide. Open it with
File > Screenshots, with the camera icon on the left, or with **Ctrl+Shift+S**. The panel shows a preview of the
image before it is saved, and has these fields.

- **Save** chooses what to save: **One image**, **2D views as separate images** or **3D rotation video** (see
  below).
- **View** chooses one of the four views, or **All views** as the window lays them out (only the enlarged view,
  if one is enlarged). It starts on the view last clicked.
- **Scale (x screen)** goes from 1 to 4. At 1 the image has the screen's pixels and matches what the views show.
  At 2 to 4 the image is drawn again at that many times the resolution, so it stays sharp in a paper or on a
  slide.
- **Information overlay** puts the views' names and slices, the readout, the orientation letters and the colour
  bars in the image, or leaves them out. The views on the screen keep their own setting.
- **Transparent background** leaves the space around the images see-through, for PNG and TIFF. JPEG has no
  transparency.
- **Image size** shows the size in pixels of the image that will be saved.

**Save...** asks for a file, which can be PNG, TIFF or, for an image without transparency, JPEG. The fields start
with the screenshot values in [Settings](09-settings.md); changing them here does not change Settings.

## Separate views and rotation videos

**2D views as separate images** saves the axial, coronal and sagittal views at once, each in its own file, so
saving `figure.png` writes `figure_axial.png`, `figure_coronal.png` and `figure_sagittal.png`. If any of them
exists, the panel asks once before replacing them.

**3D rotation video** saves the 3D view turning round, starting from where it is now, as an animated WebP or GIF.
Three fields set the video.

- **Speed** is in degrees per second (30 by default). A negative speed turns the other way.
- **Duration** is the length of the video in seconds (12 by default, one full turn at 30 degrees per second).
- **Frames per second** is 30 by default.

The preview plays the rotation in a loop at the chosen speed; **Pause** stops it on a frame and **Play** goes on.
A progress window follows the recording, and **Cancel** stops it without writing a file. The 3D view in the main
window does not move while the video is recorded. A transparent background needs WebP, because GIF has no partial
transparency. GIF also has fewer colours and plays at most 50 frames per second, so above that rate the file
dialog offers only WebP.

From Python, without a window, `viewer.screenshot(None, "figure.png", scale=2, transparent=True)` saves all
views, `viewer.screenshot_views("figure.png")` saves the three 2D views, and `viewer.record_rotation("turn.webp")`
saves the video.
