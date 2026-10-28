# Exporting images and videos

The Export sheet saves what the views show as an image or a video, for a paper, a slide or a web page. Open it with
**Export...** in the header, with File > Export... or with **Ctrl+Shift+S** (Cmd+Shift+S on macOS). The sheet is a
window of its own: a preview of the export is on its left and the choices are on its right, and the preview follows
the views while the sheet is open.

**What** chooses what to save: **One image**, **2D views as separate images** or **3D rotation video**. For one
image, **View** chooses one of the four views, or **All views** as the window lays them out: only the enlarged view
if one is enlarged, the view shown in the **1** layout, and otherwise the four views in a 2 x 2 grid, also for the
**1 + 3** layout. It starts on the view last clicked each time the sheet opens. **Image size** says how large the
image or each frame will be.

## Presets

A preset decides how the export is drawn. It sets the size of the image, which the views share when the image shows
all of them, or keeps the size the views have on the screen, a scale that multiplies that size, the background
colour, whether the information overlay is drawn, and whether the background is transparent. It can also give the
items other options while the export is drawn, such as tubes for the tractograms, and hold a position of the 3D
camera. Those options apply only to the export: the views on the screen are as they were once it is saved. The small
table under **Preset** shows the chosen preset's values.

The viewer comes with four presets.

- **Screen** draws the views as they are on the screen, at the screen's size and resolution.
- **Figure** draws them at twice the resolution on a white background, without the overlay and with a transparent
  background, for a paper.
- **Talk** draws them at twice the resolution on a black background, without the overlay, with tractograms as tubes
  with ambient occlusion, for a slide.
- **Website** draws images of 1800 x 1200 pixels on a black background, without the overlay, with tractograms as
  tubes of 0.25 mm with ambient occlusion, the smoothest FOD glyphs and blocky slices, and the 3D view seen from
  the subject's front right, 12 degrees above the horizontal.

**Edit preset...** changes the chosen preset's name, size, scale, background, overlay and transparency, and can save
the 3D view's camera as it is now. The preset is saved as a TOML file in the `export_presets` folder beside the
settings (see [Settings](09-settings.md)), so it is there the next time; a preset saved under the name of one of the
four replaces it.

**Camera** chooses where the 3D view is seen from: the preset's own position, when it has one, the 3D view's current
camera, or the axial, coronal or sagittal position. **Transparent background** leaves the space around the images
see-through, for PNG, TIFF and WebP; it starts as the preset says. **Also save the session beside the image** writes
a session file with the image's name, such as `figure.lar.json` beside `figure.png`, so that you can open the same
view again later (see [Sessions](11-sessions.md)).

**Save...** asks for a file, which can be PNG, TIFF or, for an image without transparency, JPEG, and **Cancel**
closes the sheet.

## Separate views and rotation videos

**2D views as separate images** saves the axial, coronal and sagittal views at once, each in its own file, so saving
`figure.png` writes `figure_axial.png`, `figure_coronal.png` and `figure_sagittal.png`. If any of them exists, the
sheet asks once before replacing them.

**3D rotation video** saves the 3D view turning round as an animated WebP or GIF. It starts from the camera the
sheet chooses, and three fields set the video: **Speed** in degrees per second (30 by default; a negative speed
turns the other way), **Duration** in seconds (12 by default, one full turn at 30 degrees per second) and **Frames
per second** (30 by default). A progress window follows the recording, and its **Cancel** stops it without writing
a file. A transparent background needs WebP, because GIF has no partial transparency. GIF also has fewer colours and
plays at most 50 frames per second, so above that rate the file dialog offers only WebP.

From Python, without a window, `viewer.export("Website", "slide.png", ViewKind.THREE_D)` saves one view with a
preset, `viewer.export_views("Figure", "figure.png")` saves the three 2D views and
`viewer.export_rotation("Talk", "turn.webp")` saves the video. `viewer.screenshot(None, "figure.png", scale=2)`,
`viewer.screenshot_views` and `viewer.record_rotation` take the size, scale and switches themselves instead of a
preset.
