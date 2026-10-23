---
myst:
  html_meta:
    description: "A first use of Larmor on a simulated brain: the example data, the lar command line, a Python script and the BrainViewer viewer, also on a remote server or driven by an agent."
---

# Getting started

This page takes you through a first use of Larmor on a simulated brain. You download an example dataset, then
process it with a command, with a Python script and in the viewer. The last sections show how to use the viewer
on a remote server and from an agent.

## Installing

```sh
pip install larmor-dmri
```

For the viewer, run `pip install "larmor-dmri[viewer]"` instead. To try Larmor without downloading any data,
follow the [quick start](quick-start.md), which simulates a phantom and opens it in the viewer. The
[Install](install.md) page describes the other ways to install, with uv or from source, and what the viewer needs
from the GPU.

## The example data

This page uses the simulated brain of the ISMRM 2015 Tractography Challenge, one of the
[example datasets](example-data.md). It has diffusion images at 2 mm, with one b=0 image and 32 directions at
b=1000 s/mm^2, and a T1-like image. {py:func}`~larmor.data.fetch_ismrm2015` downloads it (41 MB) into a cache
folder the first time it is called. The lines below save the diffusion image in the current folder as a MIF file,
MRtrix3's image format, and copy the T1-like image beside it. A MIF file keeps the gradient table in its header,
so the later steps need no separate b-value and b-vector files.

<!-- needs: example-data -->
```python
import shutil

from larmor.data import fetch_ismrm2015
from larmor.io import load_dwi, save_volume

files = fetch_ismrm2015()
dwi, gtab = load_dwi(files["dwi"], files["bvals"], files["bvecs"])
save_volume(dwi, "dwi.mif", gradients=gtab)
shutil.copy(files["t1"], "t1.nii.gz")
```

Save these lines as a script, `get_data.py`, and run it with `python get_data.py`. Please cite the dataset if you
use it: Maier-Hein KH et al. The challenge of mapping the human connectome based on diffusion tractography. Nature
Communications 8 (2017) 1349.

## A first command-line run

`lar --help` lists the commands of the command line, and `lar <command> --help` lists the options of one command.
The [command-line reference](cli.md) describes them all. Start with `lar info`, which prints a short summary of a
file:

<!-- needs: example-data -->
```bash
lar info dwi.mif
```

The summary gives the file's shape, voxel size, orientation and affine. For a MIF file with a gradient table, such
as this one, it also gives the number of b=0 images and the shells. The next two commands compute a brain mask,
then the DTI maps inside it:

<!-- needs: example-data -->
```bash
lar mask dwi.mif mask.nii.gz
lar dti dwi.mif dti -m mask.nii.gz
```

The second argument of `lar dti` is a prefix for the output files. Here the command writes the FA in
`dti_fa.nii.gz`, the mean, axial and radial diffusivity in `dti_md.nii.gz`, `dti_ad.nii.gz` and `dti_rd.nii.gz`,
and the principal eigenvector in `dti_pe.nii.gz`. The last file, `dti_npd.nii.gz`, marks the voxels where the fit
failed, because the fitted tensor has a smallest eigenvalue of 0 or below. A command does not overwrite existing
files unless you give `--force`.

## A first Python script

The script below does the same in Python. It computes the mask from the mean b=0 image, fits the tensor inside
the mask, and saves the FA map. It also prints the mean FA of the white matter, taken here as the voxels with an
FA above 0.3.

<!-- needs: example-data -->
```python
from larmor.dwi import DTIModel, create_brain_mask, mean_b0
from larmor.io import load_dwi, save_volume

dwi, gtab = load_dwi("dwi.mif")
mask = create_brain_mask(mean_b0(dwi, gtab))
fit = DTIModel().fit(dwi, gtab, mask=mask)
print(f"Mean FA where it is above 0.3: {fit.fa[fit.fa > 0.3].mean():.2f}")
save_volume(dwi.with_data(fit.fa), "fa.nii.gz")
```

The [user guide](guides/index.md) has a page for each task, in Python and on the command line, and the
[gallery](../examples/README.md) has longer examples with their output and images.

## A first viewer session

`lar bv` opens the viewer, BrainViewer. Each of its options loads files as one kind of item, such as `-v` for
volumes and `-o` for overlays drawn over them, and you can repeat an option to load several files. The command
below opens the T1-like image as a volume and the FA map as an overlay:

<!-- not run: opens the viewer -->
```bash
lar bv -v t1.nii.gz -o dti_fa.nii.gz
```

The Layers panel on the left lists the loaded items, and the Options panel on the right sets the options of the
selected item, such as its colormap and opacity. Click or drag in a 2D view to move the crosshair there. To load
more files into the open viewer, use File > Open, where the file type you choose decides what kind of item the
file becomes, or drag the files from the file manager onto a category in the Layers panel. The
[viewer's guide](viewer/getting-started.md) describes the window, and its other pages describe each kind of item.

From Python, `view` opens the same window. It takes paths, or objects already in memory, which it shows without
copying them:

<!-- not run: opens the viewer -->
```python
from larmor.io import load_volume
from larmor.viewer import view

t1 = load_volume("t1.nii.gz")
viewer = view(volumes=t1, overlays="dti_fa.nii.gz")
```

`view` returns when you close the window. In IPython or Jupyter, run `%gui qt` first and pass `block=False`, so
you can keep working while the window is open. You can then change what the window shows through the returned
`viewer`, for example with `viewer.load(...)` and `viewer.set(...)`.

A `Viewer` can also take screenshots without opening a window, which is useful in scripts and on servers without
a display. The [gallery](../examples/README.md) shows the viewer's features with screenshots made this way:

<!-- needs: example-data -->
```python
from larmor.viewer import Category, Viewer, ViewKind

viewer = Viewer()
viewer.load("t1.nii.gz", Category.VOLUMES)
viewer.load("dti_fa.nii.gz", Category.OVERLAYS)
viewer.screenshot(ViewKind.AXIAL, "axial.png", size=(600, 600))
```

### Using the viewer on a Linux server

To use the viewer on a remote Linux server, connect to it with X forwarding, using `ssh -X` (MobaXterm on Windows
forwards X by default), and run `lar bv` there. The server's GPU draws the views, and only the finished images
travel to your screen. If dragging feels slow, turn on **Lower resolution while moving** in the Settings panel.

Qt needs some X libraries that servers often lack. When one is missing, `lar bv` names it and the package to
install. If you have administrator rights on Debian or Ubuntu, install them all with:

<!-- not run: installs system packages -->
```bash
sudo apt install libxcb1 libx11-6 libx11-xcb1 libsm6 libice6 libxkbcommon0 libxkbcommon-x11-0 libxcb-cursor0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-render-util0 libxcb-util1 libxcb-xinerama0 libxcb-xkb1 libxcb-randr0 libxcb-shape0 libxcb-xfixes0 libxcb-sync1 libxcb-shm0 libxcb-glx0 libxcb-render0
```

Without administrator rights, download the same packages into a folder in your home directory and unpack them
there:

<!-- not run: installs system packages -->
```bash
mkdir -p ~/.local/qt-libs && cd ~/.local/qt-libs && for p in libxcb1 libx11-6 libx11-xcb1 libsm6 libice6 libxkbcommon0 libxkbcommon-x11-0 libxcb-cursor0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-render-util0 libxcb-util1 libxcb-xinerama0 libxcb-xkb1 libxcb-randr0 libxcb-shape0 libxcb-xfixes0 libxcb-sync1 libxcb-shm0 libxcb-glx0 libxcb-render0; do apt download $p; done; for f in *.deb; do dpkg -x $f .; done
```

Then start the viewer with that folder on the library path:

<!-- not run: opens the viewer -->
```bash
LD_LIBRARY_PATH=$HOME/.local/qt-libs/usr/lib/x86_64-linux-gnu lar bv
```

### Driving the viewer from an agent

An agent, such as Claude Code or any other client of the Model Context Protocol (MCP), can drive a running
viewer. It can load files, change options, move the cursor, turn the cameras and take screenshots. To allow this,
start the viewer with its MCP server:

<!-- not run: opens the viewer -->
```bash
lar bv --mcp
```

You can also turn on **Serve agents** in the Settings panel of a running viewer. The viewer then prints the
server's URL and a token:

```
Serving agents (MCP) at http://127.0.0.1:8765/mcp
Token: 9f3K...
```

The server listens on this computer only and refuses any request without the token, so other users of a shared
server cannot reach it. The token stays the same when the viewer restarts, until you replace it with **New token**
in Settings. Give the URL and the token to the agent. For Claude Code, the following command does this, and you
need to run it only once:

<!-- not run: configures an agent -->
```bash
claude mcp add --transport http Larmor http://127.0.0.1:8765/mcp --header "Authorization: Bearer 9f3K..."
```

The agent sees each of the viewer's actions as a tool with its own description, and the `scene` tool lists the
loaded items and their ids. The server uses port 8765, unless you choose another with `--mcp-port` or in Settings.

When the agent runs on a different computer from the viewer, forward the port over SSH from the agent's computer.
Use the same port number on both sides, because the server answers only at its own port; the URL then stays the
same. Paths given to tools such as `load` and `screenshot` refer to files on the viewer's computer, not the
agent's.

<!-- not run: connects to a server -->
```bash
ssh -L 8765:127.0.0.1:8765 user@server
```
