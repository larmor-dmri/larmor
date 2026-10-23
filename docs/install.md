---
myst:
  html_meta:
    description: "How to install Larmor with pip or uv, the viewer on its own, or from source, and how to run the viewer on a remote server or without a GPU."
---

# Install

Larmor can be installed as a library with pip or uv, as a desktop viewer on its own, or from source. This page
describes each way, and then how to run the viewer on a remote server or on a computer without a GPU.

Larmor needs Python 3.14 or newer. It is built and tested on Linux, macOS and Windows, and for the platforms below
it is published as prebuilt wheels, which install without a compiler:

| Platform | Wheel |
| --- | --- |
| Linux, x86-64 and aarch64 | manylinux (glibc-based distributions) |
| macOS 14 or newer, Apple silicon and Intel | arm64 and x86-64 |
| Windows, x86-64 | x86-64 |

On other platforms, such as a musl-based Linux like Alpine, pip builds Larmor from its source distribution, which
needs a C++20 compiler.

## With pip or uv

The following command installs the library and the `lar` command line:

```bash
pip install larmor-dmri
```

To install the viewer, BrainViewer, as well, add the `viewer` extra. It also installs Qt (PySide6), the pygfx
renderer, and the MCP package through which agents can drive the viewer:

```bash
pip install "larmor-dmri[viewer]"
```

With [uv](https://docs.astral.sh/uv/), use `uv pip install` in the same way inside an environment, or
`uv add larmor-dmri` in a uv project. If the computer does not have Python 3.14, `uv python install 3.14` installs
it.

## The viewer on its own

To use the viewer as a desktop application without setting up a Python environment, install the `lar` command as
a uv tool and then add the viewer to the desktop:

```bash
uv tool install "larmor-dmri[viewer]"
lar bv --install-launcher
```

The launcher opens the viewer of this install. It is an entry in the application menu on Linux, an app in
~/Applications on macOS and a shortcut in the Start menu on Windows. `lar bv --uninstall-launcher` removes it.

## From source

Installing from source compiles Larmor's C++ module, so it needs a C++20 compiler:

- **Linux:** GCC or Clang (`sudo apt install g++` on Debian and Ubuntu).
- **macOS:** the Command Line Tools (`xcode-select --install`).
- **Windows:** Visual Studio 2022 or its Build Tools, with the "Desktop development with C++" workload.

The build installs the other tools it needs, CMake, Ninja and nanobind, by itself. The C++ code also uses two
header-only libraries, Eigen and pocketfft. The source distribution on PyPI includes them, but a build from a git
clone downloads them the first time it runs, so that first build needs a network connection. Nothing else has to
be installed.

```bash
git clone https://github.com/larmor-dmri/larmor.git
cd larmor
uv sync
```

`uv sync` installs Python 3.14 if needed, then Larmor in editable mode, with the viewer's dependencies and the
development tools. Run the commands through uv, for example `uv run lar --help` and `uv run lar bv`. After a change
to the C++ code, the next `uv sync` or `uv run` compiles the changed files again. To build with pip instead, run
`pip install ".[viewer]"` in the cloned folder, in an environment with Python 3.14.

## Remote use over SSH

The viewer can run on a remote Linux server and show its window on your screen. Connect to the server with X
forwarding, using `ssh -X` (MobaXterm on Windows forwards X by default), and start `lar bv` there. The server's GPU
draws the views, and only the finished images travel to your screen. If dragging feels slow, turn on
**Lower resolution while moving** in the Settings panel.

Qt needs some X libraries that servers often lack. When one is missing, `lar bv` names it and the package that
provides it. [Getting started](getting-started.md#using-the-viewer-on-a-linux-server) shows how to install them,
with or without administrator rights.

## GPU and software rendering

The viewer draws through wgpu, a graphics library that uses Metal on macOS, Vulkan on Linux, and DirectX 12 or
Vulkan on Windows. To see which GPUs it finds, run the command below with the Python of the environment Larmor is
installed in. For the viewer installed on its own, that Python is
`uv tool run --from "larmor-dmri[viewer]" python`.

```bash
python -c "import wgpu; [print(a.summary) for a in wgpu.gpu.enumerate_adapters_sync()]"
```

- **No GPU is listed.** Install or update the GPU's driver. On Linux, the driver must include Vulkan support.
- **The computer has no GPU,** as on many servers. The viewer then draws with a software renderer, which works but
  is slow, so keep the views small and turn on **Lower resolution while moving**. On Linux, the software renderer
  is Mesa's Vulkan driver, lavapipe, from the `mesa-vulkan-drivers` package on Debian and Ubuntu.
- **The computer has several GPUs.** Choose the one the viewer draws with in the **GPU** setting of the Settings
  panel. The choice applies the next time the viewer starts.

Screenshots from Python (`Viewer.screenshot`) need no window and no display, but they still need a GPU or a
software renderer.
