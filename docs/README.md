# Larmor documentation

| Part | What it contains |
| --- | --- |
| [Install](install.md) | Installing Larmor |
| [Getting started](getting-started.md) | A first run of the command line, Python and the viewer |
| [Background](background/index.md) | The conventions Larmor uses, such as its coordinates, gradient tables, SH bases and units |
| [User guide](guides/index.md) | One page per task, each shown in Python and on the command line |
| [Examples](../examples/README.md) | Runnable examples, one for every user-facing feature |
| [Example data](example-data.md) | The public datasets that Larmor downloads for the examples |
| [Command-line reference](cli.md) | Every `lar` command and option |
| [Python API](reference/api/index.md) | One page per public package, generated from the docstrings when the website is built; also the [file formats](reference/file-formats.md) |
| [Release notes](https://github.com/larmor-dmri/larmor/releases) | What changed in each version |
| [Contributing](contributing.md) | How to contribute, with the [style guide](style.md) and [releasing](releasing.md) |

These pages are also built into the documentation website, as [Contributing](contributing.md#the-documentation-website) explains.

The command-line reference is generated from the code. After you change a command or an option, regenerate it
from the repository root:

```bash
uv run python -m typer larmor.main utils docs --name lar --output docs/cli.md
```

A test fails when the reference is out of date, and another test runs every example, so CI catches both problems.
