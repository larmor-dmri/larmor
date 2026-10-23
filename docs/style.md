# Code style

Larmor follows the standard Python conventions, [PEP 8](https://peps.python.org/pep-0008/) for code and
[PEP 257](https://peps.python.org/pep-0257/) for docstrings, and lets tools check them.

## What the tools check

[ruff](https://docs.astral.sh/ruff/) formats and checks the code, and a type checker checks the type annotations.
Between them they cover:

- layout and naming as PEP 8 describes them (for example `snake_case` functions and `CapWords` classes);
- Google-style docstrings on public functions and classes;
- imports: sorted, absolute, none unused;
- no `print` in the library (only the command line writes to the terminal, see [Logging](#logging));
- type annotations on every function, and that the types are consistent;
- common mistakes, such as a bare `except:` or a mutable default argument.

When a check is wrong for one particular line, switch it off on that line only, with a comment saying why, for
example `R = rotation_matrix(...)  # noqa: N806 (R as in the paper's formula)`.

## Public and private

A name is public when it is in a package's `__all__`, and only then, and the API reference shows exactly those
names. A module without a leading underscore can still be internal. `tests/test_layout.py` checks the rules that
follow from this:

- outside a package, a name the package exports is imported from the package (`from larmor.sd import
  find_peaks`), not from its module. Inside the package, modules import each other directly, to avoid cycles;
- a `_` name is never imported from another module. A helper shared by several modules gets a plain name in a
  private module of its package, such as `types/_affine.py`, and only that package imports the private module;
- the library's packages may use each other's internal names (`sd` uses `dwi`'s fitting helpers), but the
  command line uses only exported names, so a command never relies on something the Python API lacks.

The packages form layers, and a module imports only from layers below its own. From the top, the layers are
`cli`, `viewer`, `simulation`, `tractography`, `sd`, `dwi` and `data` (the example datasets), then `io` and
`surfaces`, then `types`, and `_log` at the bottom. Inside `io`, a format never imports the loading by path, and
`types`, `io` and `surfaces` never import scipy, scikit-image or the viewer's libraries. import-linter checks all
of this (`uv run lint-imports`, with its contracts in `pyproject.toml`).

## Names of public functions

- `read_<format>` and `write_<format>` read or write one file format, and `load_<type>` and `save_<type>` choose
  the format from the path. `open_` starts a writer, `iter_` reads a file batch by batch, and `has_` or `is_` asks
  a yes/no question about a file. `tests/test_naming.py` checks that `larmor.io` exports only such names, apart
  from `file_kind` and `output_files`.
- A computation is named after its result (`mean_b0`, `vertex_normals`) or with `fit`, `estimate`, `find`, `build`
  or `create`. The verb `simulate` is also allowed, for the entry function of a simulation engine, and
  `lar simulate` is named after it; `save_simulation` follows the `save_` rule above. No public name starts with
  `get_`.
- A name says which way a filter works: `keep_vertices` keeps the vertices where the mask is True.

## Arguments

- Functions take a `Volume` or an `ndarray` where either makes sense; the mix is deliberate.
- A simple choice is a string (`basis="tournier"`); an enum is public only where it already is.
- A count is `num_` (`num_peaks`, `num_shifts`).
- The data comes first. Options come after a `*`, so they are keyword-only:
  `fit_sh(amplitudes, directions, lmax, basis="tournier")`.

## What the reviewer judges

Some things no tool can check, and the reviewer judges them with common sense:

- names say what a thing is, so that the code reads as its own documentation. A short name that follows a
  formula is fine when it is clearer, with a comment pointing to the formula;
- an error message says what went wrong and with which file or value, so the user can fix it;
- comments explain why the code does something, not what it does;
- comments and docstrings are ASCII only: `mm^2/s`, `g^T D g`, not `mm²/s`, `gᵀDg`.

## Readable writing

Documentation, help texts and docstrings are written for someone who knows diffusion MRI but not Larmor. They use
full, connected sentences that say what something is, what it is for and how it relates to the rest, and they
explain a term before using it. A compressed sentence that lists items without saying how they relate saves words
but costs the reader time. For example, this sentence is hard to follow:

> The Phantom panel, the spline on the left (Ctrl+5), edits a phantom for Larmor's simulation: a voxel grid, named
> tissues, fibre bundles drawn in the 2D views, tissue maps and a scan.

It reads well as:

> The Phantom panel, which you open by clicking the spline icon on the left (Ctrl+5), edits a phantom for Larmor's
> simulation module. A phantom consists of a voxel grid, tissues, fibre bundles (loaded from files or drawn in the
> 2D views), tissue maps and a scanning protocol.

A page stays short by covering less, not by dropping words.

## Models: model and fit

Models follow the model and fit pattern of DIPY:

```python
model = SDModel(rfs, lmax=8)               # configuration only, no data
fit = model.fit(volume, gtab, mask=mask)   # data in (a Volume and its GradientTable), result object out
fit.fods                                   # results are named attributes
```

- The model's constructor takes configuration only; `fit` takes the data and returns a fit object.
- A fit object exposes its results as named attributes. `fit` never returns a bare array, or a tuple whose contents
  depend on the arguments.
- The model is `<Name>Model` and its result `<Name>Fit` (`DTIModel` and `DTIFit`).
- A fit object may offer `predict()`, which synthesises the signal from the fitted parameters, where the model
  supports it.

## Logging

- Larmor logs through one package logger, `logging.getLogger("larmor")`, with a `NullHandler` attached, so the
  library is silent unless the application configures logging.
- Library code never calls `print`. It reports through the logger, through `warnings.warn` for conditions the
  caller should know about, or through exceptions.
- The command line maps `--quiet`, `--info` and `--debug` to the levels ERROR, INFO and DEBUG (WARNING when none is
  given) and writes to stderr.

## Command-line conventions

- Commands are flat (`lar <command>`, no command groups), and names of more than one word use kebab-case, for
  commands and options alike (`--b0-threshold`).
- A choice of algorithm is an option, not a separate command: `lar track --algorithm prob`.
- Input and output files are positional arguments, required inputs first and outputs last; everything optional is a
  named option.
- A `.mif` file with gradients in its header provides them, and `--bvals` and `--bvecs`, if given as well,
  override them with a warning. A NIfTI file, or a `.mif` without gradients, always needs `--bvals` and `--bvecs`,
  because Larmor never looks for gradient files by name.
- An existing output file is never overwritten unless `--force` is given; without it the command fails before doing
  any work.
- Progress bars, warnings and errors go to stderr, and stdout carries only data meant to be piped. Without
  `--debug`, an error is one readable line naming the file or value
  (`ERROR: dwi.nii.gz has 64 volumes but bvals has 65 entries`), and the exit code is 1.
- Every command accepts `--quiet`, `--info` and `--debug`, and every parallel one `--nt` (number of threads, all
  cores by default).
- `-m` is `--mask` wherever a mask selects voxels, and means nothing else.
- A command only maps its options onto the Python API: it holds no logic the library lacks.

The [principles](contributing.md#principles) in the contributor guide, on dependencies, speed, the definition of
done and test data, also shape how code is written.

Units, the order of array axes and coordinate frames are a question of correctness, not style. They are part of
the core data types, and the types' docstrings document them.
