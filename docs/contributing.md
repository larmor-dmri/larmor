# Contributor workflow

This document explains how a change gets into Larmor, from picking a ticket to merging the result. Read it
before your first contribution. [Principles](#principles) sets out what counts as done and how quality is
checked, and the [style guide](style.md) describes the code style.

One rule underlies everything here: **`main` only changes through a reviewed, merged pull request.** Nobody
commits to `main` directly. That way `main` always holds code that has been reviewed and works, and anyone can
start new work from it.

## Principles

These decisions shape every change, and a reviewer checks them.

### Dependencies

- Larmor implements its own I/O for every format it supports. It has no runtime dependency on other neuroimaging
  libraries (nibabel, dipy, ...), and nibabel and MRtrix3 serve only as test oracles.
- The runtime dependencies are numpy, scipy, scikit-image, tqdm, typer, h5py and matplotlib, as listed in
  `pyproject.toml`. matplotlib is imported only in `larmor.plot` and in the viewer's Qt layer, which import-linter
  checks. A new runtime dependency is a design decision, so open an issue for it first.
- Heavy or hardware-specific dependencies are optional extras, such as `[viewer]` for the viewer. Code that needs
  an extra imports it only where it is used. If the extra is missing, the command fails with a message that names
  the install command (`lar bv` tells the user to install `larmor-dmri[viewer]`).
- The documentation's tools are in the `docs` dependency group, not in the runtime dependencies.

### Speed

- Code is first written as clear, vectorised numpy.
- C++ is used only where profiling shows a real bottleneck, and the pull request that moves a function to C++
  states the measured speed-up. Python keeps the input checks, the I/O and the orchestration, and C++ gets the hot
  loop, called through nanobind.
- When a function moves to C++, the numpy version stays in the tests as the reference, and the tests check that
  both give the same results.
- On one machine, a result is bit-identical whatever the number of threads. To make this so, parallel code
  computes each item (a voxel, a block of voxels cut by index, a streamline) on its own and never reduces across
  threads. Across machines and operating systems, results agree only to rounding, and tests that compare them use
  tolerances, each justified by the operation.

### Definition of done

A change is done when it has:

- code and tests;
- a docstring for every function and class. The public API (what a package exports in `__all__`, and the
  viewer's `view()` and the `Viewer`'s actions) uses Google style (`Args:`, `Returns:`, `Raises:`) and states
  units and conventions explicitly, such as the SH basis, coordinate frame, b-value units and array shapes.
  Everything else has a one-line docstring saying what it is or does, and the reasons go in comments and in the
  pull request;
- for a user-facing feature (a new model, command or viewer capability): a gallery example in `examples/`. CI runs
  every example, so the examples are also smoke tests.

Internal changes and bug fixes need only the first two. All code, tests included, has type annotations. Where a
check cannot pass (for example on a call into an untyped library), the line gets a targeted
`# type: ignore[<code>]` with a comment giving the reason. Whole functions or modules are never excluded.

### Docstrings of the public API

Every function, class and public method in a package's `__all__` starts with a one-line summary, followed by
Google-style sections. `Args:` names every parameter, with units, shapes and conventions where they apply,
`Returns:` says what it returns, and `Raises:` lists the exceptions it raises itself. The types live only in the
annotations, so an `Args` entry is `x: ...`, never `x (int): ...`. A method's paper goes in a `References:`
section at the end, copied from [Citing](about/citing.md). `ruff` checks that `Returns`, `Yields` and `Raises` are
present (rules DOC201, DOC402 and DOC501, which skip one-line docstrings), and `tests/test_docstrings.py` checks
the summary and the `Args` of every public name.

### Tests and reference data

- Tests never run external tools (MATLAB, MRtrix3, ...). Reference results are generated once and stored in
  `tests/data/`, with the script that made each file and the version of the tool, so that each can be made again
  (`tests/data/README.md` lists them, with the source and licence of the data).
- A reference file is at most 1 MB, because everything committed stays in the history. A small crop (a few
  voxels, a few hundred streamlines) is enough to show that Larmor and the oracle agree.
- Real data for the examples is never committed. Instead, `larmor.data` downloads public datasets into a cache
  folder (see [Example data](example-data.md)). Unit tests never use the network, and the examples that download
  data run only when `LARMOR_DATA_DIR` names the cache.
- Every test runs on every machine. A test of large data that cannot be committed runs only when an environment
  variable names it (`FREESURFER_HOME`, `LARMOR_TEST_DATA`), and says so in its skip message.

## Where work is tracked

All planned work is written down as tickets (GitHub issues) in the repository. Each ticket explains its context,
its goal, and when it counts as finished.

Tickets carry three kinds of labels:

| Label | Meaning |
| --- | --- |
| `type:feature`, `type:bug`, `type:refactor`, `type:docs`, `type:design`, `type:build`, `type:test`, `type:perf`, `type:investigation` | The kind of work |
| `area:core`, `area:io`, `area:dwi`, `area:sd`, `area:tractography`, `area:cli`, `area:viewer` | The part of Larmor it touches |
| `P0`, `P1`, `P2` | Priority: blocks other work, needed for its version, or nice to have |

Each ticket belongs to a milestone, one per version (`0.1 Foundations`, `0.2 Own I/O`, and so on). Tickets
planned for later have no milestone, and the [roadmap](about/roadmap.md) says what is planned.

If you find a problem or think of an improvement while working on something else, open a new ticket for it rather
than fixing it on the side. That keeps each pull request about one thing.

## The path of a change

Every ticket follows the same twelve steps. The commands use ticket #8 ("Add issue and pull request templates")
as an example. For another ticket, change the number, the branch name and the texts.

### 1. Pick a ticket

Pick an open ticket, preferably from the current milestone and with the highest priority, and read it completely.
Its last line says which tickets it depends on, and if one of them is still open, do that one first. Say in the
ticket that you are working on it, so that nobody else starts the same work. Work on one ticket at a time.

### 2. Update main

Your branch starts from your local `main`, so first make it match GitHub:

```bash
git switch main
git pull
```

### 3. Create a branch

A branch name has three parts, the kind of work, the ticket number and a few words, written as
`kind/number-short-name`. The kind follows the ticket's type label:

| Ticket label | Branch kind |
| --- | --- |
| `type:feature` | `feat` |
| `type:bug` | `fix` |
| `type:refactor` | `refactor` |
| `type:docs`, `type:design` | `docs` |
| `type:build` | `build` |
| `type:test` | `test` |
| `type:perf` | `perf` |
| `type:investigation` | `investigate` |

```bash
git switch -c build/8-pr-templates
```

`-c` creates the branch and switches to it. From here on, every commit goes on this branch.

### 4. Do the work

Write the code, its tests and its documentation, following [the definition of done](#definition-of-done).
Keep to what the ticket asks. If the ticket turns out much bigger than expected, stop and split it into smaller
tickets, because a pull request that does one thing is easier to review and easier to undo.

The hot loops are written in C++, in `cpp/`, so you need a C++20 compiler. CMake, Ninja and nanobind come from the
uv environment. `uv sync` and `uv run` rebuild the C++ module whenever a file in `cpp/` has changed, so there is no
separate build step: change the C++ and run the tests. The build differs a little per platform:

- **Linux:** GCC or Clang. Eigen's own kernels do the matrix products (no BLAS library).
- **macOS:** Apple Clang from the Command Line Tools (`xcode-select --install`). Eigen sends its large matrix
  products to Apple's Accelerate framework.
- **Windows:** MSVC, from Visual Studio 2022 or its Build Tools. `uv sync` finds it on its own. The commands in
  `cpp/` below use Ninja, which needs MSVC's environment: run them from the "x64 Native Tools Command Prompt",
  with `CXX=cl` set. Eigen's own kernels do the matrix products, as on Linux.

The CMake option `LARMOR_BLAS` chooses the BLAS: `auto` (Accelerate on macOS, none elsewhere), `accelerate` or
`none`. CMake downloads Eigen and pocketfft when it first configures a build. `cmake -P cpp/cmake/vendor.cmake`
puts them into `cpp/third_party/` instead, where the build then finds them and the source distribution carries
them.

The editable install is `uv sync`. `uv pip install -e .` fails with "No module named 'scikit_build_core'",
because the package builds without build isolation, in the environment itself, so that the build folder and its
incremental rebuilds survive between builds, and `uv pip` does not install the `build` group that holds the build
tools. If you want to use `uv pip` anyway, install that group in the same command, with the `dev` group for the
tests and linters: `uv pip install --group build --group dev -e .`.

The development environment includes the viewer's dependencies (PySide6 and pygfx), so the first `uv sync`
downloads a few hundred MB. The viewer's tests need no screen, because they run Qt with its offscreen platform.
The viewer's Help guides are Markdown files in `src/larmor/viewer/resources/guides/`. After a change to what the
viewer shows, take their screenshots again.

### 5. Commit in small steps

Commit each time a piece of the work is complete and makes sense on its own. Small commits are easy to follow and
easy to undo, and they are combined into one commit when the pull request is merged (step 11), so having several
costs nothing.

```bash
git status
git add .github/ISSUE_TEMPLATE/bug_report.md
git commit -m "Add the bug report template"
```

`git status` shows what changed, so that you add only what belongs to the ticket. Name the files you add instead
of using `git add .`, which also adds anything you forgot was there. [Commit messages](#commit-messages) below
explains how to write the message.

### 6. Run the checks

Run on your machine the checks that CI runs on the pull request, so that problems show up here first:

```bash
uv run pytest
uv run ruff format .
uv run ruff check .
uv run mypy
uv run lint-imports
cd cpp
uv run cmake --preset dev
uv run cmake --build --preset dev
uv run ctest --preset dev
find src tests -name '*.cpp' -o -name '*.hpp' | xargs uv run clang-format -i
uv run run-clang-tidy.py -p build/dev -quiet 'cpp/(src|tests)/'
cd ..
```

The first command runs the tests, the second formats the code, the third looks for common mistakes, the fourth
checks the type annotations, and the fifth checks that modules import only from the layers below theirs
([the style guide](style.md#public-and-private)). The commands in `cpp/` do the same for the C++ code: they build
it and run its unit tests, format it, and look for mistakes. If `ruff format` or `clang-format` changed any files,
commit them. In VS Code the Python commands are under Terminal, Run Task.

`uv run pytest` runs the tests in one process per CPU core (pytest-xdist). Add `-n 0` to run them in one process,
for example to step through a test in a debugger, or for a single test file, where starting the workers takes
longer than the tests.

The rules of ruff, mypy and import-linter are configured in `pyproject.toml` (see [the style guide](style.md)).

CI (continuous integration) runs the same checks on GitHub for every pull request and every push to `main`, as
set out in `.github/workflows/ci.yml`. It runs ruff, mypy and import-linter once on Linux, and the C++ checks and
the tests on Linux, macOS and Windows (clang-format and clang-tidy on Linux only). It also checks that `uv.lock`
matches `pyproject.toml`, so run `uv lock` after changing dependencies. A test that runs longer than two minutes
fails. The results appear at the bottom of the pull request, and the Details link next to a failed check shows
what went wrong.

macOS runs the whole test suite. The Linux and Windows runners have no GPU, so there the viewer's tests draw with a
software renderer, and the tests that build its Qt widgets (marked `qt_widgets`) and the gallery examples on a
brain are left out. Windows also leaves out `tests/test_docs.py`.

A change that only touches Markdown files skips CI, because none of the checks reads them, and such a pull request
shows no checks at all. The exception is the Markdown of the website's pages in `docs/`: a test builds the site
from them, and another compares `docs/cli.md` with the CLI. The pages the site leaves out, such as this one, still
skip CI.

A second workflow, `.github/workflows/wheels.yml`, builds and tests the wheels and the source distribution. It runs
on a version tag and by hand.

#### The documentation website

Sphinx also builds the user-facing pages in `docs/` into a website. After changing them, build the site from the
repository root, serve it, and open http://localhost:8000 in a browser:

```bash
uv run sphinx-build -b dirhtml -W -D plot_gallery=0 docs docs/_build/html
uv run python -m http.server -d docs/_build/html
```

The workflow `.github/workflows/docs.yml` runs this build, with the gallery executed, on every pull request that
changes `docs/`, `examples/` or `src/`, so a broken build fails the check.

`-b dirhtml` makes each page a folder (`/install/`), as on the published site, so its links work through the small
server but not when you open the files directly. `-W` turns every warning, such as a broken link or a page missing
from the navigation, into an error. `-D plot_gallery=0` builds the gallery's pages without running the examples
(see below). `exclude_patterns` in `docs/conf.py` leaves the pages for contributors, like this one, out of the
site, and a new page for contributors goes there too. `docs/_build/` is not committed.

Each Python API page in `docs/reference/api/` holds one line, such as `.. api-package:: larmor.dwi`.
`docs/_ext/api_reference.py` turns it into the package's names in `__all__`, documented from their docstrings, and
a test checks that the built pages show exactly those names. A new public package needs a page there and a row in
`docs/reference/api/index.md`. The build fetches the inventories of Python, numpy, scipy and nibabel to link
their names. With `LARMOR_DOCS_OFFLINE=1` set, as in the tests, it builds without the network and without those
links.

Each page starts with front matter that holds its meta description, one plain sentence for search results. A page
that only says "Coming" also has `robots: "noindex"` there, which keeps it out of search engines and the sitemap.
`docs/_extra/llms.txt` lists the pages with content for language models, and tests check that each of its links
is a page and that none is marked noindex. When a "Coming" page gets its content:

- remove its `robots: "noindex"` line;
- add it to `docs/_extra/llms.txt`.

The task pages of the user guide and the getting-started page must run as written. `tests/test_docs.py` runs
their `python` and `bash` blocks in order, in an empty folder, once as a reader of the Python tabs and once as a
reader of the command-line tabs, each time with the blocks outside the tabs. A page writes the data it works on,
often with a shared snippet from `docs/_snippets/`, so a block outside the tabs must not depend on either tab. A
comment on the line before a block changes how it runs. `<!-- not run: reason -->` marks a block that cannot run
in a test (one that opens the viewer, for example), and `<!-- needs: example-data -->` one that runs only when
`LARMOR_DATA_DIR` is set, as for the gallery. A new task page goes into `TASK_PAGES` in that test.

sphinx-gallery makes the gallery from `examples/`. It runs every example during the build (about a minute) and
writes the pages into `docs/gallery/`, which is not committed. The examples on a brain need the ISMRM 2015 data,
so they run only when `LARMOR_DATA_DIR` names the folder for it, where the data is downloaded once (41 MB).
Without it those pages would show their code without images, so the build warns and `-W` fails. To build quickly
without running any example, add `-D plot_gallery=0`. A full build looks like this:

```bash
LARMOR_DATA_DIR=~/larmor-data uv run sphinx-build -b dirhtml -W docs docs/_build/html
```

A new example goes in the folder of its section, and its page goes into `docs/_extra/llms.txt`. Each screenshot
it saves needs alt text in `ALT_TEXT` in `docs/_ext/gallery_images.py`, or the build fails. The page's meta
description is the example's first paragraph. Screenshots for the website hide the crosshair and the text
(`viewer.toggle_crosshair()`, `overlay=False`), and brain images use only the ISMRM 2015 data, whose credit the
gallery adds under them.

The landing page's carousel takes its slides from the `{carousel}` block in `docs/index.md`, one line per slide
(`file | alt text | caption`), with the images in `docs/_static/carousel/` at 900 x 600. They are screenshots of
the gallery's ISMRM 2015 examples, and `docs/update_carousel.py` renders them again (it needs `LARMOR_DATA_DIR`).

The Viewer and Agents sections show the viewer's Help guides, which live in
`src/larmor/viewer/resources/guides/`. Each page in `docs/viewer/` (and `docs/agents/connecting.md`) includes one
guide with `{include}`, so the Help panel and the site read the same file; edit the guide there, not a copy. A
guide links to another by its file name (`[Settings](09-settings.md)`). That link works in the Help panel, and
`docs/_ext/viewer_docs.py` points it at the right page of the site. Images go in the guides' `images/` folder. A
new guide needs its own page, in the toctree of `docs/viewer/index.md`, and a line in `docs/_extra/llms.txt`, and
a test fails until every guide has a page. The lists of keyboard shortcuts and of the agents' tools are generated
from the code (the shortcut registry and the viewer's methods marked for agents), so they need no editing.

Then check the [definition of done](#definition-of-done) yourself. There should be tests that would fail if the
code were wrong, docstrings for every public function and class, type annotations, and an example in `examples/`
for anything users see.

### 7. Push the branch

```bash
git push -u origin build/8-pr-templates
```

`-u` links your local branch to the one on GitHub, so later pushes on this branch need only `git push`.

### 8. Open the pull request

Open it from VS Code (Source Control panel, "Create Pull Request", from the GitHub Pull Requests extension) or in
the browser:

```bash
gh pr create --web
```

[Pull requests](#pull-requests) below explains how to write the title and the description. The description must
contain `Closes #8`, so that merging the pull request closes the ticket.

If the work is not finished but you want early feedback, open it as a draft (`gh pr create --web --draft`, or
"Create Draft" in VS Code) and mark it "Ready for review" when it is done.

GitHub numbers pull requests and tickets in one sequence, so a pull request gets the next free number. You need
that number to merge, and it is shown at the top of the pull request's page.

### 9. Read your own changes

On the pull request's page, open the "Files changed" tab and read every change as if someone else had written
it. This catches leftover `print` calls, debugging code, files that don't belong to the ticket, and missing tests.
Fix what you find before asking for a review.

### 10. Ask for a review and handle the comments

Ask for a review. [Reviews](#reviews) below lists what the reviewer looks at.

Fix each point on the same branch. Change the files, then:

```bash
git status
git add .github/pull_request_template.md
git commit -m "Explain the How I tested section in the PR template"
git push
```

The pull request updates itself with the new commit. Ask for another look, and repeat until it is approved. If
you disagree with a point, say so and explain why. Reviewers can be wrong too, and the discussion is part of the
review.

### 11. Merge

Merge only when the pull request is approved and its checks are green. Never merge with a failing or pending
check.

```bash
gh pr merge 80 --squash --delete-branch --subject "Add issue and pull request templates" --body "Bug reports, feature requests and pull requests now start from a template, so each one carries the same information.

Closes #8"
```

- `80` is the pull request's number (in this example), not the ticket's.
- `--squash` turns all the commits of the branch into one commit on `main`, so each ticket is one entry in the
  history.
- `--subject` becomes the title of that commit. Use the pull request's title.
- `--body` becomes its description, a sentence or two on why, then `Closes #8`. Without `--body`, GitHub fills
  it with the list of all the branch's commit messages, which says little.
- `--delete-branch` deletes the branch on GitHub and locally, and switches you to `main`.

### 12. Check that everything is up to date

`gh pr merge` also updates your local `main`. Check it:

```bash
git status
git log --oneline -3
```

`git status` should say your branch is up to date with `origin/main`, the log should show the merged commit on
top, and the ticket should now be closed.

## Commit messages

A commit message has a title line and, when useful, a body. The title says what the commit does, written as a
command, in under about 70 characters. The body says why, when the title does not make that obvious.

| Good | Not good | Why not |
| --- | --- | --- |
| Add the bug report template | added template | Not a command; doesn't say which template |
| Fix the TRK voxel-to-world transform | Fixes | Says nothing about what was fixed |
| Read bvals files with Windows line endings | Update bvs.py | Names a file, not a change |
| Rename dwi_to_mask to mask | fix/19 rename | That's the branch name |

A body is a second `-m`:

```bash
git commit -m "Read bvals files with Windows line endings" -m "Files written on Windows end lines with \r\n, which made the last b-value fail to parse."
```

## Pull requests

**Title.** The title says what the pull request does, written like a commit title, such as "Add issue and pull
request templates", and never the branch name. It becomes the title of the merged commit on `main` and a line in
the release notes, so users must be able to read it.

**Description.** The description has three parts, then the line that closes the ticket:

```markdown
## What

Adds templates for bug reports, feature requests and pull requests.

## Why

Every report and pull request was written differently, and important information (how to reproduce a bug, how a
change was tested) was easy to leave out.

## How I tested

Opened a new issue and a new pull request on GitHub and checked that each template appears and reads correctly.

Closes #8
```

- **What** tells the reviewer what to expect before reading the code.
- **Why** gives the reason. A year later, it is the part that explains why the code looks the way it does.
- **How I tested** is the proof that the change works: the tests added, and any check run by hand with its
  result. A change that makes something faster also shows the measured speed-up and that the results did not
  change. A change to the viewer includes a screenshot.
- **`Closes #8`** goes in the description, never in the title. `Fixes #8` and `Resolves #8` work too.

## Reviews

Every pull request is reviewed before it is merged. The reviewer checks that:

- **It works.** The tests prove the change is correct, and they would fail if the code were wrong. When in doubt,
  the reviewer breaks the code on purpose and checks that a test fails.
- **The checks pass.** CI is green on every platform.
- **It is done.** It meets the [definition of done](#definition-of-done): tests, docstrings with
  units and conventions, type annotations, and an example for anything users see.
- **It follows the conventions.** The [principles](#principles) and the [style guide](style.md), which
  includes the model and fit pattern and the command-line conventions.
- **It does one thing.** The changes belong to the ticket; anything else goes in its own ticket.
- **It is described properly.** The title, the What, Why and How I tested, and `Closes #N` are all there and
  match the changes.

For each problem, a review explains what is wrong, why it matters, and how to fix it.

## Releases

When all the tickets of a version are done, the version is tagged in git and published on GitHub, with release notes
generated from the titles of the merged pull requests. The steps are in [releasing.md](releasing.md).

## Deprecations

From the first public release on, a public name (a function, class, argument, command or option) is not removed
or changed incompatibly without warning. It is deprecated first, for at least one minor release, and removed in a
later one. Deprecating it takes three things:

- `warnings.deprecated` ([PEP 702](https://peps.python.org/pep-0702/)) on a function or class, which warns when it
  is used and lets type checkers flag the uses; for an argument, a `DeprecationWarning` when it is given, and for a
  command or option, a warning printed by the command line;
- a `.. deprecated:: X.Y` line in its docstring, saying what to use instead;
- a line in the release notes of the version that deprecates it, and another in the one that removes it.

Before 1.0 a minor release could still break the API, as [releasing.md](releasing.md) explains, but it went
through this path whenever it could. Names that are not public (not in a package's `__all__`, see the
[style guide](style.md#public-and-private)) may change at any time.

## When something goes wrong

### You committed on main instead of a branch

As long as you have not pushed `main`, you can move the commit to a branch. First check with `git status` that
nothing is left uncommitted, because the third command throws uncommitted changes away.

```bash
git switch -c build/8-pr-templates
git switch main
git reset --hard origin/main
git switch build/8-pr-templates
```

The first command creates the branch where you are, keeping the commit. The third puts `main` back to exactly
what GitHub has.

### git pull on main says the branches have diverged

Your local `main` has a commit that GitHub's `main` does not, usually because of the case above. If that change
has already been merged through a pull request, throw the local copy away with `git reset --hard origin/main`. If
it has not, move it to a branch first, as above.

### You forgot Closes #8

Before merging, add it to the description (the Edit button on the pull request's page). If the pull request is
already merged, close the ticket by hand:

```bash
gh issue close 8 --comment "Done in #80"
```

### GitHub says the branch has conflicts

Something merged into `main` after you created your branch changed the same lines as you did. Bring the new
`main` into your branch:

```bash
git switch build/8-pr-templates
git fetch
git merge origin/main
```

Git marks each conflicting place in the file between `<<<<<<<` and `>>>>>>>` lines, and VS Code shows buttons to
choose a version (Accept Current, Accept Incoming, Accept Both). Choose the right version for each place, then
commit and push:

```bash
git add .github/pull_request_template.md
git commit -m "Merge main into build/8-pr-templates"
git push
```

If you are not sure which version is right, ask before resolving.

## Quick reference

```bash
# 2. Update main
git switch main
git pull

# 3. Create the branch
git switch -c build/8-pr-templates

# 5. Commit (repeat while you work)
git status
git add .github/ISSUE_TEMPLATE/bug_report.md
git commit -m "Add the bug report template"

# 6. Run the checks
uv run pytest
uv run ruff format .
uv run ruff check .
uv run mypy
uv run lint-imports
cd cpp
uv run cmake --preset dev
uv run cmake --build --preset dev
uv run ctest --preset dev
find src tests -name '*.cpp' -o -name '*.hpp' | xargs uv run clang-format -i
uv run run-clang-tidy.py -p build/dev -quiet 'cpp/(src|tests)/'
cd ..

# 7. Push the branch
git push -u origin build/8-pr-templates

# 8. Open the pull request (title; What, Why, How I tested; Closes #8)
gh pr create --web

# 10. After each round of review comments
git status
git add .github/pull_request_template.md
git commit -m "Explain the How I tested section in the PR template"
git push

# 11. Merge (80 = the pull request's number in this example)
gh pr merge 80 --squash --delete-branch --subject "Add issue and pull request templates" --body "Bug reports, feature requests and pull requests now start from a template, so each one carries the same information.

Closes #8"

# 12. Check
git status
git log --oneline -3
```
