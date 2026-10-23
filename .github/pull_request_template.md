<!--
See docs/contributing.md, "Pull requests". The title says what this pull request does, as a command
(for example "Add the bug report template"), not the branch name.
The text inside these arrows is a hint and does not appear in the pull request.
-->

## What

<!-- What changed, so the reviewer knows what to expect before reading the code. -->

## Why

<!-- The reason for the change. A year from now, this is what explains why the code looks the way it does. -->

## How I tested

<!-- The proof that the change works: the tests added, and any check run by hand with its result.
A speed-up shows the measured timings and that the results did not change. A viewer change includes a
screenshot. -->

## Before asking for a review

<!-- Tick what applies; delete lines that do not apply to this change. -->

- [ ] The checks pass locally (`uv run pytest`, `uv run ruff format .`, `uv run ruff check .`,
  `uv run mypy`).
- [ ] New or changed public functions and classes have docstrings with units, conventions and array shapes.
- [ ] A method taken from another toolbox or paper (MRtrix3, dipy, MRIToolkit, ...) cites its original papers
  and names the toolbox it follows, in its docstring and in the documentation.
- [ ] A new user-facing feature has an example in `examples/`.
- [ ] I read my own changes in the "Files changed" tab.

<!-- Replace the number with the ticket this pull request finishes. "Closes" must stay in this description,
not the title, so that merging closes the ticket. -->

Closes #
