# Contributing to Larmor

Thank you for helping. Bug reports, questions and ideas are welcome as
[issues](https://github.com/larmor-dmri/larmor/issues); please search the existing ones first, and for a
bug give the Larmor version, your operating system and the smallest steps that show it. Security problems are
reported privately, as [SECURITY.md](SECURITY.md) explains.

For code and documentation, the full guide is [docs/contributing.md](docs/contributing.md): how a change goes from a
ticket to a merged pull request, the checks to run, how the documentation website is built, and what a review looks
at. The code style is in [docs/style.md](docs/style.md). In short:

1. Open an issue first for anything larger than a small fix, so the change can be agreed before you write it.
2. Set up with [uv](https://docs.astral.sh/uv/): `uv sync` installs Larmor in editable mode with the development
   tools, and builds its C++ module (this needs a C++ compiler).
3. Run the checks before you push: `uv run pytest`, `uv run ruff format .`, `uv run ruff check .`, `uv run mypy`
   and `uv run lint-imports`.
4. Open a pull request from a branch, with a title that says what it does, the sections of the template filled in,
   and `Closes #<issue>` in its description.

Everyone taking part follows the [code of conduct](CODE_OF_CONDUCT.md).
