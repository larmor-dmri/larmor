# Releasing a version

This document explains how to release a version of Larmor. A release gives a fixed state of `main` a version
number, so that everyone can say exactly which Larmor they used (for example in a paper), and users can read what
changed since their previous version. Each version's GitHub milestone plans what it should contain, and the
[roadmap](about/roadmap.md) gives the outline.

A release consists of three things:

1. the version number in `pyproject.toml`, which is what an installed Larmor reports about itself,
2. a git tag, `v0.1.0`, which is a permanent name for the exact commit that is the release,
3. a GitHub release, the page on the [Releases](https://github.com/larmor-dmri/larmor/releases) tab that
   shows the tag together with its release notes.

There is no changelog file. GitHub writes the release notes from the titles of the pull requests merged since the
previous release, which is why a pull request's title must say what changed (see
[Pull requests](contributing.md#pull-requests)). Only the breaking changes are written by hand, once per release.

## Version numbers

Larmor uses [semantic versioning](https://semver.org): `MAJOR.MINOR.PATCH`, for example `0.1.0`. The milestones
use only the first two parts; milestone "0.1" is version `0.1.0`.

| What changed since the previous release | Before 1.0 | From 1.0 on |
| --- | --- | --- |
| Only bug fixes, nothing new | Raise PATCH: `0.1.0` becomes `0.1.1` | Raise PATCH: `1.0.0` becomes `1.0.1` |
| New features that work with existing code and scripts | Raise MINOR: `0.1.0` becomes `0.2.0` | Raise MINOR: `1.0.0` becomes `1.1.0` |
| Something that breaks existing code or scripts | Raise MINOR: `0.1.0` becomes `0.2.0` | Raise MAJOR: `1.0.0` becomes `2.0.0` |

Before 1.0 the CLI and the API could still change, and from 1.0 on they are a promise to users. When a part goes
up, the parts after it go back to 0.

The version in `pyproject.toml` always names **the next release being prepared**, not the last one released.
Between two releases, `git describe --tags` tells you exactly which commit you have. It prints `v0.1.0` on the
release itself, and `v0.1.0-12-g1a2b3c4` twelve commits after it.

## The steps

The commands below release version **0.1.0**, whose milestone is **0.1 Foundations**. For another version, replace
these values.

### 1. Check that the version is finished

```bash
gh issue list --milestone "0.1 Foundations" --state open
gh pr list --state open
gh run list --branch main --limit 5
```

The first command must print nothing, so move the tickets that can wait to the next milestone (and update the
[roadmap](about/roadmap.md) if the outline changes). No pull request for this version may still be open, and the
newest CI run on `main` must have succeeded.

### 2. Check the version number

```bash
grep '^version' pyproject.toml
```

It normally shows `0.1.0` already. If you decided on another number (for example a bug-fix release `0.1.1`),
change it in a normal pull request first, and run `uv lock` in it, because `uv.lock` records Larmor's own version
too.

In the same pull request, set `version` and `date-released` in `CITATION.cff` to the version and the day of the
release.

Also add a line for the new version to "Versions so far" in `docs/about/changelog.md`: one or two sentences on
what it adds, in the words of its release notes.

### 3. Tag the release commit

```bash
git switch main
git pull
git log --oneline -1
git tag -a v0.1.0 -m "larmor 0.1.0"
git push origin v0.1.0
```

`-a` makes an annotated tag, which records who made it and when. `git push` does not send tags on its own, so the
last command pushes this one by name. A pushed tag is never moved. If the release contains a mistake, fix it and
release `0.1.1`.

### 4. Create the GitHub release as a draft

```bash
gh release create v0.1.0 --verify-tag --draft --title "larmor 0.1.0" --generate-notes
gh release view v0.1.0 --web
```

`--generate-notes` lists every pull request merged since the previous tag, by title and number. `--verify-tag`
stops if the tag was not pushed, instead of silently tagging the newest commit. `--draft` keeps the release hidden
until you publish it.

In the browser, click the pencil to edit the draft. If anything breaks existing code or scripts, add a section at the
top:

```markdown
## Breaking changes

- `SphericalDeconvolution` is renamed `SDModel`, and its `fit` returns an `SDFit` (#88).
```

Fix any pull request title that reads badly in the list, then click **Publish release**.

### 5. Close the milestone and start the next version

```bash
gh api 'repos/{owner}/{repo}/milestones' --jq '.[] | "\(.number) \(.title)"'
gh api --method PATCH 'repos/{owner}/{repo}/milestones/1' -f state=closed
```

The first command prints the number of each open milestone, and the second closes the released one by its number.
`gh` fills in `{owner}/{repo}` from the repository you run it in. Then open a pull request named "Start version
0.2.0" that sets `version = "0.2.0"` in `pyproject.toml` and runs `uv lock`.

## Wheels and PyPI

Pushing the tag also runs the Wheels workflow (`.github/workflows/wheels.yml`). It builds a wheel for each
platform and the source distribution, and tests each one in a fresh environment. The source distribution carries
Eigen and pocketfft (`cmake -P cpp/cmake/vendor.cmake` puts them in), so it builds without downloading any C++
library.

When every wheel and the source distribution pass, the workflow's publish job uploads them to the `larmor-dmri`
project on PyPI. The job runs in the GitHub environment `pypi` and uses trusted publishing, so no PyPI token is
stored.

## Website

The workflow `.github/workflows/docs.yml` builds the website (`sphinx-build -W`, with the gallery run on the
example data) and publishes it to the `gh-pages` branch, which GitHub Pages serves at https://larmor.org.

- On a pull request it only builds, so a broken build fails the check.
- On a push to `main` it publishes the build to `/dev/`.
- On a version tag (`v*`) it publishes the build to the site root, the stable version, and keeps `/dev/`. Files of
  an older stable version that the new one no longer has stay in the root until they are deleted from the branch.
- On every publish it writes `CNAME` (larmor.org) and `.nojekyll` at the root of the branch. It also updates
  `versions.json`, the list that the theme's version switcher reads, which holds dev and, as the preferred
  version, the latest tagged one.

The maintainer makes these settings once, after the repository is public and the first publish has created the
`gh-pages` branch:

1. Settings, Pages, Source: Deploy from a branch, `gh-pages`, folder `/ (root)`.
2. Settings, Pages, Custom domain: `larmor.org`; then tick Enforce HTTPS when GitHub offers it (after the DNS
   records resolve).
3. At the domain registrar, for `larmor.org`: A records `185.199.108.153`, `185.199.109.153`, `185.199.110.153`,
   `185.199.111.153`; AAAA records `2606:50c0:8000::153`, `2606:50c0:8001::153`, `2606:50c0:8002::153`,
   `2606:50c0:8003::153`; and a CNAME record `www` pointing to `larmor-dmri.github.io`. These are the addresses in
   GitHub's documentation on custom domains; check them there again before entering them.
