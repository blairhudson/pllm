# Repository setup and releases

## First upload

The checkout is self contained. It has no configured remote or embedded Git
history. Initialize `main`, review the files and push to a repository you own.

```bash
git init -b main
git add .
git commit -m "Initialize PLLM source, documentation, paper and CI"
```

Create the repository with your normal GitHub workflow. GitHub CLI can publish
the current directory without modifying these sources:

```bash
gh repo create pllm --private --source=. --remote=origin --push
```

Use `--public` instead after reviewing the code, research records and license.
This guide does not create a repository or publish a package on your behalf.

## Required settings

| Setting | Value |
| --- | --- |
| Default branch | `main` |
| Branch protection | Require PR review and `All checks`; prohibit force pushes |
| Vulnerability reporting | Enable private vulnerability reporting |
| Pages | Settings → Pages → Build and deployment → **GitHub Actions** |
| Pages environment | `github-pages`, restricted to `main` |
| PyPI environment | `pypi`, with required approval and version tag restrictions |
| PyPI trusted publisher | Owner and repo you created; workflow **`release.yml`**; environment **`pypi`** |

Create or claim the `pllm` PyPI project under an account you control. Repository
configuration does not reserve that name. Use PyPI's pending publisher feature
for a first publication. Do not overwrite a third party package using the same
name. The workflow uses OIDC and does not require a `PYPI_TOKEN` secret.

GitHub Pages reads the configured repository path through `configure-pages`.
A project site such as `/pllm/` and a configured custom domain are handled without
hardcoded account URLs. Pages hosts the static website only, never the inference
provider or local client gateway.

## Resolve and commit dependencies

The source was assembled without registry DNS, so dependency locks were not
invented. Before releasing, generate the real locks:

```bash
python scripts/lock_dependencies.py
git add uv.lock Cargo.lock docs/package-lock.json
git commit -m "Lock Python and documentation dependencies"
```

Alternatively, run **Resolve dependency locks** in the Actions tab, download the
`dependency-locks` artifact and commit all three files in a PR. That workflow has no
write permissions and cannot bypass branch protection.

Bootstrap CI uses real dependency resolution when a lock is absent. Once locks
exist it runs `uv sync --locked` and `npm ci`. A version tag cannot publish without
all three committed locks. Python's bounded library requirements remain in the wheel;
application and CI environments use the committed lock.

## What each workflow does

| Workflow | Trigger | Result |
| --- | --- | --- |
| `ci.yml` | PR, push to main, manual, release call | Python matrix, native arithmetic, actual HE, installed OpenAI SDK, distributions, docs and paper |
| `docs-build.yml` | Reusable workflow | Build the Pandoc paper and downloads, test/typecheck/build Fumadocs |
| `pages.yml` | Docs changes on main, manual | Build and deploy static docs with correct base path |
| `release.yml` | Push of a version tag | Validate tag/locks, rerun CI, publish to PyPI, create GitHub release |
| `wheels.yml` | Reusable or manual | Build and test native wheels on five operating system/architecture targets |
| `native-benchmark.yml` | Kernel PR changes or manual | Record actual Rust, previous C++ and NumPy timings |
| `lockfiles.yml` | Manual | Generate lockfiles for review and commit |
| `research.yml` | Manual | Run the independent lifecycle study without making it a PR runtime dependency |

All external actions are pinned to commit IDs. Dependabot proposes updates.
PRs have no publication credentials. OIDC permission is limited to the publication
or deployment job. PyPI publishes the distribution artifact that passed CI rather
than rebuilding it in the credentialed job.

## Release a version

```bash
python scripts/release.py set-version 0.17.0a1
python scripts/lock_dependencies.py
# Update CHANGELOG.md, review changes, and merge the release PR.
python scripts/release.py check v0.17.0a1 --require-locks
git tag -a v0.17.0a1 -m "PLLM 0.17.0a1"
git push origin v0.17.0a1
```

Maturin takes the version inherited from `[workspace.package]` in `Cargo.toml`. The release script updates and
checks its Python mirror in `python/pllm/_version.py` and `CITATION.cff`.
The release workflow calls the same checks used in PRs, creates distribution
attestations, publishes the verified wheel and source distribution, then attaches
these to GitHub along with the full repository source, paper PDF, paper source
and SHA256 checksums. Alpha/beta/RC tags create GitHub prereleases. Publication to
PyPI is real: an approval is not a dry run.

If publication fails, inspect the job before retrying. Do not move a published
tag, replace an existing distribution, or assume an error means no upload occurred.
Use a new version when published artifacts must change.

## Local checks

```bash
uv sync --locked --extra he --extra sdk
uv run pytest
uv run ruff check python/pllm scripts tests
uv build
uv run python scripts/check_distributions.py dist
uv run twine check --strict dist/*
make paper
uv run --no-project --python 3.13 python scripts/prepare_docs.py
cd docs && npm ci && npm test && npm run typecheck && NEXT_PUBLIC_BASE_PATH=/pllm npm run build
```

HE, installed SDK, JavaScript framework builds and remote deployment have separate
records. Do not turn a configured workflow into a claim that it already ran.

## Native release requirements

Publication uses only the portable `pllm-wheel-*` artifacts plus `pllm-sdist`.
The host validation wheel is not published as a manylinux wheel. Linux wheels
are linked through Zig for manylinux 2.28. The other platforms use native Rust
linkers. Never retag a Linux wheel to pretend it meets a different libc policy.

The Rust unit suite, compiled Python binding tests, clean wheel imports and
existing runtime checks are release requirements. A Python reference run cannot
satisfy them. The optional HE dependency has its own platform wheel availability.
