# Repository setup and releases

## Boundary

Release automation publishes Python artifacts and a static documentation site. It
does not deploy inference, preparation, or a client gateway. Do not treat a
successful package or Pages job as a production-security review.

Releases publish from `blairhudson/pllm`. Confirm `git remote -v` points to that
repository before preparing a release.

## Required repository settings

| Setting | Expected policy |
| --- | --- |
| Default branch | `main` |
| Branch protection | Require review and the configured aggregate check; prohibit force pushes |
| Vulnerability reporting | Enable private vulnerability reporting |
| Pages | Build and deploy with GitHub Actions |
| Pages environment | `github-pages`, restricted to the release branch |
| PyPI environment | `pypi`, with approval and version-tag restrictions |
| PyPI trusted publisher | Actual owner/repository, `release.yml`, environment `pypi` |

For the first release, create a pending Trusted Publisher at PyPI with these exact
values: owner `blairhudson`, repository `pllm`, workflow `release.yml`, environment
`pypi`, and project name `pllm`. Create the matching GitHub `pypi` environment and
limit deployment to protected `v*` tags. The release workflow uses OIDC; no GitHub
or PyPI secret is required. Do not add a long-lived `PYPI_TOKEN`.

GitHub Pages receives its base path from `configure-pages`. Repository paths and
custom domains therefore require no hardcoded account URL. Pages hosts static
documentation only.

## Locked dependencies

`uv.lock`, `Cargo.lock`, and `docs/package-lock.json` are committed release inputs.
After a version or dependency change, refresh all three through the repository
script and review the diff:

```bash
python scripts/lock_dependencies.py
git diff -- uv.lock Cargo.lock docs/package-lock.json
```

CI uses `uv sync --locked`, Cargo's lock, and `npm ci` when locks are present. A
version tag fails release validation if any lock is missing. Never fabricate or
hand-edit resolved checksums to make a release pass.

## Workflows

| Workflow | Trigger | Scope |
| --- | --- | --- |
| `ci.yml` | Pull request, push, manual, release call | Python matrix, native arithmetic, HE and SDK lanes, distributions, docs, paper |
| `docs-build.yml` | Reusable workflow | Paper assets plus Fumadocs content, tests, typecheck, and static build |
| `pages.yml` | Documentation changes on `main`, manual | Static Pages artifact and deployment |
| `release.yml` | Version tag | Tag/lock validation, CI, PyPI publication, GitHub release |
| `wheels.yml` | Reusable or manual | Native wheel build and installed-wheel tests on configured targets |
| `native-benchmark.yml` | Kernel changes or manual | Rust, retained C++ control, and NumPy benchmark records |
| `lockfiles.yml` | Manual | Lockfile generation for review |
| `research.yml` | Manual | Historical lifecycle study, outside normal PR runtime |

External actions are commit-pinned. Pull requests have no publication credentials.
The credentialed publication job receives artifacts already built and tested; it
does not compile source with publishing authority in scope.

## Release a version

Use the single release setup command so Cargo, Python, citation, and lock versions
move together:

```bash
uv run python scripts/release.py prepare 0.17.0a1
# Update CHANGELOG.md and VALIDATION.md, review the complete diff, then merge.
uv run python scripts/release.py check v0.17.0a1 --require-locks
git tag -a v0.17.0a1 -m "PLLM 0.17.0a1"
git push origin v0.17.0a1
```

Replace the example version with the intended new version. Do not recreate or
move a published tag. If publication is ambiguous, inspect PyPI and the workflow
before retrying; use a new version when published files must change.

The release workflow builds platform wheels and an sdist, tests installed wheels
outside the checkout, creates attestations, publishes the verified distribution,
and attaches repository and documentation assets with checksums. Pre-release tags
produce GitHub prereleases. Approval is not a dry run.

After the workflow succeeds, verify the published package independently:

```bash
uv run --isolated --no-project --with pllm==0.17.0a1 pllm --version
```

## Local release checks

Run from a clean checkout with the committed locks:

```bash
uv sync --locked --extra he --extra sdk
cargo test --locked
cargo clippy --locked --workspace --all-targets -- -D clippy::correctness
uv run pytest
uv run ruff check python/pllm scripts tests
uv build
uv run python scripts/check_distributions.py dist
uv run twine check --strict dist/*
cd docs
npm ci
npm run check:content
npm test
npm run typecheck
NEXT_PUBLIC_BASE_PATH=/pllm npm run build
```

Run paper generation only when its canonical source or release asset changed; do
not regenerate historical outputs as a side effect of an operational docs edit.
Record commands actually executed in `VALIDATION.md`. Configured workflows and old
logs are not current evidence.

## Native artifact requirements

Publication uses only wheel artifacts produced by the wheel workflow plus the
validated sdist. Linux wheels target the configured manylinux policy through Zig;
other platforms use their native Rust linkers. Never retag a wheel to claim a
different ABI or libc policy.

Release evidence must include Rust unit tests, compiled binding tests, and clean
installed-wheel imports. The explicit Python numeric reference cannot substitute
for them. The optional `he` extra has separate platform-wheel availability and is
not required by the current public seeded-inventory path.
