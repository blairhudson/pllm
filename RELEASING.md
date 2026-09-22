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
| Branch protection | Require review and the CI jobs; prohibit force pushes |
| Vulnerability reporting | Enable private vulnerability reporting |
| Cloudflare Pages | Direct Upload projects `pllm-non` and `pllm-production` |
| Non-production environment | `non-production`; receives every `main` push |
| Production environment | `production`, restricted to manual dispatches from `main` |
| PyPI environment | `pypi`, with approval and version-tag restrictions |
| PyPI trusted publisher | Actual owner/repository, `release.yml`, environment `pypi` |

For the first release, create a pending Trusted Publisher at PyPI with these exact
values: owner `blairhudson`, repository `pllm`, workflow `release.yml`, environment
`pypi`, and project name `pllm.run`. Create the matching GitHub `pypi` environment and
limit deployment to protected `v*` tags. The release workflow uses OIDC; no GitHub
or PyPI secret is required. Do not add a long-lived `PYPI_TOKEN`.

Cloudflare Pages hosts static documentation only. Every `main` push builds the
site and deploys it to `pllm-non` at `non.pllm.run`. A manual dispatch from
`main` builds and deploys production to `pllm-production` at `pllm.run`. Set
repository variable `CLOUDFLARE_ACCOUNT_ID` and environment secrets
`PLLM_NON_PAGES_API_TOKEN` and `PLLM_PRODUCTION_PAGES_API_TOKEN`; each token
needs Pages write access to its project. The pinned Restack Action performs the
deployment. Infrastructure provisioning remains separate from deployment.

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
| `ci.yml` | Pull request or manual | Python 3.11-3.13, Rust, integrations, distributions, and docs |
| `pages.yml` | `main` push or manual | Deploy non-production automatically or production manually |
| `release.yml` | Version tag | Build native wheels and sdist, then publish to PyPI |

External actions are commit-pinned. Pull requests have no publication credentials.
The credentialed publication job receives artifacts already built and tested; it
does not compile source with publishing authority in scope.

## Release a version

Use the single release setup command so Cargo, Python, citation, and lock versions
move together:

```bash
uv run python scripts/release.py prepare 0.1.0a2
# Review the complete diff, then merge.
uv run python scripts/release.py check v0.1.0a2 --require-locks --docs
git tag -a v0.1.0a2 -m "PLLM 0.1.0a2"
git push origin v0.1.0a2
```

Replace the example version with the intended new version. Do not recreate or
move a published tag. If publication is ambiguous, inspect PyPI and the workflow
before retrying; use a new version when published files must change.

The release workflow builds platform wheels and an sdist, smoke-tests every wheel
outside the checkout, validates the artifacts, and publishes them through PyPI
Trusted Publishing. Approval is not a dry run.

After the workflow succeeds, verify the published package independently:

```bash
uv run --isolated --no-project --with pllm.run==0.1.0a2 pllm --version
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
npm --prefix docs ci
uv run python scripts/docs_regen.py --check
npm --prefix docs run typecheck
NEXT_PUBLIC_BASE_PATH=/pllm npm --prefix docs run build
```

Run paper generation only when its canonical source or release asset changed; do
not regenerate historical outputs as a side effect of an operational docs edit.
Use revision-bound CI results and retained research records as execution evidence;
configured workflows and old logs are not current evidence.

## Native artifact requirements

Publication uses only artifacts produced by `release.yml`. Linux wheels target
the configured manylinux policy through Zig;
other platforms use their native Rust linkers. Never retag a wheel to claim a
different ABI or libc policy.

Release evidence must include Rust unit tests, compiled binding tests, and clean
installed-wheel imports. The explicit Python numeric reference cannot substitute
for them. The optional `he` extra has separate platform-wheel availability and is
not required by the current public seeded-inventory path.
