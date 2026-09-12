# Build and release

Use committed locks, verified native artifacts, and scoped publication permissions.


## Publication boundary

Package publication and static Pages deployment do not deploy runtime services.
This source does not assume an owner, repository URL, domain, or PyPI account.
Configure trusted publishing only for the repository and project you control.

PyPI publication uses OIDC through the `pypi` environment. Pages uses the separate
`github-pages` environment and derives its base path from repository settings.
Pull requests receive neither authority.

## Locks

`uv.lock`, `Cargo.lock`, and `docs/package-lock.json` are committed inputs. After
dependency or version changes:

```bash
python scripts/lock_dependencies.py
git diff -- uv.lock Cargo.lock docs/package-lock.json
```

Review resolved sources and checksums. Release validation fails if any lock is
missing; do not fabricate one.

## Version and tag

```bash
uv run python scripts/release.py prepare 0.17.0a1
# Update changelog and validation record, then review and merge.
uv run python scripts/release.py check v0.17.0a1 --require-locks
git tag -a v0.17.0a1 -m "PLLM 0.17.0a1"
git push origin v0.17.0a1
```

Use the intended new version, not the example unchanged. The single preparation
command keeps Cargo, Python, citation, and dependency locks aligned. Pushing the
tag runs tested wheel and sdist publication through PyPI Trusted Publishing; no
repository secret is used. Never move a published tag or replace a published
distribution; investigate ambiguous failures and release a new version when bytes
must change.

## Required local evidence

```bash
uv sync --locked --extra he --extra sdk
cargo test --locked
cargo clippy --locked --workspace --all-targets -- -D clippy::correctness
uv run pytest
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

Record actual results, platform, and revision. Historical logs and configured CI
are not current evidence. Regenerate paper/research assets only when their
canonical source changed or the release deliberately refreshes them.

Native release wheels must pass compiled binding and clean installed-wheel tests.
The Python reference backend cannot substitute. Public seeded inventory has no HE
runtime dependency; the optional HE lane validates only explicitly selected legacy
paths.
