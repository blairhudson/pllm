# Contributing

## Set up

Use the toolchain in `rust-toolchain.toml`, UV and Python 3.11 through 3.13.

```bash
uv sync --extra he --extra sdk
uv run pllm build
```

Rust sources are in `crates`; all Python code is in `python/pllm`. Do not add a
second import namespace at the root. Add public exports deliberately rather than
using wildcard imports. Protocol changes must preserve private activation scales
and one use correlation consumption.

## Check a change

```bash
cargo test -p pllm-core
cargo clippy --workspace --all-targets -- -D clippy::correctness
uv run ruff check python/pllm scripts tests
uv run pytest
uv run python scripts/check_repository.py
uv build
uv run python scripts/check_distributions.py dist
```

Use `cargo fmt --all` for Rust changes. For Python changes use the existing
project conventions and keep functions typed where practical. The core tests
must not acquire a Python dependency. Native changes require scalar comparisons,
edge cases, overflow checks and tests against the installed wheel.

## Documentation

```bash
cd docs
npm install
npm run check:content
npm test
npm run typecheck
npm run build
```

The Fumadocs site is exported as static files. Test repository base paths as well
as the root path. Edit `paper/manuscript.md`; `make paper` generates both the PDF
and web article through Pandoc.

## Evidence

Record the backend, hardware, shapes, repetitions and exact comparison in every
benchmark. Distinguish preparation, online execution, full request time, and
aggregate throughput. A passing reference test is not native validation. A
synthetic matrix does not establish checkpoint language quality.

## Releases

Only a maintainer should tag a release. Follow `RELEASING.md`, review dependency
lock changes, and let the workflow publish the tested artifacts. Do not attach
locally created API keys, model credentials, inventory databases or private
checkpoints to issues or pull requests.
