# Contributing

## Install the checkout

Install [UV](https://docs.astral.sh/uv/) and the Rust toolchain selected by
`rust-toolchain.toml`, then run this from the repository root:

```bash
uv tool install --force .
pllm --version
```

This builds the native extension and installs the current checkout as the
`pllm` command. Run the install command again after changing Python or Rust
source that you want to test through the installed CLI.

Rust sources live in `crates/`; Python sources live in `python/pllm/`. Do not
add another Python package at the repository root.

## Build a wheel

Use the standard uv/Maturin workflow from the repository root:

```bash
uv sync --group dev
uv run --no-sync maturin build --release --locked --out dist
```

Maturin's normal `--offline` and `-j 1` options are available when needed.
Build outputs stay ignored. No separate build receipt or attestation wrapper is
required. When testing an installed wheel, use `pytest -o pythonpath=` so the
checkout's Python package does not shadow the installation.
Use `uv run --no-sync` for those checks: an ordinary `uv run` can reinstall the
editable checkout in place of the wheel being tested.

## Experiments and results

Keep reusable experiment `.py` files in `examples/` and retain ordinary benchmark
outputs for reported results in `docs/evidence/`. These are the evidence record;
correctness belongs in normal tests. Preserve the benchmark's own workload,
configuration, build, and metric fields.

Do not add separate receipt collections, source copies, environment dumps,
nested archives, or custom archive validators. Temporary diagnostics belong in
ignored `target/`. Git history supplies old source and change history. Simplify
legacy collections as they are touched, retaining useful results and tests.

## Run the documentation site

From the repository root:

```bash
npm --prefix docs ci
npm --prefix docs run dev
```

The development site runs at <http://localhost:3000>. Install the paper build
tools listed in `docs/README.md`. Edit documentation under `docs/content/` and
canonical paper sources under `paper/`. The docs build regenerates publications;
to rebuild only the papers:

```bash
npm --prefix docs run papers
```

Keep working plans in `docs/plans/`, research notes in `docs/research/`, and
session handoffs in `docs/handoffs/`. Use lower-case kebab-case filenames for new
documents.
Do not commit generated PDFs, archives, figures, publication MDX, or download copies.

## Check changes

Run checks relevant to your change before opening a pull request:

```bash
cargo test
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
uv run ruff check python/pllm scripts docs/scripts docs/tests tests
uv run pytest
```

For documentation changes:

```bash
uv run python scripts/docs_regen.py
uv run python scripts/docs_regen.py --check
npm --prefix docs run typecheck
npm --prefix docs run build
```

Do not include credentials, private data, model checkpoints, or generated local
state in a pull request. Follow [SECURITY.md](SECURITY.md) for vulnerabilities
and [RELEASING.md](RELEASING.md) for releases.
