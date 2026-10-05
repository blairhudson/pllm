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
session handoffs in `docs/handoffs/`. Put measurements and reproduction records
in `docs/evidence/`; use lower-case kebab-case filenames for new documents.
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
