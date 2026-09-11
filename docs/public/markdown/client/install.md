# Install the client

Build the current public client from this source with UV and Maturin.


## Requirements

PLLM supports Python 3.11, 3.12, and 3.13. A source install needs UV and the Rust
toolchain selected by `rust-toolchain.toml`; a matching platform wheel does not
need a compiler. The client must run inside the customer boundary because it owns
plaintext, activation scales, attention state, inventory seeds, and output
reconstruction.

The public seeded-inventory path uses ordinary Rust integer arithmetic and does
not need TenSEAL.

## From this repository

```bash
cd pllm
uv sync
uv run pllm build
uv run pllm --help
```

`pllm build` reports whether `pllm._native` is loaded. It does not compile code at
service startup.

Install a persistent command from the checked source tree:

```bash
uv tool install .
pllm --help
```

Or run one isolated invocation:

```bash
uvx --from . pllm --help
```

These local-source commands do not assume that a registry project with the same
name belongs to this repository. Use a registry install only after verifying an
actual published release and its provenance.

## Optional integrations

Install the official OpenAI SDK for integration tests or direct adapter use:

```bash
uv sync --extra sdk
```

Install TenSEAL only for explicitly selected blinded confidential-weight,
direct-BFV, or historical research paths:

```bash
uv sync --extra he
```

The `he` extra does not make the current public path more private and is not part
of its online execution.

Continue with [configuration](/docs/client/configuration), the
[Python client](/docs/client/python), or [local OpenAI gateway](/docs/client/openai).
