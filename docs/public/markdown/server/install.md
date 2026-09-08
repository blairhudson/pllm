# Install a provider

Build the Rust extension and load a supported checkpoint.


## Requirements

Use Python 3.11–3.13 and UV. Building from source also requires the Rust
toolchain in `rust-toolchain.toml`. Maturin builds the extension during package
installation. A matching platform wheel needs no compiler at runtime.
SEAL and TenSEAL remain the cryptographic backend.

```bash
cd pllm
uv sync --extra he
uv run pllm build
uv run pllm serve --help
```

## Checkpoint source

The provider accepts a supported local Hugging Face snapshot or a Hub repository
ID. Local snapshots make revision selection explicit and avoid downloads at
startup. Check [model support](/docs/server/models/) before selecting a family.

```bash
export PLLM_API_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
uv run pllm serve ./models/checkpoint \
  --weights public \
  --model-id private-model \
  --host 127.0.0.1 --port 8000 \
  --local-files-only
```

The remote provider accepts private tensor requests. It is not the endpoint to
which an ordinary SDK should send its plaintext prompt. Run the
[local gateway](/docs/client/openai/) in the customer's trusted environment.
