# Install the services

Install PLLM services from PyPI and validate a public checkpoint source.


## Requirements

Use Python 3.11 through 3.13 and UV. The platform wheel includes the native
extension and needs no compiler on the runtime host.

```bash
uv venv --python 3.13
source .venv/bin/activate
uv pip install pllm
pllm serve --help
pllm preparation serve --help
```

The current public-weight path does not use SEAL/TenSEAL. Install the `he` extra
only for explicitly selected legacy confidential-weight or BFV paths.

## Checkpoint source

Both inference and preparation load the same supported public Hugging Face
snapshot. A local snapshot keeps revision and startup network behavior explicit:

```bash
export PLLM_API_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
export PLLM_PROVIDER_PUSH_API_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"

pllm serve ./models/checkpoint \
  --weights public \
  --model-id demo-model \
  --api-key "$PLLM_API_KEY" \
  --provider-push-api-key "$PLLM_PROVIDER_PUSH_API_KEY" \
  --host 127.0.0.1 \
  --port 8000 \
  --local-files-only
```

Public serving fails closed without a provider-push credential because inference
must authenticate offline corrections from preparation. Continue with
[service usage](/docs/server/usage) to start that role.

Remote model IDs are also accepted and may download at startup. Pin an explicit
revision for repeatability and review [model support](/docs/server/models) before
assuming a readable tensor layout is a complete executable graph.

Inference is not an ordinary Responses base URL. Applications send plaintext only
to the direct local client or [client-side gateway](/docs/client/openai).
