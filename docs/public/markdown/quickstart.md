# Run your first private request

Start a provider, configure the client, and generate a response.


Use this repository for both processes. There is no requirement to trust a package that happens to share the `pllm` name on a public registry. These commands use a local Hugging Face snapshot so the checkpoint revision is under your control.

## Install the runtime

From the release root:

```bash
cd pllm
uv sync --extra he
uv run pllm build
uv run pllm --version
```

A Rust toolchain is required to build this source checkout. Maturin builds the native extension during `uv sync`. Installing a matching platform wheel requires no Rust compiler. Python 3.13 matches the recorded HE experiments. See the [installation guide](/docs/server/install) for supported inputs and dependency checks.

## Start the provider

Set `PLLM_MODEL_SOURCE` to your local checkpoint directory before running this command. The directory must contain its configuration, tokenizer assets, and supported Safetensors weights.

```bash
export PLLM_API_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
export PLLM_PREPARATION_API_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
export PLLM_PROVIDER_PUSH_API_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
: "${PLLM_MODEL_SOURCE:?Set PLLM_MODEL_SOURCE to the checkpoint directory}"
uv run pllm serve "$PLLM_MODEL_SOURCE" \
  --weights public \
  --provider-push-api-key "$PLLM_PROVIDER_PUSH_API_KEY" \
  --model-id private-model \
  --local-files-only \
  --compiled-cache-dir .cache/compiled
```

## Start trusted preparation

In another trusted process, load the same public checkpoint:

```bash
uv run pllm preparation serve "$PLLM_MODEL_SOURCE" \
  --api-key "$PLLM_PREPARATION_API_KEY" \
  --inference-url http://127.0.0.1:8000 \
  --push-api-key "$PLLM_PROVIDER_PUSH_API_KEY" \
  --model-id private-model \
  --local-files-only \
  --compiled-cache-dir .cache/preparation-compiled \
  --port 8001
```

Self-host this role unless another preparation operator is trusted not to retain
masks or collude with inference. Keep all three service credentials in your secret store.

## Configure the client

In another terminal, use the same provider key. On another machine, replace the loopback URL with the provider's HTTPS address.

```bash
umask 077
uv run pllm configure \
  --server http://127.0.0.1:8000 \
  --api-key "$PLLM_API_KEY" \
  --preparation-url http://127.0.0.1:8001 \
  --preparation-api-key "$PLLM_PREPARATION_API_KEY" \
  --model private-model
uv run pllm chat
```

Each stage sends preparation and inference work concurrently. Count both links and all first-token work when measuring latency.

## Use Python

```python
from pllm import OpenAI

with OpenAI() as client:
    response = client.responses.create(
        input="Explain how this request remains private.",
        max_output_tokens=64,
    )
    print(response.output_text)
```

## Check the boundary

Your application owns the plaintext. Preparation receives fresh seeds and bound stage metadata; inference receives masked stage inputs. Timing, shapes, stage names, and approximate lengths remain visible. This implementation does not verify every service computation.
