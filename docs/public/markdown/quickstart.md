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
: "${PLLM_MODEL_SOURCE:?Set PLLM_MODEL_SOURCE to the checkpoint directory}"
uv run pllm serve "$PLLM_MODEL_SOURCE" \
  --weights public \
  --model-id private-model \
  --local-files-only \
  --compiled-cache-dir .cache/compiled
```

The provider key authenticates private requests. Keep it in your secret store. Do not enable `--allow-test-correlations` for private data: that test path lets the server learn the masks.

## Configure the client

In another terminal, use the same provider key. On another machine, replace the loopback URL with the provider's HTTPS address.

```bash
umask 077
uv run pllm configure \
  --server http://127.0.0.1:8000 \
  --api-key "$PLLM_API_KEY" \
  --model private-model
uv run pllm chat
```

The first response may wait for preparation. A fast generation interval after preparation is not the rate of the complete request.

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

Your application owns the plaintext. The configured provider receives encrypted preparation and masked stage inputs. Request timing, shapes, stage names, and approximate lengths remain visible. This implementation assumes the provider follows the protocol; it does not verify every provider computation.
