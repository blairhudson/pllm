# Run a provider

Declare the checkpoint policy and choose a supported execution path.


## Public checkpoint

```bash
pllm serve "$PLLM_MODEL_SOURCE" \
  --weights public \
  --provider-push-api-key "$PLLM_PROVIDER_PUSH_API_KEY" \
  --local-files-only \
  --model-id private-model \
  --host 127.0.0.1 \
  --port 8000
```

Set `PLLM_API_KEY` in the provider environment. Without an explicit key, the reference CLI generates a key and prints it once; this is unsuitable for captured production logs. Use a configured secret for deployment.

## Trusted preparation role

```bash
pllm preparation serve "$PLLM_MODEL_SOURCE" \
  --api-key "$PLLM_PREPARATION_API_KEY" \
  --inference-url http://127.0.0.1:8000 \
  --push-api-key "$PLLM_PROVIDER_PUSH_API_KEY" \
  --model-id private-model \
  --local-files-only \
  --host 127.0.0.1 \
  --port 8001
```

This public-weight-only service exposes no Responses or inference route. Before
chat, it receives one root seed per stage for a batch of rows, expands `r` and
`s`, pushes `W·r-s` rows to inference's fixed endpoint, waits for durable
acknowledgement, and must erase the masks. Inference seals the complete inventory
`READY`. Preparation remains idle during online chat. The client verifies that its
model, stage, and exact ring commitments match inference.

## Separate weights from adversary assumptions

`--weights public` means that model confidentiality from the client is not required. It does not mean that any malicious provider is safe.

`--weights confidential --client-trust guarded` selects blinded preparation and operational query limits. The client still sees intermediate results. These controls do not establish cryptographic model confidentiality against a modified client.

```bash
pllm serve "$PLLM_MODEL_SOURCE" \
  --weights confidential \
  --client-trust untrusted
```

This combination is rejected for arbitrary Hugging Face graphs. There is no silent fallback to guarded execution. The authenticated arithmetic preview is a simulator, not a distributed malicious security implementation.

## Compile once

```bash
pllm serve "$PLLM_MODEL_SOURCE" \
  --weights public \
  --local-files-only \
  --model-id private-model \
  --compiled-cache-dir .cache/compiled \
  --quantization-chunk-rows 64
```

Compilation uses W8A8 by default so ordinary checkpoints retain usable generation quality. `--weight-bits 4 --activation-bits 4` selects the smaller research representation. Bit equality with a clear quantized graph does not mean equality with the original floating point checkpoint.

## Health and model discovery

`/healthz` is a liveness check. Use authenticated `/v1/models` to inspect loaded model status. A healthy process is not proof that a model compiled or that either service will compute honestly.

The provider intentionally rejects plaintext `/v1/responses` for a private model. Applications call the [local gateway](/docs/client/openai) instead.
