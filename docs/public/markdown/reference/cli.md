# CLI reference

The flags below come from the supplied parser.


## Server

```text
pllm serve MODEL_PATH_OR_HUB_ID --weights public
```

| Option | Meaning |
| --- | --- |
| `--model-id` | Exposed model identifier; repeat once per loaded model |
| `--weights public\|confidential` | Who may learn the checkpoint |
| `--client-trust honest\|guarded\|untrusted` | Client behaviour assumption |
| `--activation-protection` | Explicit protocol selection |
| `--host`, `--port` | Bind address; defaults to loopback and 8000 |
| `--revision` | Hugging Face source revision |
| `--local-files-only` | Do not fetch missing checkpoint files |
| `--compiled-cache-dir` | Runtime matrix cache |
| `--quantization-chunk-rows` | Bound compile working rows; default 64 |
| `--engine-threads` | Native execution threads |
| `--max-batch-size`, `--max-wait-ms` | Stage coalescing controls |
| `--allow-test-correlations` | Insecure test option; never use for private data |

Valid activation protection values are `automatic`, `seeded-preparation`,
`precomputed-masks`, `guarded-blinded-masks`, `blinded-masks`,
`encrypted-activations`, and `authenticated-shares`. Public weights select
`seeded-preparation`; `precomputed-masks` remains a confidential-weight mode.
Valid spelling does not imply that every graph and adversary combination is
implemented.

## Configure

```text
pllm configure --server URL --api-key KEY --model MODEL_ID
```

Optional fields include `--preparation-url`, `--preparation-api-key`,
`--transport`, `--correlation-mode`,
`--correlation-prefetch`, `--token-cache-size`, `--bundle-cache-mode`,
`--bundle-cache-dir`, and `--timeout`.

`pllm sidecar` accepts the same two bundle-cache options.

## Preparation service

```text
pllm preparation serve MODEL_PATH_OR_HUB_ID --api-key KEY \
  --inference-url URL --push-api-key KEY --port 8001
```

This role is public-weight-only and exposes health, model commitment, and seeded
preparation routes. It does not expose inference, Responses, client-bundle, or
model-administration routes. Both services need the same public checkpoint.
Provider `pllm serve` accepts `--provider-push-api-key`,
`--rendezvous-timeout`, `--rendezvous-capacity`, and
`--rendezvous-max-bytes`. All three service credentials must differ.

## Chat and local gateway

```text
pllm chat [--server URL] [--model MODEL_ID] [--preparation-url URL]
pllm sidecar --local-api-key KEY --host 127.0.0.1 --port 8080
```

The sidecar reads saved provider settings. In the reference wrapper, `auto` is translated to WebSocket at startup; use `--transport http` explicitly when the deployment cannot support WebSocket upgrades.

Download [the captured CLI help](/downloads/cli-help.txt) for the complete parser output. No `pllm gateway` or `pllm build-native` alias is invented by these guides; the supplied commands are `sidecar` and `build`.
