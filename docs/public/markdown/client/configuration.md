# Configure once

The SDK and chat CLI share the same local settings.


```bash
umask 077
pllm configure \
  --server http://127.0.0.1:8000 \
  --api-key "$PLLM_API_KEY" \
  --model private-model
```

The path is `~/.config/pllm/config.toml`, or `$XDG_CONFIG_HOME/pllm/config.toml`. `PLLM_CONFIG` overrides the complete file path.

## Configuration order

Explicit SDK arguments override environment variables. Environment variables override the saved file. Unspecified fields use the defaults below.

| Setting | Environment | Default |
| --- | --- | --- |
| Provider URL | `PLLM_BASE_URL` | `http://127.0.0.1:8000` |
| Provider credential | `PLLM_API_KEY` | `pllm-local` |
| Model ID | `PLLM_MODEL` | Discover a single private model |
| Preparation URL | `PLLM_PREPARATION_BASE_URL` | None |
| Preparation credential | `PLLM_PREPARATION_API_KEY` | Provider credential |
| Prepared inventory rows per stage | `PLLM_PREPARED_INVENTORY_ROWS` | `64` |
| Transport | `PLLM_TRANSPORT` | `auto` |
| Preparation | `PLLM_CORRELATION_MODE` | `bfv` |
| Preparation horizon | `PLLM_CORRELATION_PREFETCH` | `4` |
| Token cache entries | `PLLM_TOKEN_CACHE_SIZE` | `512` |
| Bundle cache mode | `PLLM_BUNDLE_CACHE_MODE` | `read-write` |
| Bundle cache directory | `PLLM_BUNDLE_CACHE_DIR` | `$XDG_CACHE_HOME/pllm/client-bundles` or `~/.cache/pllm/client-bundles` |
| Timeout, seconds | `PLLM_TIMEOUT` | `300` |

Set an explicit credential. The default value is for local development, not an access policy.

## Configure public-weight preparation

Public-weight inference requires a trusted preparation service in addition to
the untrusted inference provider:

```bash
pllm configure \
  --server https://inference.example \
  --api-key "$PLLM_INFERENCE_KEY" \
  --preparation-url https://preparation.example \
  --preparation-api-key "$PLLM_PREPARATION_KEY"
```

Self-host the preparation service when no external preparation operator is
trusted. It needs the same public model weights, but receives only batched stage
root seeds and shape metadata before chat. It must follow the protocol, erase
expanded masks, and not collude with the inference provider. Remote URLs require HTTPS and distinct
origins; plain HTTP is accepted only on loopback for development. The client
verifies matching model and per-stage weight commitments before generation.

`PLLM_PREPARED_INVENTORY_ROWS` controls each offline stage batch. The client keeps
unreserved rows in memory for later chats and refills only while idle. Larger
values reduce refill frequency but increase preparation time, memory, and rows
burned when a reservation ends early. Restart or idle expiry discards inventory.

## Protect the settings file

The reference writer applies mode `0600` on systems that support it. The API key is stored as text, not encrypted. Set a restrictive umask before first configuration and verify the result:

```bash
umask 077
pllm configure --server http://127.0.0.1:8000 --api-key "$PLLM_API_KEY"
```

For managed environments, inject credentials at process startup and use a protected configuration directory. Do not share this directory between unrelated tenants.

## Cache public client bundles

Client bundles contain public boundary weights and tokenizer assets. PLLM verifies
the provider-advertised schema, model ID, and complete payload SHA-256 before use.
Each generation fetches the current descriptor and rebuilds in-memory model state
when its bundle fingerprint changes. Cache identity includes normalized inference
URL, model, and provider credential, keyed by a random cache-local secret so the
filename alone is not an API-key verifier. One stable record per identity is
atomically replaced on revision changes. Bundles larger than 8 GiB are rejected,
directories are created with mode `0700`, and payload files use mode `0600`.

`read-write` reads and updates the cache. `read-only` reads valid entries but does
not repair or write them. `refresh` downloads and atomically replaces an entry.
`off` bypasses disk caching. Configure with:

```bash
pllm configure \
  --bundle-cache-mode read-write \
  --bundle-cache-dir ~/.cache/pllm/client-bundles
```

If the implicit default cache directory is unavailable, PLLM skips disk and uses
the same fully verified network bundle. An explicitly configured `read-write` or
`refresh` directory fails clearly instead, so deployment mistakes are visible.

`/audit` reports `bundle_network_bytes`, `bundle_cache_hits`,
`bundle_cache_misses`, and `bundle_cache_corruptions` separately.

## Tune confidential-weight preparation cautiously

```bash
pllm configure --correlation-prefetch 4 --token-cache-size 512 --timeout 600
```

More prepared material can smooth generation but also increases memory, startup work, and unused material after cancellation. It does not reduce the amount of preparation consumed by a useful token.
