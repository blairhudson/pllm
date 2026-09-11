# Configure the client

Separate inference, preparation, inventory, cache, and timeout settings.


## Save public-path settings

```bash
umask 077
pllm configure \
  --server http://127.0.0.1:8000 \
  --api-key "$PLLM_API_KEY" \
  --preparation-url http://127.0.0.1:8001 \
  --preparation-api-key "$PLLM_PREPARATION_API_KEY" \
  --model demo-model
```

Settings are written to `~/.config/pllm/config.toml`, or
`$XDG_CONFIG_HOME/pllm/config.toml`. `PLLM_CONFIG` overrides the complete path.
The writer requests mode `0600` where the platform supports it, but values remain
plaintext on disk.

Explicit SDK or CLI arguments override environment variables. Environment
variables override the saved file; unspecified fields use these source defaults:

| Client setting | Environment | Default |
| --- | --- | --- |
| Inference URL | `PLLM_BASE_URL` | `http://127.0.0.1:8000` |
| Inference credential | `PLLM_API_KEY` | `pllm-local` |
| Model ID | `PLLM_MODEL` | Discover only when one model is available |
| Preparation URL | `PLLM_PREPARATION_BASE_URL` | None |
| Preparation credential | `PLLM_PREPARATION_API_KEY` | None; required when URL is set |
| Minimum prepared rows per stage | `PLLM_PREPARED_INVENTORY_ROWS` | `64` |
| Online transport | `PLLM_TRANSPORT` | `auto` |
| Bundle cache mode | `PLLM_BUNDLE_CACHE_MODE` | `read-write` |
| Bundle cache directory | `PLLM_BUNDLE_CACHE_DIR` | `$XDG_CACHE_HOME/pllm/client-bundles` or `~/.cache/pllm/client-bundles` |
| Direct client/chat HTTP timeout | `PLLM_TIMEOUT` | `300` seconds |
| Legacy correlation mode | `PLLM_CORRELATION_MODE` | `bfv` |
| Legacy correlation prefetch | `PLLM_CORRELATION_PREFETCH` | `4` |
| Legacy remote token cache | `PLLM_TOKEN_CACHE_SIZE` | `512` |

Always set explicit credentials outside isolated local development. Inference and
preparation credentials must differ. Remote origins must be distinct HTTPS
origins; plain HTTP is accepted only for loopback.

## Size prepared inventory

`PLLM_PREPARED_INVENTORY_ROWS` is the minimum capacity used when creating each
stage inventory. A particular response may require more: prompt token rows plus
up to `max_output_tokens - 1` decode rows. `pllm chat` calculates this before each
response. Direct clients can call `prepared_rows_for_response()` and pass its
result to `preprocess()`.

Unreserved rows remain available for later chats. Starting a response reserves
its exact requirement; early completion or failure burns only unused rows from
that reservation. Larger inventory settings increase preparation work, client and
inference memory, and rows discarded on restart or idle expiry. They do not turn
unused reserved rows back into inventory.

Default clients may prepare a spare inventory while idle. No refill starts while
an online response is active.

## Understand timeouts

`PLLM_TIMEOUT` configures HTTP operations made by the direct client and chat CLI,
including bundle transfer and offline preparation requests. Raise it when a real
checkpoint or inventory batch legitimately takes longer, but keep an upper bound.
It does not configure:

- preparation's `--push-timeout` for a correction send and acknowledgement;
- inference's `--rendezvous-timeout` for unmatched non-preloaded entries;
- inference's `--prepared-session-idle` lifetime for memory-only inventory;
- reverse-proxy HTTP or WebSocket idle limits.

Tune those at their owning process. A longer client timeout cannot restore an
inventory that expired at inference.

Persistent decode currently uses a fixed 30-second WebSocket connect timeout and
does not apply `PLLM_TIMEOUT` to each socket receive. Set proxy idle limits for
the longest legitimate decode gap. Selecting `--transport http` applies the HTTP
timeout to stage calls, but gives up persistent decode transport.

## Cache public client bundles

Client bundles contain tokenizer assets, graph metadata, and public quantized
token lookup/output-head matrices. PLLM verifies the advertised schema, model ID,
and complete payload SHA-256. Each generation checks the descriptor and rebuilds
in-memory state when its fingerprint changes.

Cache identity includes normalized inference URL, model, and provider credential,
keyed by a random cache-local secret so a filename is not an API-key verifier.
Payloads larger than 8 GiB are rejected. Directories request mode `0700`; payload
files request `0600`.

| Mode | Behavior |
| --- | --- |
| `read-write` | Read valid entries and atomically update changed bundles |
| `read-only` | Read valid entries without writing or repair |
| `refresh` | Download and atomically replace the selected entry |
| `off` | Bypass disk cache |

```bash
pllm configure \
  --bundle-cache-mode read-write \
  --bundle-cache-dir ~/.cache/pllm/client-bundles
```

If the implicit default cache is unavailable, PLLM uses the verified network
bundle without disk caching. An explicitly configured writable cache fails
clearly instead. `/audit` separates network bytes, hits, misses, and corruptions.

## Legacy settings

`PLLM_CORRELATION_MODE`, `PLLM_CORRELATION_PREFETCH`, and
`PLLM_TOKEN_CACHE_SIZE` apply to older confidential-weight/BFV execution paths.
Public seeded inventory ignores BFV correlation selection and performs public
token lookup locally. Do not tune legacy prefetch as if it controlled public
inventory rows.
