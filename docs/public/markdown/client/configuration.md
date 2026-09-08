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
| Execution strategy | `PLLM_EXECUTION_STRATEGY` | `bfv` |
| Second provider URL | `PLLM_SECONDARY_BASE_URL` | None |
| Second provider credential | `PLLM_SECONDARY_API_KEY` | Primary credential |
| Transport | `PLLM_TRANSPORT` | `auto` |
| Preparation | `PLLM_CORRELATION_MODE` | `bfv` |
| Preparation horizon | `PLLM_CORRELATION_PREFETCH` | `4` |
| Token cache entries | `PLLM_TOKEN_CACHE_SIZE` | `512` |
| Timeout, seconds | `PLLM_TIMEOUT` | `300` |

Set an explicit credential. The default value is for local development, not an access policy.

## Use two independent providers

For public weights, direct sharing avoids online HE preparation:

```bash
pllm configure \
  --execution-strategy two-provider \
  --server https://provider-a.example \
  --api-key "$PLLM_PROVIDER_A_KEY" \
  --secondary-base-url https://provider-b.example \
  --secondary-api-key "$PLLM_PROVIDER_B_KEY"
```

The providers must have independent administrative control and must not
collude. Distinct URLs, processes, ports, containers, or accounts owned by one
operator do not establish that assumption. Remote provider URLs require HTTPS;
plain HTTP is accepted only on loopback for development. The client verifies
that both providers expose identical execution plans and weight commitments.

## Protect the settings file

The reference writer applies mode `0600` on systems that support it. The API key is stored as text, not encrypted. Set a restrictive umask before first configuration and verify the result:

```bash
umask 077
pllm configure --server http://127.0.0.1:8000 --api-key "$PLLM_API_KEY"
```

For managed environments, inject credentials at process startup and use a protected configuration directory. Do not share this directory between unrelated tenants.

## Tune preparation cautiously

```bash
pllm configure --correlation-prefetch 4 --token-cache-size 512 --timeout 600
```

More prepared material can smooth generation but also increases memory, startup work, and unused material after cancellation. It does not reduce the amount of preparation consumed by a useful token.
