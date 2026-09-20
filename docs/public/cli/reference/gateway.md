# pllm gateway

Exact pllm gateway help from the installed PLLM CLI.

[View canonical HTML](https://pllm.run/cli/reference/gateway/)

Document ID: `pllm.docs.reference.cli.gateway`  
Release: `0.1.0`

## Example

```bash
pllm gateway --config client.toml
```

## Options

```text
usage: pllm gateway [-h] [--format {human,json,jsonl}] [--quiet] [--no-color]
                    [--no-input] [--dry-run] [--config CONFIG] [--host HOST]
                    [--port PORT] [--api-key API_KEY]
                    [--inference-url INFERENCE_URL]
                    [--inference-key INFERENCE_KEY]
                    [--preparation-url PREPARATION_URL]
                    [--preparation-key PREPARATION_KEY] [--model MODEL]
                    [--model-id MODEL_ID] [--local]
                    [--transport {auto,http,websocket}]
                    [--correlation-mode {bfv,local-test}]
                    [--correlation-prefetch CORRELATION_PREFETCH]
                    [--prepared-inventory-rows PREPARED_INVENTORY_ROWS]
                    [--token-cache-size TOKEN_CACHE_SIZE]
                    [--bundle-cache-mode {read-write,read-only,refresh,off}]
                    [--bundle-cache-dir BUNDLE_CACHE_DIR]
                    [--tenseal-path TENSEAL_PATH] [--timeout TIMEOUT]
                    [--revision REVISION] [--hf-cache-dir HF_CACHE_DIR]
                    [--local-files-only] [--weight-bits {4,8}]
                    [--activation-bits {4,8}]

options:
  -h, --help            show this help message and exit
  --format {human,json,jsonl}
                        result format (default: human)
  --quiet               suppress non-error diagnostics
  --no-color            disable colored output
  --no-input            fail instead of prompting
  --dry-run             report work without persistent writes or service
                        startup
  --config CONFIG       client TOML file
  --host HOST
  --port PORT
  --api-key API_KEY
  --inference-url INFERENCE_URL
  --inference-key INFERENCE_KEY
  --preparation-url PREPARATION_URL
  --preparation-key PREPARATION_KEY
  --model MODEL
  --model-id MODEL_ID
  --local               co-locate both server roles locally
  --transport {auto,http,websocket}
                        session transport
  --correlation-mode {bfv,local-test}
  --correlation-prefetch CORRELATION_PREFETCH
  --prepared-inventory-rows PREPARED_INVENTORY_ROWS
  --token-cache-size TOKEN_CACHE_SIZE
  --bundle-cache-mode {read-write,read-only,refresh,off}
  --bundle-cache-dir BUNDLE_CACHE_DIR
  --tenseal-path TENSEAL_PATH
  --timeout TIMEOUT
  --revision REVISION
  --hf-cache-dir HF_CACHE_DIR
  --local-files-only
  --weight-bits {4,8}
  --activation-bits {4,8}
```
