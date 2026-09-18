# Codex

Configure Codex CLI to use PLLM as a custom Responses API provider.

[View canonical HTML](https://pllm.run/learn/integrations/codex/)

Document ID: `pllm.docs.learn.integrations.codex`  
Release: `0.1.0`  
Build: `sha256:96b6d9446e37113d9d2892113cefbcf72b64f5f3fc8eeb331b7caddd36ad60a0`  
Source hash: `sha256:d7daaed526a26cefacfbcfd5219a1d21a5e6ccadaa4aa4727ffb420da5b1c49f`

Add this provider to user-level `~/.codex/config.toml`:

This configuration is verified with Codex CLI 0.154.0.

```toml
model = "Qwen/Qwen2.5-0.5B-Instruct"
model_provider = "pllm"
web_search = "disabled"

[model_providers.pllm]
name = "PLLM"
base_url = "http://127.0.0.1:8080/v1"
env_key = "PLLM_GATEWAY_API_KEY"
wire_api = "responses"
```

Set `PLLM_GATEWAY_API_KEY` to `local` in the Codex process environment. Provider
configuration is machine-local, not a project-scoped override. Start the
[local gateway](/learn/integrations/local-gateway/) before opening Codex.

The gateway accepts text and function-tool agent traffic. Codex must use
`wire_api = "responses"`. Keep `web_search = "disabled"`: provider-built-in web
search is unsupported. This configuration does not claim support for image
input, hosted tools, or every Codex capability.
