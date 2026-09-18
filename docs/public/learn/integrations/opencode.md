# OpenCode

Configure OpenCode to use PLLM through its Chat Completions API gateway.

[View canonical HTML](https://pllm.run/learn/integrations/opencode/)

Document ID: `pllm.docs.learn.integrations.opencode`  
Release: `0.1.0`  
Build: `sha256:a3db51eca723684328314c5428e7e991d7b5bb20ff973c6d46dfe9a79aec3a08`  
Source hash: `sha256:fb55e1f8607bb63db0b9973fd416541d07445c1a83613672187f815d4b74985f`

Add `opencode.json` to the project:

This configuration is verified with OpenCode 1.18.31.

```json
{
  "$schema": "https://opencode.ai/config.json",
  "enabled_providers": ["pllm"],
  "model": "pllm/Qwen/Qwen2.5-0.5B-Instruct",
  "provider": {
    "pllm": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "PLLM",
      "options": {
        "baseURL": "http://127.0.0.1:8080/v1",
        "apiKey": "local"
      },
      "models": {
        "Qwen/Qwen2.5-0.5B-Instruct": {
          "name": "Qwen 2.5 0.5B Instruct via PLLM"
        }
      }
    }
  }
}
```

`@ai-sdk/openai-compatible` uses the gateway's Chat Completions API route.
`enabled_providers` prevents fallback to another configured provider. Start the
[local gateway](/learn/integrations/local-gateway/) before opening OpenCode.

OpenCode custom-tool declarations map to local function tools through the
gateway adapter. Text and function-tool traffic is supported. This integration
has not established acceptance of every external tool binary or every OpenCode
agent feature; test the exact tools and model before relying on them.
