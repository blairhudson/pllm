# Local gateway

Start the trusted loopback gateway locally or connect it to separate PLLM services.

[View canonical HTML](https://pllm.run/learn/integrations/local-gateway/)

Document ID: `pllm.docs.learn.integrations.local-gateway`  
Release: `0.1.0`  
Build: `sha256:4a93c61285a110010f1bafefa367e198ed52465071615e2d4d9a91a45f2d82e2`  
Source hash: `sha256:e1ad2b7ef4325d09f45085a5da35d4145b45f48b5a631c5ebd14b45218339e07`

The gateway is client-controlled and trusted. It binds to `127.0.0.1:8080` by
default and uses the local API key `local` unless configured otherwise.

## Run all roles locally

```bash
pllm gateway --local --model Qwen/Qwen2.5-0.5B-Instruct
```

This downloads or reuses the model, starts preparation and inference children on
loopback, and keeps the gateway in the foreground. Stop the gateway to stop its
children. This mode validates transport and development behavior only. Because
both provider roles run under one operator, it does not establish non-collusion.

## Connect separate roles

Create `client.toml` for the trusted gateway:

```toml
[client]
base_url = "https://inference.example"
api_key = "inference-secret"
model = "Qwen/Qwen2.5-0.5B-Instruct"
transport = "websocket"
preparation_base_url = "https://preparation.example"
preparation_api_key = "preparation-secret"
```

Then start the loopback gateway:

```bash
pllm gateway --config client.toml
```

Run provider roles in their respective environments:

```bash
pllm serve inference --config inference.json
pllm serve preparation --config preparation.json
```

A minimal `inference.json` is:

```json
{
  "api_keys": ["inference-secret"],
  "provider_push_api_key": "push-secret",
  "engine_models": [
    {
      "engine": "masked-transformer-w4a4",
      "kind": "huggingface",
      "path": "/srv/models/Qwen2.5-0.5B-Instruct",
      "model_id": "Qwen/Qwen2.5-0.5B-Instruct"
    }
  ]
}
```

A minimal `preparation.json` is:

```json
{
  "api_keys": ["preparation-secret"],
  "engine_models": [
    {
      "engine": "masked-transformer-w4a4",
      "kind": "huggingface",
      "path": "/srv/models/Qwen2.5-0.5B-Instruct",
      "model_id": "Qwen/Qwen2.5-0.5B-Instruct"
    }
  ]
}
```

Set `PLLM_INFERENCE_URL` and `PLLM_PUSH_API_KEY` for the preparation process.
Their values must select the inference service and match its
`provider_push_api_key`. Replace example URLs, credentials, and model paths with
deployment values. Keep gateway credentials separate from both provider-role
credentials. See [deployment](/sdk/operate/deployment/status/) for remaining
production limits.
