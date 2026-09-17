# OpenAI Python SDK

Point the official OpenAI Python SDK at the trusted PLLM gateway.

[View canonical HTML](https://pllm.run/learn/integrations/openai-python/)

Document ID: `pllm.docs.learn.integrations.openai-python`  
Release: `0.1.0`  
Build: `sha256:55dedf191ed9de69d95eb69a68160a404194419bf2f721a57103e95fdce4a4e0`  
Source hash: `sha256:97dc3e67264a940b248ea2a3005ea77da1dbff2ed9fb9dad3a2aa96adfca8ef9`

Start the [local gateway](/learn/integrations/local-gateway/), then give the
official `openai` client the gateway base URL and local credential:

The executable compatibility suite uses `openai` 3.8.0.

```python
from openai import OpenAI

with OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="local",
    max_retries=0,
) as client:
    response = client.responses.create(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        input="Explain the client trust boundary in one sentence.",
        max_output_tokens=64,
    )

print(response.output_text)
```

The installed OpenAI SDK passes repository tests for synchronous and asynchronous
Responses API, Responses API streaming, Chat Completions API, and Chat Completions API streaming.
Those tests establish the exercised client paths, not complete OpenAI API or
model-modality compatibility.

Use only a loopback gateway URL. The gateway receives plaintext and belongs
inside the trusted client boundary; preparation and inference roles do not
receive the plaintext prompt.
