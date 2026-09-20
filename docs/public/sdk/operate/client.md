# Python client

Construct the native PLLM client, inspect its resources, and choose it instead of the gateway only when your process owns the private client boundary.

[View canonical HTML](https://pllm.run/sdk/operate/client/)

Document ID: `pllm.docs.operate.client`  
Release: `0.1.0`

`pllm.OpenAI` is PLLM's native synchronous client. It owns the private model runtime,
preparation inventory, credentials, and request state in the calling process. Use it
when your Python process is the trusted client boundary and can connect to PLLM's
inference and preparation roles directly.

Applications that should not own those details should instead use an ordinary
OpenAI-compatible SDK against [`pllm gateway`](/learn/integrations/local-gateway/).
The gateway remains inside the trusted client boundary; inference and preparation
URLs are not OpenAI-compatible application endpoints.

## Python SDK example

```python
from pllm import OpenAI

with OpenAI(
    base_url="http://127.0.0.1:9800",
    preparation_base_url="http://127.0.0.1:9801",
    api_key="inference-example",
    preparation_api_key="preparation-example",
    default_model="example/model",
) as client:
    resources = (
        callable(client.responses.create),
        callable(client.preprocess),
        callable(client.prepared_inventory_status),
    )
assert resources == (True, True, True)
```

API: [`pllm.OpenAI`](/sdk/reference/python/pllm/#objects-and-signatures)

Constructing the client performs no request. `responses.create(...)` resolves or
loads the selected model, ensures usable offline inventory exists, and then starts
the online client-to-inference path. Call `preprocess(...)` only to warm inventory
in advance; active responses never fall back to online preparation.

The async equivalent is `pllm.AsyncOpenAI`; use one client per trust boundary and
close it to erase live runtime state and release transports.
