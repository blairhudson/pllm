# Python SDK

Keep HE details out of application code.


## Generate text

```python
from pllm import OpenAI

with OpenAI() as client:
    client.preprocess(count=256)
    response = client.responses.create(
        input="Explain the preparation phase.",
        max_output_tokens=128,
    )
    print(response.output_text)
```

`OpenAI()` reads the saved PLLM configuration. An explicit `model` on a request overrides the selected model. For public weights, call `preprocess()` before the first chat; inference fails closed rather than doing preparation on the online path. The SDK reserves one-time rows for each execution and refills a spare inventory only while idle. Keep one client open to reuse unreserved rows across chats.

## Stream text

```python
from pllm import OpenAI

with OpenAI() as client:
    client.preprocess(count=256)
    with client.responses.create(
        input="Describe the client and provider boundary.",
        max_output_tokens=128,
        stream=True,
    ) as stream:
        for event in stream:
            if event.type == "response.output_text.delta":
                print(event.delta, end="", flush=True)
    print()
```

The direct SDK returns PLLM response types. The [local gateway](/docs/client/openai) is the path to response objects parsed by the official SDK.

## Continue a conversation

```python
from pllm import OpenAI

with OpenAI() as client:
    client.preprocess(count=256)
    first = client.responses.create(input="What does a mask hide?")
    second = client.responses.create(
        input="And what happens when it is reused?",
        previous_response_id=first.id,
    )
    print(second.output_text)
```

Conversation state belongs to this client process. Reusing the ID on another client or after losing local state is not a server history restore operation.

## Explicit connection settings

```python
import os
from pllm import OpenAI

with OpenAI(
    base_url=os.environ["PLLM_BASE_URL"],
    api_key=os.environ["PLLM_API_KEY"],
    preparation_base_url=os.environ["PLLM_PREPARATION_BASE_URL"],
    preparation_api_key=os.environ["PLLM_PREPARATION_API_KEY"],
    model="private-model",
    bundle_cache_mode="read-write",
    timeout=600,
) as client:
    client.preprocess(count=256)
    print(client.responses.create(input="Hello").output_text)
```

The constructor also accepts `preparation_base_url`, `preparation_api_key`,
`he_transport`, `correlation_mode`,
`correlation_prefetch`, `token_cache_size`, `bundle_cache_mode`,
`bundle_cache_dir`, `tenseal_path`, and `http_client`.
Remote preparation is available only for public weights. The preparation
service must be trusted not to retain masks or collude with the inference
provider. Most applications should leave these values in saved configuration.

## Async applications

```python
import asyncio
from pllm import AsyncOpenAI

async def main() -> None:
    async with AsyncOpenAI() as client:
        await client.preprocess(count=256)
        response = await client.responses.create(input="Hello")
        print(response.output_text)

asyncio.run(main())
```

The reference async wrapper delegates work to threads. It is not a separate native async inference engine. Use one client and private state per independent conversation when measuring concurrency.
