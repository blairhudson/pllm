# OpenAI Agents SDK

Run an OpenAI Agents SDK text agent through PLLM's Responses API gateway.

[View canonical HTML](https://pllm.run/learn/integrations/openai-agents/)

Document ID: `pllm.docs.learn.integrations.openai-agents`  
Release: `0.1.0`  
Build: `sha256:96b6d9446e37113d9d2892113cefbcf72b64f5f3fc8eeb331b7caddd36ad60a0`  
Source hash: `sha256:d9ffadd9be6ca996d8195ab59a362ec8f32b0d4121a5d0b31da39e93bf6f8919`

Use the Agents SDK Responses API adapter with an `AsyncOpenAI` client that targets
the local gateway:

The executable compatibility suite uses `openai-agents` 0.22.2.

```python
import asyncio

from agents import Agent, RunConfig, Runner
from agents.models.openai_responses import OpenAIResponsesModel
from openai import AsyncOpenAI


async def main() -> None:
    async with AsyncOpenAI(
        base_url="http://127.0.0.1:8080/v1",
        api_key="local",
        max_retries=0,
    ) as client:
        model = OpenAIResponsesModel(
            model="Qwen/Qwen2.5-0.5B-Instruct",
            openai_client=client,
        )
        agent = Agent(name="PLLM agent", instructions="Reply briefly.", model=model)
        result = await Runner.run(
            agent,
            "Describe the trusted client boundary.",
            run_config=RunConfig(tracing_disabled=True),
        )
    print(result.final_output)


asyncio.run(main())
```

This path passes a repository integration test against a real local Responses API
gateway. The gateway accepts text and local function-tool traffic. Do not enable
hosted tools or assume support for every Agents SDK model feature. Tracing is
disabled because this configuration does not use an OpenAI-hosted tracing
endpoint.
