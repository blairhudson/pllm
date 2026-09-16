"""Execute installed SDKs over a real local Responses gateway.

The deterministic local fixture measures API compatibility only. Its test masks
are deliberately not a private inference benchmark.
"""

from __future__ import annotations
import importlib.util
import socket
import threading
import time
import pytest
import uvicorn
from pllm.runtime.sidecar import create_sidecar_app

pytestmark = pytest.mark.sdk


@pytest.fixture
def sdk_endpoint(gateway):
    if importlib.util.find_spec("openai") is None:
        pytest.skip("Install the sdk extra for installed OpenAI SDK tests")
    app = create_sidecar_app(
        remote_base_url=gateway.base_url,
        remote_api_key=gateway.api_key,
        local_api_key="sdk-local-key",
        correlation_mode="local-test",
        correlation_prefetch=16,
    )
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error", access_log=False)
    )
    thread = threading.Thread(target=lambda: server.run(sockets=[sock]), daemon=True)
    thread.start()
    try:
        for _ in range(500):
            if server.started:
                break
            if not thread.is_alive():
                raise RuntimeError("Local SDK gateway failed to start")
            time.sleep(0.01)
        assert server.started
        yield f"http://127.0.0.1:{port}/v1"
    finally:
        server.should_exit = True
        thread.join(10)
        sock.close()
        assert not thread.is_alive()


def test_installed_openai_responses(sdk_endpoint):
    from openai import OpenAI

    with OpenAI(base_url=sdk_endpoint, api_key="sdk-local-key", max_retries=0) as client:
        result = client.responses.create(
            model="pllm-bigram-demo", input="synthetic API fixture", max_output_tokens=32
        )
        assert result.output_text == "private\n"
        assert result.status == "completed"


def test_installed_openai_stream(sdk_endpoint):
    from openai import OpenAI

    with OpenAI(base_url=sdk_endpoint, api_key="sdk-local-key", max_retries=0) as client:
        events = list(
            client.responses.create(
                model="pllm-bigram-demo",
                input="synthetic API fixture",
                max_output_tokens=32,
                stream=True,
            )
        )
        assert events[-1].type == "response.completed"
        assert (
            "".join(event.delta for event in events if event.type == "response.output_text.delta")
            == "private\n"
        )


def test_installed_openai_pllm_transport(gateway):
    from pllm import create_openai_client

    with create_openai_client(
        gateway_url=gateway.base_url,
        gateway_api_key=gateway.api_key,
        correlation_mode="local-test",
    ) as client:
        result = client.responses.create(
            model="pllm-bigram-demo",
            input="synthetic API fixture",
            max_output_tokens=32,
        )
        assert result.output_text == "private\n"


@pytest.mark.asyncio
async def test_installed_async_openai(sdk_endpoint):
    from openai import AsyncOpenAI

    async with AsyncOpenAI(base_url=sdk_endpoint, api_key="sdk-local-key", max_retries=0) as client:
        result = await client.responses.create(
            model="pllm-bigram-demo", input="synthetic API fixture", max_output_tokens=32
        )
        assert result.output_text == "private\n"


def test_installed_openai_chat_completions(sdk_endpoint):
    from openai import OpenAI

    with OpenAI(base_url=sdk_endpoint, api_key="sdk-local-key", max_retries=0) as client:
        result = client.chat.completions.create(
            model="pllm-bigram-demo",
            messages=[{"role": "user", "content": "synthetic API fixture"}],
            max_completion_tokens=32,
        )
        assert result.choices[0].message.content == "private\n"
        assert result.choices[0].finish_reason == "stop"


def test_installed_openai_chat_stream(sdk_endpoint):
    from openai import OpenAI

    with OpenAI(base_url=sdk_endpoint, api_key="sdk-local-key", max_retries=0) as client:
        chunks = list(
            client.chat.completions.create(
                model="pllm-bigram-demo",
                messages=[{"role": "user", "content": "synthetic API fixture"}],
                max_completion_tokens=32,
                stream=True,
            )
        )
    assert (
        "".join(chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices)
        == "private\n"
    )
    assert chunks[-1].choices[0].finish_reason == "stop"


@pytest.mark.asyncio
async def test_openai_agents_sdk_uses_gateway_responses(sdk_endpoint):
    agents = pytest.importorskip("agents")
    from agents.models.openai_responses import OpenAIResponsesModel
    from openai import AsyncOpenAI

    async with AsyncOpenAI(base_url=sdk_endpoint, api_key="sdk-local-key", max_retries=0) as client:
        model = OpenAIResponsesModel(model="pllm-bigram-demo", openai_client=client)
        agent = agents.Agent(name="PLLM compatibility", instructions="Reply briefly.", model=model)
        result = await agents.Runner.run(
            agent,
            "synthetic API fixture",
            run_config=agents.RunConfig(tracing_disabled=True),
        )
    assert isinstance(result.final_output, str)
    assert result.final_output
