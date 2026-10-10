"""Ordinary SDK calls own fresh negotiation and retire real HTTP role leases."""

import asyncio
from pathlib import Path
import subprocess
import sys
import threading

import pytest

from pllm import AsyncOpenAI, OpenAI
from pllm.search import PlanningPolicy
from pllm.settings import ClientSettings
from test_network_benchmark import scenario_fixture, _fixture_module

pytestmark = pytest.mark.rust


@pytest.fixture
def configured(tmp_path, monkeypatch):
    with scenario_fixture(tmp_path, hidden_size=8) as fixture:
        network = tmp_path / "network.json"
        experiment = tmp_path / "experiment.json"
        config = tmp_path / "config.toml"
        network.write_bytes(fixture["network"].canonical_bytes())
        experiment.write_bytes(fixture["request"].candidates[0].canonical_bytes())
        ClientSettings(network=str(network), experiment=str(experiment)).save(config)
        fixture["paths"]["experiment"] = experiment
        monkeypatch.setenv("PLLM_CONFIG", str(config))
        for name, secret in fixture["credentials"].items():
            monkeypatch.setenv(name, secret)
        yield fixture


def released(fixture):
    assert all(not app.state.controller._reservations for app in fixture["children"])


def attempts(fixture):
    return sum(len(app.state.controller._used_attempts) for app in fixture["children"])


def test_default_sync_requests_replan_explain_and_release(configured):
    with OpenAI() as client:
        assert attempts(configured) == 0
        assert client.models.list()["data"][0]["id"] == "network-scenarios-tiny"
        assert attempts(configured) == 0
        results = []
        for _ in range(2):
            response = client.responses.create(input="Hi", max_output_tokens=2, temperature=0)
            assert response.usage.output_tokens == 2
            assert client.responses.retrieve(response.id).output_text == response.output_text
            result = client.planning(response.id)
            assert result.status == "feasible"
            assert result.experiment.pipeline.digest() == configured["request"].candidates[0].pipeline.digest()
            assert all(secret.encode() not in result.canonical_bytes()
                       for secret in configured["credentials"].values())
            results.append(result)
            released(configured)
        assert results[0].snapshot.digest != results[1].snapshot.digest
        assert attempts(configured) == 4
        audit = client.privacy_audit.to_dict()
        assert audit["inference_stage_calls"] > 0
        assert audit["plaintext_prompt_bytes_sent"] == audit["plaintext_token_ids_sent"] == 0
    assert not client._core.records and not client._core.active


def test_close_before_iteration_and_partial_stream_retire(configured):
    with OpenAI() as client:
        stream = client.responses.create(input="Hi", max_output_tokens=2, stream=True)
        stream.close()
        assert attempts(configured) == 0
        stream = client.responses.create(input="Hi", max_output_tokens=2, stream=True)
        try:
            for event in stream:
                if event.type == "response.output_text.delta":
                    break
            else:
                pytest.fail("missing partial response")
        finally:
            stream.close()
        released(configured)
        assert attempts(configured) == 2
        assert not client._core.active
    with pytest.raises(ValueError, match="closed"):
        client.responses.create(input="Hi", max_output_tokens=2)


def test_terminal_event_retires_before_consumer_advances(configured):
    with OpenAI() as client:
        stream = client.responses.create(input="Hi", max_output_tokens=2, stream=True)
        try:
            for event in stream:
                if event.type in {"response.completed", "response.incomplete"}:
                    break
            else:
                pytest.fail("missing terminal event")
            released(configured)
            assert not client._core.active
            assert client.privacy_audit.inference_stage_calls > 0
        finally:
            stream.close()


def test_unpriced_resource_requirement_rejects_before_capacity(configured):
    with OpenAI(planning=PlanningPolicy(max_client_peak_memory_bytes=1)) as client:
        with pytest.raises(ValueError, match="NO_FEASIBLE_PLACEMENT") as caught:
            client.responses.create(input="Hi", max_output_tokens=2)
        assert caught.value.planning_result.status == "infeasible"
    assert attempts(configured) == 0
    released(configured)


@pytest.mark.parametrize("mode", ("experiment", "configured"))
def test_real_cli_automatic_gateway(configured, mode):
    result = _fixture_module.gateway_cli_probe(configured, mode)
    assert result["response_output_tokens"] == result["chat_stream_output_tokens"] == 2
    assert result["capacity_zero"] and result["stream_terminal_received"]


def test_published_automatic_example(configured):
    example = Path(__file__).resolve().parents[1] / "examples" / "automatic_client.py"
    result = subprocess.run([sys.executable, str(example)], capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stderr
    assert "Selected plan:" in result.stdout
    released(configured)


def test_configured_gateway_factory_and_conflicting_cli_override(configured):
    from argparse import Namespace
    from fastapi.testclient import TestClient
    from pllm.runtime.cli import RuntimeCLIError, run_sidecar
    from pllm.runtime.sidecar import create_sidecar_app

    with TestClient(create_sidecar_app()) as gateway:
        response = gateway.post("/v1/responses", headers={"authorization": "Bearer local"},
            json={"model": "network-scenarios-tiny", "input": "Hi", "max_output_tokens": 2, "temperature": 0})
        assert response.status_code == 200, response.text
        assert response.json()["usage"]["output_tokens"] == 2
    released(configured)
    with pytest.raises(RuntimeCLIError, match="conflicts"):
        run_sidecar(Namespace(inference_url="http://127.0.0.1:1"))
    assert attempts(configured) == 2


def test_response_cancel_burns_attempt_and_returns_terminal(configured):
    with OpenAI() as client:
        stream = client.responses.create(input="Hi", max_output_tokens=2, stream=True)
        event = next(stream)
        assert event.type == "response.created"
        cancelled = client.responses.cancel(event.response["id"])
        assert cancelled.status == "cancelled"
        assert client.responses.retrieve(cancelled.id).status == "cancelled"
        assert list(stream) == []
        released(configured)
        assert attempts(configured) == 2


@pytest.mark.asyncio
async def test_async_response_and_partial_stream_release(configured):
    async with AsyncOpenAI() as client:
        response = await client.responses.create(input="Hi", max_output_tokens=2, temperature=0)
        assert response.usage.output_tokens == 2
        assert (await client.planning(response.id)).status == "feasible"
        released(configured)
        stream = await client.responses.create(input="Hi", max_output_tokens=2, stream=True)
        try:
            async for event in stream:
                if event.type == "response.output_text.delta":
                    break
            else:
                pytest.fail("missing partial response")
        finally:
            await stream.aclose()
        released(configured)
    assert attempts(configured) == 4


@pytest.mark.asyncio
async def test_async_cancel_drains_discovery_without_late_reservation(configured, monkeypatch):
    import pllm.deployment.selection as selection

    entered, proceed = threading.Event(), threading.Event()
    discover = selection.discover

    def blocked(*args, **kwargs):
        entered.set()
        assert proceed.wait(10)
        return discover(*args, **kwargs)

    monkeypatch.setattr(selection, "discover", blocked)
    async with AsyncOpenAI() as client:
        task = asyncio.create_task(client.responses.create(input="Hi", max_output_tokens=2))
        try:
            assert await asyncio.to_thread(entered.wait, 10)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()  # The request still owns its outstanding worker.
        finally:
            proceed.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not client.sync._core.active
        assert attempts(configured) == 0
        released(configured)


def test_client_settings_policy_roundtrip_and_unknown_rejection(tmp_path, monkeypatch):
    monkeypatch.delenv("PLLM_NETWORK", raising=False)
    monkeypatch.delenv("PLLM_EXPERIMENT", raising=False)
    path = tmp_path / "client.toml"
    ClientSettings(network="network.json", experiment="experiment.json", planning={
        "max_client_peak_memory_bytes": 512 << 20,
        "objectives": ("client_weight_bytes",),
    }).save(path)
    settings = ClientSettings.load(path)
    assert settings.network == str(tmp_path / "network.json")
    assert settings.planning["max_client_peak_memory_bytes"] == 512 << 20
    path.write_text('[client.planning]\npretend_memory = 1\n')
    with pytest.raises(TypeError, match="pretend_memory"):
        ClientSettings.load(path)
