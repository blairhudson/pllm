"""Real registered providers behind existing Responses/Chat gateway routes."""

import json
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from test_network_benchmark import _fixture_module, free_port, running_http, scenario_fixture
from pllm.runtime.network_benchmark import SelectedClientFactory
from pllm.runtime.sidecar import create_sidecar_app

pytestmark = pytest.mark.rust


@pytest.fixture
def network_fixture(tmp_path):
    with scenario_fixture(tmp_path, hidden_size=8) as fixture:
        yield fixture


def factory(fixture, mode):
    return SelectedClientFactory(fixture["network"], credentials=fixture["credentials"],
                                 **{mode: fixture["result" if mode == "result" else "request"]})


def test_actual_cli_gateway_fixed_and_request_modes(network_fixture):
    for mode in ("plan", "request"):
        result = _fixture_module.gateway_cli_probe(network_fixture, mode)
        assert result["response_output_tokens"] == result["chat_stream_output_tokens"] == 2
        assert result["capacity_zero"] and result["stream_terminal_received"]


@pytest.mark.sdk
def test_official_sdk_responses_chat_and_partial_stream_close(network_fixture):
    sdk = pytest.importorskip("openai")
    fixture = network_fixture
    app = create_sidecar_app(client_factory=factory(fixture, "request"), local_api_key="client-local")
    with running_http(app, free_port()) as origin:
        with sdk.OpenAI(base_url=origin + "/v1", api_key="client-local", max_retries=0) as client:
            response = client.responses.create(model="network-scenarios-tiny", input="Hi",
                                               max_output_tokens=2, temperature=0)
            assert response.usage.output_tokens == 2
            response = client.chat.completions.create(model="network-scenarios-tiny",
                messages=[{"role": "user", "content": "Hi"}], max_tokens=2, temperature=0)
            assert response.usage.completion_tokens == 2
            for route in ("responses", "chat"):
                source = (client.responses.create(model="network-scenarios-tiny", input="Hi",
                          max_output_tokens=2, temperature=0, stream=True) if route == "responses" else
                          client.chat.completions.create(model="network-scenarios-tiny",
                          messages=[{"role": "user", "content": "Hi"}], max_tokens=2,
                          temperature=0, stream=True))
                try:
                    for event in source:
                        if (route == "responses" and event.type == "response.output_text.delta") or (
                            route == "chat" and event.choices and event.choices[0].delta.content):
                            break
                    else:
                        pytest.fail("SDK stream never delivered partial output")
                finally:
                    source.close()
        deadline = time.monotonic() + 5
        while any(child.state.controller._reservations for child in fixture["children"]):
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert sum(len(child.state.controller._used_attempts) for child in fixture["children"]) == 8


@pytest.mark.parametrize("mode", ("result", "request"))
def test_live_responses_chat_streams_admit_each_attempt_release_and_budget_gate(network_fixture, mode):
    fixture = network_fixture
    app = create_sidecar_app(client_factory=factory(fixture, mode), local_api_key="client-local")
    headers = {"authorization": "Bearer client-local"}
    with TestClient(app) as gateway:
        assert gateway.get("/v1/models", headers=headers).json()["data"][0]["id"] == "network-scenarios-tiny"
        assert all(not child.state.controller._used_attempts for child in fixture["children"])
        for route in ("responses", "chat/completions"):
            for stream in (False, True):
                body = ({"input": "Hi", "max_output_tokens": 2} if route == "responses" else
                        {"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 2})
                response = gateway.post("/v1/" + route, headers=headers, json={
                    "model": "network-scenarios-tiny", "temperature": 0, "stream": stream, **body})
                assert response.status_code == 200, response.text
                if stream:
                    assert "data: [DONE]" in response.text
                else:
                    assert response.json()["usage"].get("output_tokens", response.json()["usage"].get("completion_tokens")) == 2
                assert all(not child.state.controller._reservations for child in fixture["children"])
        counts = sum(len(child.state.controller._used_attempts) for child in fixture["children"])
        assert counts == 8
        online_before = sum(proxy.snapshot().get("online", {}).get("requests", 0) for proxy in fixture["proxies"])
        for body in ({"input": "Hi", "max_output_tokens": 3}, {"input": "x" * 1000, "max_output_tokens": 2}):
            response = gateway.post("/v1/responses", headers=headers, json={"model": "network-scenarios-tiny", **body})
            assert response.status_code == 400
            assert all(not child.state.controller._reservations for child in fixture["children"])
        online_after = sum(proxy.snapshot().get("online", {}).get("requests", 0) for proxy in fixture["proxies"])
        assert online_before == online_after  # Budget check before protected provider execution.


@pytest.mark.parametrize("route", ("responses", "chat/completions"))
def test_live_http_disconnect_after_partial_tokens_closes_lease_no_retry(network_fixture, route):
    fixture = network_fixture
    # Use slow host pair so cancellation reaches an unfinished generation.
    from pllm.runtime.network_benchmark import feasible_controls

    result = next(result for result in feasible_controls(fixture["result"])
                  if all(row["party_id"].endswith("west") for row in result.native_placement["roles"]
                         if row["role_id"] != "client"))
    app = create_sidecar_app(client_factory=SelectedClientFactory(fixture["network"], result=result,
                              credentials=fixture["credentials"]), local_api_key="client-local")
    body = ({"input": "Hi", "max_output_tokens": 2} if route == "responses" else
            {"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 2})
    with running_http(app, free_port()) as origin:
        with httpx.Client(timeout=30) as client:
            with client.stream("POST", origin + "/v1/" + route,
                headers={"authorization": "Bearer client-local"}, json={
                    "model": "network-scenarios-tiny", "temperature": 0, "stream": True, **body}) as response:
                assert response.status_code == 200
                for line in response.iter_lines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        value = json.loads(line[6:])
                        if value.get("type") == "response.output_text.delta" or any(
                            choice.get("delta", {}).get("content") for choice in value.get("choices", [])):
                            break
                else:
                    pytest.fail("no partial token reached gateway client")
        deadline = time.monotonic() + 5
        while any(child.state.controller._reservations for child in fixture["children"]):
            assert time.monotonic() < deadline, "disconnect leaked execution lease"
            time.sleep(0.01)
        assert sum(len(child.state.controller._used_attempts) for child in fixture["children"]) == 2
        assert sum(proxy.snapshot().get("reserve", {}).get("requests", 0) for proxy in fixture["proxies"]) == 2
        assert sum(proxy.snapshot().get("release", {}).get("requests", 0) for proxy in fixture["proxies"]) == 2
