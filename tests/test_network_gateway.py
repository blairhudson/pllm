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
        # The controller drops a lease before its HTTP response reaches the
        # proxy's completed-request ledger. Wait for both observable boundaries.
        while (any(child.state.controller._reservations for child in fixture["children"])
               or sum(proxy.snapshot().get("release", {}).get("requests", 0)
                      for proxy in fixture["proxies"]) < 2):
            assert time.monotonic() < deadline, "disconnect did not complete both lease releases"
            time.sleep(0.01)
        assert sum(len(child.state.controller._used_attempts) for child in fixture["children"]) == 2
        assert sum(proxy.snapshot().get("reserve", {}).get("requests", 0) for proxy in fixture["proxies"]) == 2
        assert sum(proxy.snapshot().get("release", {}).get("requests", 0) for proxy in fixture["proxies"]) == 2


def _pending_selected_gateway(tmp_path, monkeypatch, *, oversized=False, source_failure=False):
    """Real sidecar/factory/open/pending path, explicit planning/source/I/O doubles.

    Setup/planning precedes execution spies. No worker/source authentication or
    token/numeric correctness evidence is manufactured by the literal response.
    """
    from dataclasses import replace
    from pllm.deployment import execution, NetworkSnapshot, NetworkSpec, PartyOffer
    from pllm.deployment import selection
    from pllm.runtime.responses import Response
    from test_prepared_prompt import _real_activation_closure_fixture, _pending_result_stub

    options = {"template": " ".join(["hello"] * 29)} if oversized else {}
    old, core, state = _real_activation_closure_fixture(tmp_path, monkeypatch, **options)
    offers = tuple(PartyOffer(name, name, str(index + 1) * 64, 1000,
        ("trusted_client",) if name == "client" else
            ("trusted_preparation", "masked_linear_provider", "public_linear_provider"),
        256 << 20, 64 << 20, 4, 0, ("*",), ("*",))
        for index, name in enumerate(("client", "a", "b")))
    snapshot = NetworkSnapshot("local", offers, (), "pending-gateway", 0, 1000)
    network = NetworkSpec("local", snapshot)
    result, _ = _pending_result_stub(old._result.experiment,
                                    state["owned"].source_lock_digest, snapshot=snapshot)
    state["result"] = result
    def replay(cls, document):
        state["planning_calls"].append("replay")
        state["timeline"].append("replay")
        assert execution.canonical(document) == result.canonical_bytes()
        return result

    monkeypatch.setattr(execution.PlanningResult, "from_spec", classmethod(replay))
    # Real SelectedClientFactory initialization is part of fixture planning.
    # All execution-time spies below are installed AFTER this point.
    factory = SelectedClientFactory(network, result=result)
    # Existing catalog is computed during fixture planning using genuine full
    # resolution conventions (proprietary path vs public path/basename).
    model_id = factory.default_model
    state["planning_calls"].clear()
    state["timeline"].clear()
    state["leases"] = []
    actual_open = execution.open_execution

    def open_recorded(*args, **kwargs):
        lease = actual_open(*args, **kwargs, clock=lambda: state["clock"])
        state["lease"] = lease
        state["leases"].append(lease)
        return lease

    monkeypatch.setattr(selection, "open_execution", open_recorded)
    original_tokenizer = execution._TokenizerView.tokenizer

    class Counted:
        def __init__(self, inner):
            self.inner = inner

        def encode(self, text, **kwargs):
            state["timeline"].append("count")
            return self.inner.encode(text, **kwargs)

        def close(self):
            close = getattr(self.inner, "close", None)
            if close is not None:
                close()

    monkeypatch.setattr(execution._TokenizerView, "tokenizer",
                        lambda self: Counted(original_tokenizer(self)))

    def response(request, ids):
        return Response.from_dict({"id": "resp-pending-metadata-control", "model": model_id,
            "status": "completed", "output": [],
            "usage": {"input_tokens": len(ids), "output_tokens": 1, "total_tokens": len(ids) + 1}})

    state["response"] = response
    if source_failure:
        state["owned"] = replace(state["owned"], config_source_bytes=state["owned"].config_source_bytes + b" ")
    app = create_sidecar_app(client_factory=factory, local_api_key="pending-local")
    return app, model_id, old, core, state


@pytest.mark.parametrize("route", ("responses", "chat/completions"))
def test_pending_catalog_and_rendered_overlimit_real_sidecar_never_activates(tmp_path, monkeypatch, route):
    from pllm import _native, modeling
    from pllm.deployment import execution
    from pllm.runtime import delegated_client, delegated_decoder, delegated_source, responses

    app, model_id, old, core, state = _pending_selected_gateway(tmp_path, monkeypatch, oversized=True)
    headers = {"authorization": "Bearer pending-local"}
    forbidden_events = []
    rendered_prompts = []
    counted_prompts = []

    def trap(name):
        def forbidden(*args, **kwargs):
            forbidden_events.append(name)
            pytest.fail(f"overlimit gateway request reached {name}")
        return forbidden

    original_render = responses._render_prompt
    original_tokenizer = execution._TokenizerView.tokenizer

    def render_recorded(*args, **kwargs):
        rendered = original_render(*args, **kwargs)
        rendered_prompts.append(rendered)
        return rendered

    class CountedRequest:
        def __init__(self, inner):
            self.inner = inner

        def encode(self, text, **kwargs):
            lease = state["lease"]
            assert lease._state == "pending" and lease._client is None
            assert not state["topologies"]
            ids = self.inner.encode(text, **kwargs)
            counted_prompts.append((text, len(ids)))
            return ids

        def close(self):
            self.inner.close()

    def tokenizer_recorded(view):
        assert view._prompt_snapshot is not None
        return CountedRequest(original_tokenizer(view))

    try:
        with TestClient(app) as gateway:
            # Fixture metadata/composition, factory, server and client setup may
            # resolve natively. Only this request phase forbids source admission.
            # Keep bounded public metadata and real template/tokenizer reads.
            for owner, attribute, name in (
                (delegated_source.model_loader, "_source_lock", "delegated_source.model_loader._source_lock"),
                (delegated_source, "_files", "delegated_source._files"),
                (delegated_source, "_store", "delegated_source._store"),
                (delegated_source, "_locked", "delegated_source._locked"),
                (delegated_source, "_read_range", "delegated_source._read_range"),
                (delegated_source, "decoder_contract", "delegated_source.decoder_contract"),
                (delegated_decoder, "decoder_contract", "delegated_decoder.decoder_contract"),
                (_native, "gkm_decoder_contract", "_native.gkm_decoder_contract"),
                (_native, "gkm_runtime_contract", "_native.gkm_runtime_contract"),
                (_native, "resolve_experiment", "_native.resolve_experiment"),
                (delegated_client, "_admit_source", "delegated_client._admit_source"),
                (delegated_source, "admit_source", "delegated_source.admit_source"),
                (delegated_source, "_admit_source", "delegated_source._admit_source"),
                (execution.PlanningResult, "validate", "execution.PlanningResult.validate"),
                (modeling, "lower_model", "pllm.modeling.lower_model"),
                (_native, "lower_model", "_native.lower_model"),
                (delegated_source.model_loader, "resolve_model", "model_loader.resolve_model"),
            ):
                monkeypatch.setattr(owner, attribute, trap(name))
            monkeypatch.setattr(execution.PlanningResult, "from_spec",
                classmethod(trap("execution.PlanningResult.from_spec")))
            monkeypatch.setattr(type(core), "_resolve_prompt_source",
                classmethod(trap("fixture_class._resolve_prompt_source")))
            monkeypatch.setattr(responses, "_render_prompt", render_recorded)
            monkeypatch.setattr(execution._TokenizerView, "tokenizer", tokenizer_recorded)
            assert gateway.get("/v1/models").status_code == 401
            catalog = gateway.get("/v1/models", headers=headers)
            assert catalog.status_code == 200 and catalog.json()["data"][0]["id"] == model_id
            assert old._state == "pending" and not state["leases"]
            assert not state["calls"] and not state["planning_calls"]
            payload = ({"input": "hello", "max_output_tokens": 4} if route == "responses" else
                       {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 4})
            result = gateway.post("/v1/" + route, headers=headers, json={"model": model_id, **payload})
            assert result.status_code == 400, result.text
            assert "input exceeds selected budget" in result.text
            assert forbidden_events == []
            assert len(rendered_prompts) == len(counted_prompts) == 1
            assert counted_prompts[0][0] == rendered_prompts[0]
            assert counted_prompts[0][1] > state["result"].experiment.budget.max_input_tokens
            assert state["timeline"] == ["count"]
            assert not state["calls"] and old._state == "pending"
            assert not state["planning_calls"] and not state["topologies"]
            assert state["leases"] and all(lease._client is None for lease in state["leases"])
            assert all(lease.closed and lease._cleanup_complete for lease in state["leases"])
    finally:
        old.close()
        core.close()


def test_pending_admitted_real_sidecar_replays_then_authenticates_before_roles(tmp_path, monkeypatch):
    app, model_id, old, core, state = _pending_selected_gateway(tmp_path, monkeypatch)
    try:
        with TestClient(app) as gateway:
            response = gateway.post("/v1/responses", headers={"authorization": "Bearer pending-local"},
                json={"model": model_id, "input": "hello", "max_output_tokens": 4})
            assert response.status_code == 200, response.text
            assert response.json()["usage"]["input_tokens"] > 0
            timeline = state["timeline"]
            assert timeline[:6] == ["count", "replay", "validate", "lower", "resolve", "validate"]
            assert timeline.index("reserve") > timeline.index("resolve")
            assert timeline.index("arm") > timeline.index("reserve")
            assert all(lease.closed and lease._cleanup_complete for lease in state["leases"])
            assert all(not topology.rows for topology in state["topologies"])
    finally:
        old.close()
        core.close()


def test_pending_real_sidecar_source_mismatch_never_reserves(tmp_path, monkeypatch):
    app, model_id, old, core, state = _pending_selected_gateway(tmp_path, monkeypatch, source_failure=True)
    try:
        with TestClient(app) as gateway:
            response = gateway.post("/v1/responses", headers={"authorization": "Bearer pending-local"},
                json={"model": model_id, "input": "hello", "max_output_tokens": 4})
            assert response.status_code >= 400, response.text
            assert "source" in response.text.lower() or "config" in response.text.lower()
            assert "resolve" in state["calls"] and "reserve" not in state["calls"]
            assert not state["topologies"]
            assert all(lease.closed and lease._cleanup_complete for lease in state["leases"])
    finally:
        old.close()
        core.close()
