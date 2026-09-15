from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from pllm.runtime import GatewayConfig, OpenAI, create_app
from pllm.runtime.client import ProtocolError, ResponseStream


class RecordingClient:
    def __init__(self, client: TestClient):
        self.client = client
        self.rows: list[tuple[str, str, bytes]] = []

    def _record(self, method: str, url: str, kwargs: dict[str, Any]):
        if "content" in kwargs and kwargs["content"] is not None:
            body = bytes(kwargs["content"])
        elif "json" in kwargs:
            body = json.dumps(kwargs["json"], separators=(",", ":")).encode()
        else:
            body = b""
        self.rows.append((method, url, body))
        return getattr(self.client, method.lower())(url, **kwargs)

    def get(self, url: str, **kwargs: Any): return self._record("GET", url, kwargs)
    def post(self, url: str, **kwargs: Any): return self._record("POST", url, kwargs)
    def put(self, url: str, **kwargs: Any): return self._record("PUT", url, kwargs)
    def close(self): pass


@pytest.fixture
def app():
    return create_app(GatewayConfig(api_keys=("test",), allow_insecure_local_correlations=True))


@pytest.fixture
def test_client(app):
    with TestClient(app) as client:
        yield client


def auth(): return {"Authorization": "Bearer test"}


def test_auth_models_and_capabilities(test_client):
    assert test_client.get("/healthz").status_code == 200
    assert test_client.get("/v1/models").status_code == 401
    models = test_client.get("/v1/models", headers=auth()).json()
    assert models["object"] == "list"
    assert models["data"][0]["id"] == "pllm-bigram-demo"
    caps = test_client.get("/v1/runtime/capabilities", headers=auth()).json()
    assert caps["responses_api"] is True
    assert "http-binary" in caps["client_transports"]
    assert "local-test" in caps["correlation_modes"]


def test_plaintext_response_request_is_rejected_for_private_model(test_client):
    secret = "NEVER-LEAVE-CLIENT"
    response = test_client.post("/v1/responses", headers=auth(), json={"model": "pllm-bigram-demo", "input": secret})
    assert response.status_code == 426
    assert response.headers["upgrade"] == "pllm-runtime/1"
    assert response.json()["detail"]["error"]["code"] == "runtime_client_required"


def test_drop_in_client_nonstreaming_and_privacy_audit(test_client):
    recorder = RecordingClient(test_client)
    secret = "NEVER-LEAVE-CLIENT-71f95"
    client = OpenAI(
        base_url="http://testserver",
        api_key="test",
        correlation_mode="local-test",
        correlation_prefetch=16,
        http_client=recorder,
    )
    response = client.responses.create(model="pllm-bigram-demo", input=secret, max_output_tokens=32)
    assert response.output_text == "private\n"
    audit = client.privacy_audit.to_dict()
    assert audit["plaintext_prompt_bytes_sent"] == 0
    assert audit["plaintext_token_ids_sent"] == 0
    assert audit["online_steps"] == 9
    assert audit["correlation_count"] >= 9
    wire = b"\n".join(body for _, _, body in recorder.rows)
    assert secret.encode() not in wire
    assert b"user:" not in wire
    assert all("input" not in url for _, url, _ in recorder.rows)


def test_streaming_lifecycle_retrieve_previous_and_cancel(test_client):
    client = OpenAI(base_url="http://testserver", api_key="test", correlation_mode="local-test", correlation_prefetch=16, http_client=test_client)
    stream = client.responses.create(model="pllm-bigram-demo", input="first", stream=True, max_output_tokens=32)
    assert isinstance(stream, ResponseStream)
    events = list(stream)
    assert events[0].type == "response.created"
    assert events[-1].type == "response.completed"
    deltas = [event.delta for event in events if event.type == "response.output_text.delta"]
    assert "".join(deltas) == "private\n"
    response_id = events[-1].response["id"]
    assert client.responses.retrieve(response_id).output_text == "private\n"
    second = client.responses.create(model="pllm-bigram-demo", input="second", previous_response_id=response_id)
    assert second.previous_response_id == response_id
    assert client.responses.cancel(second.id).status == "cancelled"


def test_response_stream_close_closes_underlying_generator():
    finalized = False

    def prepared_events():
        nonlocal finalized
        try:
            yield "created"
        finally:
            finalized = True

    stream = ResponseStream(prepared_events())
    assert next(stream) == "created"
    stream.close()
    stream.close()
    assert finalized is True


def test_response_and_session_lifecycle_is_scoped_to_api_principal():
    app = create_app(
        GatewayConfig(
            api_keys=("first", "second"),
            allow_insecure_local_correlations=True,
        )
    )
    first = {"Authorization": "Bearer first"}
    second = {"Authorization": "Bearer second"}
    with TestClient(app) as client:
        active = client.post(
            "/v1/runtime/sessions", headers=first, json={"model": "pllm-bigram-demo"}
        ).json()
        response_id = active["response_id"]

        assert client.post(
            f"/v1/responses/{response_id}/cancel", headers=second
        ).json()["status"] == "cancelled"
        assert app.state.sessions[active["id"]].canceled is False
        assert client.post(
            f"/v1/responses/{response_id}/cancel", headers=first
        ).json()["status"] == "cancelled"
        assert app.state.sessions[active["id"]].canceled is True

        completed_session = client.post(
            "/v1/runtime/sessions", headers=first, json={"model": "pllm-bigram-demo"}
        ).json()
        completed_id = completed_session["response_id"]
        complete = client.post(
            f"/v1/runtime/sessions/{completed_session['id']}/complete",
            headers=first,
            json={"usage": {}},
        )
        assert complete.status_code == 200
        assert client.get(
            f"/v1/responses/{completed_id}", headers=second
        ).status_code == 404
        client.post(f"/v1/responses/{completed_id}/cancel", headers=second)
        assert client.get(
            f"/v1/responses/{completed_id}", headers=first
        ).json()["status"] == "completed"


def test_unknown_previous_response_is_client_side_error(test_client):
    client = OpenAI(base_url="http://testserver", api_key="test", correlation_mode="local-test", http_client=test_client)
    with pytest.raises(ProtocolError, match="previous_response_id"):
        client.responses.create(model="pllm-bigram-demo", input="x", previous_response_id="resp_missing")


def test_server_retains_only_redacted_operational_response(test_client):
    client = OpenAI(base_url="http://testserver", api_key="test", correlation_mode="local-test", correlation_prefetch=16, http_client=test_client)
    response = client.responses.create(model="pllm-bigram-demo", input="secret")
    # Client retrieval returns its private local cache.
    assert response.output_text == "private\n"
    # Raw remote retrieval contains no plaintext output.
    remote = test_client.get(f"/v1/responses/{response.id}", headers=auth()).json()
    assert remote["runtime_redacted"] is True
    assert remote["output"] == []


def test_metrics_track_online_work(test_client):
    client = OpenAI(base_url="http://testserver", api_key="test", correlation_mode="local-test", correlation_prefetch=16, http_client=test_client)
    client.responses.create(model="pllm-bigram-demo", input="secret")
    assert test_client.get("/metrics").status_code == 401
    metrics = test_client.get("/metrics", headers=auth()).json()
    assert metrics["sessions"]["online_steps"] == 9
    assert metrics["stage_schedulers"]["pllm-bigram-demo"]["items"] == 9


def test_prefixed_backend_model_is_unprefixed_upstream_and_public_in_response():
    from pllm.runtime.backends.base import BackendModel
    from pllm.runtime.backends.registry import BackendRegistry

    class Adapter:
        capabilities = type("Capabilities", (), {"to_dict": lambda self: {"responses": True}})()

        def __init__(self):
            self.bodies = []

        async def list_models(self):
            return [BackendModel("same", "backend", "fake", "trusted_backend")]

        async def create_response(self, body):
            self.bodies.append(body)
            return {
                "id": "resp_prefixed", "object": "response", "status": "completed",
                "model": body["model"], "output": [],
            }

        async def stream_response(self, body):
            self.bodies.append(body)
            yield {
                "type": "response.completed",
                "response": {
                    "id": "resp_stream", "object": "response", "status": "completed",
                    "model": body["model"], "output": [],
                },
            }

    a, b = Adapter(), Adapter()
    registry = BackendRegistry()
    registry.add("a", a)
    registry.add("b", b)
    import asyncio
    asyncio.run(registry.refresh())
    app = create_app(GatewayConfig(api_keys=("x",)), backend_registry=registry)
    with TestClient(app) as client:
        response = client.post(
            "/v1/responses",
            headers={"Authorization": "Bearer x"},
            json={"model": "b/same", "input": "hello"},
        )
        assert response.status_code == 200
        assert b.bodies[-1]["model"] == "same"
        assert response.json()["model"] == "b/same"

        with client.stream(
            "POST",
            "/v1/responses",
            headers={"Authorization": "Bearer x"},
            json={"model": "b/same", "input": "hello", "stream": True},
        ) as streamed:
            text = "".join(streamed.iter_text())
        assert b.bodies[-1]["model"] == "same"
        assert '"model":"b/same"' in text.replace(" ", "")
