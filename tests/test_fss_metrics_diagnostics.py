"""Mocked public diagnostics only; no native or historical-cause evidence."""

from contextlib import contextmanager
from types import SimpleNamespace

import httpx
import pytest

from pllm.deployment.network import NetworkError
from pllm.runtime import fss_worker as worker
from pllm.runtime.fss_protocol import ROLES
from test_fss_conformance_sdk import assert_retired
from test_fss_worker import cluster, double  # noqa: F401 -- shared protocol doubles


@pytest.mark.asyncio
@pytest.mark.parametrize(("error", "code"), [
    (NetworkError("CLIENT_MEMORY_PRESSURE"), "CLIENT_MEMORY_PRESSURE"),
    (NetworkError("CLIENT_MEMORY_OBSERVATION_UNAVAILABLE"), "CLIENT_MEMORY_OBSERVATION_UNAVAILABLE"),
    (NetworkError("CLIENT_MEMORY_CAPACITY_CHANGED"), "CLIENT_MEMORY_CAPACITY_CHANGED"),
    (NetworkError("private-seed-credential"), "FSS_METRICS_FAILED"),
    (ValueError("CLIENT_MEMORY_PRESSURE"), "FSS_METRICS_FAILED"),
])
async def test_metrics_reservation_denial_has_only_allowlisted_code(cluster, monkeypatch, error, code):
    role = ROLES[0]
    # Complete real mocked-owner retirement before denying the next observation.
    assert all(r.status_code == 200 for r in (await cluster.admit()).values())
    for owner in ROLES:
        assert (await cluster.post(owner, "cancel")).status_code == 200
    assert (await cluster.metrics(role))["active_sessions"] == 0
    assert all(r["closed"] for r in cluster.double.reservations)

    @contextmanager
    def deny(size):
        raise error
        yield  # pragma: no cover -- context manager never acquires an owner

    with monkeypatch.context() as patch:
        patch.setattr(worker, "reserve_incremental_memory", deny)
        client = cluster.clients[role]
        unauthorized = await client.get("/v1/fss/metrics", headers={"authorization": "Bearer invalid"})
        assert unauthorized.status_code == 401
        assert "x-pllm-error-code" not in unauthorized.headers
        response = await client.get("/v1/fss/metrics")
        assert response.status_code == 400
        assert response.content == b""
        assert response.headers["x-pllm-error-code"] == code
    assert (await cluster.metrics(role))["active_sessions"] == 0


@pytest.mark.parametrize("public_code", ["CLIENT_MEMORY_PRESSURE", "private-seed-credential", None])
def test_retirement_assertion_reports_sanitized_role_status_code(monkeypatch, public_code):
    core = SimpleNamespace(role_connections={role: ("http://127.0.0.1:1234/v1", "private-key")
                                             for role in ROLES}, _active=None)
    headers = {} if public_code is None else {"x-pllm-error-code": public_code}

    def get(*args, **kwargs):
        return httpx.Response(400, headers=headers, content=b"private-payload-seed-credential")

    monkeypatch.setattr(httpx, "get", get)
    expected = "CLIENT_MEMORY_PRESSURE" if public_code == "CLIENT_MEMORY_PRESSURE" else "UNAVAILABLE"
    with pytest.raises(AssertionError) as caught:
        assert_retired(core)
    assert str(caught.value) == (
        f"Qwen2 FSS retirement metrics failed: role={ROLES[0]} status=400 code={expected}")


def test_retirement_assertion_still_checks_client_owner(monkeypatch):
    core = SimpleNamespace(role_connections={role: ("http://127.0.0.1:1234/v1", "private-key")
                                             for role in ROLES}, _active=object())
    monkeypatch.setattr(httpx, "get", lambda *args, **kwargs: httpx.Response(200, json={"active_sessions": 0}))
    with pytest.raises(AssertionError):
        assert_retired(core)
