"""Generated F3 arithmetic through ordinary SDK/gateway and real role children.

Not full5, checkpoint-performance, role-independence, or full-state evidence.
The exact nonzero h2/l1 tensors reproduce fss_decoder/tests.rs::fixture;
its independently established source-table oracle selects (0, 0) for input 1.
Public EOS metadata is 0; it must not truncate the fixed two selections.
"""
import asyncio
import dataclasses
import json
import os
import threading
import time

import httpx
import pytest

from pllm import AsyncOpenAI, OpenAI, Model, _native
from pllm.configuration import Deployment, ExecutionBudget, Experiment
from pllm.profiles import FuseFssSourceCpu
from pllm.runtime import fss_transport as transport
from pllm.runtime.client import ProtocolError
from pllm.runtime.fss_protocol import ROLES

PROFILE = "fusefss.gpt2-complete-ring50-q12.v1"
F3_EXPECTED = (0, 0)
READY = all(callable(getattr(_native, name, None)) for name in (
    "FssSourceBuilder", "FssClientRequest", "FssRoleRequest", "fss_transport_admission", "fss_source_contract"))
pytestmark = pytest.mark.skipif(not READY, reason="pending installed wheel with native FSS role/source/channel symbols")


def fixture_experiment(path, *, requests=8):
    from examples.papers.fusefss_smoke import experiment
    return experiment(path, requests=requests)


def active_sessions(core):
    counts = []
    for role in ROLES:
        url, key = core.role_connections[role]
        response = httpx.get(url.removesuffix("/v1") + "/v1/fss/metrics",
            headers={"authorization": f"Bearer {key}"}, timeout=10, trust_env=False)
        response.raise_for_status()
        counts.append(response.json()["active_sessions"])
    return counts


def assert_retired(core):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if active_sessions(core) == [0, 0, 0]:
            assert core._active is None
            return
        time.sleep(0.02)
    raise AssertionError("FSS native/HTTP owners did not retire")


def test_sdk_exact_f3_final_only_fixed_count_and_three_real_children(tmp_path, monkeypatch):
    from pllm.runtime import fss_source, servers
    experiment = fixture_experiment(tmp_path)
    expected_pins = dict(fss_source.resolve_client_source(experiment).snapshot_pins)
    spawned_pins, spawn = [], servers._spawn_role_process

    def observe_spawn(command, **kwargs):
        pins = json.loads(kwargs["environment"]["PLLM_FSS_SOURCE_PINS_JSON"])
        assert pins == expected_pins
        spawned_pins.append(pins)
        return spawn(command, **kwargs)

    def no_parent_tensor_import(*args, **kwargs):
        pytest.fail("client/launcher imported role tensor values")

    monkeypatch.setattr(servers, "_spawn_role_process", observe_spawn)
    # Children are isolated installed interpreters; they do not inherit this
    # parent-only guard. Public metadata/tokenizer/source-lock reads remain real.
    monkeypatch.setattr(fss_source, "load_role_source", no_parent_tensor_import)
    captured, calls = [], []
    original = transport._Peers.post

    def observe(self, index, action, *args, **kwargs):
        calls.append((index, action, kwargs.get("phase")))  # Public control metadata only.
        return original(self, index, action, *args, **kwargs)

    monkeypatch.setattr(transport._Peers, "post", observe)
    with OpenAI(experiment=experiment, timeout=300) as client:
        topology = client._owned_topology
        statuses = topology.statuses
        assert {s.role for s in statuses} == set(ROLES)
        assert len({s.pid for s in statuses}) == 3
        assert all(s.running and s.pid != os.getpid() for s in statuses)
        assert spawned_pins == [expected_pins] * 3
        client._core._result_hook = captured.append
        response = client.responses.create(input="hello", max_output_tokens=2, store=False)
        result, = captured
        assert tuple(f.name for f in dataclasses.fields(result)) == ("selected",)
        assert result.selected == F3_EXPECTED
        assert response.usage.input_tokens == 1
        assert response.usage.output_tokens == 2  # Includes private feedback after EOS=0.
        assert response.output_text == "zero zero"
        assert response.status == "incomplete"
        assert not client._core.records
        assert_retired(client._core)
        audit = client._core.audit.to_dict()
        assert audit["plaintext_prompt_bytes_sent"] == audit["plaintext_token_ids_sent"] == 0
        assert "generation_decode_online_body_bytes" not in audit
        online = [(index, action) for index, action, phase in calls if phase == "online"]
        assert online == [(0, "input"), (1, "input"), (0, "start"), (1, "start"), (0, "finish"), (1, "finish")]
        # No per-selection input/feedback roundtrip through client, and dealer
        # has already retired before either initial private frame is sent.
        assert calls.index((2, "finish", "setup")) < calls.index((0, "input", "online"))
        for holder in ("worker_a", "worker_b"):
            assert audit[f"peer_link.dealer.{holder}.online_bytes"] == 0
            assert audit[f"peer_link.{holder}.dealer.online_bytes"] == 0
        assert audit["peer_link.worker_a.worker_b.online_bytes"] > 0
        assert audit["peer_link.worker_b.worker_a.online_bytes"] > 0
        with pytest.raises(Exception):
            client.responses.create(input="hello", previous_response_id=response.id, max_output_tokens=1)
        assert_retired(client._core)
    assert all(not s.running for s in topology.statuses)


@pytest.mark.parametrize("online_schedule", ["serial", "wavefront-v1", "tensor-v1"])
def test_dense_rope_sdk_preserves_native_oracle_and_private_feedback(tmp_path, online_schedule):
    from examples.papers.fusefss_smoke import dense_experiment
    captured = []
    experiment = dense_experiment(tmp_path).with_params(pipeline__linear__online_schedule=online_schedule)
    with OpenAI(experiment=experiment, timeout=300) as client:
        topology = client._owned_topology
        client._core._result_hook = captured.append
        response = client.responses.create(input="hello", max_output_tokens=2, store=False)
        assert captured[0].selected == (1, 1)  # Independent dense native fixture oracle.
        assert response.usage.output_tokens == 2
        assert_retired(client._core)
    assert all(not status.running for status in topology.statuses)


def test_ordinary_benchmark_prices_dealer_cpu_and_buffered_disclosure(tmp_path):
    from pllm.runtime.benchmark_cli import run_loopback_benchmark
    report = run_loopback_benchmark(experiment=fixture_experiment(tmp_path),
        model="generated-fss", model_id=None, tiny=False, prompt="hello", max_output_tokens=2,
        warmups=0, repetitions=1, timeout_seconds=300, backend="native", capture_output_digest=True)
    assert report["checks"]["passed"], report["checks"]
    contract = report["runtime_contract"]
    assert contract["complete_program_admitted"] is True
    assert len(contract["native_program_digest"]) == 64
    assert len(contract["native_runtime_contract_digest"]) == 64
    assert contract["completed_program_geometry"] == {"input_tokens": 1, "output_tokens": 2}
    # The client never imports role tensors merely to fill an estimate.
    assert contract["program_resources"] is None
    assert contract["native_admission"]["numeric"] == "not_performed"  # Metadata preflight only.
    run, = report["runs"]
    assert run["tokens"]["output_tokens"] == 2
    assert all(run["processes"][role]["cpu_seconds"] > 0 for role in ROLES)
    assert all(run["processes"][role]["rss_peak_bytes"] > 0 for role in ROLES)
    assert run["durations"]["final_disclosure_seconds"] > 0
    assert run["durations"]["online_seconds"] is run["durations"]["ttft_seconds"] is None
    assert report["summary"]["median_tokens_per_second"] is None
    ledger, = report["topology_accounting"]["runs"]
    assert ledger["aggregate_run_window_cpu_seconds"] > 0
    assert ledger["storage_bytes"]["fss_storage.worker_a.read_bytes"] > 0
    phases = ledger["phase_timing"]
    assert phases["preparation_seconds"] > 0
    assert phases["delivery_callback_seconds"] > 0
    assert phases["seal_callback_seconds"] > 0
    assert phases["preparation_outside_callbacks_seconds"] > 0
    assert all(value > 0 for value in phases["online_seconds_by_holder"].values())
    assert phases["online_seconds"] == max(phases["online_seconds_by_holder"].values())
    assert phases["online_scope"] == "maximum_holder_native_run_wall_not_client_online_interval"
    assert run["processes"]["dealer"]["phase_timing"]["preparation_calls"] == 1


@pytest.mark.parametrize("failure", ["partial_admission", "cancel", "malformed", "replay"])
def test_failed_request_publishes_nothing_and_does_not_leave_reusable_state(tmp_path, monkeypatch, failure):
    original, injected, captured = transport._Peers.post, [], []

    def fail(self, index, action, *args, **kwargs):
        target = (index == 1 and action == "admit") if failure == "partial_admission" else (index == 0 and action == "input")
        if target and not injected:
            injected.append(True)
            if failure == "partial_admission":
                raise ProtocolError("injected admission interruption")
            original(self, index, action, *args, **kwargs)
            if failure == "cancel":
                raise asyncio.CancelledError()
            if failure == "malformed":
                return b"invalid acknowledgement"
            return original(self, index, action, *args, **kwargs)
        return original(self, index, action, *args, **kwargs)

    with OpenAI(experiment=fixture_experiment(tmp_path), timeout=300) as client:
        client._core._result_hook = captured.append
        monkeypatch.setattr(transport._Peers, "post", fail)
        with pytest.raises((ValueError, ProtocolError, asyncio.CancelledError)):
            with client.responses.create(input="hello", max_output_tokens=1, stream=True, store=False) as stream:
                for _ in stream:
                    pytest.fail("failed native execution exposed a response event")
        assert injected and not captured and not client._core.records
        assert_retired(client._core)
        monkeypatch.setattr(transport._Peers, "post", original)
        # Fresh request must succeed; no continuation/tape/nonce state reused.
        response = client.responses.create(input="hello", max_output_tokens=2, store=False)
        assert captured[-1].selected == F3_EXPECTED and response.usage.output_tokens == 2
        assert_retired(client._core)


@pytest.mark.asyncio
async def test_async_cancellation_drains_and_closes_children(tmp_path, monkeypatch):
    entered, release, original = threading.Event(), threading.Event(), transport._Peers.post

    def observe(self, index, action, *args, **kwargs):
        if index == 0 and action == "start":
            result = original(self, index, action, *args, **kwargs)
            entered.set()
            if not release.wait(30):
                raise ProtocolError("test cancellation gate expired")
            return result
        return original(self, index, action, *args, **kwargs)

    monkeypatch.setattr(transport._Peers, "post", observe)
    async with AsyncOpenAI(experiment=fixture_experiment(tmp_path), timeout=300) as client:
        topology, core = client.sync._owned_topology, client.sync._core
        task = asyncio.create_task(client.responses.create(input="hello", max_output_tokens=2, store=False))
        try:
            deadline = time.monotonic() + 120
            while not entered.is_set():
                if task.done():
                    await task  # Surface actual admission/readiness error, not a misleading timeout.
                    pytest.fail("request completed without entering the online holder phase")
                assert time.monotonic() < deadline, "request never reached online cancellation gate"
                await asyncio.sleep(0.01)
            task.cancel()
            await asyncio.sleep(0.05)
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert not core.records
            await asyncio.to_thread(assert_retired, core)
        finally:
            release.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    assert all(not s.running for s in topology.statuses)


@pytest.mark.asyncio
@pytest.mark.parametrize("online_schedule", ["serial", "wavefront-v1", "tensor-v1"])
async def test_real_role_children_through_ordinary_gateway_final_buffered_sse(tmp_path, online_schedule):
    from pllm.runtime.servers import build_roles
    experiment = fixture_experiment(tmp_path).with_params(pipeline__linear__online_schedule=online_schedule)
    topology = build_roles(experiment, log_dir=tmp_path / "logs")
    await asyncio.to_thread(topology.start)
    role_secrets = {*topology._role_credentials.values(), *topology._peer_credentials.values()}
    try:
        app = topology.gateway_app(local_api_key="generated-gateway-auth")
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gateway",
                    headers={"authorization": "Bearer generated-gateway-auth"}, timeout=300) as client:
                models = await client.get("/v1/models")
                assert models.status_code == 200
                model = models.json()["data"][0]
                assert model["runtime"]["client_runtime"] == "compiled_fss_v1"
                assert model["runtime"]["kv_reuse"] is False
                response = await client.post("/v1/responses", json={"model": model["id"], "input": "hello",
                    "max_output_tokens": 2, "stream": True, "store": False})
                assert response.status_code == 200, response.text
                events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ") and line != "data: [DONE]"]
                final = events[-1]
                assert final["type"] == "response.incomplete"
                assert final["response"]["usage"]["output_tokens"] == 2
                assert "".join(e["delta"] for e in events if e["type"] == "response.output_text.delta") == "zero zero"
                forbidden = {"logits", "kv", "activations", "token_shares", "executed"}
                assert not forbidden.intersection(final["response"])
    finally:
        await asyncio.to_thread(topology.close)
    assert all(not s.running for s in topology.statuses)
    public = repr(topology.statuses) + "".join(path.read_text() for path in (tmp_path / "logs").glob("*.log"))
    assert all(secret not in public for secret in role_secrets)
