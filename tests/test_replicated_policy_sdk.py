"""Installed-wheel gates: generated source only, ordinary SDK workers.

Run after the coordinated native wheel update with:
target/puma-sdk-integration-env-v1/bin/python -I -m pytest -o pythonpath=. tests/test_replicated_policy_sdk.py
These deliberately do not skip on stale wheels: absent policy bindings are failures.
"""
import asyncio
import json
import threading

import pytest

from pllm import AsyncOpenAI, OpenAI
from pllm.configuration import Experiment
from pllm.runtime.replicated_protocol import execution_binding, STREAM_MEMORY, PEER_COUNTERS, ROLES, FRAME_LIMIT
from pllm.runtime.replicated_source import admit_source
from test_replicated_sdk import experiment_fixture, active_sessions


COMBINED = {"range_policy": "checked", "max_abs_activation": 256,
            "lookup": "tiled", "selection": "tiled", "peer_transport": "streams"}


def combined_experiment(path, *, rejecting=False):
    """Freeze a generated source before admission; never tune on a request.

    Rejecting source: every prompt embedding is within ±1, complete head selects
    visible token 4 (world), whose embedding is ±1.5. One output succeeds; the
    second full-prefix pass fails privately and must mask the earlier output.
    This mirrors the independently proved native acceptance fixture without
    exporting any production tensor/share or reading a validity bit in Python.
    """
    baseline = experiment_fixture(path, weights="public", active=not rejecting)
    if rejecting:
        import numpy as np
        from safetensors.numpy import load_file, save_file
        from pathlib import Path

        source = Path(baseline.pipeline.model.source) / "model.safetensors"
        values = load_file(source)
        for key, value in values.items():
            value[:] = 1 if "norm" in key else 0
            if key == "model.embed_tokens.weight":
                value[:] = np.array([[0.25, -0.25]] * 4 + [[1.5, -1.5]], dtype=np.float32)
            elif key == "lm_head.weight":
                value[:] = np.array([[-0.25, 0.25]] * 4 + [[0.25, -0.25]], dtype=np.float32)
        save_file(values, source, metadata={"format": "pt"})
    policy = {**COMBINED, "max_abs_activation": 1 if rejecting else 256}
    return baseline.with_params(**{f"pipeline__components__linear__{name}": value
                                   for name, value in policy.items()})


def observe_publishers(monkeypatch):
    """Observe lifetime only; delegate every share and acceptance call unchanged."""
    from pllm.runtime.replicated_source import ReplicatedSource

    original = ReplicatedSource.publisher
    owners = []

    class Observed:
        def __init__(self, owner):
            self.owner, self.finish_calls, self.close_calls = owner, 0, 0

        def __getattr__(self, name):
            return getattr(self.owner, name)

        def finish(self, *frames):
            self.finish_calls += 1
            return self.owner.finish(*frames)

        def close(self):
            self.close_calls += 1
            self.owner.close()

    def publish(self, *args, **kwargs):
        owner, manifest = original(self, *args, **kwargs)
        observed = Observed(owner)
        owners.append(observed)
        return observed, manifest

    monkeypatch.setattr(ReplicatedSource, "publisher", publish)
    return owners


def assert_publishers_retired(owners):
    assert owners
    for observed in owners:
        assert observed.close_calls == 1
        with pytest.raises(ValueError, match="retired"):
            observed.owner.next_chunk()


def assert_no_failed_output(sdk, captured, records, accepted):
    assert len(captured) == accepted and len(sdk._core.records) == records
    assert sdk._core._active is None
    assert active_sessions(sdk._core) == [0, 0, 0]


def observe_public_work(monkeypatch):
    """Keep only public HTTP schedule/lengths and aggregate peer byte counts.

    No ciphertext, PRSS material, private inputs, validity bits or individual
    range sites enter this observer. Native publisher acceptance stays real.
    """
    from pllm.runtime import replicated_transport as transport
    from pllm.runtime.delegated_protocol import unpack

    original, requests = transport._Peers.post, {}

    def post(self, index, action, payload=b"", **kwargs):
        result = original(self, index, action, payload, **kwargs)
        work = requests.setdefault(self.request_id, {"schedule": [], "admission": [], "final": {}})
        if action == "admit":
            public = json.loads(unpack(payload, limit=transport.ADMISSION_LIMIT)["manifest"])
            assert len(public["session"]) == 32
            work["admission"].append((index, {name: public[name] for name in (
                "schema", "weights", "execution", "rows", "outputs", "source", "composition", "program")}))
        # Fresh random session bytes have variable JSON decimal widths. Compare
        # their admitted public program/geometry above, not nonce serialization.
        work["schedule"].append((index, action, None if action == "admit" else len(payload), len(result)))
        if action == "finish":
            value = unpack(result, limit=FRAME_LIMIT)
            work["final"][ROLES[index]] = value["peer_bytes"]
        return result

    monkeypatch.setattr(transport._Peers, "post", post)
    return requests


def directed_peer_bytes(work):
    assert set(work["final"]) == set(ROLES)
    counts = {}
    for source in ROLES:
        for destination in ROLES:
            if source == destination:
                assert not any(work["final"][source][destination].values())
                continue
            for phase in ("setup", "online"):
                # Each stream initiator owns request + ACK byte accounting.
                # The opposite stream's ACK travels in this direction too.
                value = (work["final"][source][destination][f"{phase}_upload_bytes"]
                         + work["final"][destination][source][f"{phase}_download_bytes"])
                assert type(value) is int and value > 0
                counts[f"peer_link.{source}.{destination}.{phase}_bytes"] = value
    assert set(counts) == PEER_COUNTERS
    return counts


def public_worker_counts(core):
    import httpx
    from pllm.runtime.replicated_peer import TRANSPORT

    counts = {}
    for role, (url, key) in core.role_connections.items():
        response = httpx.get(url.removesuffix("/v1") + "/v1/replicated/metrics", timeout=5,
            headers={"authorization": f"Bearer {key}"}, trust_env=False)
        response.raise_for_status()
        public = response.json()
        # Public role/process/work counters only; no per-site validity/debug data.
        assert set(public) == {"schema", "cpu_ns", "peak_rss_bytes", "active_sessions", "role",
                               "peer_transport", "peer_counts"}
        assert public["role"] == role and public["active_sessions"] == 0
        assert public["peer_transport"] == TRANSPORT
        assert set(public["peer_counts"]) == {"frames", "controls"}
        assert all(type(size) is int and size >= 0 for size in public["peer_counts"].values())
        assert public["peer_counts"]["frames"] > 0
        if role == ROLES[0]:
            assert public["peer_counts"]["controls"] > 0  # only A initiates control barriers
        counts[role] = public["peer_counts"]
    assert set(counts) == set(ROLES)
    return counts


def assert_subprocess_roles(topology):
    import os

    statuses = topology.statuses
    assert {status.role for status in statuses} == set(ROLES)
    assert all(status.running and type(status.pid) is int and status.pid != os.getpid() for status in statuses)
    assert len({status.pid for status in statuses}) == 3


@pytest.mark.parametrize("choices", [
    {"range_policy": "checked"}, {"lookup": "tiled"}, {"selection": "tiled"}, {"peer_transport": "streams"},
    COMBINED,
])
def test_generated_public_policy_ordinary_sdk_final_only(tmp_path, choices):
    baseline = experiment_fixture(tmp_path, weights="public", active=True)
    experiment = baseline.with_params(**{f"pipeline__components__linear__{name}": value
                                        for name, value in choices.items()})
    assert Experiment.from_spec(experiment.to_spec()).canonical_bytes() == experiment.canonical_bytes()
    source = admit_source(experiment)
    expected = execution_binding(experiment)
    assert source.contract["execution"] == expected
    assert source.contract.get("peer_stream_workspace_bytes_per_worker", 0) == (
        STREAM_MEMORY if choices.get("peer_transport") == "streams" else 0)
    publisher, manifest = source.publisher(experiment, (3, 4), 2, lambda: False)
    try:
        assert json.loads(manifest)["execution"] == expected
    finally:
        publisher.close()
    captured = []
    with OpenAI(experiment=experiment, timeout=300) as client:
        client._core._result_hook = captured.append
        for prompt, tokens in (("hello hello", (4, 4)), ("world hello", (1, 1))):
            response = client.responses.create(input=prompt, max_output_tokens=2)
            assert response.usage.output_tokens == 2
            assert captured[-1].selected == tokens  # no aggregate bit escapes native finish
            assert active_sessions(client._core) == [0, 0, 0]
        audit = client._core.audit.to_dict()
        assert audit["plaintext_prompt_bytes_sent"] == audit["plaintext_token_ids_sent"] == 0


def test_combined_checked_rejection_masks_earlier_output_and_retires_sdk_owners(tmp_path, monkeypatch):
    experiment = combined_experiment(tmp_path, rejecting=True)
    owners = observe_publishers(monkeypatch)
    captured = []
    with OpenAI(experiment=experiment, timeout=300) as sdk:
        topology = sdk._owned_topology
        sdk._core._result_hook = captured.append
        assert sdk.responses.create(input="hello", max_output_tokens=1).output_text == "world"
        assert captured[-1].selected == (4,)  # first selection is valid, visible, non-EOS
        records = len(sdk._core.records)
        for stream in (False, True):
            with pytest.raises(ValueError, match="failed protected range checks"):
                if stream:
                    with sdk.responses.create(input="hello", max_output_tokens=2, stream=True) as events:
                        for _ in events:
                            pytest.fail("a rejected range request exposed an event")
                else:
                    sdk.responses.create(input="hello", max_output_tokens=2)
            assert_no_failed_output(sdk, captured, records, 1)
            assert owners[-1].finish_calls == 1  # rejection came from final native acceptance
            assert_publishers_retired(owners)
        assert sdk.responses.create(input="hello", max_output_tokens=1).output_text == "world"
        assert captured[-1].selected == (4,)
        assert active_sessions(sdk._core) == [0, 0, 0]
    assert all(not status.running for status in topology.statuses)
    assert_publishers_retired(owners)


def test_combined_true_private_input_rejection_has_identical_public_work(tmp_path, monkeypatch):
    experiment = combined_experiment(tmp_path, rejecting=True)
    owners, work, captured = observe_publishers(monkeypatch), observe_public_work(monkeypatch), []
    with OpenAI(experiment=experiment, timeout=300) as sdk:
        topology = sdk._owned_topology
        assert_subprocess_roles(topology)
        sdk._core._result_hook = captured.append
        # Same frozen source, admitted policy, one input and one output. Only
        # true private token differs: world embeds outside ±1, hello inside.
        assert sdk.responses.create(input="hello", max_output_tokens=1).output_text == "world"
        accepted_work, = work.values()
        before = sdk._core.audit.to_dict()
        before_counts = public_worker_counts(sdk._core)
        records = len(sdk._core.records)
        with pytest.raises(ValueError, match="^replicated request failed protected range checks$"):
            with sdk.responses.create(input="world", max_output_tokens=1, stream=True) as events:
                for _ in events:
                    pytest.fail("invalid true input exposed response events")
        assert_no_failed_output(sdk, captured, records, 1)
        assert len(work) == 2  # fresh immutable session, not replay/reused shares
        _, rejected_work = work.values()
        assert rejected_work["admission"] == accepted_work["admission"]
        assert rejected_work["schedule"] == accepted_work["schedule"]
        accepted_bytes = directed_peer_bytes(accepted_work)
        rejected_bytes = directed_peer_bytes(rejected_work)
        after = sdk._core.audit.to_dict()
        after_counts = public_worker_counts(sdk._core)
        # Native work is fixed by the admitted program. Individual link routing
        # and batching vary even between identical valid inputs (see separate
        # repeatability control). Compare aggregate work, not per-role routing.
        for name in ("frames", "controls"):
            first = sum(before_counts[role][name] for role in ROLES)
            second = sum(after_counts[role][name] - before_counts[role][name] for role in ROLES)
            assert first == second
        for key, size in accepted_bytes.items():
            assert before[key] == size
            assert after[key] - before[key] == rejected_bytes[key]
            if key.endswith(".setup_bytes"):
                assert rejected_bytes[key] == size
        assert all(owner.finish_calls == 1 for owner in owners)
        assert_publishers_retired(owners)
        assert after["plaintext_prompt_bytes_sent"] == after["plaintext_token_ids_sent"] == 0
    assert all(not status.running for status in topology.statuses)


def test_combined_stream_admission_prices_workspace_once_per_worker(tmp_path):
    from pllm.runtime.benchmark_memory import benchmark_memory, _PROCESS_BYTES, _slack

    stream = combined_experiment(tmp_path)
    for choice in ("http", "streams"):
        experiment = stream.with_params(pipeline__components__linear__peer_transport=choice)
        contract = admit_source(experiment).contract
        report = benchmark_memory(experiment, backend="native", memory_budget_bytes=4 << 30)
        estimate = report["estimate"]
        margin = estimate["allocation_slack_percent"]
        base = _PROCESS_BYTES + contract["execution_payload_estimate_bytes"]
        workspace = STREAM_MEMORY if choice == "streams" else 0
        assert contract.get("peer_stream_workspace_bytes_per_worker", 0) == workspace
        assert estimate["components"].get("peer_stream_workspace_bytes_per_worker", 0) == workspace
        assert estimate["client_peak_bytes"] == _slack(base, margin)
        assert estimate["provider_peak_bytes"] == {role: _slack(base + workspace, margin) for role in ROLES}
        assert estimate["native_total_peak_bytes"] == _slack(base, margin) + 3 * _slack(base + workspace, margin)


def test_combined_gateway_responses_stream_and_chat_keep_final_only_acceptance(tmp_path):
    from fastapi.testclient import TestClient
    from pllm.runtime.sidecar import create_sidecar_app

    captured = []
    with OpenAI(experiment=combined_experiment(tmp_path), timeout=300) as sdk:
        topology = sdk._owned_topology
        assert_subprocess_roles(topology)
        sdk._core._result_hook = captured.append
        app = create_sidecar_app(client=sdk, local_api_key="policy-gateway-test")
        with TestClient(app) as client:
            headers = {"authorization": "Bearer policy-gateway-test"}
            body = {"model": sdk._core.default_model, "input": "hello hello", "max_output_tokens": 2, "temperature": 0}
            assert client.post("/v1/responses", json=body).status_code == 401
            assert not captured and active_sessions(sdk._core) == [0, 0, 0]
            for stream in (False, True):
                response = client.post("/v1/responses", json={**body, "stream": stream}, headers=headers)
                assert response.status_code == 200, response.text
                if stream:
                    events = [json.loads(line[6:]) for line in response.text.splitlines()
                              if line.startswith("data: ") and line != "data: [DONE]"]
                    assert events[-1]["type"] == "response.incomplete"
                    value = events[-1]["response"]
                else:
                    value = response.json()
                assert value["usage"]["output_tokens"] == 2
                assert captured[-1].selected == (4, 4)
                assert active_sessions(sdk._core) == [0, 0, 0]
            response = client.post("/v1/chat/completions", headers=headers, json={"model": body["model"],
                "messages": [{"role": "user", "content": "hello hello"}], "max_completion_tokens": 2, "temperature": 0})
            assert response.status_code == 200, response.text
            assert response.json()["usage"]["completion_tokens"] == 2
            assert captured[-1].selected == (4, 4)
    assert all(not status.running for status in topology.statuses)


def test_combined_gateway_checked_rejection_exposes_only_error_and_retires(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from pllm.runtime.sidecar import create_sidecar_app

    owners, captured = observe_publishers(monkeypatch), []
    with OpenAI(experiment=combined_experiment(tmp_path, rejecting=True), timeout=300) as sdk:
        topology = sdk._owned_topology
        sdk._core._result_hook = captured.append
        app = create_sidecar_app(client=sdk, local_api_key="policy-gateway-test")
        with TestClient(app, raise_server_exceptions=False) as client:
            headers = {"authorization": "Bearer policy-gateway-test"}
            for stream in (False, True):
                response = client.post("/v1/responses", headers=headers, json={"model": sdk._core.default_model,
                    "input": "hello", "max_output_tokens": 2, "temperature": 0, "stream": stream})
                assert "error" in response.text
                assert "world" not in response.text
                assert not any(kind in response.text for kind in (
                    "response.created", "response.output", "response.incomplete", "response.completed"))
                assert_no_failed_output(sdk, captured, 0, 0)
                assert owners[-1].finish_calls == 1
                assert_publishers_retired(owners)
            response = client.post("/v1/chat/completions", headers=headers, json={"model": sdk._core.default_model,
                "messages": [{"role": "user", "content": "hello"}], "max_completion_tokens": 2, "temperature": 0})
            assert response.status_code >= 400 and "error" in response.text and "world" not in response.text
            assert "choices" not in response.json()
            assert_no_failed_output(sdk, captured, 0, 0)
            assert owners[-1].finish_calls == 1
        assert sdk.responses.create(input="hello", max_output_tokens=1).output_text == "world"
        assert captured[-1].selected == (4,)
        assert active_sessions(sdk._core) == [0, 0, 0]
    assert all(not status.running for status in topology.statuses)
    assert_publishers_retired(owners)


@pytest.mark.parametrize("swap", ["range_policy", "selection", "source"])
def test_combined_sdk_authenticated_publication_substitution_retires_all_roles(tmp_path, monkeypatch, swap):
    from pllm.runtime import replicated_transport as transport
    from pllm.runtime.client import ProtocolError
    from pllm.runtime.delegated_protocol import pack, unpack

    original, injected = transport._Peers.post, []

    def substitute(self, index, action, payload=b"", **kwargs):
        if index == 1 and action == "admit" and not injected:
            value = unpack(payload, limit=transport.ADMISSION_LIMIT)
            public = json.loads(value["manifest"])
            if swap == "selection":
                public["execution"]["selection"] = "expanded"
            elif swap == "range_policy":
                public["execution"]["range_policy"] = "static"
            else:
                public["source"][0] ^= 1
            value["manifest"] = json.dumps(public).encode()
            payload = pack(value, limit=transport.ADMISSION_LIMIT)
            injected.append(True)
        return original(self, index, action, payload, **kwargs)

    owners, captured = observe_publishers(monkeypatch), []
    with OpenAI(experiment=combined_experiment(tmp_path), timeout=300) as sdk:
        topology = sdk._owned_topology
        sdk._core._result_hook = captured.append
        monkeypatch.setattr(transport._Peers, "post", substitute)
        with pytest.raises((ValueError, ProtocolError)):
            with sdk.responses.create(input="hello hello", max_output_tokens=2, stream=True) as events:
                for _ in events:
                    pytest.fail("substituted source or selector exposed an event")
        assert injected
        assert_no_failed_output(sdk, captured, 0, 0)
        assert owners[-1].finish_calls == 0
        assert_publishers_retired(owners)
        monkeypatch.setattr(transport._Peers, "post", original)
        sdk.responses.create(input="hello hello", max_output_tokens=2)
        assert captured[-1].selected == (4, 4)
        assert active_sessions(sdk._core) == [0, 0, 0]
    assert all(not status.running for status in topology.statuses)
    assert_publishers_retired(owners)


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["input", "finish"])
async def test_combined_async_cancellation_before_final_acceptance_drains_owners(tmp_path, monkeypatch, phase):
    from pllm.runtime import replicated_transport as transport
    from pllm.runtime.client import ProtocolError

    entered, release = threading.Event(), threading.Event()
    original, gated = transport._Peers.post, []

    def observe(self, index, action, *args, **kwargs):
        result = original(self, index, action, *args, **kwargs)
        if index == 0 and action == phase and not gated:
            gated.append(True)
            entered.set()  # actual authenticated input/first final-share RPC completed
            assert release.wait(45), "test cancellation gate was not released"
        return result

    owners, captured = observe_publishers(monkeypatch), []
    monkeypatch.setattr(transport._Peers, "post", observe)
    experiment = combined_experiment(tmp_path)
    async with AsyncOpenAI(experiment=experiment, timeout=300) as sdk:
        core = sdk.sync._core
        topology = sdk.sync._owned_topology
        core._result_hook = captured.append
        task = asyncio.create_task(sdk.responses.create(input="hello hello", max_output_tokens=2))
        try:
            assert await asyncio.to_thread(entered.wait, 60)
            attempt = core._active
            assert attempt is not None
            # Single-seat SDK rejects concurrent attempt without touching first
            # request's native owner, cancellation scope or authenticated peers.
            with pytest.raises(ProtocolError) as conflict:
                await sdk.responses.create(input="world hello", max_output_tokens=2)
            assert conflict.value.status_code == 409
            assert core._active is attempt and not attempt()
            assert len(owners) == 1 and owners[0].close_calls == 0
            assert not captured and not core.records
            # A different SDK core can get past its local seat and contend on
            # these same real subprocess roles. Its partial admission/cancel
            # must target its own session, never retire first attempt's peers.
            expected_active = [1, 1, 1] if phase == "input" else [0, 1, 1]
            async with AsyncOpenAI(experiment=experiment, role_connections=core.role_connections,
                                   timeout=300) as contender:
                with pytest.raises(ProtocolError) as remote_conflict:
                    await contender.responses.create(input="world hello", max_output_tokens=2)
                assert remote_conflict.value.status_code in {400, 409}
                assert contender.sync._core._active is None and not contender.sync._core.records
                assert len(owners) == 2 and owners[1].finish_calls == 0
                assert_publishers_retired(owners[1:])
                assert await asyncio.to_thread(active_sessions, core) == expected_active
            assert owners[0].close_calls == 0 and core._active is attempt and not attempt()
            task.cancel()
            await asyncio.sleep(0)
            assert attempt(), "async cancellation was not signalled to the request owner"
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert core._active is None and not core.records and not captured
            assert await asyncio.to_thread(active_sessions, core) == [0, 0, 0]
            assert owners[-1].finish_calls == 0
            assert_publishers_retired(owners)
            # A fresh attempt after drain must not inherit cancellation, prompt
            # feedback or a stale request seat. Use opposite deterministic head.
            monkeypatch.setattr(transport._Peers, "post", original)
            response = await sdk.responses.create(input="world hello", max_output_tokens=2)
            assert response.usage.output_tokens == 2
            assert len(captured) == 1 and captured[0].selected == (1, 1)
            attempt.cancel()  # stale handle cannot evict newly accepted state
            assert core._active is None and len(core.records) == 1
            assert await asyncio.to_thread(active_sessions, core) == [0, 0, 0]
            assert_publishers_retired(owners)
        finally:
            release.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    assert all(not status.running for status in topology.statuses)


def test_combined_response_benchmark_retains_checked_rejection_as_failed_case(tmp_path):
    from pllm.metrics import BenchmarkCase, BenchmarkSuite, BenchmarkTarget, Sampling, Scenario, benchmark_suite

    accepted = combined_experiment(tmp_path / "accepted")
    rejected = combined_experiment(tmp_path / "rejected", rejecting=True)
    # Prove this source admits and the first output succeeds; failure in the
    # benchmark's two-output request is not an unsupported-policy/admission skip.
    with OpenAI(experiment=rejected, timeout=300) as sdk:
        assert sdk.responses.create(input="hello", max_output_tokens=1).output_text == "world"
        assert active_sessions(sdk._core) == [0, 0, 0]
    # Four conservative process allowances plus request/stream owners exceed
    # 2 GiB even for this tiny graph. This is an explicit sampling budget, not
    # a native capacity change or a preflight/watchdog bypass.
    suite = BenchmarkSuite([
        BenchmarkCase("combined-accepted", BenchmarkTarget.response(accepted,
            scenario=Scenario.cold(max_output_tokens=2), backend="native")),
        BenchmarkCase("combined-rejected", BenchmarkTarget.response(rejected,
            scenario=Scenario.cold(max_output_tokens=2), backend="native")),
    ], Sampling(processes=1, timeout_seconds=300, memory_bytes=4 << 30))
    output = tmp_path / "response-report.json"
    report = benchmark_suite(suite, inputs={"combined-accepted": {"prompt": "hello hello"},
                                           "combined-rejected": {"prompt": "hello"}}, output=output)
    assert len(report.results) == 2 and output.is_file()
    positive, negative = (row.to_dict() for row in report.results)
    assert positive["status"] == "completed", positive
    assert negative["status"] == "failed", negative
    assert negative["execution"]["failure_code"] in {"target_execution_failed", "response_checks_failed"}
    assert "world" not in json.dumps(negative)
    metrics = {entry["id"]: entry for entry in positive["metrics"]}
    assert metrics["response.decode_tps"]["value"] is None
    assert metrics["response.decode_tps"]["unavailable_reason"] == "final_only_private_token_disclosure"
    assert positive["execution"]["details"]["runtime_contract"]["numeric"]["numeric_profile"] == (
        "puma.public-q18.checked-range.rmsnorm-epsilon1.full-prefix.final-greedy-ids.v1")
    details = positive["execution"]["details"]
    totals = details["runtime_audit"]["totals"]
    assert PEER_COUNTERS <= totals.keys()
    assert all(type(totals[key]) is int and totals[key] > 0 for key in PEER_COUNTERS)
    for role in ROLES:
        for phase in ("setup", "online"):
            for direction in ("upload", "download"):
                key = f"role_link.{role}.{phase}_{direction}_bytes"
                # Online control/finish RPCs deliberately upload empty bodies;
                # execution traffic goes directly between workers, not client.
                expected_zero = phase == "online" and direction == "upload"
                assert type(totals[key]) is int
                assert totals[key] == 0 if expected_zero else totals[key] > 0
    assert metrics["network.covered_body_bytes"]["value"] > 0
    assert metrics["request.0.decode_tps"]["value"] is None
    assert metrics["request.0.decode_tps"]["unavailable_reason"] == "final_only_private_token_disclosure"
    assert details["topology_admission"]["estimate"]["components"]["peer_stream_workspace_bytes_per_worker"] == STREAM_MEMORY
    for role in ("worker_a", "worker_b", "worker_c"):
        assert metrics[f"{role}.cold_cpu"]["value"] > 0
        assert metrics[f"{role}.peak_rss"]["value"] > 0
