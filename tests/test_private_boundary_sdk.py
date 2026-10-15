"""Generated checkpoints over actual five-role child processes and SDK lifecycle.

Run with the unified installed wheel, not pytest's source-tree native extension:
target/frontier-sdk-env/bin/python -m pytest -o pythonpath= tests/test_private_boundary_sdk.py
Transport observation below never changes requests, replies, or decoder execution.
"""
from __future__ import annotations

import asyncio
import json
import os
import threading
from collections import defaultdict

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from pllm import Deployment, ExecutionBudget, Experiment, Model
from pllm.profiles import MaskedLinearCpu
from pllm.roles import PreparedBoundaryRoles, PrivateLookupAndFullHead
from pllm.runtime.prepared_queue import PreparedQueueClosed
from pllm.runtime.private_boundary_protocol import ROLES, canonical, unpack_frame
from pllm.runtime.private_boundary_transport import PrivateBoundaryTransport
from pllm.runtime.servers import build_roles
from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint
from pllm.runtime.topology_accounting import BOUNDARY_COVERAGE, body_coverage_identity, private_boundary_body_accounting

pytestmark = pytest.mark.integration


def experiment(root, deployment, *, private, boundary_options=None):
    pipeline = MaskedLinearCpu(Model.path(str(root), model_id="boundary-sdk"),
        **({"boundary": PrivateLookupAndFullHead(**(boundary_options or {})),
            "topology": PreparedBoundaryRoles()} if private else {}))
    return Experiment("private-sdk" if private else "ordinary-sdk", pipeline,
        Deployment.local(root=str(deployment)),
        ExecutionBudget(requests=16, max_input_tokens=32, max_new_tokens=3))


def test_tiny_source_identity_survives_supervised_roles(tmp_path):
    from pllm.model_loader import resolve_model

    model = Model.tiny()
    selected = Experiment("tiny-boundary-source", MaskedLinearCpu(model,
        boundary=PrivateLookupAndFullHead(), topology=PreparedBoundaryRoles()),
        Deployment.local(root=str(tmp_path / "roles")),
        ExecutionBudget(requests=1, max_input_tokens=256, max_new_tokens=1))
    expected_source = resolve_model(model).source_lock_digest
    with build_roles(selected, engine_threads=1) as topology:
        with topology.client(bundle_cache_dir=str(tmp_path / "cache"),
                             background_inventory_refill=False) as client:
            response = client.responses.create(input="Hello", max_output_tokens=1, temperature=0)
            assert response.usage.output_tokens == 1
            bundle = client._core._transformer_state(topology.model_id).bundle
            assert bundle.manifest["metadata"]["source_lock_digest"] == expected_source
            assert bundle.manifest["metadata"]["source_lock"]["kind"] == "tiny"


@pytest.fixture(scope="module", params=[
    pytest.param((family, options), id=f"{family}-{label}")
    for family in ("qwen2", "qwen3", "llama")
    for label, options in (("serial-u32", {}), ("parallel-residues", {
        "head_dispatch": "parallel", "head_output_encoding": "uniform_residues"}))
])
def roles(request, tmp_path_factory):
    family, boundary_options = request.param
    folder = tmp_path_factory.mktemp("boundary-sdk-" + family)
    checkpoint = create_tiny_llama_checkpoint(folder / "model", model_type=family, seed=179,
        hidden_size=8, intermediate_size=16, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=4, vocab_size=258,
        with_qkv_bias=family == "qwen2", qk_norm=family == "qwen3")
    topology = build_roles(experiment(checkpoint, folder / "private", private=True,
        boundary_options=boundary_options), engine_threads=1)
    try:
        topology.start()
        statuses = topology.statuses
        assert {row.role for row in statuses} == {"inference", "preparation", *ROLES}
        assert all(row.running and row.pid != os.getpid() for row in statuses)
        assert len({row.pid for row in statuses}) == 4
        yield topology, checkpoint, folder
    finally:
        topology.close()
        assert topology.closed
        assert all(not row.running for row in topology.statuses)


@pytest.fixture
def exchanges(monkeypatch):
    records = []
    original = PrivateBoundaryTransport._post

    def observe(owner, index, path, payload, phase, **kwargs):
        status, reply = original(owner, index, path, payload, phase, **kwargs)
        records.append((ROLES[index], path, phase, bytes(payload), status, bytes(reply)))
        return status, reply

    monkeypatch.setattr(PrivateBoundaryTransport, "_post", observe)
    return records


def assert_private_client(client):
    state = client._core._transformer_state("boundary-sdk")
    assert state.bundle.cfg["vocab_size"] == 258
    assert state.bundle.stages["lm_head"].out_features == 258
    assert all(row.client_weight is None and row.client_aux_weight is None
               and row.client_weight_scales is None and row.client_aux_scales is None
               for row in state.bundle.stages.values())
    assert state.bundle.privacy["private_boundary"]["lookup"]["records"] == 258
    assert not state.bundle.privacy["dense_weights_in_bundle"]
    assert state.bundle.privacy["plaintext_activations_remote"] is False
    assert state.bundle.privacy["plaintext_logits_remote"] is False
    assert not state.token_cache


def assert_private_audit(client, *, closed=False):
    audit = client.privacy_audit.to_dict()
    for counter in ("plaintext_prompt_bytes_sent", "plaintext_token_ids_sent",
                    "preparation_requests_during_online"):
        assert audit[counter] == 0, counter
    assert audit["inference_stage_calls"] > 0
    assert audit["private_boundary_body_coverage_failures"] == 0
    assert client._core._online_active == 0
    assert not client._core._active_boundary_transports
    if closed:
        assert not client._core._transformer_states
        assert not client._core._transformer_conversations
    else:
        state = client._core._transformer_states["boundary-sdk"]
        assert state.active_prepared_responses == 0
        assert not state.token_cache
    return audit


def assert_capped(response):
    value = response if isinstance(response, dict) else response.to_dict()
    assert value["status"] == "incomplete"
    assert value["incomplete_details"] == {"reason": "max_output_tokens"}
    assert value["model"] == "boundary-sdk"
    assert value["usage"]["output_tokens"] == 3


def decoder_state(client, response):
    """Inspect retained actual results; never replace the decoder or its sampler."""
    saved = client._core._transformer_conversations[response.id]
    assert saved.next_logits.dtype == np.float32 and saved.next_logits.shape[-1] == 258
    return (saved.token_ids[:], saved.pending_token_ids[:], saved.snapshot.position,
        [saved.next_logits.copy(), *(value[:row.length].copy() for row in saved.snapshot.caches
                                    for value in (row.key, row.value))])


def assert_decoder_parity(actual, expected):
    assert actual[:3] == expected[:3]
    for left, right in zip(actual[3], expected[3], strict=True):
        np.testing.assert_array_equal(left.view(np.uint32), right.view(np.uint32))


def assert_boundary_sessions_retired(topology, records, action):
    opened = [(role, json.loads(payload)) for role, path, _, payload, status, _ in records
              if path == "/v1/private-boundary/sessions" and status == 200]
    assert len(opened) == 2
    for role, row in opened:
        path = f"/v1/private-boundary/sessions/{row['session']}/{action}"
        matching = [record for record in records if record[0] == role and record[1] == path]
        assert len(matching) == 1 and matching[0][4] == 200
        reply = json.loads(matching[0][5])
        assert reply["status"] == action
        url, key = topology.boundary_connections()[role]
        with httpx.Client(base_url=url) as verifier:
            replay = verifier.post(path, content=canonical({"schema": "pllm.private_boundary.v1." + action,
                "role": role, "session": row["session"], "epoch": row["epoch"]}),
                headers={"authorization": "Bearer " + key})
        assert replay.status_code >= 400  # Actual worker confirms owner was burned.


def assert_sender_ledger(client, records):
    audit = client.privacy_audit.to_dict()
    measured = defaultdict(int)
    for role, _, phase, payload, status, reply in records:
        assert status == 200
        measured[f"role_link.{role}.{phase}_upload_bytes"] += len(payload)
        measured[f"role_link.{role}.{phase}_download_bytes"] += len(reply)
        if phase == "online":
            frame = unpack_frame(payload)
            assert frame["operation"] in {"lookup", "head"}
            assert "ids" not in frame and "scales" not in frame
    for role in ROLES:
        for phase in ("setup", "online", "teardown"):
            for direction in ("upload", "download"):
                key = f"role_link.{role}.{phase}_{direction}_bytes"
                assert audit[key] == measured[key] > 0
    ledger = private_boundary_body_accounting({"privacy": audit})
    assert ledger["schema"] == "pllm.topology_body_accounting.v4"
    assert body_coverage_identity(ledger) == BOUNDARY_COVERAGE
    assert ledger["all_link_serialized_body_bytes"] == sum(edge["serialized_body_bytes"]
        for edge in ledger["body_bytes_by_edge"])
    assert ledger["online_all_link_serialized_body_bytes"] == (
        audit["inference_upload_bytes"] + audit["inference_download_bytes"]
        + sum(value for key, value in measured.items() if ".online_" in key))
    assert ledger["total_wire_bytes"] is None  # HTTP bodies are not TCP/header measurements.


def test_generated_five_role_sdk_matches_ordinary_full_vocab(roles, exchanges):
    topology, checkpoint, folder = roles
    with topology.client(bundle_cache_dir=folder / "cache-private", background_inventory_refill=False) as client:
        assert_private_client(client)
        startup = client.privacy_audit.to_dict()
        assert not exchanges  # No boundary operation has occurred before this snapshot.
        assert all(startup.get(f"role_link.{role}.{phase}_{direction}_bytes") == 0
            for role in ROLES for phase in ("setup", "online", "teardown")
            for direction in ("upload", "download"))
        startup_ledger = private_boundary_body_accounting({"privacy": startup}, initial_preparation=True)
        assert body_coverage_identity(startup_ledger) == BOUNDARY_COVERAGE
        assert startup_ledger["all_link_serialized_body_bytes"] == sum(
            edge["serialized_body_bytes"] for edge in startup_ledger["body_bytes_by_edge"])
        actual = client.responses.create(input="aba", max_output_tokens=3, temperature=0)
        assert_capped(actual)
        actual_state = decoder_state(client, actual)
        audit = assert_private_audit(client)
        assert audit["private_boundary_lookup_rows"] == actual.usage.input_tokens + actual.usage.output_tokens - 1
        assert audit["private_boundary_head_calls"] == actual.usage.output_tokens
        assert_boundary_sessions_retired(topology, exchanges, "complete")
        assert_sender_ledger(client, exchanges)
    with build_roles(experiment(checkpoint, folder / "ordinary", private=False), engine_threads=1) as ordinary:
        with ordinary.client(bundle_cache_dir=folder / "cache-ordinary", background_inventory_refill=False) as client:
            expected = client.responses.create(input="aba", max_output_tokens=3, temperature=0)
            expected_state = decoder_state(client, expected)
    assert_capped(expected)
    assert actual.output_text == expected.output_text
    assert actual.usage.to_dict() == expected.usage.to_dict()
    assert_decoder_parity(actual_state, expected_state)


@pytest.mark.parametrize("termination", ["partial-stream", "cancel-response", "close-client"])
def test_paused_boundary_stream_retires_both_real_workers(roles, exchanges, termination):
    topology, _, folder = roles
    client = topology.client(bundle_cache_dir=folder / "cache-lifecycle", background_inventory_refill=False)
    stream = None
    try:
        stream = client.responses.create(input="aba", max_output_tokens=3, temperature=0, stream=True)
        created = next(stream)
        assert created.type == "response.created"
        # Pause only after real boundary lookup/full-head work delivered output.
        for event in stream:
            assert event.type not in {"response.completed", "response.incomplete"}
            if event.type == "response.output_text.delta":
                break
        else:
            pytest.fail("stream produced no text delta before termination")
        online_exchanges = sum(row[2] == "online" for row in exchanges)
        if termination == "partial-stream":
            stream.close()
            stream.close()
        elif termination == "cancel-response":
            cancelled = client.responses.cancel(created.response["id"])
            assert cancelled.status == "cancelled"
            with pytest.raises(PreparedQueueClosed, match="cancelled"):
                list(stream)
            # Unsealed failed responses retire their private history and state.
            with pytest.raises(KeyError):
                client.responses.retrieve(cancelled.id)
        else:
            client.close()
            client.close()
            stream.close()
        assert_boundary_sessions_retired(topology, exchanges, "cancel")
        assert sum(row[2] == "online" for row in exchanges) == online_exchanges
        assert_private_audit(client, closed=termination == "close-client")
        if termination != "close-client":
            offset = len(exchanges)
            recovered = client.responses.create(input="aba", max_output_tokens=3, temperature=0)
            assert_capped(recovered)
            assert_boundary_sessions_retired(topology, exchanges[offset:], "complete")
            assert_private_audit(client)
    finally:
        if stream is not None:
            stream.close()
        client.close()


def test_gateway_request_uses_same_real_boundary_and_teardown(roles, exchanges):
    topology, _, folder = roles
    app = topology.gateway_app(local_api_key="boundary-gateway-test",
        bundle_cache_dir=folder / "cache-gateway", background_inventory_refill=False)
    with TestClient(app) as gateway:
        result = gateway.post("/v1/responses", headers={"authorization": "Bearer boundary-gateway-test"},
            json={"model": "boundary-sdk", "input": "aba", "max_output_tokens": 3, "temperature": 0})
        assert result.status_code == 200, result.text
        response = result.json()
        assert_capped(response)
    assert_boundary_sessions_retired(topology, exchanges, "complete")


def test_async_cancel_drains_parallel_head_pair_before_sdk_retirement(tmp_path, exchanges, monkeypatch):
    from pllm.runtime.client import AsyncResponsesResource

    checkpoint = create_tiny_llama_checkpoint(tmp_path / "model", model_type="qwen2", seed=179,
        hidden_size=8, intermediate_size=16, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=4, vocab_size=258,
        with_qkv_bias=True)
    selected = experiment(checkpoint, tmp_path / "async", private=True, boundary_options={
        "head_dispatch": "parallel", "head_output_encoding": "uniform_residues"})
    entered = [threading.Event(), threading.Event()]
    release = threading.Event()
    original = PrivateBoundaryTransport._post

    def blocked(owner, index, path, payload, phase, **kwargs):
        if path.endswith("/execute") and unpack_frame(payload)["operation"] == "head":
            entered[index].set()
            assert release.wait(15), "paired head exchange did not drain"
        return original(owner, index, path, payload, phase, **kwargs)

    monkeypatch.setattr(PrivateBoundaryTransport, "_post", blocked)
    with build_roles(selected, engine_threads=1) as topology, topology.client(
            bundle_cache_dir=tmp_path / "cache-async", background_inventory_refill=False) as client:
        async def run():
            task = asyncio.create_task(AsyncResponsesResource(client.responses).create(
                input="aba", max_output_tokens=3, temperature=0))
            try:
                assert all(await asyncio.gather(*(asyncio.to_thread(event.wait, 15) for event in entered)))
                task.cancel()
                await asyncio.sleep(0.05)
                assert not task.done(), "SDK cancellation escaped before paired calls drained"
                assert client._core._active_boundary_transports
                release.set()
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(task, 15)
            finally:
                release.set()
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)

        asyncio.run(run())
        assert_boundary_sessions_retired(topology, exchanges, "cancel")
        assert_private_audit(client)
        assert not client._core._transformer_conversations
        assert not any(thread.is_alive() and thread.name.startswith("pllm-boundary-head")
                       for thread in threading.enumerate())


def test_native_wan_five_role_response_conserves_boundary_directions(tmp_path, exchanges):
    """Actual native shared TCP pacing at default 100/40 Mbps, not estimates."""
    from pllm.deployment import WanConditions

    checkpoint = create_tiny_llama_checkpoint(tmp_path / "model", model_type="qwen2", seed=179,
        hidden_size=8, intermediate_size=16, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=4, vocab_size=258,
        with_qkv_bias=True)
    conditions = WanConditions()
    topology = build_roles(experiment(checkpoint, tmp_path / "wan", private=True,
        boundary_options={"head_dispatch": "parallel", "head_output_encoding": "uniform_residues"}),
        engine_threads=1, wan=conditions)
    try:
        with topology, topology.client(bundle_cache_dir=tmp_path / "cache-wan",
                background_inventory_refill=False) as client:
            assert len(topology.statuses) == 4 and all(row.running for row in topology.statuses)
            assert_private_client(client)
            result = client.responses.create(input="aba", max_output_tokens=3, temperature=0)
            assert_capped(result)
            audit = assert_private_audit(client)
            assert audit["private_boundary_head_calls"] == 3
            assert_boundary_sessions_retired(topology, exchanges, "complete")
            assert_sender_ledger(client, exchanges)
            sample = topology.resource_samples()["wan_emulation"]
            assert sample["backend"] == "native-shared-tcp-pacer" and sample["enforced"] is True
            assert sample["conditions_digest"] == conditions.digest
            assert set(sample["role_parties"]) == {"client", "inference", "preparation", *ROLES}
            streams = sample["directed_stream_bytes"]
            assert sample["total_stream_bytes"] == sum(streams.values()) > 0
            for role in ROLES:
                for source, target, direction in (("client", role, "upload"), (role, "client", "download")):
                    bodies = sum(audit[f"role_link.{role}.{phase}_{direction}_bytes"]
                                 for phase in ("setup", "online", "teardown"))
                    assert streams[f"{source}->{target}"] >= bodies > 0
            assignment = sample["role_parties"]
            for party, counters in sample["parties"].items():
                expected = {"upload": 0, "download": 0}
                for link, count in streams.items():
                    source, target = link.split("->")
                    if assignment[source] != assignment[target]:
                        if assignment[source] == party:
                            expected["upload"] += count
                        if assignment[target] == party:
                            expected["download"] += count
                for direction, rate in (("upload", 5_000_000), ("download", 12_500_000)):
                    assert counters[direction]["admitted_bytes"] == expected[direction]
                    assert counters[direction]["bytes_per_second"] == rate
    finally:
        topology.close()
        assert topology.closed and all(not row.running for row in topology.statuses)


def test_composed_artifacts_zlib_paged_demand_packed_residue_response(tmp_path, exchanges):
    """Keep the composed benchmark controls enabled on actual generated source."""
    from pllm.preparation import ModelAwareCorrections, PreparedInventory
    from pllm.protocols import ClientBundleTransport, MaskedLinear
    from pllm.quantization import SymmetricPerRow

    checkpoint = create_tiny_llama_checkpoint(tmp_path / "model", model_type="qwen2", seed=179,
        hidden_size=8, intermediate_size=16, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=4, vocab_size=258,
        with_qkv_bias=True)
    pipeline = MaskedLinearCpu(Model.path(str(checkpoint), model_id="boundary-sdk"),
        boundary=PrivateLookupAndFullHead(head_dispatch="parallel", head_output_encoding="uniform_residues"),
        topology=PreparedBoundaryRoles(),
        linear=MaskedLinear(prefill_pruning="terminal", request_encoding="stage_packed",
                            output_encoding="row_residues"),
        quantization=SymmetricPerRow(weight_bits=8, activation_bits=8, causal_reduction="prefix_f32"),
        preparation=ModelAwareCorrections(storage="paged"),
        inventory=PreparedInventory(refill="on-demand", allocation="demand", stage_window=1,
                                    stage_window_bytes=16 << 20),
        delivery=ClientBundleTransport("artifacts", compression="zlib", storage="paged"))
    composed = Experiment("composed-boundary-sdk", pipeline, Deployment.local(root=str(tmp_path / "composed")),
        ExecutionBudget(requests=2, max_input_tokens=32, max_new_tokens=3))
    topology = build_roles(composed, engine_threads=1)
    try:
        with topology, topology.client(bundle_cache_dir=tmp_path / "cache-composed",
                background_inventory_refill=False) as client:
            assert_private_client(client)
            state = client._core._transformer_state("boundary-sdk")
            assert state.bundle.weight_storage == "paged"
            assert client._core.bundle_compression == "artifacts-zlib"
            assert client._core.prepared_request_encoding == "stage_packed"
            assert state.bundle.privacy["prepared_output_encoding"] == "row_residues"
            body = [stage for stage in state.bundle.stages.values()
                    if stage.id not in {"token_lookup", "lm_head"}]
            assert body and all(stage.output_residue_bits is not None for stage in body)
            actual = client.responses.create(input="aba", max_output_tokens=3, temperature=0)
            assert_capped(actual)
            actual_state = decoder_state(client, actual)
            audit = assert_private_audit(client)
            assert audit["bundle_network_bytes"] > 0
            assert audit["prepared_stage_rows_issued"] > 0
            assert audit["prepared_stage_rows_reserved"] == (
                audit["prepared_stage_rows_claimed"] + audit["prepared_stage_rows_burned"])
            assert audit["private_boundary_head_calls"] == 3
            assert_boundary_sessions_retired(topology, exchanges, "complete")
            assert_sender_ledger(client, exchanges)
    finally:
        topology.close()
        assert topology.closed and all(not row.running for row in topology.statuses)
    comparator = MaskedLinearCpu(pipeline.model,
        quantization=SymmetricPerRow(weight_bits=8, activation_bits=8, causal_reduction="prefix_f32"))
    ordinary = Experiment("ordinary-composed-control", comparator,
        Deployment.local(root=str(tmp_path / "control")), ExecutionBudget(1, 32, 3))
    with build_roles(ordinary, engine_threads=1) as control:
        with control.client(bundle_cache_dir=tmp_path / "cache-control",
                background_inventory_refill=False) as client:
            expected = client.responses.create(input="aba", max_output_tokens=3, temperature=0)
            expected_state = decoder_state(client, expected)
    assert_capped(expected)
    assert actual.output_text == expected.output_text
    assert actual.usage.to_dict() == expected.usage.to_dict()
    assert_decoder_parity(actual_state, expected_state)
