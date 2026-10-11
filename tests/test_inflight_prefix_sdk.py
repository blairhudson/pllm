"""Lightweight SDK-owner checks; no checkpoints, builds, installs or timing runs."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from pllm.runtime.inflight_prefix import (
    InflightBinding, InflightCancelled, InflightPrefixOwner,
    InflightProducerFailed, InflightResources,
)
from pllm.runtime.transformer_client import LayerCache, RuntimeSnapshot, TransformerClientError, _snapshot_state_basis


BINDING = InflightBinding("a" * 64, "b" * 64, "c" * 64, "d" * 64, "e" * 64)
RESOURCES = InflightResources(2, 1, 2, 16, 32)


def result(ids=(1, 2), binding=BINDING):
    execution = hashlib.sha256(
        b"pllm.decoder.execution-basis.v1\0" + binding.native_prefill_digest.encode()
        + np.arange(len(ids), dtype="<i8").tobytes() + np.asarray(ids, dtype="<i8").tobytes()
    ).hexdigest()
    basis = _snapshot_state_basis("completed_prefill", len(ids), binding.state_binding_digest,
                                  binding.native_prefill_digest, execution, None, len(ids))
    keys = np.repeat(np.asarray(ids, dtype=np.float32), 2).reshape(-1, 1, 2)
    caches = [LayerCache(keys.copy(), (keys / 2).copy(), len(ids)) for _ in range(2)]
    return RuntimeSnapshot(len(ids), caches, {}, binding.state_binding_digest, basis), np.arange(16, dtype=np.float32)


def validate(snapshot):
    # Native equivalent geometry validator; actual seal/provenance checks are real.
    assert snapshot.semantic_binding_digest == BINDING.state_binding_digest
    assert snapshot.state_basis.valid()
    assert len(snapshot.caches) == 2


def make(**kwargs):
    client = object()
    return client, InflightPrefixOwner(client, **kwargs)


def register(owner, client, ids=(1, 2), **kwargs):
    return owner.lookup_or_register(client, BINDING, ids, RESOURCES,
                                    admit=kwargs.pop("admit", lambda p, s: None), **kwargs)


def publish(request, ids=(1, 2)):
    snapshot, logits = result(ids)
    request.publish(snapshot, logits, validate_snapshot=validate)
    request.finish_producer()
    return snapshot, logits


def assert_drained(owner):
    c = owner.counters()
    assert c["reserved_bytes"] == c["requests"] == c["nodes"] == c["active_producers"] == 0


def test_many_threads_join_one_pending_producer_independent_branches():
    client, owner = make(max_requests=32)
    gate = threading.Barrier(17)

    def enroll(i):
        request = register(owner, client, (1, 2, i % 13 + 3))
        gate.wait(timeout=5)
        return request

    producer = register(owner, client)
    with ThreadPoolExecutor(max_workers=16) as pool:
        tasks = [pool.submit(enroll, i) for i in range(16)]
        gate.wait(timeout=5)
        requests = [task.result(timeout=5) for task in tasks]
        assert all(not r.is_producer and r.prefix_tokens == 2 for r in requests)
        original, scores = publish(producer)
        futures = [pool.submit(r.wait) for r in [producer, *requests]]
        leases = [task.result(timeout=5) for task in futures]
    expected_peak = RESOURCES.producer_bytes(2) + 17 * RESOURCES.branch_bytes(2)
    assert owner.counters()["peak_reserved_bytes"] == expected_peak
    assert owner.counters()["publications"] == owner.counters()["producer_registrations"] == 1
    leases[0].snapshot.caches[0].key.fill(99)
    leases[0].logits.fill(99)
    original.caches[0].key.fill(88)
    scores.fill(88)
    assert all(x.snapshot.caches[0].key[0, 0, 0] == 1 and x.logits[0] == 0 for x in leases[1:])
    roots = producer._node.owned_arrays
    for lease in leases:
        lease.close()
        lease.close()
        assert all(not a.any() for a in lease._owned_arrays)
    assert all(not a.any() for a in roots)
    assert_drained(owner)


def test_longest_complete_pending_prefix_and_no_speculative_split():
    client, owner = make()
    first = register(owner, client, (1,))
    second = register(owner, client, (1, 2))  # joins existing full one-token prefill
    assert not second.is_producer and second.prefix_tokens == 1
    unrelated = register(owner, client, (2, 3))
    assert unrelated.is_producer
    for r in (first, unrelated):
        r.fail()
        r.finish_producer()
    for r in (first, second, unrelated):
        r.release()
    assert_drained(owner)


def test_completed_cache_and_publication_never_provide_extra_join():
    client, owner = make()
    assert register(owner, client, completed_cache_hit=True) is None
    assert owner.counters()["producer_registrations"] == 0
    producer = register(owner, client)
    publish(producer)
    next_request = register(owner, client)
    assert next_request.is_producer  # even while prior published branch remains live
    publish(next_request)
    producer.release()
    next_request.release()
    assert owner.counters()["cache_bypasses"] == 1
    assert_drained(owner)


def test_one_waiter_cancellation_leaves_producer_and_survivor():
    client, owner = make()
    producer = register(owner, client)
    cancelled = register(owner, client)
    survivor = register(owner, client)
    with ThreadPoolExecutor(max_workers=2) as pool:
        wait = pool.submit(cancelled.wait)
        cancelled.cancel()
        with pytest.raises(InflightCancelled):
            wait.result(timeout=5)
        assert not producer.producer_cancel.is_set()
        # Even producer response cancellation cannot erase another waiter.
        producer.cancel()
        assert not producer.producer_cancel.is_set()
        publish(producer)
        lease = survivor.wait()
        assert lease.snapshot.position == 2
        lease.close()
    assert_drained(owner)


def test_last_cancel_keeps_retiring_producer_counted_until_safe_completion():
    client, owner = make(max_nodes=1)
    producer = register(owner, client)
    producer.cancel()
    assert producer.producer_cancel.is_set()
    c = owner.counters()
    assert c["requests"] == 0 and c["nodes"] == c["active_producers"] == 1
    assert c["reserved_bytes"] == RESOURCES.producer_bytes(2)
    assert register(owner, client) is None
    with pytest.raises(InflightCancelled):
        producer.publish(*result(), validate_snapshot=validate)
    producer.finish_producer()
    producer.finish_producer()
    assert_drained(owner)


def test_failure_broadcast_and_fresh_retry_not_poisoned_by_old_node():
    client, owner = make()
    producer = register(owner, client)
    waiter = register(owner, client)
    producer.fail()
    fresh = register(owner, client)
    assert fresh.is_producer
    # Retirement of same-key older node must not remove new pending entry.
    producer.finish_producer()
    producer.release()
    with pytest.raises(InflightProducerFailed):
        waiter.wait()
    joined = register(owner, client)
    assert not joined.is_producer
    publish(fresh)
    joined.wait().close()
    fresh.release()
    assert_drained(owner)


def test_finish_without_publication_broadcasts_failure():
    client, owner = make()
    producer = register(owner, client)
    waiter = register(owner, client)
    producer.finish_producer()
    with pytest.raises(InflightProducerFailed):
        waiter.wait()
    producer.release()
    assert_drained(owner)


def test_timeout_cancels_only_timed_waiter():
    client, owner = make()
    producer = register(owner, client)
    waiter = register(owner, client)
    with pytest.raises(TimeoutError):
        waiter.wait(timeout=0)
    assert not producer.producer_cancel.is_set()
    publish(producer)
    producer.wait().close()
    assert_drained(owner)


def test_close_wakes_waiters_and_drains_active_runtime_and_lease_scopes():
    client, owner = make()
    producer = register(owner, client)
    waiter = register(owner, client)
    with ThreadPoolExecutor(max_workers=2) as pool:
        future = pool.submit(waiter.wait)
        assert not owner.close(wait=False)
        with pytest.raises(InflightCancelled):
            future.result(timeout=5)
        assert producer.producer_cancel.is_set()
        assert owner.counters()["reserved_bytes"] == RESOURCES.producer_bytes(2)
        producer.finish_producer()
        assert owner.close(timeout=0)
    assert not any(owner._secret)
    assert_drained(owner)
    with pytest.raises(InflightCancelled):
        register(owner, client)


def test_close_does_not_zeroize_branch_while_branch_runtime_active():
    client, owner = make()
    producer = register(owner, client)
    publish(producer)
    lease = producer.wait()
    assert not owner.close(timeout=0)
    assert lease.logits[1] == 1
    assert owner.counters()["requests"] == 1
    lease.close()  # caller reset branch runtime first
    assert owner.close(timeout=0)
    assert not lease.logits.any()
    assert_drained(owner)


def test_same_request_concurrent_wait_is_one_copy():
    client, owner = make()
    producer = register(owner, client)
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(producer.wait) for _ in range(8)]
        publish(producer)
        leases = [f.result(timeout=5) for f in futures]
    assert all(lease is leases[0] for lease in leases)
    assert owner.counters()["copy_bytes"] == 2 * RESOURCES.payload_bytes(2)
    producer.release()
    assert_drained(owner)


@pytest.mark.parametrize("mutation", ["token", "phase", "native", "source", "owner", "seal", "verified",
                                      "nan", "shape", "alias", "logits", "shared", "capacity"])
def test_bad_provenance_numeric_ownership_broadcasts_failure(mutation):
    client, owner = make()
    producer = register(owner, client)
    waiter = register(owner, client)
    snapshot, logits = result((2, 1) if mutation == "token" else (1, 2))
    basis = snapshot.state_basis
    if mutation in {"phase", "native", "source", "owner", "verified"}:
        snapshot.state_basis = _snapshot_state_basis(
            "incremental" if mutation == "phase" else basis.phase, basis.position,
            "f" * 64 if mutation == "source" else basis.source_binding_digest,
            "f" * 64 if mutation == "native" else basis.native_phase_digest,
            basis.execution_digest, "response" if mutation == "owner" else None, basis.prefill_extent,
            verification_failure_bits=1 if mutation == "verified" else 0,
            verification_lineage=1 if mutation == "verified" else 0,
            verification_inventory_digest="f" * 64 if mutation == "verified" else "")
    elif mutation == "seal":
        snapshot.state_basis = replace(basis, seal="0" * 64)
    elif mutation == "nan":
        snapshot.caches[0].key[0, 0, 0] = np.nan
    elif mutation == "shape":
        snapshot.caches[0].key = np.zeros((2, 1, 3), np.float32)
    elif mutation == "alias":
        snapshot.caches[0].value = snapshot.caches[0].key
    elif mutation == "logits":
        logits = logits.astype(np.float64)
    elif mutation == "shared":
        snapshot.shared_kv["x"] = (np.ones(2), np.ones(2))
    elif mutation == "capacity":
        snapshot.caches[0].key = np.zeros((32, 1, 2), np.float32)
    with pytest.raises(ValueError):
        producer.publish(snapshot, logits, validate_snapshot=validate)
    producer.finish_producer()
    producer.release()
    with pytest.raises(InflightProducerFailed):
        waiter.wait()
    assert_drained(owner)


def test_publication_mutation_detected_and_owned_roots_erased():
    client, owner = make()
    producer = register(owner, client)
    waiter = register(owner, client)
    publish(producer)
    roots = producer._node.owned_arrays
    roots[0].setflags(write=True)
    roots[0].fill(7)
    with pytest.raises(ValueError, match="mutated"):
        waiter.wait()
    producer.release()
    assert all(not a.any() for a in roots)
    assert_drained(owner)


def test_branch_descriptor_mutation_never_erases_foreign_array():
    client, owner = make()
    producer = register(owner, client)
    publish(producer)
    lease = producer.wait()
    roots = lease._owned_arrays
    foreign = np.ones((2, 1, 2), np.float32)
    lease.snapshot.caches[0].key = foreign
    lease.logits = np.ones(16, np.float32)
    lease.close()
    assert foreign.all() and lease.logits.all()
    assert all(not a.any() for a in roots)
    assert_drained(owner)


def test_caps_and_native_admission_fail_before_allocation():
    client, owner = make(max_bytes=RESOURCES.producer_bytes(2) + RESOURCES.branch_bytes(2) - 1)
    calls = []
    assert register(owner, client, admit=lambda *args: calls.append(args)) is None
    assert not calls and owner.counters()["producer_registrations"] == 0
    client, owner = make()
    def deny(p, s):
        raise MemoryError("native resource limit")
    with pytest.raises(MemoryError):
        register(owner, client, admit=deny)
    assert_drained(owner)
    producer = register(owner, client)
    waiter = register(owner, client, (1, 2, 3), admit=lambda p, s: calls.append((p, s)))
    assert calls == [(2, 1)]
    producer.finish_producer()
    producer.release()
    waiter.release()


def test_bindings_and_client_capability_do_not_cross_share():
    client, owner = make()
    with pytest.raises(ValueError, match="trusted client"):
        register(owner, object())
    first = register(owner, client)
    for field in BINDING.__dataclass_fields__:
        binding = replace(BINDING, **{field: "f" * 64})
        other = owner.lookup_or_register(client, binding, (1, 2), RESOURCES, admit=lambda p, s: None)
        assert other.is_producer
        other.finish_producer()
        other.release()
    first.finish_producer()
    first.release()
    assert_drained(owner)


@pytest.mark.parametrize("ids", [(), (True,), (1.5,), (-1,), (16,), (1 << 32,), tuple([1] * 4097)])
def test_invalid_tokens_reject_without_scope(ids):
    client, owner = make()
    with pytest.raises(ValueError):
        register(owner, client, ids)
    assert_drained(owner)


@pytest.mark.parametrize("kwargs", [{"max_requests": True}, {"max_requests": 65},
                                    {"max_bytes": 0}, {"max_bytes": (256 << 20) + 1}, {"max_nodes": 0}])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        make(**kwargs)


def test_counters_no_private_values_and_bounded_fields():
    client, owner = make()
    producer = register(owner, client)
    publish(producer)
    producer.release()
    counters = owner.counters()
    assert all(type(value) is int for value in counters.values())
    assert not {"tokens", "digest", "key", "logits", "execution_digest"} & counters.keys()
    assert_drained(owner)


def test_branch_copy_failure_does_not_poison_other_waiter(monkeypatch):
    client, owner = make()
    producer = register(owner, client)
    failed = register(owner, client)
    survivor = register(owner, client)
    publish(producer)
    copy = owner._copy
    def broken(*args):
        raise MemoryError("branch allocation failure")
    monkeypatch.setattr(owner, "_copy", broken)
    with pytest.raises(MemoryError):
        failed.wait()
    monkeypatch.setattr(owner, "_copy", copy)
    assert owner.counters()["failures"] == 0
    survivor.wait().close()
    producer.release()
    assert_drained(owner)


def test_release_while_waiting_wakes_without_leaking():
    client, owner = make()
    producer = register(owner, client)
    waiter = register(owner, client)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(waiter.wait)
        waiter.release()
        with pytest.raises(InflightCancelled):
            future.result(timeout=5)
    producer.finish_producer()
    producer.release()
    assert_drained(owner)


def test_blocking_close_drains_after_active_producer_finally():
    client, owner = make()
    producer = register(owner, client)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(owner.close)
        assert producer.producer_cancel.wait(timeout=5)
        assert not future.done()
        producer.finish_producer()
        assert future.result(timeout=5)
    assert_drained(owner)


def test_native_validator_rejects_and_producer_copy_is_cleaned():
    client, owner = make()
    producer = register(owner, client)
    calls = []
    def reject_copy(snapshot):
        calls.append(snapshot)
        if len(calls) == 2:
            raise ValueError("native shape/source rejection")
    with pytest.raises(ValueError, match="native shape"):
        producer.publish(*result(), validate_snapshot=reject_copy)
    assert all(not a.any() for row in calls[1].caches for a in (row.key, row.value))
    producer.finish_producer()
    producer.release()
    assert_drained(owner)


def mock_runtime(**kwargs):
    values = dict(_causal_reduction="prefix_f32", _mlp_tiling=None, nonlinear_evaluator=None,
                  _verification_bits=0, _source_plan=SimpleNamespace(digest="a" * 64),
                  _graphs={"prefill": {"public": "graph"}},
                  _schedule={"composition_digest": "b" * 64, "prefill": {"public": "schedule"}},
                  _state_binding_digest="d" * 64)
    return SimpleNamespace(**dict(values, **kwargs))


def test_binding_from_runtime_exact_phase_identity_and_no_runtime_retention():
    runtime = mock_runtime()
    binding = InflightBinding.from_runtime(runtime, source_digest="a" * 64,
                                          composition_digest="b" * 64, compiled_digest="c" * 64)
    runtime._graphs["prefill"]["public"] = "changed"
    changed = InflightBinding.from_runtime(runtime, source_digest="a" * 64,
                                          composition_digest="b" * 64, compiled_digest="c" * 64)
    assert binding.native_prefill_digest != changed.native_prefill_digest
    assert not hasattr(binding, "runtime")


@pytest.mark.parametrize("field,value", [("_causal_reduction", None), ("_mlp_tiling", object()),
                                          ("nonlinear_evaluator", object()), ("_verification_bits", 1),
                                          ("_capsule", object()), ("_tiered", object()),
                                          ("_retained", object()), ("_window_contracts", {1: 1}),
                                          ("_hybrid_contracts", {1: 1})])
def test_unsupported_runtime_contract_rejected(field, value):
    with pytest.raises(ValueError, match="untiled unverified"):
        InflightBinding.from_runtime(mock_runtime(**{field: value}), source_digest="a" * 64,
                                     composition_digest="b" * 64, compiled_digest="c" * 64)


def test_request_bound_rejects_before_native_admission():
    client, owner = make(max_requests=1)
    producer = register(owner, client)
    calls = []
    assert register(owner, client, admit=lambda *args: calls.append(args)) is None
    assert not calls
    producer.finish_producer()
    producer.release()
    assert_drained(owner)


def test_published_arrays_survive_cancel_until_producer_runtime_reset():
    client, owner = make()
    producer = register(owner, client)
    snapshot, logits = result()
    producer.publish(snapshot, logits, validate_snapshot=validate)
    roots = producer._node.owned_arrays
    producer.cancel()
    assert producer.producer_cancel.is_set()
    assert roots[-1][1] == 1
    assert owner.counters()["reserved_bytes"] == RESOURCES.producer_bytes(2)
    producer.finish_producer()
    assert all(not a.any() for a in roots)
    assert_drained(owner)


def test_concurrent_register_cancel_publish_finish_stress():
    # Repeat race orderings without sleeps, timing measurements or native work.
    for _ in range(20):
        client, owner = make(max_requests=16)
        producer = register(owner, client)
        requests = [register(owner, client) for _ in range(12)]
        barrier = threading.Barrier(13)
        def branch(request):
            barrier.wait(timeout=5)
            try:
                request.wait().close()
            except InflightCancelled:
                pass
            finally:
                request.release()
        with ThreadPoolExecutor(max_workers=12) as pool:
            futures = [pool.submit(branch, request) for request in requests]
            barrier.wait(timeout=5)
            for request in requests[::2]:
                request.cancel()
            publish(producer)
            producer.release()
            for future in futures:
                future.result(timeout=5)
        assert_drained(owner)


def mock_compiled(monkeypatch):
    from pllm.configuration import Model
    from pllm.profiles import MaskedLinearCpu
    from pllm.quantization import SymmetricPerRow
    from pllm.runtime.model_binding import CompiledRuntimeModel

    composition = MaskedLinearCpu(Model.path("unused-generated-metadata"),
                                 quantization=SymmetricPerRow(causal_reduction="prefix_f32"))
    def states(capacity):
        return [dict(kind=kind, layer=layer, shape=[1, 1, capacity, 2], maximum_sequence=capacity)
                for layer in range(2) for kind in ("key", "value")]
    graphs = dict(config_digest="0" * 64,
                  prefill=dict(maximum_key_sequence=8, query_sequence=8, operations=[],
                               state_inputs=[], state_outputs=states(8)),
                  decode=dict(maximum_key_sequence=16, query_sequence=1, operations=[],
                              state_inputs=states(16), state_outputs=states(16)))
    schedule = dict(composition_digest=composition.digest(), prefill={"steps": []}, decode={"steps": []})
    document = dict(mode="full_kv_batched_suffix", source_plan_digest="a" * 64,
                    composition_digest=composition.digest(), source_schedule_digest="f" * 64, token_bound=8)
    class Plan:
        digest = "a" * 64
        def to_dict(self):
            return graphs
        def continuation_schedule(self, pipeline):
            assert pipeline.canonical_bytes() == composition.canonical_bytes()
            return SimpleNamespace(to_dict=lambda: document)
    compiled = object.__new__(CompiledRuntimeModel)
    compiled._plan = Plan()
    compiled._bundle = SimpleNamespace(cfg=dict(vocab_size=16, num_hidden_layers=2,
                                               num_key_value_heads=1, head_dim=2),
                                       privacy=dict(body_fingerprint="b" * 64, stage_commitment="c" * 64))
    compiled._fingerprint, compiled._digest = "b" * 64, "c" * 64
    compiled._runtime_schedule_digest = "f" * 64
    compiled._canonical_composition = composition.canonical_bytes()
    monkeypatch.setattr(CompiledRuntimeModel, "_validated_schedule", lambda self: schedule)
    def no_runtime(*args, **kwargs):
        raise AssertionError("compiled helpers must not allocate runtime")
    monkeypatch.setattr(CompiledRuntimeModel, "runtime", no_runtime)
    return compiled, graphs, schedule, document


def test_compiled_helpers_match_runtime_metadata_and_use_decode_capacity(monkeypatch):
    from pllm.runtime.model_binding import _state_binding_digest

    compiled, graphs, schedule, _ = mock_compiled(monkeypatch)
    binding = InflightBinding.from_compiled(compiled)
    resources = InflightResources.from_compiled(compiled)
    runtime = mock_runtime(_graphs=graphs, _schedule=schedule, _source_plan=compiled._plan,
                          _state_binding_digest=_state_binding_digest(compiled._plan, compiled._bundle,
                                                                     schedule["composition_digest"]))
    assert binding == InflightBinding.from_runtime(runtime, source_digest=compiled.bundle_fingerprint,
                                                  composition_digest=schedule["composition_digest"],
                                                  compiled_digest=compiled.digest)
    assert resources == InflightResources(2, 1, 2, 16, 16)
    assert resources.runtime_overlap_bytes == 2 * 16 * resources.kv_row_bytes + resources.logits_bytes
    assert resources.capacity_tokens != graphs["prefill"]["query_sequence"]
    client, owner = make()
    producer = owner.lookup_or_register(client, binding, (1, 2), resources, admit=lambda p, s: None)
    waiter = owner.lookup_or_register(client, binding, (1, 2, 3), resources, admit=lambda p, s: None)
    producer.publish(*result(binding=binding), validate_snapshot=lambda saved: None)
    producer.finish_producer()
    waiter.wait().close()
    producer.release()
    assert_drained(owner)


@pytest.mark.parametrize("mutation", ["heterogeneous", "hybrid", "shared_batch", "duplicate", "missing_pair",
                                      "noncontiguous_layers", "capacity", "bundle_heads", "bundle_layers",
                                      "native_mode", "native_source", "native_schedule", "native_composition",
                                      "native_tokens", "verified", "tiling", "retained"])
def test_compiled_helpers_reject_unsupported_geometry_and_contract(monkeypatch, mutation):
    compiled, graphs, schedule, document = mock_compiled(monkeypatch)
    if mutation == "heterogeneous":
        graphs["decode"]["state_inputs"][0]["shape"][1] = 2
    elif mutation == "hybrid":
        graphs["decode"]["state_inputs"][0]["kind"] = "recurrent"
    elif mutation == "shared_batch":
        graphs["decode"]["state_inputs"][0]["shape"][0] = 2
    elif mutation == "duplicate":
        graphs["prefill"]["state_outputs"].append(dict(graphs["prefill"]["state_outputs"][0]))
    elif mutation == "missing_pair":
        graphs["decode"]["state_outputs"].pop()
    elif mutation == "noncontiguous_layers":
        for phase in ("prefill", "decode"):
            for direction in ("state_inputs", "state_outputs"):
                for row in graphs[phase][direction]:
                    row["layer"] += 1
    elif mutation == "capacity":
        graphs["decode"]["maximum_key_sequence"] = 17
    elif mutation == "bundle_heads":
        compiled._bundle.cfg["num_key_value_heads"] = 2
    elif mutation == "bundle_layers":
        compiled._bundle.cfg["num_hidden_layers"] = 3
    elif mutation.startswith("native_"):
        key = {"native_mode": "mode", "native_source": "source_plan_digest",
               "native_schedule": "source_schedule_digest", "native_composition": "composition_digest",
               "native_tokens": "token_bound"}[mutation]
        document[key] = 9 if mutation == "native_tokens" else "wrong"
    elif mutation == "verified":
        compiled._bundle.privacy["verification_target_failure_bits"] = 1
    elif mutation == "tiling":
        schedule["mlp_tiling"] = {}
    elif mutation == "retained":
        schedule["retained_state"] = {}
    for helper in (InflightBinding.from_compiled, InflightResources.from_compiled):
        with pytest.raises((ValueError, KeyError, TransformerClientError)):
            helper(compiled)


def test_compiled_helpers_revalidate_bound_metadata_first(monkeypatch):
    from pllm.runtime.model_binding import CompiledRuntimeModel
    compiled, _, _, _ = mock_compiled(monkeypatch)
    def changed(self):
        raise ValueError("bound plan or bundle changed since compilation")
    monkeypatch.setattr(CompiledRuntimeModel, "_validated_schedule", changed)
    for helper in (InflightBinding.from_compiled, InflightResources.from_compiled):
        with pytest.raises(ValueError, match="changed since compilation"):
            helper(compiled)


@pytest.mark.parametrize("field,value", [("causal_reduction", None), ("verification_component", "verified"),
                                          ("verification_target_failure_bits", 1), ("nonlinear_layout", "shared")])
def test_compiled_helpers_reject_unsupported_resolved_composition(monkeypatch, field, value):
    import pllm.profiles as profiles
    from pllm.configuration import Pipeline

    compiled, _, _, _ = mock_compiled(monkeypatch)
    composition = Pipeline.from_spec(json.loads(compiled._canonical_composition))
    options = replace(profiles.resolve_runtime_composition(composition), **{field: value})
    monkeypatch.setattr(profiles, "resolve_runtime_composition", lambda pipeline: options)
    for helper in (InflightBinding.from_compiled, InflightResources.from_compiled):
        with pytest.raises(ValueError, match="untiled unverified"):
            helper(compiled)


def test_producer_release_keeps_publication_and_full_producer_overlap_until_reset():
    client, owner = make()
    producer = register(owner, client)
    waiter = register(owner, client)
    producer.publish(*result(), validate_snapshot=validate)
    roots = producer._node.owned_arrays
    producer.release()  # No lease; producer runtime still executing.
    assert owner.counters()["reserved_bytes"] == RESOURCES.producer_bytes(2) + RESOURCES.branch_bytes(2)
    assert roots[-1][1] == 1 and not producer.producer_cancel.is_set()
    lease = waiter.wait()
    assert not owner.close(wait=False)
    assert roots[-1][1] == lease.logits[1] == 1
    lease.close()  # Branch runtime reset; producer still owns counted overlap.
    assert owner.counters()["reserved_bytes"] == RESOURCES.producer_bytes(2)
    assert roots[-1][1] == 1
    producer.finish_producer()  # Producer runtime reset completes safe retirement.
    assert not any(a.any() for a in roots)
    assert owner.close(timeout=0)
    assert_drained(owner)


def generated_native_worker(family):
    """Parent-only generated parity worker; all-link/material claims excluded."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from probe_inflight_prefix import inflight_generated
    from pllm.runtime.model_binding import compile_runtime_model
    from pllm.runtime.inflight_prefix import _erase
    from pllm.state import ClientPrefixReuse

    with tempfile.TemporaryDirectory(prefix="pllm-inflight-sdk-") as folder, ExitStack() as cleanup:
        fixture = inflight_generated(Path(folder) / "model", family, 12, 2)
        composition = fixture.composition.with_params(cache=ClientPrefixReuse(
            max_bytes=1 << 20, fixed_input_tokens=12, inflight=True))
        compiled = compile_runtime_model(fixture.plan, fixture.bundle, composition=composition)
        # Derive both helpers before FIRST runtime/remote callback allocation.
        binding = InflightBinding.from_compiled(compiled)
        resources = InflightResources.from_compiled(compiled)
        assert resources.capacity_tokens == compiled._plan.to_dict()["decode"]["maximum_key_sequence"]
        def runtime():
            r = compiled.runtime(fixture.remote)
            cleanup.callback(r.reset)
            assert binding == InflightBinding.from_runtime(r, source_digest=compiled.bundle_fingerprint,
                composition_digest=composition.digest(), compiled_digest=compiled.digest)
            r.install_continuation(compiled._plan.continuation_schedule(composition))
            return r
        client = object()
        owner = InflightPrefixOwner(client)
        cleanup.callback(owner.close, wait=False)
        prefix = [1, 2, 3, 4, 5, 6, 7, 8]
        producer = owner.lookup_or_register(client, binding, prefix, resources, admit=lambda p, s: None)
        cleanup.callback(producer.release)
        cleanup.callback(producer.finish_producer)
        native = compiled._plan.continuation_schedule(composition)
        def admit(p, s):
            if s:
                native.admit(p, s)
        requests = []
        prompts = [prefix, prefix + [9, 10, 11, 12]]
        for prompt in prompts:
            request = owner.lookup_or_register(client, binding, prompt, resources, admit=admit)
            assert not request.is_producer and request.prefix_tokens == len(prefix)
            cleanup.callback(request.release)
            requests.append(request)
        r = runtime()
        _, scores, _ = r.prepare_ids(prefix)
        original, logits = r.snapshot(), scores.copy()
        cleanup.callback(_erase, original, logits)
        assert original.state_basis.native_phase_digest == binding.native_prefill_digest
        producer.publish(original, logits, validate_snapshot=r._validate_full_snapshot)
        _erase(original, logits)
        r.reset()
        producer.finish_producer()
        producer.release()
        for request, prompt in zip(requests, prompts, strict=True):
            lease = request.wait()
            branch, cold = runtime(), runtime()
            branch.restore(lease.snapshot)
            scores = lease.logits.copy()
            if len(prompt) > len(prefix):
                scores = branch.continue_ids(prompt[len(prefix):])[-1].copy()
            _, golden, _ = cold.prepare_ids(prompt)
            assert np.array_equal(scores.view(np.uint32), golden.view(np.uint32))
            assert branch.position == cold.position == len(prompt)
            for left, right in zip(branch.caches, cold.caches, strict=True):
                for a, b in ((left.key[:left.length], right.key[:right.length]),
                             (left.value[:left.length], right.value[:right.length])):
                    assert np.array_equal(a.view(np.uint32), b.view(np.uint32))
            selected = int(np.argmax(scores))
            next_scores = branch.forward_ids([selected])[-1]
            next_golden = cold.forward_ids([selected])[-1]
            assert np.array_equal(next_scores.view(np.uint32), next_golden.view(np.uint32))
            assert branch.position == cold.position == len(prompt) + 1
            # Last newly selected output token stays pending, not executed in KV.
            assert int(np.argmax(next_scores)) == int(np.argmax(next_golden))
            for left, right in zip(branch.caches, cold.caches, strict=True):
                for a, b in ((left.key[:left.length], right.key[:right.length]),
                             (left.value[:left.length], right.value[:right.length])):
                    assert np.array_equal(a.view(np.uint32), b.view(np.uint32))
            branch.reset()
            cold.reset()
            scores.fill(0)
            lease.close()
        assert owner.close(timeout=0)
        assert_drained(owner)
        return dict(family=family, metadata_matches_runtime=True, exact_prefill_and_joined_logits_kv=True,
                    pending_token_not_executed=True, owner_drained=True)


@pytest.mark.skipif(os.environ.get("PLLM_INFLIGHT_SDK_NATIVE") != "1",
                    reason="parent runs generated parity after final installed wheel")
@pytest.mark.parametrize("family", ["qwen2", "qwen3", "llama"])
def test_generated_compiled_metadata_exact_native_prefill_and_join(family):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from frontier_probe_support import fresh_workers
    with fresh_workers(Path(__file__).resolve(), memory_bytes=512 << 20, timeout=240) as (run, _, _):
        report = run({"family": family})
    assert report == dict(family=family, metadata_matches_runtime=True, exact_prefill_and_joined_logits_kv=True,
                          pending_token_not_executed=True, owner_drained=True)


if __name__ == "__main__":
    if sys.argv[1:] != ["--worker"]:
        raise SystemExit("only parent-generated --worker invocation is supported")
    print(json.dumps(generated_native_worker(json.load(sys.stdin)["family"])))
