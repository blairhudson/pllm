"""Batch audit integration without native owners, model files or sockets."""

import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from pllm import Deployment, ExecutionBudget, Experiment, Model
from pllm.profiles import OpenWeightVerifiableCpu
from pllm.protocols import OpenWeightVerifiableOutsourcing
from pllm.runtime import delegated_client as adapter
from pllm.runtime import delegated_transport as transport
from pllm.runtime.client import ProtocolError
from pllm.runtime.delegated_protocol import BATCH_EXECUTION
from pllm.runtime.delegated_decoder import GkmDecoderResult


@pytest.fixture(params=[None, BATCH_EXECUTION], ids=["scalar", "batch"])
def client(request, monkeypatch):
    monkeypatch.setattr(Experiment, "resolve", lambda _: SimpleNamespace(
        client_runtime="compiled_gkm_v1", model="fake-gkm"))
    params = {} if request.param is None else {"execution": request.param}
    experiment = Experiment("batch-audit", OpenWeightVerifiableCpu(Model.path("/unused"),
        linear=OpenWeightVerifiableOutsourcing(**params)), Deployment.local(root="/unused"),
        ExecutionBudget(requests=4, max_input_tokens=2, max_new_tokens=1))
    core = adapter.DelegatedClient(experiment, {
        "worker_a": ("http://127.0.0.1:1", "a" * 32),
        "worker_b": ("http://127.0.0.1:2", "b" * 32)})
    yield core, request.param
    core.close()


def peers(batch=True):
    owner = object.__new__(transport._Peers)
    owner.roles = ("worker_a", "worker_b")
    owner.counts = {f"role_link.{role}.{phase}_{direction}_bytes": 0
        for role in owner.roles for phase in ("setup", "online", "teardown")
        for direction in ("upload", "download")}
    owner.phase_bytes = {"prefill": 0, "decode": 0}
    owner.unknown = set()
    owner.calls, owner.server_ns = 4, 23
    owner.calls_known = owner.server_known = True
    owner.batch = {"descriptors": [{"rows": 2}, {"rows": 1}]} if batch else None
    owner.batch_index = owner.batch_exchanges = 2
    owner.scalar_claims = 3
    return owner


def test_real_transport_snapshot_passes_selected_schema_and_exports_distinct_totals(client):
    core, execution = client
    batch = execution is not None
    owner = peers(batch)
    snapshot = owner.audit()
    attempt = adapter._Attempt(core, {})
    attempt.record_audit(snapshot)
    assert attempt.transport_audit == snapshot
    core.audit.begin_transport()
    core.audit.finish_transport(attempt.transport_audit)
    exported = core.audit.to_dict()
    assert exported["inference_stage_calls"] == 4
    if batch:
        assert exported["batch_exchanges"] == 2
        assert exported["native_scalar_claims"] == 3
        assert exported["online_steps"] == 3
        assert core.audit.batch_exchanges == 2 and core.audit.native_scalar_claims == 3
    else:
        assert exported["online_steps"] == 2
        assert core.counter_fields == adapter._COUNTER_FIELDS
        assert "batch_exchanges" not in exported and "native_scalar_claims" not in exported


def test_batch_counter_totals_and_missing_measurements_stay_unknown(client):
    core, execution = client
    if execution is None:
        return
    snapshot = peers().audit()
    for _ in range(2):
        core.audit.begin_transport()
        core.audit.finish_transport(snapshot)
    assert core.audit.to_dict()["batch_exchanges"] == 4
    assert core.audit.to_dict()["native_scalar_claims"] == 6
    core.audit.begin_transport()
    core.audit.finish_transport({"inference_stage_calls": 1})
    assert "batch_exchanges" not in core.audit.to_dict()
    assert "native_scalar_claims" not in core.audit.to_dict()
    core.audit.finish_transport(snapshot)
    assert "batch_exchanges" not in core.audit.to_dict()


@pytest.mark.parametrize("name", ["batch_exchanges", "native_scalar_claims"])
@pytest.mark.parametrize("bad", [True, -1, 1.0, None, 1 << 63])
def test_new_counters_retain_exact_integer_and_bound_validation(client, name, bad):
    core, _ = client
    attempt = adapter._Attempt(core, {})
    with pytest.raises(ValueError, match="counter snapshot"):
        attempt.record_audit({name: bad})
    assert attempt.transport_audit is None


def test_snapshot_stays_single_use_and_selector_specific(client):
    core, execution = client
    attempt = adapter._Attempt(core, {})
    unsupported = "batch_exchanges" if execution is None else "unbounded_custom_metric"
    with pytest.raises(ValueError, match="counter snapshot"):
        attempt.record_audit({unsupported: 1})
    with pytest.raises(ValueError, match="counter snapshot"):
        attempt.record_audit({"arbitrary_metric": 1})
    snapshot = peers(execution is not None).audit()
    attempt.record_audit(snapshot)
    with pytest.raises(ValueError, match="counter snapshot"):
        attempt.record_audit(snapshot)
    assert attempt.transport_audit == snapshot


def test_failed_exchange_omits_unknown_batch_counts_not_fake_zero(client):
    core, execution = client
    if execution is None:
        return
    owner = peers()
    owner.calls_known = owner.server_known = False
    snapshot = owner.audit()
    assert not {"batch_exchanges", "native_scalar_claims", "inference_stage_calls"} & snapshot.keys()
    attempt = adapter._Attempt(core, {})
    attempt.record_audit(snapshot)
    core.audit.finish_transport(attempt.transport_audit)
    assert "batch_exchanges" not in core.audit.to_dict()


def test_batch_alias_mismatch_is_rejected_without_recording_snapshot(client):
    core, execution = client
    if execution is None:
        return
    snapshot = peers().audit()
    snapshot["online_steps"] = snapshot["batch_exchanges"]
    attempt = adapter._Attempt(core, {})
    with pytest.raises(ValueError, match="scalar-step counters differ"):
        attempt.record_audit(snapshot)
    assert attempt.transport_audit is None


def test_accepted_response_records_final_transport_snapshot_without_schema_failure(client, monkeypatch):
    core, execution = client
    source = SimpleNamespace(config={"model_type": "qwen2", "num_hidden_layers": 1,
        "num_attention_heads": 1, "num_key_value_heads": 1, "hidden_size": 2},
        contract={"vocabulary": 8})
    owner = peers(execution is not None)

    def execute(source, ids, count, attempt):
        assert ids == (1, 2) and count == 1
        attempt.record_audit(owner.audit())
        state = ((2, 1, 2), bytes(16), bytes(16))
        return GkmDecoderResult((bytes(32),), (state,), (3,), ids, 2, b"fake-native-acceptance")

    monkeypatch.setattr(core, "_ensure_source", lambda: source)
    monkeypatch.setattr(core, "tokenizer_scope", nullcontext)
    monkeypatch.setattr(core, "_encode", lambda *a: (SimpleNamespace(decode=lambda ids: "ok"), (1, 2)))
    monkeypatch.setattr(core, "_execute", execute)
    response = core.create({"input": "hello", "max_output_tokens": 1})
    assert response.output_text == "ok" and core._active is None
    snapshot = core.audit.to_dict()
    assert snapshot["inference_stage_calls"] == 4
    assert snapshot["online_steps"] == (3 if execution is not None else 2)
    assert snapshot["generation_completed_responses"] == 1
    if execution is not None:
        assert snapshot["native_scalar_claims"] == 3 and snapshot["batch_exchanges"] == 2


def test_completed_http_pair_is_counted_before_boundary_cancellation():
    owner = peers()
    descriptor = {"rows": 2, "request_bytes": [1, 1], "reply_bytes": [1, 1]}
    owner.batch = {"descriptors": [descriptor]}
    owner.batch_index = owner.batch_exchanges = owner.scalar_claims = owner.calls = 0

    def check():
        if owner.calls == 2:
            raise ProtocolError("cancelled after completed pair", 499)

    def post(*args, **kwargs):
        owner.calls += 1
        return b"d"

    owner.check, owner.post = check, post
    with pytest.raises(ProtocolError, match="cancelled"):
        owner.exchange_batch((b"q", b"q"), descriptor)
    snapshot = owner.audit()
    assert snapshot["inference_stage_calls"] == 2 and snapshot["batch_exchanges"] == 1
    assert snapshot["native_scalar_claims"] == snapshot["online_steps"] == 2


def exchange_fixture():
    owner = peers()
    owner.check = lambda: None
    accepted = {"contract": "a", "decoder_contract": "b", "numeric_profile": "c",
                "source": "d", "publication": "e", "context": "f", "request": "g",
                "steps": 3, "sharing": "full"}
    completions = [{**accepted, "schema_version": "pllm.gkm.sdk-worker-completion.v1", "worker": role}
                   for role in (0, 1)]
    owner.finish = lambda: completions
    native = SimpleNamespace(finish=lambda: json.dumps(accepted).encode())
    return transport._NativeExchange(native, owner, 0), accepted, completions


def test_batch_finish_reconciles_accepted_scalar_steps_not_aggregate_exchanges():
    exchange, accepted, _ = exchange_fixture()
    assert json.loads(exchange.finish())["steps"] == accepted["steps"] == 3
    assert exchange.peers.batch_exchanges == 2 and exchange.peers.calls == 4


@pytest.mark.parametrize("field,bad", [("steps", 2), ("steps", True), ("steps", 3.0),
    ("scalar_claims", 2), ("batch_exchanges", 1), ("batch_index", 1),
    ("calls", 3), ("calls_known", False)])
def test_batch_finish_rejects_count_mismatches_even_with_matching_worker_receipts(field, bad):
    exchange, accepted, completions = exchange_fixture()
    if field == "steps":
        accepted["steps"] = bad
        for completion in completions:
            completion["steps"] = bad
    else:
        setattr(exchange.peers, field, bad)
    with pytest.raises(ProtocolError, match="accepted scalar steps or role calls"):
        exchange.finish()


def test_scalar_finish_keeps_existing_receipt_semantics():
    exchange, accepted, _ = exchange_fixture()
    exchange.peers.batch = None
    # Scalar snapshots retain their historical online_steps meaning; this finish
    # path must not reinterpret legacy scalar role-call counts as batch metadata.
    exchange.peers.calls = 6
    assert json.loads(exchange.finish()) == accepted
