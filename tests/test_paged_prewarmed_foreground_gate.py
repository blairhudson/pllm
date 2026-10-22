"""Ordinary SDK owner/admission tests, not a Qwen3 numeric or RSS experiment.

Use real component serialization, HTTP authority checks, reservations, one-use
ledgers and SDK streaming owners. Only model execution/bundle loading and remote
services are fixtures. Source admission is not exercised or claimed; no private
native pointer is constructed, and every native callable is trapped during tests. A compiled Qwen3
run remains a separate admission/numeric/peak-memory gate.
"""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
import json
from types import SimpleNamespace

import httpx
import numpy as np
import pytest

from pllm import Experiment, _native
from pllm.profiles import resolve_runtime_composition
from pllm.runtime.client import ModelError, ResponsesResource, RuntimeClient
from pllm.runtime.preparation_protocol import PreparationRequest, seeded_ring_profile
from pllm.runtime.prepared_queue import PreparedExecutionQueue, prepared_attempt
from pllm.runtime.transformer_client import ClientBundle, PreparedInventory, PreparedStageRows, StageMetadata
from pllm.runtime.types import Response, ResponseEvent

from examples.papers.low_memory_qwen3 import recipes


STAGES = ("layers.0.self_attn.q_proj", "layers.0.mlp.down_proj", "lm_head")
MODEL = "foreground-lifecycle-fixture"
PRIVATE = "caller-private-prompt-not-for-provider-controls"


class DeferredExecutor:
    """Deterministic idle scheduler; never execute preparation as a test side effect."""
    def __init__(self):
        self.calls = []

    def submit(self, function, *args):
        future = Future()
        self.calls.append((function, args, future))
        return future

    def shutdown(self, **kwargs):
        for _, _, future in self.calls:
            future.cancel()


@pytest.fixture
def harness(monkeypatch, tmp_path):
    def forbidden_native(*args, **kwargs):
        pytest.fail("native execution is outside this lifecycle test")

    for name, value in vars(_native).copy().items():
        if not name.startswith("_") and callable(value):
            monkeypatch.setattr(_native, name, forbidden_native)

    # These are genuine public configuration objects, not a forged resolved
    # ExperimentProfile. Use the normal endpoint SDK mode for the model-less
    # lifecycle fixture; do not bypass native source/composition admission.
    declared = Experiment.from_spec(json.loads(json.dumps(recipes()["paged-prepared"].to_spec())))
    runtime = resolve_runtime_composition(declared.pipeline)
    assert runtime is not None and runtime.inventory_policy == "prewarm"
    assert declared.pipeline.components["delivery"].get_params()["storage"] == "paged"
    assert runtime.weight_bits == runtime.activation_bits == 8
    profile = seeded_ring_profile(2 * 127 * 127)
    stages = {sid: StageMetadata(sid, "linear", 2, 2, 8, 8, profile.ring,
              profile.modulus, profile.wire_bits, np.ones(2, dtype=np.float32),
              weight_digest="w" * 64, seeded_profile=profile) for sid in STAGES}
    body = "b" * 64
    commitment = "c" * 64
    metadata = {"client_runtime": "masked_transformer_v1", "privacy_mode": "public",
                "body_fingerprint": body, "stage_commitment": commitment,
                "architecture": "fixture", "stage_count": len(stages)}
    bundle = ClientBundle(MODEL, {"architecture": "fixture", "stages": list(stages)},
        {"bos_token_id": 0}, {"kind": "fixture"}, stages, {},
        {"mode": "public", "protocol": "seeded-inventory/v1", "body_fingerprint": body,
         "stage_commitment": commitment, "weight_bits": 8, "activation_bits": 8}, weight_storage="paged")
    state = SimpleNamespace(
        http=[], remote_rows={}, live_status={}, live_sequences={}, session_error=None, cancel_error=None,
        authority_override={}, authority_callback=None, leases=[], retired=[], handler=None)

    def handler(request):
        state.http.append(request)
        path = request.url.path
        if request.method == "GET" and path == f"/v1/runtime/models/{MODEL}":
            if state.authority_callback is not None:
                state.authority_callback()
            return httpx.Response(200, json={"id": MODEL, "metadata": metadata,
                "client_bundle": {"schema": 2, "sha256": "f" * 64, "size": 1, "etag": '"' + "f" * 64 + '"'}})
        if request.method == "GET" and path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": MODEL,
                "runtime": {**metadata, **state.authority_override},
                "preparation": {**metadata, "protocol": "seeded-inventory/v1",
                                "weight_bits": 8, "activation_bits": 8, **state.authority_override}}]})
        if request.method == "GET" and "/inventories/" in path:
            inventory_id = path.rsplit("/", 1)[-1]
            status = state.live_status.get(inventory_id, 200)
            if state.live_sequences.get(inventory_id):
                status = state.live_sequences[inventory_id].pop(0)
            return httpx.Response(status, json={"status": "ready", "next_row": state.remote_rows.get(inventory_id, 0)})
        if request.method == "POST" and path == "/v1/runtime/sessions":
            if state.session_error:
                raise state.session_error
            values = json.loads(request.content)
            assert values["execution"] == "seeded-preparation"
            assert not {"input", "prompt", "token_ids", "seed"} & values.keys()
            state.remote_rows[values["inventory_id"]] = values["inventory_start"] + values["inventory_rows"]
            return httpx.Response(200, json={"id": "session", "response_id": "response"})
        if request.method == "POST" and path == "/v1/responses/response/cancel":
            return httpx.Response(200, json=Response(id="response", model=MODEL, output=[], status="cancelled").to_dict())
        if request.method == "POST" and path.endswith("/cancel"):
            if state.cancel_error and "/sessions/" in path:
                raise state.cancel_error
            return httpx.Response(200, json={"status": "cancelled"})
        pytest.fail(f"unexpected request, including forbidden material issuance: {request.method} {path}")

    inference = httpx.Client(base_url="https://inference.test", transport=httpx.MockTransport(handler))
    preparation = httpx.Client(base_url="https://preparation.test", transport=httpx.MockTransport(handler))
    core = RuntimeClient(base_url=str(inference.base_url), api_key="inference-key", default_model=MODEL,
        preparation_api_key="preparation-key", http_client=inference, preparation_http_client=preparation,
        prepared_inventory_rows=runtime.prepared_inventory_rows,
        background_inventory_refill=runtime.background_inventory_refill,
        bundle_compression=runtime.bundle_compression, bundle_cache_dir=tmp_path / "cache")
    # Replace only the model-loading boundary, never _transformer_state, authority,
    # _ensure_prepared_inventory, _open_transformer_session, reserve or retirement.
    monkeypatch.setattr(core, "_load_client_bundle_record", lambda *a, **kw: (bundle, "f" * 64))
    bound = core._transformer_state(MODEL)
    core._provider_executor.shutdown(wait=True)
    core._provider_executor = DeferredExecutor()

    def inventory(capacity=71, name="committed"):
        values = {}
        for sid in stages:
            request = PreparationRequest("11" * 16, name, MODEL, body, sid, "w" * 64,
                capacity, 2, 2, 8, 8, profile.signed_output_bound, profile.ring,
                profile.modulus, profile.wire_bits, b"fixture-not-issued".ljust(32, b"!"))
            # Already committed synthetic material; never invoke mask issuance.
            values[sid] = PreparedStageRows(request, np.zeros((capacity, 2), dtype=np.uint32),
                                           np.zeros((capacity, 2), dtype=np.uint32))
        value = PreparedInventory(name, capacity, values, _audit=core.audit.record_inventory)
        state.remote_rows[name] = 0
        return value

    bound.prepared_inventory = inventory()
    state.core, state.bound, state.bundle, state.inventory = core, bound, bundle, inventory
    state.declared = declared
    state.sdk = ResponsesResource(core)
    state.http.clear()
    yield state
    core.close()
    inference.close()
    preparation.close()


def probe(h, monkeypatch, *, rows=71, used=3, error=None, callback=None, seeded_owner=True):
    """Fixture numeric executor reached by genuine SDK dispatch and owner wrapper."""
    core = h.core
    core._prepared_queue = PreparedExecutionQueue() if seeded_owner else None

    def execute(body, messages, descriptor, **kwargs):
        # All authority/one-use operations below are the real implementation.
        state = core._transformer_state(MODEL)
        core._ensure_prepared_inventory(MODEL, state, rows)
        session, state, lease = core._open_transformer_session(MODEL, max_output_tokens=8, required_rows=rows)
        h.leases.append(lease)
        attempt = prepared_attempt.get()
        if attempt is not None:
            attempt.enter()
            attempt.response_id, attempt.model_id = session["response_id"], MODEL
        completed = False
        try:
            yield ResponseEvent.from_dict({"type": "response.created", "sequence_number": 0,
                "response": Response(id="response", model=MODEL, output=[], status="in_progress").to_dict()})
            for sid in STAGES:
                if attempt is not None:
                    attempt.check_cancelled()
                lease.take(sid, used)
            if callback is not None:
                callback()
            if error is not None:
                raise error
            completed = True
            yield ResponseEvent.from_dict({"type": "response.completed", "sequence_number": 1,
                "response": Response(id="response", model=MODEL, output=[], status="completed").to_dict()})
        finally:
            if not completed:
                core._cancel_transformer_session(session["id"])
            core._finish_prepared_response(MODEL, state, lease)
            h.retired.append((core._foreground_active, core._online_active))

    monkeypatch.setattr(core, "_transformer_events", execute)
    return h.sdk.create(model=MODEL, input=PRIVATE, max_output_tokens=8, stream=True)


def no_material_or_plaintext(h):
    assert not any(request.method == "POST" and not request.url.path.endswith("/cancel")
                   and ("/preparation/" in request.url.path or request.url.path == "/v1/runtime/inventories")
                   for request in h.http)
    assert all(PRIVATE.encode() not in request.content for request in h.http)


@pytest.mark.parametrize("seeded_owner", [True, False])
@pytest.mark.parametrize("condition", ["missing", "underflow", "stale", "replay", "cancelled"])
def test_sdk_underflow_fails_before_session_or_material_and_preserves_authority(harness, monkeypatch, seeded_owner, condition):
    h = harness
    inventory = h.bound.prepared_inventory
    if condition == "missing":
        h.bound.prepared_inventory = None
    elif condition == "underflow":
        inventory.reserve(1).close()
    elif condition == "stale":
        h.live_status[inventory.id] = 410
    elif condition == "replay":
        h.remote_rows[inventory.id] = 1
    else:
        inventory.cancel()
    before = inventory.status().copy()
    with probe(h, monkeypatch, seeded_owner=seeded_owner) as stream:
        with pytest.raises(ModelError, match="prepared inventory unavailable"):
            next(stream)
    assert inventory.status() == before
    assert not any(request.method == "POST" and request.url.path == "/v1/runtime/sessions" for request in h.http)
    assert not h.core._provider_executor.calls and not h.core._pending_prepared_refills
    assert h.core._foreground_active == h.core._online_active == h.core._preparation_active == 0
    no_material_or_plaintext(h)


@pytest.mark.parametrize("seeded_owner", [True, False])
def test_callback_cross_thread_preparation_is_blocked_before_manifest_or_issuance(harness, monkeypatch, seeded_owner):
    h = harness
    with probe(h, monkeypatch, seeded_owner=seeded_owner) as stream:
        next(stream)
        before = len(h.http)
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(h.core.preprocess, MODEL, 71)
            with pytest.raises(ModelError, match="inference is online"):
                future.result(timeout=2)
        assert len(h.http) == before
        assert h.core._foreground_active == 1
        assert h.core._preparation_active == 0
    assert h.retired == [(1, 0)]
    assert h.core._foreground_active == 0
    no_material_or_plaintext(h)


def test_idle_epoch_gap_and_final_cleanup_keep_same_global_foreground_owner(harness, monkeypatch):
    h = harness
    with probe(h, monkeypatch) as stream:
        next(stream)
        h.core._finish_prepared_response(MODEL, h.bound, h.leases[0])
        assert h.core._foreground_active == 1 and h.core._online_active == 0
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(h.core._prepare_inventory_locked, MODEL, h.bound, 71)
            with pytest.raises(ModelError, match="foreground response"):
                future.result(timeout=2)
        before = h.bound.prepared_inventory.status().copy()
        before_http = len(h.http)
        h.core._background_refill(MODEL, h.bound)
        assert len(h.http) == before_http and not h.bound.refill_in_progress
        with pytest.raises(ModelError, match="foreground response"):
            h.core._fresh_decode_epoch_inventory(MODEL, h.bound, 71, None)
        assert h.bound.prepared_inventory.status() == before
    assert h.retired == [(1, 0)]
    assert h.core._foreground_active == 0
    no_material_or_plaintext(h)


@pytest.mark.parametrize("termination", ["eos", "cancel", "original_error"])
def test_used_rows_consumed_rest_burned_and_original_failure_wins(harness, monkeypatch, termination):
    h = harness
    original = ValueError("original model/transport failure") if termination == "original_error" else None
    inventory = h.bound.prepared_inventory
    retained = sum(stage.retained_mask_bytes for stage in inventory.stages.values())
    with probe(h, monkeypatch, rows=10, used=3, error=original) as stream:
        next(stream)
        if termination == "cancel":
            stream.close()
        elif original is not None:
            with pytest.raises(ValueError) as error:
                next(stream)
            assert error.value is original
        else:
            assert next(stream).type == "response.completed"
            # Terminal delivery is still a callback boundary, not idle.
            assert h.core._foreground_active == 1
            list(stream)
    status = inventory.status()
    claimed = 0 if termination == "cancel" else 3
    assert status["reserved"] == 0 and status["available"] == 61
    assert status["consumed"] == claimed and status["burned"] == 10 - claimed
    assert status["stage_rows_claimed"] == len(STAGES) * claimed
    assert status["stage_rows_burned"] == len(STAGES) * (10 - claimed)
    assert sum(stage.retained_mask_bytes for stage in inventory.stages.values()) == retained
    assert h.retired == [(1, 0)] and h.core._foreground_active == 0
    no_material_or_plaintext(h)


def test_lost_session_ack_burns_reservation_closes_admission_and_never_retries(harness, monkeypatch):
    h = harness
    original = httpx.ReadTimeout("lost session ACK")
    h.session_error = original
    with probe(h, monkeypatch) as stream:
        with pytest.raises(httpx.ReadTimeout) as error:
            next(stream)
    assert error.value is original
    status = h.bound.prepared_inventory.status()
    assert status["reserved"] == status["consumed"] == 0 and status["burned"] == 71
    assert h.core._closing and h.core._online_active == h.core._foreground_active == 0
    assert sum(request.url.path == "/v1/runtime/sessions" for request in h.http) == 1
    with pytest.raises(ModelError, match="admission is closed"):
        h.core.preprocess(MODEL, 71)
    no_material_or_plaintext(h)


def test_lost_cancel_ack_keeps_original_error_and_closed_admission(harness, monkeypatch):
    h = harness
    original = ValueError("original prefill failure")
    h.cancel_error = httpx.ReadTimeout("lost cancellation ACK")
    with probe(h, monkeypatch, error=original) as stream:
        next(stream)
        with pytest.raises(ValueError) as error:
            next(stream)
    assert error.value is original and h.core._closing
    assert h.bound.prepared_inventory.status()["reserved"] == 0
    assert h.core._foreground_active == h.core._online_active == 0
    no_material_or_plaintext(h)


def test_public_cancel_keeps_foreground_owner_until_real_retirement(harness, monkeypatch):
    from pllm.runtime.prepared_queue import PreparedQueueClosed

    h = harness
    with probe(h, monkeypatch) as stream:
        next(stream)
        assert h.sdk.cancel("response").status == "cancelled"
        assert h.core._foreground_active == h.core._online_active == 1
        with pytest.raises(ModelError, match="inference is online"):
            h.core.preprocess(MODEL, 71)
        with pytest.raises(PreparedQueueClosed, match="cancelled"):
            next(stream)
    assert h.core._foreground_active == h.core._online_active == 0
    status = h.bound.prepared_inventory.status()
    assert status["reserved"] == status["consumed"] == 0
    assert status["stage_rows_burned"] == 71 * len(STAGES)
    no_material_or_plaintext(h)


def test_lazy_stream_preserves_original_cancellation_context_until_retirement(harness, monkeypatch):
    from pllm.runtime.prepared_queue import PreparedCancellation, PreparedQueueClosed, prepared_cancellation

    h = harness
    signal = PreparedCancellation()
    token = prepared_cancellation.set(signal)
    try:
        stream = probe(h, monkeypatch)
    finally:
        prepared_cancellation.reset(token)
    with stream:
        next(stream)
        assert len(signal._attempts) == 1
        signal.cancel()
        assert h.core._foreground_active == 1
        with pytest.raises(PreparedQueueClosed, match="cancelled"):
            next(stream)
    assert not signal._attempts
    assert h.core._foreground_active == h.core._online_active == 0
    no_material_or_plaintext(h)


@pytest.mark.parametrize("field", ["body_fingerprint", "stage_commitment", "architecture", "weight_bits", "activation_bits"])
def test_authority_body_or_head_stage_contract_mismatch_is_not_bypassed(harness, monkeypatch, field):
    from pllm.runtime.client import ProtocolError

    h = harness
    h.bound.preparation_verified = False
    h.authority_override[field] = "wrong-source-head-or-body-contract"
    with probe(h, monkeypatch) as stream:
        with pytest.raises(ProtocolError, match="commitments do not match"):
            next(stream)
    assert not h.leases and h.core._foreground_active == 0
    no_material_or_plaintext(h)


def test_committed_spare_can_be_reserved_but_never_generated_in_foreground(harness, monkeypatch):
    h = harness
    h.bound.prepared_inventory.reserve(1).close()
    spare = h.inventory(name="committed-spare")
    h.bound.prepared_inventory_spare = spare
    with probe(h, monkeypatch) as stream:
        next(stream)
        assert h.leases[0].inventory_id == spare.id
        assert h.bound.prepared_inventory is spare
    no_material_or_plaintext(h)


def test_committed_disk_page_read_is_allowed_online_without_material_generation(harness, monkeypatch, tmp_path):
    h = harness
    path = tmp_path / "committed-public-page"
    path.write_bytes(b"already-committed-public-fixture")

    def read():
        assert h.core._foreground_active == h.core._online_active == 1
        with path.open("rb") as file:
            assert file.read() == b"already-committed-public-fixture"

    with probe(h, monkeypatch, callback=read) as stream:
        list(stream)
    no_material_or_plaintext(h)


def test_idle_refill_is_scheduled_only_after_foreground_runtime_retirement(harness, monkeypatch):
    h = harness
    with probe(h, monkeypatch) as stream:
        next(stream)
        assert not h.core._provider_executor.calls
        next(stream)
        assert not h.core._provider_executor.calls
        list(stream)
    assert h.retired == [(1, 0)]
    assert h.core._foreground_active == 0
    function, args, _ = h.core._provider_executor.calls[0]
    assert function == h.core._background_refill and args == (MODEL, h.bound)
    no_material_or_plaintext(h)


def test_foreground_start_during_existing_preparation_fails_without_retry(harness, monkeypatch):
    h = harness
    h.core._begin_preparation()
    try:
        with probe(h, monkeypatch) as stream:
            with pytest.raises(ModelError, match="refill is in progress"):
                next(stream)
        assert h.core._foreground_active == 0 and not h.http
    finally:
        h.core._end_preparation()


def test_explicit_preparation_owns_slow_authority_callback_and_original_error(harness):
    h = harness
    original = ValueError("original authority-control failure")

    def callback():
        assert h.core._preparation_active == 1 and h.core._foreground_active == 0
        # This is a genuine nested SDK entry during the HTTP control callback,
        # before provider classification or generation. No flag drops in flight.
        with h.sdk.create(model=MODEL, input=PRIVATE, stream=True) as stream:
            with pytest.raises(ModelError, match="refill is in progress"):
                next(stream)
        raise original

    h.authority_callback = callback
    with pytest.raises(ValueError) as error:
        h.core.preprocess(MODEL, 71)
    assert error.value is original
    assert h.core._preparation_active == h.core._foreground_active == 0
    assert not any(request.method == "POST" for request in h.http)
    no_material_or_plaintext(h)


def test_nested_request_retains_outer_owner_after_inner_exhaustion(harness, monkeypatch):
    h = harness
    monkeypatch.setattr(h.core, "_events", lambda body: iter([body["input"]]))
    with nullcontext(h.core.events({"input": "outer"})) as outer:
        assert next(outer) == "outer"
        assert list(h.core.events({"input": "inner"})) == ["inner"]
        assert h.core._foreground_active == 1
        with pytest.raises(ModelError, match="foreground response"):
            h.core._begin_preparation()
        outer.close()
    assert h.core._foreground_active == 0


@pytest.mark.parametrize("continuation", [False, True])
def test_real_prefill_and_continuation_dispatch_fail_before_numeric_execution_or_issuance(harness, monkeypatch, continuation):
    h = harness
    h.bound.prepared_inventory = None
    # Mock the tokenizer/compiler front-end, not the real transformer event,
    # ownership, admission or authority paths. Returning None is the supported
    # legacy uncompiled interface; no native session/pointer is fabricated.
    monkeypatch.setattr(h.core, "_render_cached_decoder_prompt", lambda *a, **kw: PRIVATE)
    monkeypatch.setattr(h.core, "_response_tokenizer", lambda _: SimpleNamespace(encode=lambda *a, **kw: [1] * 64))
    monkeypatch.setattr(h.core, "_compiled_public_decoder", lambda *a, **kw: None)
    options = {}
    if continuation:
        h.core.histories["previous"] = "client-private previous response"
        h.core.message_histories["previous"] = []
        options["previous_response_id"] = "previous"
    with h.sdk.create(model=MODEL, input=PRIVATE, max_output_tokens=8, stream=True, **options) as stream:
        with pytest.raises(ModelError, match="prepared inventory unavailable"):
            next(stream)
    assert h.core._foreground_active == h.core._online_active == 0
    assert not any(request.method == "POST" for request in h.http)
    no_material_or_plaintext(h)


def test_inventory_expiring_between_check_and_reservation_fails_without_retry(harness, monkeypatch):
    h = harness
    inventory = h.bound.prepared_inventory
    h.live_sequences[inventory.id] = [200, 410]
    with probe(h, monkeypatch) as stream:
        with pytest.raises(ModelError, match="became unavailable before reservation"):
            next(stream)
    assert not h.leases
    assert inventory.status()["stage_rows_discarded"] == 71 * len(STAGES)
    assert not h.core._provider_executor.calls
    assert h.core._foreground_active == h.core._online_active == 0
    no_material_or_plaintext(h)


@pytest.mark.parametrize("stage", [*STAGES, "embed_tokens"])
def test_stage_exhaustion_or_unadmitted_lookup_never_falls_back_to_generation(harness, monkeypatch, stage):
    from pllm.runtime.transformer_client import TransformerClientError

    h = harness

    def exhaust():
        h.leases[0].take(stage, 72)

    with probe(h, monkeypatch, callback=exhaust) as stream:
        next(stream)
        with pytest.raises(TransformerClientError, match="exhausted|invalid"):
            next(stream)
    status = h.bound.prepared_inventory.status()
    assert status["reserved"] == 0
    assert status["stage_rows_claimed"] + status["stage_rows_burned"] == 71 * len(STAGES)
    assert h.core._foreground_active == h.core._online_active == 0
    no_material_or_plaintext(h)
