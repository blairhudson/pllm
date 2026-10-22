"""Close/error/generation supplements to C8's independent SDK-entry matrix.

Use real SDK/context admission, authority fixtures, correlation owners and frame
authentication. HTTP/model doubles are explicit. No native execution, material
issuance, checkpoint admission, model quality or memory result is claimed.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import threading
from types import SimpleNamespace

import httpx
import numpy as np
import pytest

from pllm import OpenAI
from pllm.runtime.client import (
    ModelError, ResponseStream, _BlindedCorrelationProvider, _CorrelationSource,
    _TransformerCryptoState,
)
from pllm.runtime.masked_runtime import MaskCorrelation, MaskedLinearRequest, MaskedLinearResponse
from pllm.runtime.protocol import ProtocolEnvelope, pack_envelope, unpack_envelope
from pllm.runtime.security import derive_session_key
from pllm.runtime.stage_protocol import BlindedStageCorrelation

from test_foreground_owner_regressions import endpoint, legacy  # noqa: F401
from test_paged_prewarmed_foreground_gate import MODEL, PRIVATE, STAGES, harness  # noqa: F401


def test_concurrent_close_rejects_without_retiring_borrow_or_owned_topology(endpoint, monkeypatch):
    e = endpoint
    releases = []
    monkeypatch.setattr(e.sdk, "_owned_topology", SimpleNamespace(close=lambda: releases.append("topology")))
    with e.sdk.responses.create(input=PRIVATE, max_output_tokens=2, stream=True) as stream:
        next(stream)
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(e.sdk.close)
            with pytest.raises(ModelError, match="retired foreground"):
                future.result(timeout=2)
        assert not e.core._closing and not releases
        assert e.bound.prepared_inventory is e.inventory
        assert not e.inventory._cancelled and not e.leases[0]._closed
        assert e.core._foreground_active == e.core._online_active == 1
    assert e.leases[0]._closed and e.retirement == [1]
    e.sdk.close()
    assert releases == ["topology"] and e.inventory._cancelled


def test_same_thread_model_callback_close_rejects_without_deadlock(endpoint, monkeypatch):
    e = endpoint
    execute = e.core._transformer_events
    callbacks = []

    def model_callback(*args, **kwargs):
        source = execute(*args, **kwargs)
        try:
            first = next(source)
            with pytest.raises(ModelError, match="retired foreground"):
                e.sdk.close()
            callbacks.append(e.core._foreground_active)
            assert not e.inventory._cancelled and not e.leases[0]._closed
            yield first
            yield from source
        finally:
            source.close()

    monkeypatch.setattr(e.core, "_transformer_events", model_callback)
    response = e.sdk.responses.create(input=PRIVATE, max_output_tokens=2)
    assert response.status == "completed" and callbacks == [1]
    assert e.retirement == [1] and e.leases[0]._closed


def test_bundle_retirement_runs_outside_client_coordination_locks(endpoint, monkeypatch):
    e = endpoint
    original = type(e.bound.bundle).close
    observed = []

    def inspect_locks():
        values = []
        for lock in (e.core._activity_lock, e.core._transformer_state_lock):
            acquired = lock.acquire(blocking=False)
            values.append(acquired)
            if acquired:
                lock.release()
        return values

    def close_bundle(bundle):
        if bundle is e.bound.bundle:
            with ThreadPoolExecutor(max_workers=1) as pool:
                observed.append(pool.submit(inspect_locks).result(timeout=2))
        return original(bundle)

    monkeypatch.setattr(type(e.bound.bundle), "close", close_bundle)
    e.sdk.close()
    assert observed == [[True, True]]


def test_authority_lookup_preparation_borrow_rejects_close_and_new_foreground(endpoint):
    e = endpoint
    entered, release = threading.Event(), threading.Event()

    def authority_callback():
        entered.set()
        assert release.wait(timeout=2)

    e.h.authority_callback = authority_callback
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(e.sdk.preprocess, count=1)
        try:
            assert entered.wait(timeout=2)
            assert e.core._preparation_active == 1 and e.core._foreground_active == 0
            with pytest.raises(ModelError, match="preparation owners"):
                e.sdk.close()
            assert not e.core._closing and not e.inventory._cancelled
            before = len(e.h.http)
            with e.sdk.responses.create(input=PRIVATE, max_output_tokens=2, stream=True) as stream:
                with pytest.raises(ModelError, match="refill is in progress"):
                    next(stream)
            assert len(e.h.http) == before and not e.leases
        finally:
            release.set()
        assert future.result(timeout=2)["generated"] == 0
    e.h.authority_callback = None
    assert e.core._preparation_active == e.core._foreground_active == 0
    assert e.sdk.responses.create(input=PRIVATE, max_output_tokens=2).status == "completed"


def test_stream_context_preserves_caller_and_records_both_retirement_errors():
    retired = []
    original = ValueError("caller owns original failure")

    def source():
        try:
            yield 1
        finally:
            retired.append("iterator")
            raise RuntimeError("iterator retirement failed")

    def callback():
        retired.append("callback")
        raise LookupError("callback retirement failed")

    iterator = source()
    with pytest.raises(ValueError) as caught:
        with ResponseStream(iterator, close=callback) as stream:
            assert next(stream) == 1
            raise original
    assert caught.value is original and iterator.gi_frame is None
    assert retired == ["iterator", "callback"]
    assert any("iterator retirement failed" in note for note in original.__notes__)
    assert any("callback retirement failed" in note for note in original.__notes__)


def test_direct_stream_close_preserves_first_cleanup_error_and_attempts_callback():
    first = RuntimeError("first retirement failure")
    callbacks = []

    def source():
        try:
            yield 1
        finally:
            raise first

    def callback():
        callbacks.append(True)
        raise LookupError("secondary callback failure")

    stream = ResponseStream(source(), close=callback)
    next(stream)
    with pytest.raises(RuntimeError) as caught:
        stream.close()
    assert caught.value is first and callbacks == [True]
    assert any("secondary callback failure" in note for note in first.__notes__)


def test_close_hook_lookup_failure_still_attempts_callback_retirement():
    first = RuntimeError("close hook unavailable")
    callbacks = []

    class Iterator:
        def __iter__(self):
            return self

        def __next__(self):
            raise StopIteration

        @property
        def close(self):
            raise first

    stream = ResponseStream(Iterator(), close=lambda: callbacks.append(True))
    with pytest.raises(RuntimeError) as caught:
        stream.close()
    assert caught.value is first and callbacks == [True]


def test_sdk_context_caller_error_keeps_live_stream_until_explicit_retirement(endpoint):
    e = endpoint
    original = ValueError("caller left stream open")
    stream = e.sdk.responses.create(input=PRIVATE, max_output_tokens=2, stream=True)
    try:
        with pytest.raises(ValueError) as caught:
            with e.sdk:
                next(stream)
                raise original
        assert caught.value is original
        assert any("retired foreground" in note for note in original.__notes__)
        assert not e.core._closing and not e.inventory._cancelled and not e.leases[0]._closed
    finally:
        stream.close()
    assert e.leases[0]._closed and e.core._foreground_active == 0


def committed_legacy(legacy):
    """Normal explicit idle entry establishes state; fixture adds a committed row."""
    assert legacy.sdk.preprocess(count=0)["generated"] == 0
    core = legacy.sdk._core
    state = core._crypto_states["legacy"]
    width = state.model.tokenizer.vocab_size
    row = MaskCorrelation("fixture-committed-once", np.arange(width, dtype=np.int64),
                          np.zeros(width, dtype=np.int64))
    state.pool.put(row)
    return core, state, row


@pytest.fixture
def legacy_http(legacy, monkeypatch):
    owner = legacy.sdk._core
    # Public HTTP-only configuration: never let an auto WebSocket connection
    # escape the mock transport. Keep C8's fixture and its own SDK unchanged.
    with OpenAI(base_url=owner.base_url, api_key=owner.api_key, default_model="legacy",
                correlation_mode="local-test", session_transport="http", http_client=owner.http) as sdk:
        monkeypatch.setattr(legacy, "sdk", sdk)
        yield legacy


def test_legacy_first_committed_response_and_idle_entry_use_real_authenticated_channel(legacy_http, monkeypatch):
    legacy = legacy_http
    core, state, row = committed_legacy(legacy)
    transport = core.http._transport
    ordinary = transport.handle_request

    def model_service(request):
        if request.url.path.endswith("/execute"):
            legacy.requests.append(request)
            frame = unpack_envelope(request.content)
            key = derive_session_key(core.api_key, frame.session_id)
            frame.verify(key)
            masked = MaskedLinearRequest.unpack(frame.payload)
            assert masked.correlation_id == row.id and "ab".encode() not in frame.payload
            logits = np.zeros(state.model.tokenizer.vocab_size, dtype=np.int64)
            logits[state.model.tokenizer.eos_token_id] = 1
            result = MaskedLinearResponse(row.id, logits)
            reply = ProtocolEnvelope.create(request_id=frame.request_id, session_id=frame.session_id,
                model=frame.model, kind="masked.linear.result", sequence=frame.sequence,
                payload=result.pack(), metadata={}, key=key)
            return httpx.Response(200, content=pack_envelope(reply))
        return ordinary(request)

    monkeypatch.setattr(transport, "handle_request", model_service)
    with legacy.sdk as sdk:
        response = sdk.responses.create(input="ab", max_output_tokens=1, temperature=0)
        assert response.status == "completed" and row.consumed
        assert state.pool.consumed == 1 and len(state.pool) == 0
    assert not legacy.foreground_at_generation


def test_legacy_missing_bfv_context_rejects_before_session_or_key_generation(legacy, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("foreground constructed a fresh BFV context")

    monkeypatch.setattr("pllm.runtime.client.BFVCorrelationClient", forbidden)
    legacy.sdk._core.correlation_mode = "bfv"
    with pytest.raises(ModelError, match="legacy correlation inventory unavailable"):
        legacy.sdk.responses.create(input="ab", max_output_tokens=1)
    assert not any(request.method != "GET" for request in legacy.requests)


def test_legacy_explicit_replenishment_cannot_issue_from_active_sdk_callback(legacy):
    core, state, row = committed_legacy(legacy)
    source = _CorrelationSource(client=core, session_id="legacy-session", state=state, prefetch=1)
    with legacy.sdk.responses.create(input="ab", max_output_tokens=1, stream=True) as stream:
        next(stream)
        before = len(legacy.requests)
        with pytest.raises(ModelError, match="foreground response"):
            source.replenish(1)
        assert len(legacy.requests) == before and not row.consumed and len(state.pool) == 1
    assert not legacy.foreground_at_generation


def test_proprietary_generation_entry_rejects_but_committed_queue_still_consumes(endpoint):
    e = endpoint
    # Component-owner units, not a claim that this proprietary toy checkpoint
    # passed a public/source compiler gate. Real SDK response supplies the lease.
    bundle = replace(e.bound.bundle, privacy={"mode": "proprietary", "protocol": "blinded_ole_w4a4"})
    empty = _TransformerCryptoState(bundle, "f" * 64, "local-test", "proprietary", "blinded_ole_w4a4")
    ready = _TransformerCryptoState(bundle, "f" * 64, "local-test", "proprietary", "blinded_ole_w4a4")
    stage = replace(bundle.stages[STAGES[0]], modulus=257, ring="prime", seeded_profile=None)
    row = BlindedStageCorrelation("committed", stage.id, ready.blinded_owner_id,
        np.zeros(2, np.uint32), np.zeros(2, np.uint32), 257, "prime")
    ready.queues[stage.id].append(row)
    providers = [_BlindedCorrelationProvider(client=e.core, session_id="owner-unit", state=state, prefetch=1)
                 for state in (empty, ready)]
    with e.sdk.responses.create(input=PRIVATE, max_output_tokens=2, stream=True) as stream:
        next(stream)
        before = len(e.h.http)
        with pytest.raises(ModelError, match="inference is online"):
            providers[0].take_many(stage, 1)
        with pytest.raises(ModelError, match="inference is online"):
            providers[0]._create(stage, 1)
        assert providers[1].take_many(stage, 1) == [row] and row.consumed
        assert not empty.queues[stage.id] and len(e.h.http) == before
        assert e.core._preparation_active == 0 and e.core._foreground_active == 1
