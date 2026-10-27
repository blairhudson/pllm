"""Adapter ownership tests with a controlled finish gate, not native/network evidence.

The real source tokenizer runs locally. Only native profile resolution, source
admission, and the not-yet-integrated two-worker transport are injected here.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import gc
import json
import struct
import threading
import time
from types import SimpleNamespace
import weakref

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from pllm import Deployment, ExecutionBudget, Experiment, Model
from pllm.profiles import OpenWeightVerifiableCpu
from pllm.runtime import delegated_client as adapter
from pllm.runtime.client import (
    AsyncResponsesResource, ModelsResource, ProtocolError, ResponsesResource, RuntimeResource,
)
from pllm.runtime.delegated_decoder import GkmDecoderResult
from pllm.runtime.http_gateway import GatewayError, iter_chat_sse
from pllm.runtime.prepared_queue import PreparedCancellation, prepared_cancellation
from pllm.runtime.responses import sse_event
from pllm.runtime.topology_accounting import measured_decode_window, two_worker_body_accounting
from pllm.runtime.transformer_client import TokenizersJSONTokenizer
from pllm.runtime.types import Response


CONNECTIONS = {"worker_a": ("http://127.0.0.1:7101", "a-secret"),
               "worker_b": ("http://127.0.0.1:7102", "b-secret")}


class Source:
    """Metadata-only stand-in with genuine WordLevel encode/decode semantics."""

    def __init__(self, experiment):
        self.model = experiment.pipeline.model
        self.contract = {"input_rows": experiment.budget.max_input_tokens,
                         "decode_steps": experiment.budget.max_new_tokens - 1, "vocabulary": 8}
        self.config = {"model_type": "qwen2", "num_hidden_layers": 2, "num_attention_heads": 2,
                       "num_key_value_heads": 1, "hidden_size": 4}
        self.config_source_bytes = json.dumps(self.config).encode()
        self.source_float32_bytes = 128
        self.source = SimpleNamespace(manifest=SimpleNamespace(context_length=128))
        self.tokenizer_descriptor = {"add_bos_token": False}
        self.checkpoint_digest = "1" * 64
        self.source_lock_digest = "2" * 64
        self.tokenizer_digest = "3" * 64
        self.tokenizers_created = 0
        self.tokenizers_closed = 0
        self.encodings = []
        self.decodings = []
        self.ids_override = None

    def tokenizer(self):
        self.tokenizers_created += 1
        inner = Tokenizer(WordLevel({"<bos>": 0, "<eos>": 1, "<unk>": 2,
                                    "user": 3, ":": 4, "hello": 5, "world": 6, "after": 7},
                                   unk_token="<unk>"))
        inner.pre_tokenizer = Whitespace()
        inner.add_special_tokens(["<bos>", "<eos>", "<unk>"])
        tokenizer = TokenizersJSONTokenizer(inner, 0, 1)
        owner = self

        class ObservedTokenizer:
            def encode(self, text, *, add_bos=False):
                owner.encodings.append((text, add_bos))
                return (tokenizer.encode(text, add_bos=add_bos)
                        if owner.ids_override is None else owner.ids_override)

            def decode(self, ids):
                owner.decodings.append(ids)
                return tokenizer.decode(ids)

            def close(self):
                owner.tokenizers_closed += 1

        return ObservedTokenizer()

    def load_values(self):
        raise AssertionError("the adapter must leave tensor materialization to the transport")


def accepted(ids, count):
    selected = tuple([1, 6, *([7] * 7)][:count])
    position = len(ids) + count - 1
    logits = tuple(struct.pack("<8f", *range(8)) for _ in selected)
    state = ((position, 1, 2), bytes(position * 8), bytes(position * 8))
    return GkmDecoderResult(logits, (state, state), selected,
                             ids + selected[:-1], position, b"test-native-finish-receipt")


def prompt_lease(h):
    """Ordinary lease with pure topology/result doubles; no planning/native calls."""
    from pllm.deployment.execution import ExecutionLease, _LEASE_KEY, _LeaseResponses

    result = SimpleNamespace(experiment=h.experiment, snapshot=None, validate=lambda **kw: None)
    topology = SimpleNamespace(close=lambda: None, cleanup_complete=True)
    view = SimpleNamespace(tokenizer_descriptor=h.source.tokenizer_descriptor,
        original_config=h.source.config, manifest={"chat_template": None},
        tokenizer=h.source.tokenizer, source_lock_digest=h.source.source_lock_digest)
    lease = ExecutionLease(result, SimpleNamespace(backend="local", snapshot=None),
        lambda: 0, topology, view, "pure-test", _key=_LEASE_KEY)
    lease._client = SimpleNamespace(_core=h.core)
    return lease, _LeaseResponses(lease, h.responses)


def test_prepared_admission_and_execution_share_one_encode(harness):
    h = harness
    lease, resource = prompt_lease(h)
    response = resource.create(input="hello")
    assert response.usage.input_tokens == 5
    assert h.source.encodings == [("user: hello\nassistant:", False)]
    assert h.calls == [((3, 4, 5, 2, 4), 3)]
    assert h.source.tokenizers_created == h.source.tokenizers_closed == 1
    assert lease._requests == 1


def test_prepared_admission_refuses_actual_rendered_count_without_source_or_attempt(harness):
    from pllm.deployment.execution import NetworkError
    h = harness
    lease, resource = prompt_lease(h)
    lease._result.experiment = replace(h.experiment,
        budget=replace(h.experiment.budget, max_input_tokens=4))
    with pytest.raises(NetworkError, match="input exceeds"):
        resource.create(input="hello")
    assert len(h.source.encodings) == 1 and h.source.tokenizers_closed == 1
    assert h.admitted == [] and h.calls == [] and h.core._active is None
    assert lease._requests == 0


def test_prepared_unstarted_iterator_keeps_owner_until_cancellation(harness):
    h = harness
    _, resource = prompt_lease(h)
    stream = resource.create(input="hello", stream=True)
    payload = h.core._active._prepared_prompt._payload
    assert payload.ids == (3, 4, 5, 2, 4)
    assert h.source.tokenizers_closed == 0 and h.admitted == []
    stream.close()
    assert h.source.tokenizers_closed == 1
    assert h.core._active is None and h.calls == []


@pytest.mark.asyncio
async def test_prepared_async_scope_survives_creation_until_retirement(harness):
    from pllm.deployment.execution import _LeaseResponses
    h = harness
    lease, _ = prompt_lease(h)
    resource = _LeaseResponses(lease, AsyncResponsesResource(h.responses))
    stream = await resource.create(input="hello", stream=True)
    assert h.source.tokenizers_closed == 0
    assert h.core._active._prepared_prompt._payload.ids == (3, 4, 5, 2, 4)
    await stream.aclose()
    assert h.source.tokenizers_closed == 1 and h.calls == []


def test_prepared_changed_source_refuses_before_transport(harness):
    h = harness
    _, resource = prompt_lease(h)
    stream = resource.create(input="hello", stream=True)
    h.source.tokenizer_descriptor["chat_template"] = "changed"
    with pytest.raises(GatewayError, match="association"):
        list(stream)
    assert h.calls == [] and h.core._active is None
    assert h.source.tokenizers_closed == 1


def test_reference_input_witness_uses_same_prepared_cohort(harness):
    from pllm.runtime.reference_benchmark import _source_ids
    h = harness
    assert _source_ids(h.source, ("hello",)) == ((3, 4, 5, 2, 4),)
    assert h.source.encodings == [("user: hello\nassistant:", False)]
    assert h.source.tokenizers_created == h.source.tokenizers_closed == 1
    assert h.calls == [] and h.admitted == []


@pytest.mark.asyncio
async def test_prepared_async_execution_cancel_drains_before_owner_retirement(harness, monkeypatch):
    from pllm.deployment.execution import _LeaseResponses
    h = harness
    lease, _ = prompt_lease(h)
    entered = threading.Event()
    def execute(*args):
        cancelled = args[-1]
        entered.set()
        assert h.source.tokenizers_closed == 0
        assert cancelled.cancelled.wait(2)
        assert h.source.tokenizers_closed == 0
        raise ProtocolError("cancelled", 499)
    monkeypatch.setattr(adapter, "_execute_request", execute)
    resource = _LeaseResponses(lease, AsyncResponsesResource(h.responses))
    task = asyncio.create_task(resource.create(input="hello"))
    assert await asyncio.to_thread(entered.wait, 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert h.source.tokenizers_closed == 1
    assert h.core._active is None


def transport_counters():
    values = {
        "inference_upload_bytes": 20, "inference_download_bytes": 28,
        "inference_control_upload_bytes": 36, "inference_control_download_bytes": 10,
        "inference_stage_calls": 6, "online_steps": 12, "inference_server_ns": 100,
        "generation_prefill_online_body_bytes": 30, "generation_decode_online_body_bytes": 18,
    }
    for role in ("worker_a", "worker_b"):
        values.update({f"role_link.{role}.{name}": value for name, value in {
            "setup_upload_bytes": 11, "setup_download_bytes": 3,
            "online_upload_bytes": 10, "online_download_bytes": 14,
            "teardown_upload_bytes": 7, "teardown_download_bytes": 2,
        }.items()})
    return values


@pytest.fixture
def harness(monkeypatch):
    experiment = Experiment(name="gkm-adapter-unit",
        pipeline=OpenWeightVerifiableCpu(Model.path("/unused/gkm-source")),
        deployment=Deployment.local(root="local://gkm-adapter-unit"),
        budget=ExecutionBudget(requests=4, max_input_tokens=8, max_new_tokens=3))
    resolved = []

    def resolve(value):
        resolved.append(value)
        return SimpleNamespace(client_runtime="compiled_gkm_v1", model="gkm-fixture")

    monkeypatch.setattr(Experiment, "resolve", resolve)
    source = Source(experiment)
    admitted, calls = [], []

    def admit(value):
        admitted.append(value)
        return source

    def execute(value, loaded, ids, count, connections, timeout, cancelled):
        assert value is experiment and loaded is source
        # Independent WordLevel fixture: user/:/hello/<unk assistant>/:.
        # The no-template admission fallback includes its assistant marker.
        assert type(ids) is tuple and ids == (3, 4, 5, 2, 4)
        assert dict(connections) == CONNECTIONS and timeout == 0.5
        assert not cancelled()
        calls.append((ids, count))
        cancelled.record_audit(transport_counters())
        return accepted(ids, count)

    monkeypatch.setattr(adapter, "_admit_source", admit)
    monkeypatch.setattr(adapter, "_execute_request", execute)
    core = adapter.DelegatedClient(experiment, CONNECTIONS, timeout=0.5)
    value = SimpleNamespace(experiment=experiment, core=core, source=source, resolved=resolved,
                            admitted=admitted, calls=calls, execute=execute,
                            responses=ResponsesResource(core))
    yield value
    core.close()


def test_resource_compatibility_fixed_count_eos_and_local_rendering(harness):
    h = harness
    assert h.resolved == [h.experiment]
    assert not h.admitted
    response = h.responses.create(input="hello", metadata={"case": "fixed-eos"})
    assert response.output_text == "world after"
    assert h.source.decodings == [[1, 6, 7]]  # EOS did not stop token selection.
    assert h.source.encodings == [("user: hello\nassistant:", False)]
    assert h.calls == [((3, 4, 5, 2, 4), 3)]
    assert h.admitted == [h.experiment]
    assert response.usage.to_dict() == {
        "input_tokens": 5, "output_tokens": 3, "total_tokens": 8,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens_details": {"reasoning_tokens": 0},
    }
    assert response.status == "incomplete"
    assert response.incomplete_details == {"reason": "max_output_tokens"}
    assert response.temperature == 0 and response.top_p == 1
    assert response.metadata == {"case": "fixed-eos"}
    assert h.responses.retrieve(response.id).output_text == response.output_text
    assert h.responses.cancel(response.id).status == "incomplete"
    assert h.source.tokenizers_created == h.source.tokenizers_closed == 1
    assert not h.core._transformer_states


@pytest.mark.parametrize("streaming", [False, True])
def test_nothing_observable_until_native_finish(harness, monkeypatch, streaming):
    h = harness
    entered, finish = threading.Event(), threading.Event()
    constructed, observed, event_times = [], [], []
    original = Response.text_response

    def text_response(cls, **kwargs):
        constructed.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(Response, "text_response", classmethod(text_response))
    h.core._result_hook = observed.append

    def execute(*args):
        entered.set()
        assert finish.wait(3)
        return h.execute(*args)

    monkeypatch.setattr(adapter, "_execute_request", execute)
    stream = h.responses.create(input="hello", stream=True) if streaming else None

    def request():
        value = next(stream) if streaming else h.responses.create(input="hello")
        event_times.append(time.monotonic_ns())
        return value

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(request)
        try:
            assert entered.wait(3)
            assert not future.done()
            assert not constructed and not observed and not event_times
            assert not h.core.records and not h.core._transformer_states
            assert h.core._active.response_id is None
            assert h.core.audit.generation_output_tokens == 0
            assert h.core.audit.inference_upload_bytes is None
            with pytest.raises(KeyError):
                h.responses.retrieve("unpublished")
            with pytest.raises(KeyError):
                h.responses.cancel("unpublished")
            with pytest.raises(ProtocolError, match="exclusive request"):
                h.responses.create(input="hello")
        finally:
            finish_time = time.monotonic_ns()
            finish.set()
        value = future.result(3)
    assert len(constructed) == len(observed) == 1
    assert event_times[0] >= finish_time
    if streaming:
        events = [value, *stream]
        assert [item.sequence_number for item in events] == list(range(len(events)))
        assert events[0].type == "response.created"
        assert events[0].response["output"] == []
        assert events[-1].type == "response.incomplete"
        assert [event.delta for event in events if event.type == "response.output_text.delta"] == ["world after"]
        assert b"event: response.created" in sse_event(events[0])
        chat = b"".join(iter_chat_sse(events, "gkm-fixture", include_usage=True))
        assert b'"finish_reason":"length"' in chat
        assert b'"completion_tokens":3' in chat
        stream.close()
    assert h.core._active is None


@pytest.mark.parametrize("streaming", [False, True])
def test_verification_failure_retires_without_response_state_or_events(harness, monkeypatch, streaming):
    h = harness
    observed = []
    h.core._result_hook = observed.append

    def fail(*args):
        args[-1].record_audit(transport_counters())
        raise ValueError("native request-end verification failed")

    monkeypatch.setattr(adapter, "_execute_request", fail)
    with pytest.raises(ValueError, match="verification failed"):
        result = h.responses.create(input="hello", stream=streaming)
        if streaming:
            next(result)
    assert not observed and not h.core.records and not h.source.decodings
    assert h.core._active is None and not h.core._tokenizer_scopes
    assert h.core.audit.generation_completed_responses == 0
    assert h.core.audit.role_link_bodies["worker_a"]["teardown_upload_bytes"] == 7
    monkeypatch.setattr(adapter, "_execute_request", h.execute)
    assert h.responses.create(input="hello").output_text == "world after"


@pytest.mark.parametrize("body", [
    {}, {"input": ""}, {"input": []}, {"input": True},
    {"model": "other"}, {"max_output_tokens": 4}, {"max_output_tokens": True},
    {"max_output_tokens": 0}, {"max_output_tokens": 1.5},
    {"temperature": 0.1}, {"temperature": float("nan")}, {"top_p": 0.5},
    {"temperature": float("inf")}, {"temperature": True},
    {"precision": "i8"}, {"seed": 1}, {"stop": "<eos>"},
    {"background": True}, {"stream": "true"}, {"store": None},
    {"instructions": False}, {"instructions": 2},
    {"truncation": "auto"}, {"truncation": {}}, {"top_logprobs": []},
    {"prompt_cache_key": "reuse"}, {"previous_response_id": "unknown"},
    {"reasoning": {"effort": "high"}}, {"include": ["message.output_text.logprobs"]},
    {"max_tool_calls": 1}, {"tool_choice": "required"},
    {"tools": [{"type": "function", "name": "tool"}]},
    {"input": [{"type": "item_reference", "id": "msg-old"}]},
    {"input": [{"type": "compaction", "encrypted_content": "old"}]},
    {"input": [{"type": "function_call_output", "call_id": "old", "output": "data"}]},
    {"input": [{"role": "tool", "content": "data"}]},
    {"input": [{"role": [], "content": "data"}]},
    {"input": [{"role": "user", "content": [{"type": "input_image"}]}]},
    {"input": [{"type": "input_text", "text": 123}]},
    {"input": [{"role": "user", "content": "hello", "reference": "old"}]},
    {"metadata": {"key": 1}}, {"metadata": {str(i): "x" for i in range(17)}},
])
def test_body_admission_precedes_source_and_material(harness, body):
    h = harness
    request = body if not body else {"input": "hello", **body}
    with pytest.raises(GatewayError):
        h.responses.create(**request)
    assert not h.admitted and not h.calls
    assert not h.core.records and h.core._active is None


def test_body_byte_depth_and_cardinality_admission(harness):
    h = harness
    deep = []
    for _ in range(20):
        deep = [deep]
    for body in ({"input": "x" * (adapter._BODY_BYTES + 1)},
                 {"input": "hello", "client_metadata": {"deep": deep}},
                 {"input": "hello", "client_metadata": {"wide": [0] * 1025}},
                 {"input": "hello", "client_metadata": {"integer": 1 << 10000}}):
        with pytest.raises(GatewayError):
            h.core.create(body)
    assert not h.admitted and not h.calls


@pytest.mark.parametrize("ids", [[], list(range(9)), [True], [-1], [8], [1.0]])
def test_token_admission_precedes_transport(harness, ids):
    h = harness
    h.source.ids_override = ids
    with pytest.raises(GatewayError, match="tokenized input"):
        h.responses.create(input="hello")
    assert not h.calls and not h.core.records
    assert h.source.tokenizers_closed == 1


def test_input_snapshot_and_shared_normalization(harness, monkeypatch):
    h = harness
    body = {"input": [{"role": "developer", "content": [{"type": "input_text", "text": "world"}]},
                      {"role": "user", "content": "hello"}], "instructions": "after",
            "stream": True}
    stream = h.core.create(body)
    body["input"][1]["content"] = "changed"
    body["instructions"] = "changed"
    h.source.ids_override = [3, 4, 5, 2, 4]
    list(stream)
    assert h.source.encodings == [("system: after\nsystem: world\nuser: hello\nassistant:", False)]


def test_unstarted_stream_owns_one_seat_and_close_allocates_no_source(harness):
    h = harness
    stream = h.responses.create(input="hello", stream=True)
    with pytest.raises(ProtocolError, match="exclusive request"):
        h.responses.create(input="hello", stream=True)
    stream.close()
    assert not h.admitted and not h.calls
    assert h.core._active is None
    assert h.responses.create(input="hello").output_text == "world after"


def test_close_signals_cancellation_then_drains_before_releasing_owner(harness, monkeypatch):
    h = harness
    entered, cancelled, release, retired = (threading.Event() for _ in range(4))

    def execute(*args):
        handle = args[-1]
        entered.set()
        deadline = time.monotonic() + 3
        try:
            while not handle() and time.monotonic() < deadline:
                cancelled.wait(0.005)
            assert handle()
            cancelled.set()
            assert release.wait(3)
            raise ProtocolError("GKM request cancelled", 499)
        finally:
            handle.record_audit(transport_counters())
            retired.set()

    monkeypatch.setattr(adapter, "_execute_request", execute)
    with ThreadPoolExecutor(max_workers=2) as pool:
        request = pool.submit(h.responses.create, input="hello")
        assert entered.wait(3)
        closing = pool.submit(h.core.close)
        try:
            assert cancelled.wait(3)
            assert not retired.is_set() and not closing.done()
            assert h.core._active is not None
            with pytest.raises(ProtocolError, match="closed"):
                h.responses.create(input="hello")
        finally:
            release.set()
        with pytest.raises(ProtocolError, match="cancelled"):
            request.result(3)
        closing.result(3)
    assert retired.is_set() and h.core._active is None
    assert h.core._source is None and not h.core.records
    h.core.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_async_cancellation_drains_even_repeated_cancels(harness, monkeypatch, streaming):
    h = harness
    entered, cancelled, release, retired = (threading.Event() for _ in range(4))

    def execute(*args):
        handle = args[-1]
        entered.set()
        deadline = time.monotonic() + 3
        try:
            while not handle() and time.monotonic() < deadline:
                cancelled.wait(0.005)
            assert handle()
            cancelled.set()
            assert release.wait(3)
            raise ProtocolError("GKM request cancelled", 499)
        finally:
            handle.record_audit(transport_counters())
            retired.set()

    monkeypatch.setattr(adapter, "_execute_request", execute)
    resource = AsyncResponsesResource(h.responses)
    stream = await resource.create(input="hello", stream=True) if streaming else None
    task = asyncio.create_task(anext(stream) if streaming else resource.create(input="hello"))
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        task.cancel()
        assert await asyncio.to_thread(cancelled.wait, 3)
        assert not task.done() and not retired.is_set()
        assert h.core._active is not None
        task.cancel()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    if stream is not None:
        await stream.aclose()
    assert retired.is_set() and h.core._active is None
    assert not h.core.records and not h.source.decodings
    assert h.core.audit.generation_completed_responses == 0
    monkeypatch.setattr(adapter, "_execute_request", h.execute)
    assert (await resource.create(input="hello")).output_text == "world after"


@pytest.mark.asyncio
async def test_async_cancel_racing_accepted_response_evicts_unobserved_cache(harness):
    h = harness
    completed, release = threading.Event(), threading.Event()

    class DelayedDelivery(ResponsesResource):
        def create(self, **kwargs):
            response = super().create(**kwargs)
            completed.set()
            assert release.wait(3)
            return response

    resource = AsyncResponsesResource(DelayedDelivery(h.core))
    task = asyncio.create_task(resource.create(input="hello"))
    try:
        assert await asyncio.to_thread(completed.wait, 3)
        assert len(h.core.records) == 1
        task.cancel()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not h.core.records and h.core._active is None


def test_pre_cancelled_context_and_post_finish_cancel_never_publish(harness):
    h = harness
    cancellation = PreparedCancellation()
    token = prepared_cancellation.set(cancellation)
    try:
        cancellation.cancel()
        with pytest.raises(ProtocolError, match="cancelled"):
            h.responses.create(input="hello")
        assert not h.admitted and not h.calls
    finally:
        prepared_cancellation.reset(token)
    h.core._result_hook = lambda result: h.core._active.cancel()
    with pytest.raises(ProtocolError, match="cancelled"):
        h.responses.create(input="hello")
    assert not h.core.records and not h.source.decodings


@pytest.mark.asyncio
async def test_async_buffered_stream_delivery_and_early_close(harness):
    h = harness
    resource = AsyncResponsesResource(h.responses)
    stream = await resource.create(input="hello", stream=True)
    events = [event async for event in stream]
    assert events[-1].type == "response.incomplete"
    response_id = events[-1].response["id"]
    assert (await resource.retrieve(response_id)).output_text == "world after"
    stream = await resource.create(input="hello", stream=True)
    first = await anext(stream)
    assert first.response["id"] in h.core.records
    await stream.aclose()
    with pytest.raises(KeyError):
        await resource.retrieve(first.response["id"])
    assert h.core._active is None


@pytest.mark.parametrize("streaming", [False, True])
def test_close_racing_iterator_construction_is_clean(harness, monkeypatch, streaming):
    h = harness
    original = h.core.events

    def events(body):
        attempt = original(body)
        h.core.close()
        return attempt

    monkeypatch.setattr(h.core, "events", events)
    if streaming:
        stream = h.responses.create(input="hello", stream=True)
        assert list(stream) == []
        stream.close()
    else:
        with pytest.raises(ProtocolError, match="without a terminal"):
            h.responses.create(input="hello")
    assert h.core._active is None and not h.admitted and not h.calls


def test_private_observation_hook_can_close_without_leaking_state(harness):
    h = harness
    h.core._result_hook = lambda result: h.core.close()
    with pytest.raises(ProtocolError, match="cancelled"):
        h.responses.create(input="hello")
    assert h.core._active is None and h.core._source is None
    assert h.core._result_hook is None and not h.core.records


def test_result_owner_is_hook_only_and_retired_before_first_event(harness):
    h = harness
    references = []
    h.core._result_hook = lambda result: references.append(weakref.ref(result))
    stream = h.responses.create(input="hello", stream=True)
    first = next(stream)
    gc.collect()
    assert references[0]() is None
    assert h.core._active.body is None
    # Accepted logits/KV are gone, but the prepared public prompt owner follows
    # the iterator until terminal delivery/cancellation, not merely construction.
    assert h.source.tokenizers_closed == 0
    assert first.response["id"] in h.core.records
    stream.close()  # Buffered, undelivered response is no longer retained.
    assert not h.core.records and h.core._active is None
    assert h.source.tokenizers_closed == 1


@pytest.mark.parametrize("change", [
    {"acceptance": b""}, {"selected": (1,)}, {"executed": (3, 4, 5)},
    {"position": 99}, {"logits": (b"small",) * 3}, {"kv": ()},
    {"kv": (((1, 1, 2), b"", b""),) * 2},
])
def test_malformed_accepted_result_cannot_publish(harness, monkeypatch, change):
    h = harness
    monkeypatch.setattr(adapter, "_execute_request", lambda *args: replace(accepted(args[2], args[3]), **change))
    with pytest.raises(ProtocolError, match="GKM"):
        h.responses.create(input="hello")
    assert not h.core.records and not h.source.decodings and h.core._active is None


def test_store_retention_is_bounded_text_only_and_defensive(harness, monkeypatch):
    h = harness
    first = h.responses.create(input="hello")
    first.output[0]["content"][0]["text"] = "mutated"
    assert h.responses.retrieve(first.id).output_text == "world after"
    copy = h.responses.retrieve(first.id)
    copy.output.clear()
    assert h.responses.retrieve(first.id).output_text == "world after"
    for prior in (first.id, "unknown"):
        with pytest.raises(GatewayError, match="continuation"):
            h.responses.create(input="hello", previous_response_id=prior)
    assert len(h.calls) == 1
    h.core.evict_response(first.id)
    with pytest.raises(KeyError):
        h.responses.retrieve(first.id)
    ephemeral = h.responses.create(input="hello", store=False)
    with pytest.raises(KeyError):
        h.responses.retrieve(ephemeral.id)
    assert not h.core.records
    monkeypatch.setattr(adapter, "_RECORD_COUNT", 2)
    retained = [h.responses.create(input="hello") for _ in range(3)]
    with pytest.raises(KeyError):
        h.responses.retrieve(retained[0].id)
    assert list(h.core.records) == [response.id for response in retained[1:]]
    monkeypatch.setattr(adapter, "_RECORD_BYTES", 1)
    h.responses.create(input="hello")
    assert not h.core.records


def test_audit_uses_observed_directed_bodies_and_phase_counters(harness):
    h = harness
    h.responses.create(input="hello")
    audit = RuntimeResource(h.core).privacy_audit
    counters = audit.to_dict()
    assert counters["plaintext_prompt_bytes_sent"] == counters["plaintext_token_ids_sent"] == 0
    assert counters["generation_attribution_supported_responses"] == 1
    assert measured_decode_window(counters, output_tokens=3,
                                  roles=("client", "worker_a", "worker_b")) == (18, 2)
    ledger = two_worker_body_accounting({"privacy": counters})
    assert ledger["tracked_body_counter_set_present"] is True
    assert ledger["online_all_link_serialized_body_bytes"] == 48
    assert ledger["total_wire_bytes"] is None
    h.responses.create(input="hello")
    assert audit.inference_upload_bytes == 40
    assert audit.role_link_bodies["worker_b"]["teardown_upload_bytes"] == 14
    assert audit.generation_completed_responses == 2


def test_unknown_transport_accounting_never_becomes_fabricated_zero(harness, monkeypatch):
    h = harness
    monkeypatch.setattr(adapter, "_execute_request", lambda *args: accepted(args[2], args[3]))
    h.responses.create(input="hello")
    counters = h.core.audit.to_dict()
    assert "generation_prefill_online_body_bytes" not in counters
    assert "inference_upload_bytes" not in counters
    assert not h.core.audit.role_link_bodies
    assert h.core.audit.generation_attribution_supported_responses == 0
    ledger = two_worker_body_accounting({"privacy": counters})
    assert ledger["tracked_body_counter_set_present"] is False
    assert ledger["all_link_serialized_body_bytes"] is None
    monkeypatch.setattr(adapter, "_execute_request", h.execute)
    h.responses.create(input="hello")
    assert "inference_upload_bytes" not in h.core.audit.to_dict()  # Prior unknown work stays unknown.


@pytest.mark.parametrize("counters", [{"inference_upload_bytes": -1},
    {"inference_upload_bytes": True}, {"unbounded_custom_metric": 1},
    {"inference_upload_bytes": 1 << 64}])
def test_transport_audit_admission(harness, monkeypatch, counters):
    h = harness

    def execute(*args):
        args[-1].record_audit(counters)
        return accepted(args[2], args[3])

    monkeypatch.setattr(adapter, "_execute_request", execute)
    with pytest.raises(ValueError, match="counter snapshot"):
        h.responses.create(input="hello")
    assert not h.core.records
    assert "inference_upload_bytes" not in h.core.audit.to_dict()


def test_partial_transport_audit_does_not_invent_other_edges(harness, monkeypatch):
    h = harness

    def execute(*args):
        args[-1].record_audit({"role_link.worker_a.setup_upload_bytes": 17})
        return accepted(args[2], args[3])

    monkeypatch.setattr(adapter, "_execute_request", execute)
    h.responses.create(input="hello")
    assert h.core.audit.role_link_bodies == {"worker_a": {"setup_upload_bytes": 17}}
    assert "generation_decode_online_body_bytes" not in h.core.audit.to_dict()
    assert h.core.audit.generation_attribution_supported_responses == 0


def test_source_listing_tokenizer_scopes_and_inventory_controls(harness):
    h = harness
    runtime = ModelsResource(h.core).list()["data"][0]["runtime"]
    assert runtime["verification_window"] == "request_end"
    assert runtime["streaming"] == "buffered_until_verified"
    assert runtime["lookup_dtype"] == "original_float32"
    assert runtime["kv_reuse"] is False
    assert runtime["source_lock_digest"] == h.source.source_lock_digest
    with h.core.tokenizer_scope():
        with h.core.tokenizer_scope():
            assert h.core._response_input_tokens("gkm-fixture", "hello") == 5
        h.responses.create(input="hello")
        # Generation transferred custody from this sizing scope to its attempt;
        # the terminal response already retired that exact public tokenizer.
        assert h.source.tokenizers_created == 1 and h.source.tokenizers_closed == 1
    assert h.source.tokenizers_closed == 1
    assert h.core.prepared_inventory_status("gkm-fixture") == {
        "status": "not-applicable", "capacity": 0, "available": 0, "reserved": 0, "burned": 0,
    }
    for call in (
        lambda: RuntimeResource(h.core).preprocess(model="gkm-fixture"),
        lambda: h.core.prepare_response("gkm-fixture", "hello", 3),
        lambda: h.core.prepared_rows_for_response("gkm-fixture", "hello", 3),
    ):
        with pytest.raises(GatewayError, match="preparation inventory"):
            call()
    with pytest.raises(ProtocolError, match="control endpoint"):
        RuntimeResource(h.core).capabilities()
    h.core.close()
    assert not h.core.records and h.core._source is None
    with pytest.raises(ProtocolError, match="closed"):
        h.responses.create(input="hello")


def test_native_resolution_and_connection_checks_precede_source(harness, monkeypatch):
    h = harness

    def unavailable(value):
        raise ValueError("native GKM capability unavailable")

    monkeypatch.setattr(Experiment, "resolve", unavailable)
    with pytest.raises(ValueError, match="capability unavailable"):
        adapter.DelegatedClient(h.experiment, CONNECTIONS)
    assert not h.admitted
    monkeypatch.setattr(Experiment, "resolve", lambda value: h.core.experiment)
    for connections in ({}, {"worker_a": CONNECTIONS["worker_a"]},
                        dict(CONNECTIONS, worker_b=CONNECTIONS["worker_a"]),
                        dict(CONNECTIONS, worker_b=("file:///local", "key")),
                        dict(CONNECTIONS, worker_b=("http://localhost:12", ""))):
        with pytest.raises(ValueError):
            adapter.DelegatedClient(h.experiment, connections)
    for timeout in (None, True, 0, -1, float("inf")):
        with pytest.raises(ValueError, match="timeout"):
            adapter.DelegatedClient(h.experiment, CONNECTIONS, timeout=timeout)
    assert not h.admitted


def test_explicit_source_and_wrong_source_admission(harness, monkeypatch):
    h = harness
    h.core.close()

    def forbidden(experiment):
        raise AssertionError("explicit source must not invoke the source factory")

    monkeypatch.setattr(adapter, "_admit_source", forbidden)
    core = adapter.DelegatedClient(h.experiment, CONNECTIONS, timeout=0.5, source=h.source)
    try:
        assert core.create({"input": "hello"}).output_text == "world after"
    finally:
        core.close()
    h.source.contract["input_rows"] = 7
    core = adapter.DelegatedClient(h.experiment, CONNECTIONS, source=h.source)
    try:
        with pytest.raises(ValueError, match="source differs"):
            core.create({"input": "hello"})
        assert core._active is None
        assert len(h.calls) == 1
    finally:
        core.close()
