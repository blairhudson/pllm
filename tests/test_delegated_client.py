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
from pllm.runtime import tokenizer_admission
from pllm.runtime.client import (
    AsyncResponsesResource, ModelsResource, ProtocolError, ResponsesResource, RuntimeResource,
)
from pllm.runtime.delegated_decoder import GkmDecoderResult
from pllm.runtime.http_gateway import GatewayError, iter_chat_sse
from pllm.runtime.prepared_queue import PreparedCancellation, prepared_cancellation
from pllm.runtime.responses import ResponsesError, sse_event
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
                       "num_key_value_heads": 1, "hidden_size": 4, "vocab_size": 8,
                       "bos_token_id": 0, "eos_token_id": 1, "max_position_embeddings": 128}
        self.config_source_bytes = json.dumps(self.config).encode()
        self.source_float32_bytes = 128
        self.source = SimpleNamespace(manifest=SimpleNamespace(context_length=128))
        inner = Tokenizer(WordLevel({"<bos>": 0, "<eos>": 1, "<unk>": 2,
                                    "user": 3, ":": 4, "hello": 5, "world": 6, "after": 7},
                                   unk_token="<unk>"))
        inner.pre_tokenizer = Whitespace()
        inner.add_special_tokens(["<bos>", "<eos>", "<unk>"])
        self.tokenizer_options = {"model_max_length": 128, "add_bos_token": False,
                                  "bos_token": "<bos>", "eos_token": "<eos>"}
        self.tokenizer_descriptor = tokenizer_admission._source_descriptor(
            inner.to_str().encode(), self.config, self.tokenizer_options,
            policy=tokenizer_admission._DescriptorPolicy.DELEGATED,
            vocabulary=8, max_entries=256, allow_model_padding=False,
            manifest_template=None, error=ValueError, label="GKM adapter fixture")
        self.checkpoint_digest = "1" * 64
        self.source_lock_digest = "2" * 64
        self.tokenizer_digest = "3" * 64
        self.tokenizers_created = 0
        self.tokenizers_closed = 0
        self.encodings = []
        self.decodings = []
        self.ids_override = None

    def tokenizer(self):
        descriptor = self.tokenizer_descriptor
        return self.observe_tokenizer(TokenizersJSONTokenizer(
            Tokenizer.from_buffer(descriptor["model"]),
            descriptor["bos_token_id"], descriptor["eos_token_id"]))

    def observe_tokenizer(self, tokenizer):
        """Observe request use only, including the genuine provisional tokenizer."""
        self.tokenizers_created += 1
        owner = self

        class ObservedTokenizer:
            def __init__(self, inner):
                self.inner = inner

            def encode(self, text, *, add_bos=False):
                owner.encodings.append((text, add_bos))
                return (self.inner.encode(text, add_bos=add_bos)
                        if owner.ids_override is None else owner.ids_override)

            def decode(self, ids):
                owner.decodings.append(ids)
                return self.inner.decode(ids)

            def close(self):
                inner, self.inner = self.inner, None
                if inner is None:
                    return
                owner.tokenizers_closed += 1
                close = getattr(inner, "close", None)
                if close is not None:
                    close()

        return ObservedTokenizer(tokenizer)

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
    from pllm.deployment.execution import ExecutionLease, _LEASE_KEY, _LeaseResponses, _TokenizerView

    result = SimpleNamespace(experiment=h.experiment, snapshot=None, validate=lambda **kw: None)
    topology = SimpleNamespace(close=lambda: None, cleanup_complete=True)
    view = _TokenizerView(h.source.tokenizer_descriptor, h.source.config,
        SimpleNamespace(chat_template=None), original_config=h.source.config,
        source_lock_digest=h.source.source_lock_digest)
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
    with pytest.raises(tokenizer_admission._PromptMetadataError,
                       match="^prepared metadata/source association changed$"):
        list(stream)
    assert h.calls == [] and h.core._active is None
    assert h.source.tokenizers_created == h.source.tokenizers_closed == 1
    assert not h.core.records


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
def harness(monkeypatch, tmp_path):
    experiment = Experiment(name="gkm-adapter-unit",
        pipeline=OpenWeightVerifiableCpu(Model.path(str(tmp_path))),
        deployment=Deployment.local(root="local://gkm-adapter-unit"),
        budget=ExecutionBudget(requests=4, max_input_tokens=8, max_new_tokens=3))
    resolved = []

    def resolve(value):
        resolved.append(value)
        return SimpleNamespace(client_runtime="compiled_gkm_v1", model="gkm-fixture")

    monkeypatch.setattr(Experiment, "resolve", resolve)
    source = Source(experiment)
    (tmp_path / "config.json").write_bytes(source.config_source_bytes)
    (tmp_path / "tokenizer.json").write_bytes(source.tokenizer_descriptor["model"])
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(source.tokenizer_options))
    from pllm.runtime.responses import _prompt_snapshot

    model_spec = _prompt_snapshot(experiment.pipeline.model.to_spec())
    root = tmp_path.resolve()
    construct = tokenizer_admission._PromptMetadataSnapshot.tokenizer

    def observed_snapshot_tokenizer(snapshot):
        tokenizer = construct(snapshot)
        if snapshot.root == root and snapshot.model_spec == model_spec:
            return source.observe_tokenizer(tokenizer)
        return tokenizer

    monkeypatch.setattr(tokenizer_admission._PromptMetadataSnapshot, "tokenizer",
                        observed_snapshot_tokenizer)
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


def test_harness_metadata_matches_owned_source_without_resolving_it(harness, tmp_path):
    h = harness
    assert {path.name for path in tmp_path.iterdir()} == {
        "config.json", "tokenizer.json", "tokenizer_config.json"}
    assert h.source.model is h.experiment.pipeline.model
    assert (tmp_path / "config.json").read_bytes() == h.source.config_source_bytes
    assert (tmp_path / "tokenizer.json").read_bytes() == h.source.tokenizer_descriptor["model"]
    snapshot = tokenizer_admission._capture_prompt_metadata(h.experiment.pipeline.model,
        policy=tokenizer_admission._DescriptorPolicy.DELEGATED, capacity="capsule",
        input_rows=8, outputs=3, expected_source_lock=h.source.source_lock_digest)
    snapshot.check_source(h.source)
    assert snapshot.tokenizer_descriptor == h.source.tokenizer_descriptor
    assert snapshot.tokenizer_descriptor["type"] == "tokenizer_json"
    assert snapshot.tokenizer_descriptor["vocab_size"] == 8
    assert snapshot.tokenizer_descriptor["bos_token_id"] == 0
    assert snapshot.tokenizer_descriptor["eos_token_id"] == 1
    assert snapshot.tokenizer_descriptor["add_bos_token"] is False
    assert snapshot.tokenizer_descriptor["chat_template"] is None
    assert snapshot.config == h.source.config
    lease, _ = prompt_lease(h)
    lease._tokenizer._check_prompt_snapshot(snapshot, h.experiment.pipeline.model)
    lease._tokenizer._check_owned_prompt_metadata(h.source, h.experiment.pipeline.model)
    assert h.source.encodings == h.source.decodings == []
    assert h.source.tokenizers_created == h.source.tokenizers_closed == 0
    assert not h.core._source_checked and h.admitted == h.calls == []
    assert h.resolved == [h.experiment]  # Only the existing client profile double.


@pytest.mark.parametrize("owner", ["snapshot", "source"])
@pytest.mark.parametrize("ids_override", [None, [7]])
def test_harness_shared_tokenizer_observer_preserves_ids_and_retirement(harness, owner, ids_override):
    h = harness
    snapshot = tokenizer_admission._capture_prompt_metadata(h.experiment.pipeline.model,
        policy=tokenizer_admission._DescriptorPolicy.DELEGATED, capacity="capsule",
        input_rows=8, outputs=3)
    assert h.source.tokenizers_created == 0  # Parsing metadata is not request use.
    tokenizer = snapshot.tokenizer() if owner == "snapshot" else h.source.tokenizer()
    h.source.ids_override = ids_override
    expected = [3, 4, 5, 2, 4] if ids_override is None else ids_override
    assert tokenizer.encode("user: hello\nassistant:", add_bos=False) == expected
    assert tokenizer.decode([1, 6, 7]) == "world after"
    tokenizer.close()
    tokenizer.close()
    assert tokenizer.inner is None
    assert h.source.encodings == [("user: hello\nassistant:", False)]
    assert h.source.decodings == [[1, 6, 7]]
    assert h.source.tokenizers_created == h.source.tokenizers_closed == 1
    assert not h.core._source_checked and h.admitted == h.calls == []


def test_harness_snapshot_instrumentation_matches_both_model_and_root(harness, tmp_path):
    h = harness
    snapshot = tokenizer_admission._capture_prompt_metadata(h.experiment.pipeline.model,
        policy=tokenizer_admission._DescriptorPolicy.DELEGATED, capacity="capsule",
        input_rows=8, outputs=3)
    # Only test the observer's filter; these copies authorize no request/source.
    for other in (replace(snapshot, root=tmp_path.parent), replace(snapshot, model_spec=())):
        tokenizer = other.tokenizer()
        assert isinstance(tokenizer, tokenizer_admission._PromptTokenizer)
        try:
            assert tokenizer.encode("hello", add_bos=False) == [5]
        finally:
            tokenizer.close()
    assert h.source.encodings == h.source.decodings == []
    assert h.source.tokenizers_created == h.source.tokenizers_closed == 0
    assert h.admitted == h.calls == []


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


@pytest.mark.parametrize("method,stage", [
    ("_prepare_request_prompt", "prepare_prompt"), ("_ensure_source", "validate_source"),
    ("_encode", "bind_encode"), ("_execute", "transport"), ("_validate_result", "accept_result"),
])
def test_request_protocol_error_keeps_identity_traceback_and_bounded_stage(harness, monkeypatch, method, stage):
    h = harness
    h.core._ensure_source()  # Pure admitted fixture; never open an on-disk checkpoint.
    body = {"private": "secret-prompt-coefficients-/tensor/path"}
    error = ProtocolError("secret-worker-message", body=body)

    def fail(*args, **kwargs):
        if method == "_execute":
            args[-1].record_audit(transport_counters())
        raise error

    monkeypatch.setattr(h.core, method, fail)
    stream = h.responses.create(model="gkm-fixture", input="hello", max_output_tokens=2,
                                stream=True, temperature=0)
    with pytest.raises(ProtocolError) as caught:
        next(stream)
    assert caught.value is error and error.body is body
    frames = []
    tb = error.__traceback__
    while tb is not None:
        frames.append(tb.tb_frame.f_code.co_name)
        tb = tb.tb_next
    assert "fail" in frames and "_produce" in frames
    assert error.request_failure_context == {
        "stage": stage, "role": "client", "code": "gkm_stage_failed", "http_status": None}
    assert error.status_code == 500  # Default SDK status is not HTTP-response evidence.
    assert h.core._active is None and not h.core._tokenizer_scopes
    assert h.source.tokenizers_created == h.source.tokenizers_closed
    assert not h.core.records and h.core.audit.generation_completed_responses == 0
    if stage == "transport":
        assert h.core.audit.role_link_bodies["worker_a"]["teardown_upload_bytes"] == 7
    stream.close()


@pytest.mark.parametrize("role", ["worker_a", "worker_b"])
def test_deeper_worker_context_survives_client_stage(harness, monkeypatch, role):
    harness.core._ensure_source()
    error = ProtocolError("private response", 503, b"private raw body")
    context = {"stage": "transport", "role": role,
               "code": "worker_http_rejected", "http_status": 503}
    error.annotate_request_failure(context)

    def fail(*args):
        args[-1].record_audit(transport_counters())
        raise error

    monkeypatch.setattr(adapter, "_execute_request", fail)
    with pytest.raises(ProtocolError) as caught:
        harness.responses.create(input="hello")
    assert caught.value is error and error.request_failure_context == context
    assert error.body == b"private raw body"
    assert harness.core.audit.role_link_bodies[role]["teardown_download_bytes"] == 2


def test_sdk_shaped_stream_keeps_model_and_greedy_defaults(harness):
    harness.core._ensure_source()
    stream = harness.responses.create(model="gkm-fixture", input="hello", max_output_tokens=2,
                                      stream=True, temperature=0)
    events = list(stream)
    terminal = events[-1].response
    assert terminal["model"] == "gkm-fixture"
    assert terminal["temperature"] == 0 and terminal["top_p"] == 1
    assert terminal["incomplete_details"] == {"reason": "max_output_tokens"}
    assert harness.calls == [((3, 4, 5, 2, 4), 2)]
    assert harness.core.audit.generation_completed_responses == 1
    assert harness.core.audit.generation_output_tokens == 2


def test_transport_audit_cleanup_not_labeled_as_request_stage(harness, monkeypatch):
    harness.core._ensure_source()
    error = ProtocolError("private cleanup")

    def fail(*args):
        raise error

    monkeypatch.setattr(harness.core.audit, "finish_transport", fail)
    with pytest.raises(ProtocolError) as caught:
        harness.responses.create(input="hello")
    assert caught.value is error and error.request_failure_context is None
    assert harness.core._active is None


def test_decode_protocol_error_has_no_request_stage(harness, monkeypatch):
    h = harness
    h.core._ensure_source()
    original = h.source.tokenizer
    error = ProtocolError("private decoded message")

    def fail_decode(*args):
        raise error

    def tokenizer():
        inner = original()
        return SimpleNamespace(encode=inner.encode, decode=fail_decode, close=inner.close)

    monkeypatch.setattr(h.source, "tokenizer", tokenizer)
    with pytest.raises(ProtocolError) as caught:
        h.responses.create(input="hello")
    assert caught.value is error and error.request_failure_context is None
    assert h.core._active is None and not h.core.records
    assert h.source.tokenizers_created == h.source.tokenizers_closed


def test_scope_cleanup_replacing_transport_error_not_mislabeled(harness, monkeypatch):
    from contextlib import contextmanager

    h = harness
    h.core._ensure_source()
    original_scope = h.core.tokenizer_scope
    request_error, cleanup_error = ProtocolError("private request"), ProtocolError("private cleanup")

    def fail(*args):
        raise request_error

    @contextmanager
    def scope():
        try:
            with original_scope():
                yield
        finally:
            raise cleanup_error

    monkeypatch.setattr(h.core, "_execute", fail)
    monkeypatch.setattr(h.core, "tokenizer_scope", scope)
    with pytest.raises(ProtocolError) as caught:
        h.responses.create(input="hello")
    assert caught.value is cleanup_error and cleanup_error.request_failure_context is None
    assert h.core._active is None and not h.core._tokenizer_scopes


@pytest.mark.parametrize("runtime", ["compiled_fss_v1", "compiled_replicated_v1"])
def test_non_gkm_client_does_not_inherit_gkm_stage_labels(harness, monkeypatch, runtime):
    harness.core._ensure_source()
    error = ProtocolError("private other-method failure")

    def fail(*args):
        raise error

    monkeypatch.setattr(harness.core, "runtime_id", runtime)
    monkeypatch.setattr(harness.core, "_execute", fail)
    with pytest.raises(ProtocolError) as caught:
        harness.responses.create(input="hello")
    assert caught.value is error and error.request_failure_context is None


@pytest.mark.parametrize("role_index", [0, 1])
@pytest.mark.parametrize("namespace", ["delegation", "fss", "replicated"])
def test_actual_worker_http_rejection_context_keeps_counters(role_index, namespace):
    """HTTP boundary double only: no network, I/O owner or native request."""
    from contextlib import asynccontextmanager
    from pllm.runtime.delegated_transport import _Peers

    @asynccontextmanager
    async def stream(*args, **kwargs):
        async def chunks():
            yield b"secret-worker-body"

        yield SimpleNamespace(status_code=503, headers={},
                              request=SimpleNamespace(content=b"request"), aiter_raw=chunks)

    peers = _Peers()
    peers.roles = ("worker_a", "worker_b")
    peers.namespace, peers.request_id, peers.phase = namespace, "fixed", "prefill"
    peers.clients = [SimpleNamespace(stream=stream), SimpleNamespace(stream=stream)]
    prefix = f"role_link.{peers.roles[role_index]}.online_"
    peers.counts = {prefix + direction + "_bytes": 0 for direction in ("upload", "download")}
    with pytest.raises(ProtocolError) as caught:
        asyncio.run(peers._post(role_index, "step", b"request", phase="online", limit=64, timeout=1))
    context = {
        "stage": "transport", "role": peers.roles[role_index],
        "code": "worker_http_rejected", "http_status": 503}
    assert caught.value.request_failure_context == (context if namespace == "delegation" else None)
    assert caught.value.status_code == 503
    assert peers.counts == {prefix + "upload_bytes": 7, prefix + "download_bytes": 18}
    assert peers.phase_bytes == {"prefill": 25, "decode": 0}
    assert peers.calls == 0 and peers.server_ns == 0 and not peers.unknown


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
    with pytest.raises(ValueError, match="verification failed") as caught:
        result = h.responses.create(input="hello", stream=streaming)
        if streaming:
            next(result)
    assert getattr(caught.value, "request_failure_context", None) is None
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


@pytest.mark.parametrize("ids,error_type,message", [
    ([], ResponsesError, "tokenized input contains invalid prompt IDs"),
    (list(range(9)), GatewayError, "request input exceeds selected budget"),
    ([True], ResponsesError, "tokenized input contains invalid prompt IDs"),
    ([-1], ResponsesError, "tokenized input contains invalid prompt IDs"),
    ([8], GatewayError, "request input exceeds selected budget"),
    ([1.0], ResponsesError, "tokenized input contains invalid prompt IDs"),
])
def test_token_admission_precedes_transport(harness, ids, error_type, message):
    h = harness
    h.source.ids_override = ids
    with pytest.raises(error_type, match=f"^{message}$"):
        h.responses.create(input="hello")
    assert h.admitted == h.calls == [] and h.core._active is None
    assert h.source.encodings == [("user: hello\nassistant:", False)]
    assert not h.source.decodings and not h.core.records
    assert h.source.tokenizers_created == h.source.tokenizers_closed == 1


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
            with pytest.raises(ProtocolError, match="^client already owns request/tokenizer cleanup$"):
                h.core._response_input_tokens("gkm-fixture", "hello")
            assert h.source.tokenizers_created == h.source.tokenizers_closed == 0
            assert h.source.encodings == h.source.decodings == []
        assert h.source.tokenizers_created == h.source.tokenizers_closed == 0
    assert h.source.tokenizers_created == h.source.tokenizers_closed == 0
    assert h.core._active is None and not h.core._tokenizer_scopes
    assert h.calls == [] and not h.core.records
    assert h.core._response_input_tokens("gkm-fixture", "hello") == 5
    assert h.source.tokenizers_created == h.source.tokenizers_closed == 1
    assert h.source.encodings == [("user: hello\nassistant:", False)]
    assert h.source.decodings == [] and h.calls == []
    h.responses.create(input="hello")
    assert h.source.tokenizers_created == h.source.tokenizers_closed == 2
    assert h.source.encodings == [("user: hello\nassistant:", False)] * 2
    assert h.source.decodings == [[1, 6, 7]]
    assert h.calls == [((3, 4, 5, 2, 4), 3)]
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


def _count_dashboard(h):
    """Real dashboard request/history path over the existing tiny adapter fixture."""
    from pllm.runtime import benchmark_failure, dashboard
    from pllm.runtime.benchmark_history import BenchmarkHistory

    config = dashboard.DashboardConfig("127.0.0.1", 1, "unused", "gkm-fixture", 3,
                                       backend="native")
    store = SimpleNamespace(finish_run_window=lambda _: {}, snapshot=lambda: {
        "services": {"pllm-preparation": {}}, "traffic": {}})
    runtime = dashboard.DashboardRuntime(config, store)
    runtime._topology = SimpleNamespace(requires_preparation=False, statuses=[],
        startup_breadcrumbs={"phase": "ready", "spawned_roles": [], "ready_roles": []})
    runtime._client = SimpleNamespace(_core=h.core, responses=h.responses,
                                      tokenizer_scope=h.core.tokenizer_scope)
    runtime._active_run = dashboard._RunCapture("count-run", 3, False, 1, {}, 1, 0)
    runtime._has_started_run = True
    runtime._state.update(phase="preparing", run_id="count-run", prompt="hello")
    runtime._delegated = True
    runtime.memory_preflight = {"estimate": {"max_input_tokens": 5}}
    runtime._audit_snapshot = lambda: {}
    runtime._services_healthy = lambda: True
    runtime._capture_online_telemetry = lambda: None
    runtime._benchmark_failure_diagnostics = benchmark_failure.FailureDiagnostics(runtime)
    runtime.history = BenchmarkHistory(":memory:")
    return runtime


@pytest.mark.parametrize("input_limit", [4, 5])
def test_dashboard_real_count_retires_before_execution_scope(harness, monkeypatch, input_limit):
    from pllm.runtime import dashboard

    h = harness
    runtime = _count_dashboard(h)
    runtime.memory_preflight["estimate"]["max_input_tokens"] = input_limit
    observed, cleanup = [], []
    observe_tokenizer = h.source.observe_tokenizer

    def observe(tokenizer):
        observed.append((bool(h.core._tokenizer_scopes), h.core._count_preparation is not None))
        return observe_tokenizer(tokenizer)

    def execute(*args):
        assert h.core._tokenizer_scopes and h.core._count_preparation is None
        assert h.source.tokenizers_closed == 1  # Preflight already retired.
        return h.execute(*args)

    evict = h.core.evict_response

    def evict_response(response_id):
        assert h.core._tokenizer_scopes  # Dashboard cleanup still inside execution scope.
        cleanup.append(response_id)
        return evict(response_id)

    monkeypatch.setattr(h.source, "observe_tokenizer", observe)
    monkeypatch.setattr(adapter, "_execute_request", execute)
    monkeypatch.setattr(h.core, "evict_response", evict_response)
    monkeypatch.setattr(dashboard.time, "sleep", lambda _: None)
    try:
        runtime._execute_run("count-run", "hello", 3)
        saved = runtime.history.get("count-run")
        assert runtime._active_run is None and runtime._state["phase"] == "ready"
        assert h.core._count_preparation is None and not h.core._tokenizer_scopes
        assert h.core._active is None and not h.core.records
        if input_limit == 5:
            assert saved["status"] == "completed" and saved["failure"] is None
            assert saved["tokens"] == {"input_tokens": 5, "output_tokens": 3,
                                       "total_tokens": 8, "authoritative": True}
            assert runtime._state["text"] == "world after"
            assert observed == [(False, True), (True, False)]
            assert h.calls == [((3, 4, 5, 2, 4), 3)] and len(cleanup) == 1
            assert h.source.tokenizers_created == h.source.tokenizers_closed == 2
            assert h.source.encodings == [("user: hello\nassistant:", False)] * 2
        else:
            assert saved["status"] == "failed" and saved["failure"]["type"] == "ValueError"
            assert runtime._state["error"] == "ValueError: input exceeds the memory-admitted workload"
            assert saved["timestamps"]["online_started_at_ns"] is None
            assert saved["timestamps"]["preparation_finished_at_ns"] is not None
            assert observed == [(False, True)] and cleanup == h.calls == h.admitted == []
            assert h.source.tokenizers_created == h.source.tokenizers_closed == 1
            assert h.source.encodings == [("user: hello\nassistant:", False)]
    finally:
        runtime.history.close()


@pytest.mark.parametrize("failure", ["external_scope", "closed_client"])
def test_dashboard_real_count_failure_preserves_origin_before_finish(harness, monkeypatch, tmp_path, failure):
    from contextlib import ExitStack
    from pllm._cli.benchmark_output import BenchmarkOutput
    from pllm.runtime import benchmark_cli, benchmark_failure, dashboard

    h = harness
    runtime = _count_dashboard(h)
    errors = []
    annotations = []
    finish = runtime._finish_run
    annotate = ProtocolError.annotate_request_failure
    expected = {"stage": "input_count", "role": "client", "http_status": None,
                "code": "count_owner_busy" if failure == "external_scope" else "input_count_failed"}

    def observe_annotation(error, context):
        annotate(error, context)
        annotations.append((error, error.__traceback__, error.request_failure_context))

    def finish_run(capture, status, error):
        assert isinstance(error, ProtocolError)
        errors.append(error)
        assert error.request_failure_context == expected
        assert runtime._benchmark_failure_diagnostics.capture()["request_failure_context"] == expected
        assert bool(h.core._tokenizer_scopes) == (failure == "external_scope")
        first = runtime._benchmark_failure_diagnostics.capture()
        runtime._topology.startup_breadcrumbs["phase"] = "role_startup"
        assert benchmark_failure.capture_failure(runtime, error=ValueError("private-cleanup-secret")) == first
        finish(capture, status, error)

    monkeypatch.setattr(ProtocolError, "annotate_request_failure", observe_annotation)
    monkeypatch.setattr(runtime, "_finish_run", finish_run)
    monkeypatch.setattr(dashboard.time, "sleep", lambda _: None)
    try:
        with ExitStack() as scope:
            if failure == "external_scope":
                scope.enter_context(h.core.tokenizer_scope())
                with pytest.raises(ProtocolError) as direct:
                    h.core._response_input_tokens("gkm-fixture", "hello")
                assert direct.value.status_code == 409
                assert direct.value.request_failure_context == expected
            else:
                h.core.close()  # Real _admit/_check_open failure BEFORE busy guard.
            runtime._execute_run("count-run", "hello", 3)
            assert len(errors) == 1
            error = errors[0]
            own_annotations = [item for item in annotations if item[0] is error]
            assert own_annotations[-1][0] is error
            assert error.__traceback__ is own_annotations[-1][1]  # Dashboard used bare raise.
            assert error.request_failure_context is own_annotations[0][2]  # Deepest label retained.
            assert error.status_code == 409  # Local SDK status, never HTTP evidence.
            frames = []
            tb = error.__traceback__
            while tb is not None:
                frames.append(tb.tb_frame.f_code.co_name)
                tb = tb.tb_next
            assert frames.count("_execute_run") == 1 and "_response_input_tokens" in frames
            if failure == "closed_client":
                assert "_admit" in frames and "_check_open" in frames
            saved = runtime.history.get("count-run")
            assert saved["failure"]["diagnostics"]["request_failure_context"] == expected
            assert saved["failure"]["type"] == "ProtocolError"
            path = tmp_path / "count-failure.json"
            output = BenchmarkOutput(path, force=False, configuration={}, experiments=[None])
            output.start(1)
            output.failed(benchmark_cli.LoopbackBenchmarkError("private-cleanup-secret",
                failure_diagnostics=saved["failure"]["diagnostics"]))
            artifact = json.loads(path.read_text())
            assert artifact["failure"]["diagnostics"]["request_failure_context"] == expected
            assert "private-cleanup-secret" not in json.dumps(saved) + path.read_text()
            assert runtime._benchmark_failure_diagnostics.capture()["request_failure_context"] == expected
            assert h.source.encodings == h.source.decodings == h.calls == h.admitted == []
            assert h.source.tokenizers_created == h.source.tokenizers_closed == 0
        assert not h.core._tokenizer_scopes and h.core._count_preparation is None
    finally:
        runtime.history.close()


def test_real_count_busy_history_crosses_run_once_worker_receiver_and_artifact(harness, monkeypatch, tmp_path):
    """Continuous real failure propagation; HTTP and process launch stay in memory."""
    from collections import deque
    from multiprocessing.reduction import ForkingPickler
    import signal

    import httpx

    from pllm._cli.benchmark_output import BenchmarkOutput
    from pllm.runtime import benchmark_cli, benchmark_isolation, dashboard

    h = harness
    runtime = _count_dashboard(h)
    secret = "private-count-message-body-extra-/tensor/path"
    sdk_errors, produced_errors, requests = [], [], []
    queued, transmitted, closed = deque(), [], []
    initialize = ProtocolError.__init__
    finish = runtime._finish_run
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def initialize_error(error, message, *args, **kwargs):
        initialize(error, message, *args, **kwargs)
        if message == "client already owns request/tokenizer cleanup":
            # Poison only the genuine guard's exception, never replace the guard/count.
            error.args = (secret,)
            error.body = {"private": secret}
            error.private_extra = secret
            sdk_errors.append(error)

    def finish_run(capture, status, error):
        assert error is sdk_errors[0] and error.status_code == 409
        assert h.core._tokenizer_scopes and h.core._count_preparation is None
        assert runtime._benchmark_failure_diagnostics.capture()["request_failure_context"] == error.request_failure_context
        finish(capture, status, error)

    def respond(request):
        requests.append(request.url.path)
        if request.url.path == "/api/run":
            payload = json.loads(request.content)
            run_id = payload["request_id"]
            runtime._active_run.run_id = run_id
            runtime._state["run_id"] = run_id
            runtime._execute_run(run_id, payload["prompt"], payload["max_output_tokens"])
            return httpx.Response(200, json={"status": "started"})
        if request.url.path == "/api/snapshot":
            return httpx.Response(200, json={"run": {**runtime._state, "active_run": None}})
        if request.url.path.startswith("/api/runs/"):
            return httpx.Response(200, json=runtime.history.get(request.url.path.rsplit("/", 1)[1]))
        raise AssertionError(request.url.path)

    def loopback_boundary(**options):
        # Bypass only role/server startup; the real HTTP polling/failure conversion runs.
        with httpx.Client(transport=httpx.MockTransport(respond), base_url="http://dashboard") as client:
            try:
                return benchmark_cli._run_once(client, SimpleNamespace(poll=lambda: None),
                    prompt="hello", max_output_tokens=3, timeout_seconds=10)
            except benchmark_cli.LoopbackBenchmarkError as error:
                produced_errors.append(error)
                error.private_extra = secret
                error.failure_diagnostics["private_extra"] = secret
                raise

    def send_message(message):
        encoded = bytes(ForkingPickler.dumps(message))
        queued.append(encoded)
        transmitted.append(encoded)

    receive = SimpleNamespace(recv=lambda: ForkingPickler.loads(queued.popleft()),
                              close=lambda: closed.append("receive"))
    send = SimpleNamespace(send=send_message, close=lambda: closed.append("send"))

    def process(*, target, args):
        assert target is benchmark_isolation._worker
        return SimpleNamespace(start=lambda: target(*args), join=lambda **kw: None,
                               is_alive=lambda: False)

    context = SimpleNamespace(Pipe=lambda **kw: (receive, send), Process=process)
    monkeypatch.setattr(ProtocolError, "__init__", initialize_error)
    monkeypatch.setattr(runtime, "_finish_run", finish_run)
    monkeypatch.setattr(dashboard.time, "sleep", lambda _: None)
    monkeypatch.setattr(benchmark_cli, "run_loopback_benchmark", loopback_boundary)
    monkeypatch.setattr(benchmark_isolation.multiprocessing, "get_context", lambda _: context)
    try:
        with h.core.tokenizer_scope():
            with pytest.raises(benchmark_cli.LoopbackBenchmarkError) as received:
                benchmark_isolation.run_isolated_loopback_benchmark()
            assert h.core._tokenizer_scopes  # External owner was not waived or retired.
        assert len(sdk_errors) == len(produced_errors) == 1
        original = sdk_errors[0]
        expected = {"stage": "input_count", "role": "client",
                    "code": "count_owner_busy", "http_status": None}
        assert original.request_failure_context == expected  # Dashboard fallback kept deeper origin.
        assert original.body == {"private": secret} and str(original) == secret
        frames = []
        tb = original.__traceback__
        while tb is not None:
            frames.append(tb.tb_frame.f_code.co_name)
            tb = tb.tb_next
        assert frames.count("_execute_run") == 1 and "_response_input_tokens" in frames
        assert received.value is not produced_errors[0]
        assert str(produced_errors[0]) == "benchmark run failed: ProtocolError"
        assert received.value.failure_diagnostics["request_failure_context"] == expected
        assert not hasattr(received.value, "private_extra")
        assert [ForkingPickler.loads(item)[0] for item in transmitted] == ["failure_diagnostics", "error"]
        assert secret.encode() not in b"".join(transmitted)
        assert not queued and closed == ["send", "send", "receive"]
        assert requests[:2] == ["/api/run", "/api/snapshot"] and len(requests) == 3
        saved = runtime.history.get(runtime._state["run_id"])
        assert saved["failure"]["diagnostics"]["request_failure_context"] == expected
        path = tmp_path / "count-isolated-failure.json"
        output = BenchmarkOutput(path, force=False, configuration={}, experiments=[None])
        output.start(1)
        output.failed(received.value)
        artifact = json.loads(path.read_text())
        assert artifact["failure"]["diagnostics"]["request_failure_context"] == expected
        assert secret not in json.dumps(saved) + path.read_text()
        assert h.source.encodings == h.source.decodings == h.calls == h.admitted == []
        assert not h.core._tokenizer_scopes and h.core._count_preparation is None
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        runtime.history.close()


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
