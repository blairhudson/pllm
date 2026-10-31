"""Bounded Responses adapter for the request-verified GKM runtime.

The transport owns native execution and retirement. It returns a GkmDecoderResult
only after native finish succeeds; no token, Response, or event callback crosses
that boundary. Streaming is a buffered serialization of the accepted result.
Source admission is metadata-only and lazy. Neither accepted KV nor prompt/token
history is retained by this client.

Internal integration seam: ``cancelled`` passed to execute_request is callable
and has ``record_audit(counters)``. The transport may report one final, flat,
per-request snapshot in its finally block, including failed/retired work. Missing
transport counters remain unavailable, not zero. The optional private
``_result_hook`` observes one bounded accepted result; production leaves it None.
"""

from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
import json
import math
import threading
import time
from types import MappingProxyType
from typing import Any

from pllm.configuration import Experiment

from .client import (
    PrivacyAudit, ProtocolError, ResponseStream, _ScopedTokenizer,
    _normalized_inference_endpoint, _response_contract_fields,
)
from .delegated_decoder import GkmDecoderResult, result_payload_limit
from .delegated_protocol import BATCH_EXECUTION, clear_failure_frames
from .http_gateway import GatewayError, validate_response_body
from .prepared_queue import PreparedCancellation, prepared_cancellation
from .responses import (normalize_input, _owned_prompt_source, _prepare_prompt,
                        _prepared_prompt_request, ResponsesError)
from .types import Response, response_events


_delegated_stack = threading.local()


class _DelegatedExtent:
    """Actual physical call stack, NOT retirement/completion authority."""

    def __init__(self, core, attempt=None):
        self.core, self.attempt = core, attempt
        self.parent = None
        self.active = self.failed = False
        self.normal_stop = False
        self.tickets = []
        self.generation = (attempt._generation if attempt is not None
                           else getattr(core, "_attempt_generation", 0))

    def __enter__(self):
        return self

    def activate(self):
        self.parent = getattr(_delegated_stack, "frame", None)
        _delegated_stack.frame = self
        self.active = True

    def __exit__(self, kind, primary, traceback):
        if not self.active:
            return False
        assert _delegated_stack.frame is self
        _delegated_stack.frame = self.parent
        self.active = False
        if self.failed or (primary is not None and not self.normal_stop):
            parent = self.parent
            while parent is not None:
                parent.failed = True
                parent = parent.parent
            for lease, ticket in self.tickets:
                lease._fail_deferred_close(ticket)
            return False
        # No attempt execution lock or live same-core boundary remains here.
        for lease, ticket in self.tickets:
            try:
                lease._resume_deferred_close(ticket, self.core, self.generation)
            except BaseException as cleanup:
                # Service runs AFTER pop: poison every still-live ancestor,
                # including other cores and ancestors without a ticket yet.
                parent = self.parent
                while parent is not None:
                    parent.failed = True
                    parent = parent.parent
                if primary is None:
                    raise
                primary.add_note(f"deferred lease cleanup failed: {type(cleanup).__name__}")
        return False


def _defer_delegated_close(core, lease):
    """Recognize only a live inherited next/retire/core-close on THIS thread."""
    frame = getattr(_delegated_stack, "frame", None)
    outer = attempt = None
    while frame is not None:
        if frame.active and frame.core is core:
            outer = frame
            if attempt is None and type(frame.attempt) is _Attempt and frame.attempt.owner is core:
                attempt = frame.attempt
        frame = frame.parent
    if outer is None:
        return False
    ticket = lease._defer_from_delegated(core, attempt, outer.generation)
    if not any(owner is lease and saved is ticket for owner, saved in outer.tickets):
        outer.tickets.append((lease, ticket))
    if attempt is not None:
        attempt.cancel()  # Original only; never reread/cancel a later active seat.
    return True


def _has_delegated_extent(core):
    frame = getattr(_delegated_stack, "frame", None)
    while frame is not None:
        if frame.active and frame.core is core:
            return True
        frame = frame.parent
    return False


_BODY_BYTES = 64 << 10
_OUTPUT_BYTES = 1 << 20
_RECORD_BYTES = 4 << 20
_RECORD_COUNT = 64
_TRANSPORT_FIELDS = frozenset({
    "inference_upload_bytes", "inference_download_bytes",
    "inference_control_upload_bytes", "inference_control_download_bytes",
    "inference_server_ns", "inference_stage_calls", "online_steps",
    "generation_prefill_online_body_bytes", "generation_decode_online_body_bytes",
})
_BATCH_TRANSPORT_FIELDS = frozenset({"batch_exchanges", "native_scalar_claims"})
_ROLE_FIELDS = frozenset(
    f"role_link.{role}.{phase}_{direction}_bytes"
    for role in ("worker_a", "worker_b")
    for phase in ("setup", "online", "teardown")
    for direction in ("upload", "download")
)
_COUNTER_FIELDS = _TRANSPORT_FIELDS | _ROLE_FIELDS


def _counter_fields(roles):
    return _TRANSPORT_FIELDS | frozenset(
        f"role_link.{role}.{phase}_{direction}_bytes" for role in roles
        for phase in ("setup", "online", "teardown") for direction in ("upload", "download")
    )


def _admit_source(experiment):
    from .delegated_source import admit_source

    return admit_source(experiment)


def _execute_request(experiment, source, token_ids, output_tokens, connections, timeout, cancelled):
    from .delegated_transport import execute_request

    return execute_request(experiment, source, token_ids, output_tokens,
                           connections, timeout, cancelled)


def _unsupported(param, message):
    return GatewayError(message, param=param, code="unsupported_feature")


def _body_snapshot(body):
    """Bound work before shared validators serialize or walk caller-owned input."""
    remaining, nodes = _BODY_BYTES, 1024

    def charge(size):
        nonlocal remaining
        remaining -= size
        if remaining < 0:
            raise GatewayError("GKM request exceeds its 64 KiB body limit", status_code=413)

    def visit(value, depth=0):
        nonlocal nodes
        nodes -= 1
        if nodes < 0 or depth > 16:
            raise GatewayError("GKM request exceeds its structure limit", status_code=413)
        if type(value) is str:
            if len(value) > remaining:
                charge(len(value))
            charge(len(json.dumps(value, ensure_ascii=True)))
            return value
        if value is None or type(value) in (bool, int, float):
            if type(value) is int and value.bit_length() > 64:
                raise GatewayError("GKM request integer exceeds its limit")
            if type(value) is float and not math.isfinite(value):
                raise GatewayError("GKM request numbers must be finite")
            charge(len(json.dumps(value)))
            return value
        if type(value) in (list, dict):
            if len(value) > nodes:
                raise GatewayError("GKM request exceeds its structure limit", status_code=413)
            charge(2 + 2 * len(value))
            if type(value) is list:
                return [visit(item, depth + 1) for item in value]
            if any(type(key) is not str for key in value):
                raise GatewayError("GKM request object keys must be strings")
            return {visit(key, depth + 1): visit(item, depth + 1)
                    for key, item in value.items()}
        raise GatewayError("GKM request must contain only JSON values")

    if type(body) is not dict:
        raise GatewayError("Request body must be a JSON object")
    try:
        return visit(body)
    except RuntimeError as error:
        raise GatewayError("GKM request changed during admission") from error


def _text_input(value):
    """Reject input features that normalize_input would ignore or stringify."""
    if isinstance(value, str):
        return
    for item in value:
        kind = item.get("type", "message")
        if kind in ("input_text", "text"):
            if set(item) - {"type", "text"} or not isinstance(item.get("text"), str):
                raise GatewayError("GKM text items require only type and string text", param="input")
            continue
        if kind != "message" or set(item) - {"type", "role", "content"}:
            raise _unsupported("input", "GKM requires fresh text messages; input references and tools are unsupported")
        if item.get("role", "user") not in ("system", "developer", "user", "assistant"):
            raise _unsupported("input", "GKM input role is unsupported")
        content = item.get("content")
        if isinstance(content, str):
            continue
        if not isinstance(content, list) or not content:
            raise GatewayError("GKM message content must be text", param="input")
        for part in content:
            if (not isinstance(part, dict) or set(part) - {"type", "text"}
                    or part.get("type") not in ("input_text", "output_text", "text")
                    or not isinstance(part.get("text"), str)):
                raise _unsupported("input", "GKM supports only string text content")


class _DelegatedAudit(PrivacyAudit):
    """Ordinary audit attributes, omitting transport measurements when unknown."""

    def __init__(self, lock, fields=_COUNTER_FIELDS):
        super().__init__()
        self._lock = lock
        self._totals = dict.fromkeys(fields, 0)

    def begin_transport(self):
        for name in _TRANSPORT_FIELDS | (_BATCH_TRANSPORT_FIELDS & self._totals.keys()):
            setattr(self, name, None)
        self.role_link_bodies.clear()

    def finish_transport(self, snapshot):
        snapshot = snapshot or {}
        for name, prior in self._totals.items():
            value = snapshot.get(name)
            total = None if value is None or prior is None else prior + value
            self._totals[name] = total
            if name in _TRANSPORT_FIELDS | _BATCH_TRANSPORT_FIELDS:
                setattr(self, name, total)
            elif total is not None and name.startswith("role_link."):
                _, role, metric = name.split(".", 2)
                self.role_link_bodies.setdefault(role, {})[metric] = total

    def to_dict(self):
        with self._lock:
            values = {name: value for name in self.__dataclass_fields__
                      if not name.startswith("_") and name != "role_link_bodies"
                      and (value := getattr(self, name)) is not None}
            for role, counters in self.role_link_bodies.items():
                values.update({f"role_link.{role}.{name}": value for name, value in counters.items()})
            values.update({name: value for name, value in self._totals.items()
                            if (name.startswith("peer_link.") or name in _BATCH_TRANSPORT_FIELDS)
                            and value is not None})
            return values


class _NoProviderHTTP:
    def __getattr__(self, name):
        raise ProtocolError("GKM runtime has no ordinary provider control endpoint", 400)


class _Attempt:
    """One serialized iterator, cancellation handle, and exclusive request seat."""

    def __init__(self, owner, body):
        self.owner = owner
        owner._attempt_generation = getattr(owner, "_attempt_generation", 0) + 1
        self._generation = owner._attempt_generation
        self.body = body
        self.streaming = body.get("stream", False)
        self.lock = threading.RLock()
        self.cancelled = threading.Event()
        self.cancellation = prepared_cancellation.get() or PreparedCancellation()
        self.buffer = None
        self.response_id = None
        self.closed = False
        self._state_lock = threading.Lock()
        self._completion = threading.Event()
        self._retired = False
        self._outstanding_calls = 0
        self._invocation_owner = self._invocation_thread = None
        self.running_thread = None
        self.transport_audit = None
        self._fss_control_boundary = self._fss_control_origin = None
        self._fss_control_failed = False
        prepared = _prepared_prompt_request.get()
        self._prepared_prompt = None if prepared is None else prepared.claim()

    def record_control_boundary(self, role, action, status):
        """Best-effort one-slot diagnostic, relative to first control boundary."""
        try:
            if (self.owner.runtime_id != "compiled_fss_v1" or self.closed
                    or self.owner._active is not self or self._fss_control_failed):
                return
            from .benchmark_failure import sanitize_fss_control_boundary
            if sanitize_fss_control_boundary((role, action, status, 0.0)) is None:
                self._fss_control_boundary = None
                return
            now = time.monotonic()
            if type(now) not in (int, float) or not math.isfinite(now):
                self._fss_control_boundary = None
                return
            origin = now if self._fss_control_origin is None else self._fss_control_origin
            marker = (role, action, status, now - origin)
            if sanitize_fss_control_boundary(marker) is not None:
                self._fss_control_origin = origin
                self._fss_control_boundary = marker
            else:
                self._fss_control_boundary = None
        except BaseException:
            self._fss_control_boundary = None
            pass  # Neither diagnostic clocks nor callbacks are execution authority.

    def fail_control_boundary(self):
        """Freeze/pass immutable data BEFORE native/HTTP cleanup, never owners."""
        try:
            if (self.owner.runtime_id != "compiled_fss_v1" or self.closed
                    or self.owner._active is not self or self._fss_control_failed):
                return
            marker = self._fss_control_boundary
            if marker is not None:
                # Time remains that of the last boundary, not cleanup/failure age.
                role, action, status, elapsed = marker
                # A failure BETWEEN calls cannot relabel a validated call as a
                # failure inside that operation (e.g. the final deadline check).
                if status == "entered":
                    marker = (role, action, "failed", elapsed)
                self._fss_control_boundary = marker
            self._fss_control_failed = True
            observer = getattr(self.owner, "_fss_failure_observer", None)
            if observer is not None:
                observer(marker)
        except BaseException:
            pass  # Preserve original source/transport exception, even on unwind.

    def __call__(self):
        return self.cancelled.is_set() or self.cancellation.event.is_set()

    def cancel(self):
        self.cancelled.set()

    def check_cancelled(self):
        if self():
            raise ProtocolError("GKM request cancelled", 499)

    def record_audit(self, counters):
        """One final transport snapshot, not cumulative counters from prior requests."""
        if (self.closed or self.transport_audit is not None or type(counters) is not dict
                or len(counters) > len(self.owner.counter_fields)
                or any(name not in self.owner.counter_fields or type(value) is not int
                       or not 0 <= value < 1 << 63 for name, value in counters.items())):
            raise ValueError("GKM transport audit must be one bounded nonnegative counter snapshot")
        if (_BATCH_TRANSPORT_FIELDS <= self.owner.counter_fields
                and "online_steps" in counters and "native_scalar_claims" in counters
                and counters["online_steps"] != counters["native_scalar_claims"]):
            raise ValueError("GKM transport audit scalar-step counters differ")
        self.transport_audit = dict(counters)

    def __iter__(self):
        return self

    def _register_invocation(self):
        # Include waiters and post-unlock finalizers, not only published owners.
        with self._state_lock:
            if type(self.owner) is DelegatedClient and self._completion.is_set():
                return False
            self._outstanding_calls += 1
            return True

    def _finish_invocation(self, token, parent_token, parent_thread, parent_running):
        displaced = None
        with self._state_lock:
            if self._invocation_owner is token:
                displaced = self._invocation_owner
                self._invocation_owner = parent_token
                self._invocation_thread = parent_thread
                self.running_thread = parent_running
            self._outstanding_calls -= 1
            if (type(self.owner) is DelegatedClient and self._retired
                    and self._outstanding_calls == 0):
                self._completion.set()
        displaced = None  # Superseded references drop outside the state mutex.

    def __next__(self):
        token = object()  # Private per-call identity, allocated before state lock.
        registered = acquired = released = False
        parent_token = parent_thread = parent_running = previous = None
        with _DelegatedExtent(self.owner, self) as extent:
            try:
                registered = self._register_invocation()
                if not registered:
                    extent.normal_stop = True
                    raise StopIteration
                acquired = self.lock.acquire()  # Never under _state_lock.
                if not acquired:
                    raise RuntimeError("attempt mutex acquisition did not succeed")
                extent.activate()
                ident = threading.get_ident()
                with self._state_lock:
                    previous = self._invocation_owner
                    if previous is not None and self._invocation_thread == ident:
                        parent_token, parent_thread = previous, self._invocation_thread
                        parent_running = self.running_thread
                    # A foreign previous owner is stale, NOT a recursive parent.
                    self._invocation_owner = token
                    self._invocation_thread = ident
                    self.running_thread = ident if not self.closed else parent_running
                return self._advance_locked(extent)
            finally:
                try:
                    if acquired:
                        self.lock.release()
                        released = True
                finally:
                    # An unobserved/broken release is NOT successful drain.
                    if registered and (not acquired or released):
                        self._finish_invocation(token, parent_token, parent_thread, parent_running)

    def _advance_locked(self, extent):
        if self.closed:
            extent.normal_stop = True
            raise StopIteration
        retiring = False
        try:
            self.check_cancelled()
            if self.buffer is None:
                self.buffer = iter(self.owner._produce(self))
            self.check_cancelled()
            event = next(self.buffer)
            if event.type in {"response.completed", "response.incomplete"}:
                retiring = True
                self._retire(delivered=True)
            return event
        except BaseException as error:
            try:
                self.fail_control_boundary()
            except BaseException:
                extent.failed = True
                pass
            if type(self.owner) is not DelegatedClient:
                # HEAD legacy selection, INCLUDING terminal retirement failure:
                # its existing second cleanup may replace the first exception.
                self._retire(delivered=False)
            elif not retiring:
                try:
                    self._retire(delivered=False)
                except BaseException as cleanup:
                    extent.failed = True
                    if cleanup is error:
                        raise
                    error.add_note(f"attempt cleanup retained owner: {type(cleanup).__name__}")
            if isinstance(error, StopIteration) and not retiring and not extent.failed:
                extent.normal_stop = True  # Only after actual retirement succeeded.
            raise
    def _retire(self, *, delivered):
        if self.closed:
            return
        if self._prepared_prompt is not None:
            self._prepared_prompt.close()
            self._prepared_prompt = None
        gkm = type(self.owner) is DelegatedClient
        if not gkm:
            self.closed = True
        self._fss_control_boundary = self._fss_control_origin = None
        self.body = self.buffer = self.transport_audit = None
        self.cancellation.unregister(self)
        with self.owner._lock:
            if not delivered and self.response_id is not None:
                self.owner.evict_response(self.response_id)
            if not gkm and self.owner._active is self:
                self.owner._active = None
            if self.owner._closing:
                self.owner._clear_owned_state()
            if gkm and self.owner._active is self:
                self.owner._active = None
        if gkm:
            with self._state_lock:
                self.closed = self._retired = True

    def close(self):
        self.cancel()
        prepared = self._prepared_prompt
        generation = None
        if prepared is not None:
            with prepared._lock:
                if prepared._close_thread is not None:
                    generation = prepared._close_event
        token = object()
        registered = acquired = released = False
        parent_token = parent_thread = parent_running = previous = None
        with _DelegatedExtent(self.owner, self) as extent:
            try:
                registered = self._register_invocation()
                if not registered:
                    if generation is not None and generation._error is not None:
                        raise generation._error
                    return
                acquired = self.lock.acquire()
                if not acquired:
                    raise RuntimeError("attempt mutex acquisition did not succeed")
                extent.activate()
                if generation is not None and generation._error is not None:
                    raise generation._error
                ident = threading.get_ident()
                with self._state_lock:
                    previous = self._invocation_owner
                    recursive = previous is not None and self._invocation_thread == ident
                    if not recursive:
                        self._invocation_owner = token
                        self._invocation_thread = ident
                        # Legacy close has no GKM running/completion semantics.
                        self.running_thread = None
                    if not recursive and type(self.owner) is DelegatedClient:
                        self.running_thread = ident
                if recursive:
                    return  # Cancellation-only; do not replace the live parent.
                self._retire(delivered=False)
            finally:
                try:
                    if acquired:
                        self.lock.release()
                        released = True
                finally:
                    if registered and (not acquired or released):
                        self._finish_invocation(token, parent_token, parent_thread, parent_running)


class _CountPreparation:
    """One count-only owner; no request seat or source admission authority."""

    __slots__ = ("state", "owner", "reservation")

    def __init__(self, owner):
        self.state = "preparing"
        self.owner = owner
        self.reservation = None


class DelegatedClient:
    """RuntimeClient-compatible, one active fixed-length greedy request at a time."""

    request_body_limit_bytes = _BODY_BYTES
    runtime_id = "compiled_gkm_v1"
    role_names = ("worker_a", "worker_b")
    from .tokenizer_admission import _DescriptorPolicy
    _prompt_descriptor_policy = _DescriptorPolicy.DELEGATED
    counter_fields = _COUNTER_FIELDS

    def __init__(self, experiment: Experiment, role_connections, timeout=60.0, source=None):
        if not isinstance(experiment, Experiment):
            raise TypeError("GKM runtime requires an Experiment")
        profile = experiment.resolve()  # Native capability admission is mandatory.
        if profile.client_runtime != self.runtime_id:
            raise ValueError("GKM runtime requires the compiled_gkm_v1 profile")
        budget = experiment.budget
        self._capacity = experiment.pipeline.components["linear"].params.get("capacity", "capsule")
        if (self.runtime_id == "compiled_gkm_v1" and
                experiment.pipeline.components["linear"].params.get("execution") == BATCH_EXECUTION):
            self.counter_fields = self.counter_fields | _BATCH_TRANSPORT_FIELDS
        expanded = self._capacity == "decoder"
        if not (1 <= budget.max_input_tokens <= (512 if expanded else 32)
                and 1 <= budget.max_new_tokens <= (65 if expanded else 9)
                and 1 <= budget.requests <= 64):
            raise ValueError("GKM Experiment exceeds its bounded workload")
        if (type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0):
            raise ValueError("GKM timeout must be positive and finite")
        if role_connections is None or set(role_connections) != set(self.role_names):
            raise ValueError("GKM runtime requires both worker connections")
        connections = {}
        for role, value in role_connections.items():
            if (type(value) is not tuple or len(value) != 2
                    or any(type(item) is not str or not item or len(item) > 4096 for item in value)):
                raise ValueError("GKM worker connections require URL and API key strings")
            connections[role] = (_normalized_inference_endpoint(value[0]).rstrip("/"), value[1])
        if len({value[0] for value in connections.values()}) != len(self.role_names):
            raise ValueError("GKM workers require distinct endpoints")
        self.experiment = profile
        self._declaration = experiment
        self._experiment_budget = budget
        self.default_model = profile.model
        self.role_connections = MappingProxyType(connections)
        self.timeout = float(timeout)
        self._source = source
        self._source_checked = False
        self._lock = threading.RLock()
        self._closing = False
        self._active = None
        self._drain_attempt = None
        self._compact_cleanup = None
        self._failed_request_cleanup = None
        self._count_preparation = None
        self._fss_failure_observer = None
        self._tokenizer_owner = ContextVar("pllm_gkm_tokenizer_owner", default=None)
        self._tokenizer_scopes = set()
        self._result_hook = None
        self.records = OrderedDict()
        self.audit = _DelegatedAudit(self._lock, self.counter_fields)
        self.prepared_inventory_rows = 0
        self.prefill_cache_mode = "off"
        self.prefill_cache_bytes = 0
        self.decode_cache = None
        self.http, self.headers = _NoProviderHTTP(), {}
        # The dashboard's ordinary bundle sampler must see no retained body/KV.
        self._transformer_state_lock = self._lock
        self._transformer_states = MappingProxyType({})

    def _check_open(self):
        if self._closing:
            raise ProtocolError("GKM client is closed", 409)
        if getattr(self, "_count_preparation", None) is not None:
            raise ProtocolError("client has pending token count preparation/cleanup", 409)
        failed = getattr(self, "_failed_request_cleanup", None)
        if failed is not None and failed.pending_retirement:
            raise ProtocolError("client has pending request cleanup", 409)

    def _fss_control_snapshot(self):
        if not self._lock.acquire(blocking=False):
            return None
        try:
            attempt = self._active
            return (attempt._fss_control_boundary
                    if type(attempt) is _Attempt and not attempt.closed else None)
        finally:
            self._lock.release()

    def _model(self, model):
        self._check_open()
        if model != self.default_model:
            raise GatewayError("request model conflicts with Experiment model", param="model")

    def _admit(self, body):
        body = _body_snapshot(body)
        body.setdefault("model", self.default_model)
        try:
            body = validate_response_body(body)
        except GatewayError:
            raise
        except (TypeError, ValueError) as error:
            raise GatewayError("Malformed GKM response body") from error
        self._model(body["model"])
        if body.get("previous_response_id") is not None:
            raise _unsupported("previous_response_id", "GKM does not retain continuation state")
        for name in ("tools", "include", "reasoning"):
            if body.get(name):
                raise _unsupported(name, f"GKM does not support {name}")
        for name in ("max_tool_calls", "prompt_cache_key"):
            if body.get(name) is not None:
                raise _unsupported(name, f"GKM does not support {name}")
        if body.get("tool_choice", "auto") not in ("auto", "none"):
            raise _unsupported("tool_choice", "GKM does not support tool selection")
        if body.get("truncation", "disabled") != "disabled":
            raise _unsupported("truncation", "GKM requires the complete untruncated input")
        for name, expected in (("temperature", 0), ("top_p", 1)):
            if body.get(name) is not None and body[name] != expected:
                raise _unsupported(name, "GKM requires fixed-length greedy generation")
            body[name] = expected
        count = body.setdefault("max_output_tokens", self._experiment_budget.max_new_tokens)
        if count > self._experiment_budget.max_new_tokens:
            raise GatewayError("max_output_tokens exceeds Experiment budget", param="max_output_tokens")
        metadata = body.get("metadata")
        if metadata is not None and (len(metadata) > 16 or any(
                type(value) is not str or len(key) > 64 or len(value) > 512
                for key, value in metadata.items())):
            raise GatewayError("metadata requires at most 16 bounded string pairs", param="metadata")
        _text_input(body["input"])
        normalize_input(body["input"], instructions=body.get("instructions"))
        return body

    @classmethod
    def _resolve_prompt_source(cls, declaration):
        return _admit_source(declaration)

    def _ensure_source(self):
        with self._lock:
            self._check_open()
            if not self._source_checked:
                source = self._source if self._source is not None else self._resolve_prompt_source(self._declaration)
                contract = source.contract
                expanded = self._capacity == "decoder"
                if (source.model != self._declaration.pipeline.model
                        or getattr(source, "capacity", "capsule") != self._capacity
                        or contract["input_rows"] != self._experiment_budget.max_input_tokens
                        or contract["decode_steps"] != self._experiment_budget.max_new_tokens - 1
                        or not 1 <= contract["vocabulary"] <= (262144 if expanded else 256)
                        or not 0 < source.source_float32_bytes <= (4 << 30 if expanded else 8 << 20)):
                    raise ValueError("GKM source differs from the admitted Experiment")
                result_payload_limit(contract, capacity=self._capacity)
                self._source, self._source_checked = source, True
            return self._source

    def _prompt_metadata_owner(self):
        """Borrow ONLY an already checked owner. Never lazily resolve here."""
        with self._lock:
            self._check_open()
            return self._source if self._source_checked else None

    def _prepare_count_prompt(self, body, slot):
        from . import tokenizer_admission
        from .responses import ResponsesError, _prepare_prompt, _owned_prompt_source

        owner = slot.owner
        if owner._closed or owner._failure is not None or owner._payload is not None:
            raise ResponsesError("prompt preparation owner is unavailable")
        # The count slot already owns its actual reservation. Borrow only an
        # already checked source; this path never resolves or admits a source.
        with self._lock:
            source = self._source if self._source_checked else None
        snapshot = None
        if source is not None:
            descriptor = source.tokenizer_descriptor
            tokenizer = source.tokenizer()
            owner._adopt_tokenizer(tokenizer)
            association = _owned_prompt_source(source)
        else:
            from pllm.deployment.execution import _TokenizerView

            snapshot = tokenizer_admission._capture_prompt_metadata(
                self._declaration.pipeline.model, policy=self._prompt_descriptor_policy,
                capacity=self._capacity, input_rows=self._experiment_budget.max_input_tokens,
                outputs=self._experiment_budget.max_new_tokens,
                manifest_template=None, expected_source_lock=None)
            view = _TokenizerView._from_prompt_snapshot(snapshot)
            descriptor = view.tokenizer_descriptor
            tokenizer = view.tokenizer()
            owner._adopt_tokenizer(tokenizer)
            association = snapshot.association()
        prepared = _prepare_prompt(body, descriptor=descriptor, tokenizer=tokenizer,
            source=association, _pending_source=snapshot, _preparation_owner=owner)
        ids = prepared._payload.ids
        if (not 1 <= len(ids) <= self._experiment_budget.max_input_tokens
                or any(i >= descriptor["vocab_size"] for i in ids)):
            raise GatewayError("request input exceeds selected budget", param="input")
        return prepared

    def _prepare_request_prompt(self, body, *, attempt=None, messages=None, tools=None,
                                metadata_view=None, source=None, _count_slot=None,
                                _preparation_owner=None):
        if _count_slot is not None:
            return self._prepare_count_prompt(body, _count_slot)
        from . import tokenizer_admission
        from .responses import ResponsesError, _PreparedPromptUse, _prepare_prompt, _owned_prompt_source

        if attempt is not None and attempt._prepared_prompt is not None:
            owner = attempt._prepared_prompt
            if (owner._payload is None or owner._failure is not None or owner._closed
                    or owner._used or (_preparation_owner is not None and _preparation_owner is not owner)):
                raise ResponsesError("attempt already owns incomplete or retired prompt preparation")
            return owner
        if attempt is not None:
            owner = _PreparedPromptUse._empty() if _preparation_owner is None else _preparation_owner
            attempt._prepared_prompt = owner  # Real attempt roots BEFORE acquisition/construction.
        else:
            if _preparation_owner is None:
                raise ResponsesError("prompt preparation requires a caller-rooted owner")
            owner = _preparation_owner
        if owner._closed or owner._failure is not None or owner._payload is not None:
            raise ResponsesError("prompt preparation owner is unavailable")
        tokenizer = snapshot = view = prepared = None
        try:
            source = self._prompt_metadata_owner() if source is None else source
            if source is not None:
                if metadata_view is not None:
                    metadata_view._check_owned_prompt_metadata(source, self._declaration.pipeline.model)
                descriptor = source.tokenizer_descriptor
                tokenizer = source.tokenizer()
                owner._adopt_tokenizer(tokenizer)
                association = _owned_prompt_source(source)
            else:
                # Existing metadata allowances, held until the prepared owner
                # retires. No source inventory/contract/payload admission yet.
                from pllm.deployment.execution import _TokenizerView

                owner._reserve_metadata(
                    tokenizer_admission.DECODER_METADATA_MEMORY_BYTES
                    if self._capacity == "decoder" else 1 << 20)
                snapshot = tokenizer_admission._capture_prompt_metadata(
                    self._declaration.pipeline.model, policy=self._prompt_descriptor_policy,
                    capacity=self._capacity, input_rows=self._experiment_budget.max_input_tokens,
                    outputs=self._experiment_budget.max_new_tokens,
                    manifest_template=None if metadata_view is None else metadata_view.manifest.get("chat_template"),
                    expected_source_lock=None if metadata_view is None else metadata_view.source_lock_digest)
                if metadata_view is not None:
                    metadata_view._check_prompt_snapshot(snapshot, self._declaration.pipeline.model)
                view = _TokenizerView._from_prompt_snapshot(snapshot)
                descriptor = view.tokenizer_descriptor
                tokenizer = view.tokenizer()
                owner._adopt_tokenizer(tokenizer)
                association = snapshot.association()
            prepared = _prepare_prompt(body, descriptor=descriptor, tokenizer=tokenizer,
                source=association, messages=messages, tools=tools,
                _pending_source=snapshot, _preparation_owner=owner)
            ids = prepared._payload.ids
            if (not 1 <= len(ids) <= self._experiment_budget.max_input_tokens
                    or any(i >= descriptor["vocab_size"] for i in ids)):
                raise GatewayError("request input exceeds selected budget", param="input")
            if attempt is not None:
                prepared.claim()
            return prepared
        except BaseException as error:
            # Preserve failure on the already rooted owner. Actual attempt/count/
            # lease retirement runs AFTER these constructor/render frames unwind.
            owner._failure = error
            raise
    @contextmanager
    def tokenizer_scope(self):
        existing = self._tokenizer_owner.get()
        if existing is not None:
            self._check_open()
            yield
            return
        owner = _ScopedTokenizer()
        with self._lock:
            self._check_open()
            if self._tokenizer_scopes:
                raise ProtocolError("GKM client already owns a tokenizer scope", 409)
            self._tokenizer_scopes.add(owner)
        token = self._tokenizer_owner.set(owner)
        try:
            yield
        finally:
            try:
                owner.close()
            finally:
                self._tokenizer_owner.reset(token)
                with self._lock:
                    self._tokenizer_scopes.discard(owner)

    def _encode(self, body, source, *, attempt=None):
        prepared = None if attempt is None else attempt._prepared_prompt
        if prepared is not None:
            try:
                prepared._bind_owned_source(body, source)
                tokenizer, ids = prepared.consume(body, _owned_prompt_source(source))
            except ResponsesError as error:
                raise GatewayError(str(error), param="input") from error
            if not 1 <= len(ids) <= self._experiment_budget.max_input_tokens or any(
                    token >= source.contract["vocabulary"] for token in ids):
                raise GatewayError("prepared input exceeds source/input budget", param="input")
            return tokenizer, ids
        tokenizer = self._tokenizer_owner.get().get(source)
        descriptor = source.tokenizer_descriptor
        try:
            prepared = _prepare_prompt(body, descriptor=descriptor, tokenizer=tokenizer,
                                       source=_owned_prompt_source(source))
        except ResponsesError as error:
            raise GatewayError(str(error), param="input") from error
        # Direct callers already have an owned tokenizer scope. The immutable
        # payload is used here too, without transferring/closing its borrowed tokenizer.
        _, ids = prepared.consume(body, _owned_prompt_source(source))
        if (type(ids) not in (list, tuple) or not 1 <= len(ids) <= self._experiment_budget.max_input_tokens
                or any(type(token) is not int or not 0 <= token < source.contract["vocabulary"]
                       for token in ids)):
            raise GatewayError("GKM tokenized input exceeds its complete-table/input budget", param="input")
        if attempt is not None:
            # The generator's lexical tokenizer scope ends before event delivery.
            # Transfer actual custody to the attempt until acceptance/cancellation retires it.
            self._tokenizer_owner.get()._detach(source)
            attempt._prepared_prompt = prepared.claim()
        return tokenizer, tuple(ids)

    def _response_input_tokens(self, model, input):
        from contextlib import ExitStack
        from pllm.deployment.resources import reserve_incremental_memory
        from . import tokenizer_admission
        from .responses import _PreparedPromptUse

        with self._lock:
            body = self._admit({"model": model, "input": input})
            cleanup = getattr(self, "_compact_cleanup", None)
            if (self._active is not None or self._tokenizer_scopes
                    or (cleanup is not None and not cleanup.closed)):
                raise ProtocolError("client already owns request/tokenizer cleanup", 409)
            slot = _CountPreparation(_PreparedPromptUse._empty())
            self._count_preparation = slot
        try:
            slot.owner._metadata_owner = ExitStack()
            slot.reservation = reserve_incremental_memory(
                tokenizer_admission.DECODER_METADATA_MEMORY_BYTES
                if self._capacity == "decoder" else 1 << 20)
            slot.owner._metadata_owner.enter_context(slot.reservation)
            prepared = self._prepare_request_prompt(body, _count_slot=slot)
            count = len(prepared._payload.ids)
        except BaseException as error:
            # Constructor failures may own native/metadata temporaries only in
            # inactive traceback frames. Retire those under the SAME allowance.
            slot.owner._failure = error
            self._retire_count_preparation(slot, primary=error)
            if isinstance(error, ResponsesError):
                raise GatewayError(str(error), param="input") from error
            raise
        self._retire_count_preparation(slot)
        return count

    def _retire_count_preparation(self, slot, *, primary=None):
        with self._lock:
            slot.state = "draining"
        try:
            slot.owner.close()
        except BaseException as cleanup:
            slot.reservation.retain_failed((slot.owner,))
            with self._lock:
                slot.state = "pending"
            if primary is None:
                raise
            primary.add_note(f"token count cleanup retained owner: {type(cleanup).__name__}")
        else:
            with self._lock:
                self._count_preparation = None

    def _validate_result(self, result, source, ids, count):
        """Check bounded transport shape; native finish is the verification authority."""
        if not isinstance(result, GkmDecoderResult):
            raise ProtocolError("GKM transport did not return an accepted decoder result")
        position = len(ids) + count - 1
        cfg = source.config
        cfg = cfg.get("text_config", cfg)
        heads = cfg.get("num_key_value_heads", cfg["num_attention_heads"])
        width = (cfg["head_dim"] if cfg["model_type"] == "qwen3"
                 else cfg["hidden_size"] // cfg["num_attention_heads"])
        shape = (position, heads, width)
        if (type(result.acceptance) is not bytes or not 0 < len(result.acceptance) <= 16 << 10
                or type(result.selected) is not tuple or len(result.selected) != count
                or any(type(t) is not int or not 0 <= t < source.contract["vocabulary"] for t in result.selected)
                or type(result.executed) is not tuple or result.executed != ids + result.selected[:-1]
                or type(result.position) is not int or result.position != position
                or type(result.logits) is not tuple or len(result.logits) != count
                or any(type(row) is not bytes or len(row) != 4 * source.contract["vocabulary"]
                       for row in result.logits)
                or type(result.kv) is not tuple or len(result.kv) != cfg["num_hidden_layers"]):
            raise ProtocolError("GKM accepted result differs from the fixed-length full-state contract")
        size = sum(map(len, result.logits)) + len(result.acceptance)
        for row in result.kv:
            if (type(row) is not tuple or len(row) != 3 or row[0] != shape
                    or any(type(data) is not bytes or len(data) != math.prod(shape) * 4 for data in row[1:])):
                raise ProtocolError("GKM accepted full-KV shape differs from the request")
            size += len(row[1]) + len(row[2])
        if size > result_payload_limit(source.contract, capacity=self._capacity):
            raise ProtocolError("GKM accepted result exceeds its bounded owner")

    def _execute(self, source, ids, count, attempt):
        return _execute_request(self._declaration, source, ids, count,
                                self.role_connections, self.timeout, attempt)

    def _produce(self, attempt):
        result = tokenizer = source = None
        ids = ()
        body = attempt.body
        try:
            attempt.check_cancelled()
            with self.tokenizer_scope():
                self._prepare_request_prompt(body, attempt=attempt)
                attempt.check_cancelled()
                source = self._ensure_source()
                tokenizer, ids = self._encode(body, source, attempt=attempt)
                count = body["max_output_tokens"]
                attempt.check_cancelled()
                with self._lock:
                    self.audit.generation_requested_output_tokens += count
                    self.audit.generation_requested_prefill_tokens += len(ids)
                    self.audit.generation_requested_decode_steps += count - 1
                    self.audit.begin_transport()
                try:
                    result = self._execute(source, ids, count, attempt)
                finally:
                    with self._lock:
                        self.audit.finish_transport(attempt.transport_audit)
                attempt.check_cancelled()
                self._validate_result(result, source, ids, count)
                if self._result_hook is not None:
                    self._result_hook(result)
                attempt.check_cancelled()
                text = tokenizer.decode(list(result.selected))
                if type(text) is not str or len(text) > _OUTPUT_BYTES or len(text.encode()) > _OUTPUT_BYTES:
                    raise ProtocolError("GKM decoded output exceeds its text limit")
                response = Response.text_response(model=self.default_model, text=text,
                                                   input_tokens=len(ids), output_tokens=count)
                for name, value in _response_contract_fields(body, max_output_tokens=count).items():
                    setattr(response, name, value)
                # EOS is a selected token, never an early stop in this contract.
                response.status = "incomplete"
                response.incomplete_details = {"reason": "max_output_tokens"}
                events = tuple(response_events(response, [text] if text else []))
                with self._lock:
                    self._check_open()
                    attempt.check_cancelled()
                    if response.store:
                        payload = response.model_dump_json().encode()
                        self.records[response.id] = payload
                        while len(self.records) > _RECORD_COUNT or sum(map(len, self.records.values())) > _RECORD_BYTES:
                            self.records.popitem(last=False)
                    attempt.response_id = response.id
                    self.audit.generation_completed_responses += 1
                    self.audit.generation_output_tokens += count
                    self.audit.generation_decode_output_tokens += count - 1
                    self.audit.generation_executed_decode_steps += count - 1
                    self.audit.generation_remote_prefill_tokens += len(ids)
                    self.audit.generation_remote_decode_steps += count - 1
                    snapshot = attempt.transport_audit or {}
                    phase_fields = ("generation_prefill_online_body_bytes", "generation_decode_online_body_bytes")
                    online_fields = tuple(name for name in _ROLE_FIELDS if ".online_" in name)
                    if (all(name in snapshot for name in (*phase_fields, *online_fields))
                            and sum(snapshot[name] for name in phase_fields)
                            == sum(snapshot[name] for name in online_fields)):
                        self.audit.generation_attribution_supported_responses += 1
                return events
        except BaseException as error:
            clear_failure_frames(error)
            raise
        finally:
            # No accepted logits/KV in the iterator, client cache, or error frame.
            result = tokenizer = source = body = None
            ids = ()
            attempt.body = None

    def events(self, body):
        with self._lock:
            self._check_open()
            cleanup = getattr(self, "_compact_cleanup", None)
            if cleanup is not None and not cleanup.closed:
                raise ProtocolError("client has pending compact cleanup", 409)
            if self._active is not None:
                raise ProtocolError("GKM client already owns an exclusive request", 409)
            if type(self) is DelegatedClient:
                previous = getattr(self, "_drain_attempt", None)
                if previous is not None and not previous._completion.is_set():
                    raise ProtocolError("GKM request execution has not drained", 409)
            attempt = _Attempt(self, self._admit(body))
            self._active = attempt
            if type(self) is DelegatedClient:
                self._drain_attempt = attempt
            attempt.cancellation.register(attempt)
        return attempt

    def create(self, body):
        events = self.events(body)
        if events.streaming:
            return ResponseStream(events)
        try:
            for event in events:
                if event.type in {"response.completed", "response.incomplete"}:
                    return Response.from_dict(event.response)
        finally:
            events.close()
        raise ProtocolError("GKM response ended without a terminal event")

    def retrieve(self, response_id):
        with self._lock:
            self._check_open()
            return Response.from_dict(json.loads(self.records[response_id]))

    def cancel(self, response_id):
        # Identifiers only become observable after verification and retirement.
        # An accepted terminal response must never be relabeled as cancelled.
        return self.retrieve(response_id)

    def continuation_response(self, response_id):
        raise _unsupported("previous_response_id", "GKM does not retain continuation state")

    def evict_response(self, response_id):
        with self._lock:
            self.records.pop(response_id, None)

    def _accept_decode_response(self, response_id):
        pass  # Existing async resource delivery hook; no reusable decode state.

    def _abort_decode_response(self, response_id, *, primary=None):
        self.evict_response(response_id)

    def list_models(self):
        from pllm.protocols import OpenWeightVerifiableOutsourcing

        source = self._ensure_source()
        runtime = {
            "client_runtime": "compiled_gkm_v1", "privacy_mode": "gkm_public",
            "protocol": "gkm_i16_q11_ring64", "generation_policy": "fixed_length_greedy",
            "verification_window": "request_end", "streaming": "buffered_until_verified",
            "sharing": self._declaration.pipeline.components["linear"].params["sharing"],
            "kv_reuse": False, "kv": "full", "output_head": "full",
            "lookup_dtype": "original_float32", "eos_policy": "fixed_count",
            "numeric_profile": OpenWeightVerifiableOutsourcing.NUMERIC_PROFILE,
            "local_numeric_profile": OpenWeightVerifiableOutsourcing.LOCAL_NUMERIC_PROFILE,
            "checkpoint_digest": source.checkpoint_digest,
            "source_lock_digest": source.source_lock_digest,
            "tokenizer_digest": source.tokenizer_digest,
        }
        return {"object": "list", "data": [{"id": self.default_model, "object": "model",
                "owned_by": "client", "runtime": runtime,
                "context_length": source.source.manifest.context_length}]}

    def prepared_inventory_status(self, model):
        self._model(model)
        return {"status": "not-applicable", "capacity": 0, "available": 0,
                "reserved": 0, "burned": 0}

    def preprocess(self, model, count=0, *, stages=None):
        self._model(model)
        raise _unsupported("preprocess", "GKM has no preparation inventory")

    def prepare_response(self, model, input, max_output_tokens, *, instructions=None, store=True):
        self._model(model)
        raise _unsupported("prepare_response", "GKM admits material within each exclusive request; no preparation inventory")

    def prepared_rows_for_response(self, model, input, max_output_tokens, *, instructions=None, store=True):
        self._model(model)
        raise _unsupported("prepared_rows_for_response", "GKM has no preparation inventory")

    def _new_compact_cleanup_scope(self, attempt):
        from pllm.deployment.resources import reserve_cleanup_slot

        with self._lock:
            self._check_open()
            if type(attempt) is not _Attempt or self._active is not attempt or attempt.owner is not self or attempt.closed:
                raise ProtocolError("compact cleanup requires the active request seat", 409)
            previous = getattr(self, "_compact_cleanup", None)
            if previous is not None and not previous.closed:
                raise ProtocolError("client has pending compact cleanup", 409)
            scope = reserve_cleanup_slot()
            self._compact_cleanup = scope
            return scope

    def _retain_failed_request_reservation(self, attempt, reservation):
        from pllm.deployment.resources import _MemoryReservation

        with self._lock:
            # Handoff is allowed WHILE closing: close() waits for this active
            # producer first, then reads/drains the retained cleanup handle.
            if (type(attempt) is not _Attempt or attempt.owner is not self
                    or self._active is not attempt or attempt.closed):
                raise ProtocolError("failed cleanup requires the active request seat", 409)
            if type(reservation) is not _MemoryReservation or not reservation.pending_retirement:
                raise ProtocolError("failed cleanup requires actual retained admission", 409)
            previous = getattr(self, "_failed_request_cleanup", None)
            if previous is not None and previous is not reservation and previous.pending_retirement:
                raise ProtocolError("client already owns pending request cleanup", 409)
            # ONE handle for this exclusive seat; real root remains its admitted
            # module ledger token, not this client/callback/exception reference.
            self._failed_request_cleanup = reservation

    def _lease_drain_snapshot(self, attempt=None, *, seal=False):
        """Exact original attempt status, independent of its execution lock.

        Sealing precedes lease close so no new request can appear between capture
        and cancellation. The last attempt survives removal from _active.
        """
        if type(self) is not DelegatedClient:
            raise ProtocolError("lease drain requires the exact delegated client", 409)
        with self._lock:
            if seal:
                self._closing = True
            if attempt is None:
                attempt = self._active or getattr(self, "_drain_attempt", None)
            if attempt is not None and (type(attempt) is not _Attempt or attempt.owner is not self):
                raise ProtocolError("lease drain requires the original request attempt", 409)
        if attempt is None:
            return None, None, None, False
        with attempt._state_lock:
            return attempt, attempt.running_thread, attempt._completion, attempt._retired

    def close(self):
        with _DelegatedExtent(self) as extent:
            extent.activate()
            self._close_owners()

    def _close_owners(self):
        with self._lock:
            count = getattr(self, "_count_preparation", None)
            if count is not None:
                if count.state != "pending":
                    raise ProtocolError("token count preparation/cleanup is busy; retry close", 409)
                count.state = "draining"
            self._closing = True
            self._fss_failure_observer = None
            active = self._active
            if active is None and type(self) is DelegatedClient:
                previous = getattr(self, "_drain_attempt", None)
                if previous is not None and not previous._completion.is_set():
                    active = previous
            if active is not None:
                active.cancel()
        if count is not None:
            try:
                count.reservation.drain_failed()
            except BaseException:
                with self._lock:
                    count.state = "pending"
                raise
            with self._lock:
                self._count_preparation = None
        if active is not None:
            active.close()  # Cancellation is signalled before waiting for execution to drain.
            if active.running_thread == threading.get_ident():
                return
        # Producer/attempt drained first. Never invoke native cleanup while
        # holding the client, attempt, admission or slot lock.
        errors = []  # bounded2: both owners attempted, no callback under locks
        cleanup = getattr(self, "_compact_cleanup", None)
        failed = getattr(self, "_failed_request_cleanup", None)
        for owner, method in ((cleanup, "drain"), (failed, "drain_failed")):
            if owner is not None:
                try:
                    getattr(owner, method)()
                except BaseException as error:
                    errors.append(error)
        if errors:
            first = errors[0]
            for error in errors[1:]:
                first.add_note(f"request cleanup also failed: {type(error).__name__}")
            raise first  # failed retirement stays visible and ledger-rooted
        self._clear_owned_state()

    def _clear_owned_state(self):
        with self._lock:
            if getattr(self, "_count_preparation", None) is not None:
                return
            cleanup = getattr(self, "_compact_cleanup", None)
            if cleanup is not None and not cleanup.closed:
                return  # _Attempt retirement cannot erase failed cleanup custody.
            failed = getattr(self, "_failed_request_cleanup", None)
            if failed is not None and failed.pending_retirement:
                return
            self._compact_cleanup = None
            self._failed_request_cleanup = None
            try:
                for owner in self._tokenizer_scopes:
                    owner.close()
            finally:
                self._tokenizer_scopes.clear()
                self.records.clear()
                self._source = self._result_hook = None
                self._source_checked = False


__all__ = ["DelegatedClient"]
