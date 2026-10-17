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
from types import MappingProxyType
from typing import Any

from pllm.configuration import Experiment

from .client import (
    PrivacyAudit, ProtocolError, ResponseStream, _ScopedTokenizer,
    _normalized_inference_endpoint, _response_contract_fields,
)
from .delegated_decoder import GkmDecoderResult, result_payload_limit
from .delegated_protocol import clear_failure_frames
from .http_gateway import GatewayError, validate_response_body
from .prepared_queue import PreparedCancellation, prepared_cancellation
from .responses import normalize_input, prompt_text
from .types import Response, response_events


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
_ROLE_FIELDS = frozenset(
    f"role_link.{role}.{phase}_{direction}_bytes"
    for role in ("worker_a", "worker_b")
    for phase in ("setup", "online", "teardown")
    for direction in ("upload", "download")
)
_COUNTER_FIELDS = _TRANSPORT_FIELDS | _ROLE_FIELDS


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

    def __init__(self, lock):
        super().__init__()
        self._lock = lock
        self._totals = dict.fromkeys(_COUNTER_FIELDS, 0)

    def begin_transport(self):
        for name in _TRANSPORT_FIELDS:
            setattr(self, name, None)
        self.role_link_bodies.clear()

    def finish_transport(self, snapshot):
        snapshot = snapshot or {}
        for name, prior in self._totals.items():
            value = snapshot.get(name)
            total = None if value is None or prior is None else prior + value
            self._totals[name] = total
            if name in _TRANSPORT_FIELDS:
                setattr(self, name, total)
            elif total is not None:
                _, role, metric = name.split(".", 2)
                self.role_link_bodies.setdefault(role, {})[metric] = total

    def to_dict(self):
        with self._lock:
            values = {name: value for name in self.__dataclass_fields__
                      if not name.startswith("_") and name != "role_link_bodies"
                      and (value := getattr(self, name)) is not None}
            for role, counters in self.role_link_bodies.items():
                values.update({f"role_link.{role}.{name}": value for name, value in counters.items()})
            return values


class _NoProviderHTTP:
    def __getattr__(self, name):
        raise ProtocolError("GKM runtime has no ordinary provider control endpoint", 400)


class _Attempt:
    """One serialized iterator, cancellation handle, and exclusive request seat."""

    def __init__(self, owner, body):
        self.owner = owner
        self.body = body
        self.streaming = body.get("stream", False)
        self.lock = threading.RLock()
        self.cancelled = threading.Event()
        self.cancellation = prepared_cancellation.get() or PreparedCancellation()
        self.buffer = None
        self.response_id = None
        self.closed = False
        self.running_thread = None
        self.transport_audit = None

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
                or len(counters) > len(_COUNTER_FIELDS)
                or any(name not in _COUNTER_FIELDS or type(value) is not int
                       or not 0 <= value < 1 << 63 for name, value in counters.items())):
            raise ValueError("GKM transport audit must be one bounded nonnegative counter snapshot")
        self.transport_audit = dict(counters)

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            if self.closed:
                raise StopIteration
            self.running_thread = threading.get_ident()
            try:
                self.check_cancelled()
                if self.buffer is None:
                    self.buffer = iter(self.owner._produce(self))
                self.check_cancelled()
                event = next(self.buffer)
                if event.type in {"response.completed", "response.incomplete"}:
                    self._retire(delivered=True)
                return event
            except BaseException:
                self._retire(delivered=False)
                raise
            finally:
                self.running_thread = None

    def _retire(self, *, delivered):
        if self.closed:
            return
        self.closed = True
        self.body = self.buffer = self.transport_audit = None
        self.cancellation.unregister(self)
        with self.owner._lock:
            if not delivered and self.response_id is not None:
                self.owner.evict_response(self.response_id)
            if self.owner._active is self:
                self.owner._active = None
            if self.owner._closing:
                self.owner._clear_owned_state()

    def close(self):
        self.cancel()
        with self.lock:
            # A private observation hook can request cancellation, but must not
            # release the seat while the executing stack still owns its result.
            if self.running_thread == threading.get_ident():
                return
            self._retire(delivered=False)


class DelegatedClient:
    """RuntimeClient-compatible, one active fixed-length greedy request at a time."""

    request_body_limit_bytes = _BODY_BYTES

    def __init__(self, experiment: Experiment, role_connections, timeout=60.0, source=None):
        if not isinstance(experiment, Experiment):
            raise TypeError("GKM runtime requires an Experiment")
        profile = experiment.resolve()  # Native capability admission is mandatory.
        if profile.client_runtime != "compiled_gkm_v1":
            raise ValueError("GKM runtime requires the compiled_gkm_v1 profile")
        budget = experiment.budget
        self._capacity = experiment.pipeline.components["linear"].params.get("capacity", "capsule")
        expanded = self._capacity == "decoder"
        if not (1 <= budget.max_input_tokens <= (512 if expanded else 32)
                and 1 <= budget.max_new_tokens <= (65 if expanded else 9)
                and 1 <= budget.requests <= 64):
            raise ValueError("GKM Experiment exceeds its bounded workload")
        if (type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0):
            raise ValueError("GKM timeout must be positive and finite")
        if role_connections is None or set(role_connections) != {"worker_a", "worker_b"}:
            raise ValueError("GKM runtime requires both worker connections")
        connections = {}
        for role, value in role_connections.items():
            if (type(value) is not tuple or len(value) != 2
                    or any(type(item) is not str or not item or len(item) > 4096 for item in value)):
                raise ValueError("GKM worker connections require URL and API key strings")
            connections[role] = (_normalized_inference_endpoint(value[0]).rstrip("/"), value[1])
        if connections["worker_a"][0] == connections["worker_b"][0]:
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
        self._tokenizer_owner = ContextVar("pllm_gkm_tokenizer_owner", default=None)
        self._tokenizer_scopes = set()
        self._result_hook = None
        self.records = OrderedDict()
        self.audit = _DelegatedAudit(self._lock)
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

    def _ensure_source(self):
        with self._lock:
            self._check_open()
            if not self._source_checked:
                source = self._source if self._source is not None else _admit_source(self._declaration)
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

    def _encode(self, body, source):
        tokenizer = self._tokenizer_owner.get().get(source)
        rendered = prompt_text(normalize_input(body["input"], instructions=body.get("instructions")))
        ids = tokenizer.encode(rendered, add_bos=source.tokenizer_descriptor.get("add_bos_token", True))
        if (type(ids) not in (list, tuple) or not 1 <= len(ids) <= self._experiment_budget.max_input_tokens
                or any(type(token) is not int or not 0 <= token < source.contract["vocabulary"]
                       for token in ids)):
            raise GatewayError("GKM tokenized input exceeds its complete-table/input budget", param="input")
        return tokenizer, tuple(ids)

    def _response_input_tokens(self, model, input):
        body = self._admit({"model": model, "input": input})
        with self.tokenizer_scope():
            _, ids = self._encode(body, self._ensure_source())
            return len(ids)

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

    def _produce(self, attempt):
        result = tokenizer = source = None
        ids = ()
        body = attempt.body
        try:
            attempt.check_cancelled()
            with self.tokenizer_scope():
                source = self._ensure_source()
                tokenizer, ids = self._encode(body, source)
                count = body["max_output_tokens"]
                attempt.check_cancelled()
                with self._lock:
                    self.audit.generation_requested_output_tokens += count
                    self.audit.generation_requested_prefill_tokens += len(ids)
                    self.audit.generation_requested_decode_steps += count - 1
                    self.audit.begin_transport()
                try:
                    result = _execute_request(self._declaration, source, ids, count,
                                              self.role_connections, self.timeout, attempt)
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
            if self._active is not None:
                raise ProtocolError("GKM client already owns an exclusive request", 409)
            attempt = _Attempt(self, self._admit(body))
            self._active = attempt
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
            "sharing": "full", "kv_reuse": False, "kv": "full", "output_head": "full",
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

    def close(self):
        with self._lock:
            self._closing = True
            active = self._active
            if active is not None:
                active.cancel()
        if active is not None:
            active.close()  # Cancellation is signalled before waiting for execution to drain.
            if active.running_thread == threading.get_ident():
                return
        self._clear_owned_state()

    def _clear_owned_state(self):
        with self._lock:
            try:
                for owner in self._tokenizer_scopes:
                    owner.close()
            finally:
                self._tokenizer_scopes.clear()
                self.records.clear()
                self._source = self._result_hook = None
                self._source_checked = False


__all__ = ["DelegatedClient"]
