"""Client-owned per-request planning, execution and bounded diagnostic ownership."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from contextlib import ExitStack, nullcontext
from contextvars import copy_context
import threading

from pllm.deployment.network import NetworkError


async def _drain(task):
    """A cancelled caller still owns the worker until its cleanup finishes."""
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
        except Exception:
            break
    return task.result()


async def _owned_thread(function, *args, cancel):
    task = asyncio.create_task(asyncio.to_thread(function, *args))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        cancel()
        try:
            await _drain(task)
        except BaseException:
            pass
        raise


class _OwnedAttempt:
    """One generator context, serialized next/close, including gateway threads."""

    def __init__(self, owner, body, cancelled):
        self.owner, self.cancelled = owner, cancelled
        self.context = copy_context()
        self.lock = threading.RLock()
        self.source = self._events(body)
        self.closed = False
        self.response_id = None
        self.last_response = None
        self.client = None

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            if self.closed:
                raise StopIteration
            try:
                return self.context.run(next, self.source)
            except BaseException:
                self.close()
                raise

    def _events(self, body):
        from .http_gateway import resource_dict, validate_response_body

        owner = self.owner
        with owner.lock:
            if owner._closing:
                raise NetworkError("client is closed")
            if len(owner.active) >= 8:
                raise NetworkError("client already owns eight response attempts")
            owner.active.add(self)
        stack = ExitStack()
        try:
            body = dict(body)
            body.setdefault("model", owner.default_model)
            if body.get("previous_response_id"):
                raise NetworkError("per-response network leases do not retain continuation state")
            body = validate_response_body(body)
            models = getattr(owner.factory, "model_ids", None)
            if models is not None and body["model"] not in models:
                raise NetworkError("request model conflicts with configured deployment")
            if self.cancelled.is_set():
                return
            if hasattr(owner.factory, "open"):
                self.client, decision = stack.enter_context(owner.factory.open(cancelled=self.cancelled))
            else:
                self.client = stack.enter_context(owner.factory())
                decision = None
            if self.cancelled.is_set():
                return
            # Keep the lease resource wrapper: calling core.events directly
            # would bypass source, token count and one-attempt admission.
            stream = self.client.responses.create(**{**body, "stream": True})
            stack.callback(stream.close)
            for event in stream:
                if self.cancelled.is_set():
                    return
                value = resource_dict(event)
                response = value.get("response")
                if isinstance(response, dict):
                    self.response_id = response.get("id", self.response_id)
                    self.last_response = response
                owner._remember(event, decision)
                if value.get("type") in {"response.completed", "response.incomplete",
                                         "response.cancelled", "response.failed"}:
                    # Release even when the consumer stops at the terminal event.
                    stack.close()
                    self._retire()
                yield event
        finally:
            try:
                stack.close()
            finally:
                self._retire()

    def _retire(self):
        with self.owner.lock:
            if self.client is not None:
                self.owner._add_audit(self.client.privacy_audit)
            self.client = None
            self.owner.active.discard(self)

    def close(self):
        self.cancelled.set()
        with self.lock:
            if self.closed:
                return
            self.closed = True
            self.context.run(self.source.close)


class PlannedClient:
    """SDK/gateway adapter; all arithmetic and protocol work uses ordinary clients.

    Stored responses and immutable explanations have separate 4 MiB ceilings.
    Capacity, inventories and response-local decoder state retire each attempt.
    """

    _RECORD_BYTES = 4 << 20

    def __init__(self, factory):
        from .client import PrivacyAudit

        self.factory = factory
        self.default_model = getattr(factory, "default_model", None)
        self.lock = threading.RLock()
        self.active = set()
        self.records = OrderedDict()
        self._decisions = OrderedDict()
        self._audit = PrivacyAudit()
        self._closing = False
        self._experiment_budget = None
        self.prefill_cache_mode = "off"
        self.prepared_inventory_rows = 0

    def tokenizer_scope(self):
        return nullcontext()  # Each ordinary request owns its actual tokenizer.

    def _remember(self, event, decision):
        from .http_gateway import resource_dict
        from pllm.deployment.network import canonical

        response = resource_dict(event).get("response")
        if not isinstance(response, dict) or response.get("status") not in {
            "completed", "incomplete", "cancelled"
        }:
            return
        identifier = response["id"]
        with self.lock:
            if response.get("store", True):
                self._bounded_put(self.records, identifier, canonical(response))
            if decision is not None:
                self._bounded_put(self._decisions, identifier, decision.canonical_bytes())

    def _bounded_put(self, records, key, value):
        if len(value) > self._RECORD_BYTES:
            return
        records[key] = value
        while len(records) > 64 or sum(map(len, records.values())) > self._RECORD_BYTES:
            records.popitem(last=False)

    def _add_audit(self, other):
        for name, value in other.to_dict().items():
            if name.startswith("role_link."):
                _, role, metric = name.split(".", 2)
                phases = self._audit.role_link_bodies.setdefault(role, {})
                phases[metric] = phases.get(metric, 0) + value
            else:
                setattr(self._audit, name, getattr(self._audit, name) + value)

    @property
    def audit(self):
        return self._audit

    def create(self, body, cancelled=None):
        from .client import ResponseStream, ProtocolError
        from .types import Response

        attempt = _OwnedAttempt(self, body, cancelled or threading.Event())
        stream = ResponseStream(attempt)
        if body.get("stream"):
            return stream
        final = None
        try:
            for event in stream:
                if event.type in {"response.completed", "response.incomplete"}:
                    final = Response.from_dict(event.response)
        finally:
            stream.close()
        if final is None:
            raise ProtocolError("response stream ended without a terminal response")
        return final

    async def async_create(self, body):
        cancelled = threading.Event()
        if not body.get("stream"):
            return await _owned_thread(self.create, body, cancelled, cancel=cancelled.set)
        stream = self.create(body, cancelled)

        async def iterate():
            from .client import _next_or

            sentinel = object()
            try:
                while True:
                    value = await _owned_thread(_next_or, stream, sentinel, cancel=cancelled.set)
                    if value is sentinel:
                        break
                    yield value
            finally:
                await _owned_thread(stream.close, cancel=cancelled.set)

        return iterate()

    def planning(self, response_id):
        from pllm.plan import PlanningResult
        from pllm.deployment.network import strict_load

        with self.lock:
            return PlanningResult.from_spec(strict_load(self._decisions[response_id]))

    def list_models(self):
        return {"object": "list", "data": [
            {"id": model, "object": "model", "created": 0, "owned_by": "pllm"}
            for model in getattr(self.factory, "model_ids", ())
        ]}

    def retrieve(self, response_id):
        from .types import Response
        from pllm.deployment.network import strict_load

        with self.lock:
            return Response.from_dict(strict_load(self.records[response_id]))

    def evict_response(self, response_id):
        with self.lock:
            self.records.pop(response_id, None)
            self._decisions.pop(response_id, None)

    def cancel(self, response_id):
        from .types import Response

        with self.lock:
            streams = tuple(self.active)
        for stream in streams:
            if stream.response_id == response_id:
                stream.close()
                response = dict(stream.last_response)
                if response.get("status") not in {"completed", "incomplete", "cancelled"}:
                    response["status"] = "cancelled"
                self._remember({"response": response}, None)
                return Response.from_dict(response)
        return self.retrieve(response_id)

    def preprocess(self, *args, **kwargs):
        raise NetworkError("automatic admission prepares within each bounded response attempt")

    prepare_response = preprocess
    prepared_rows_for_response = preprocess
    prepared_inventory_status = preprocess

    def close(self):
        with self.lock:
            self._closing = True
            streams = tuple(self.active)
            for stream in streams:
                stream.cancelled.set()
        error = None
        try:
            for stream in streams:
                try:
                    stream.close()
                except BaseException as exc:
                    if error is None:
                        error = exc
        finally:
            with self.lock:
                self.records.clear()
                self._decisions.clear()
        if error is not None:
            raise error
