"""Response-scoped, client-local ownership for an exact conversation chain."""

from __future__ import annotations

import asyncio
import math
import threading
from typing import Any

from .types import Response


class _TurnCancelled(Exception):
    pass


class Conversation:
    """Own one completed response and its KV; retire it after the next commits.

    The byte bound covers retained KV and next-token logits. Active execution,
    old/new snapshot overlap, text, tokenizer and the separate prefix cache keep
    their existing owners and admission. This handle never changes numerics or
    transfers private state to a provider.
    """

    def __init__(self, client, *, max_state_bytes: int = 32 << 20, **defaults: Any):
        if type(max_state_bytes) is not int or not 1 <= max_state_bytes <= 256 << 20:
            raise ValueError("conversation state bound must be 1 byte–256 MiB")
        self._validate_options(defaults)
        core = client._core
        if core._experiment_budget is None or core.prefill_cache_mode != "prefix":
            raise ValueError("conversation requires an Experiment with bounded ClientPrefixReuse")
        self._core = core
        self._defaults = dict(defaults)
        self._max_state_bytes = max_state_bytes
        self._response_id: str | None = None
        self._state_bytes = 0
        self._closed = False
        self._lock = threading.Lock()

    @staticmethod
    def _validate_options(options):
        if set(options) & {"previous_response_id", "store", "stream", "input"}:
            raise ValueError(
                "conversation owns input, previous_response_id, store and non-streaming execution"
            )

    @property
    def max_state_bytes(self) -> int:
        return self._max_state_bytes

    @property
    def response_id(self) -> str | None:
        return self._response_id

    @property
    def state_bytes(self) -> int:
        """Current retained KV/logit payload, excluding execution and text owners."""
        return self._state_bytes

    def _admit(self, options):
        core = self._core
        if self._closed or core._closing:
            raise RuntimeError("conversation is closed")
        model = core.experiment.model
        if options.get("model", model) != model:
            raise ValueError("conversation model differs from its Experiment")
        budget = core._experiment_budget
        state = core._transformer_state(model)
        compiled = core._compiled_public_decoder(
            state, max_input_tokens=budget.max_input_tokens, max_new_tokens=budget.max_new_tokens
        )
        if compiled is None:
            raise ValueError("conversation requires a compiled full-KV decoder")
        # Native continuation legality excludes hybrid, sliding and retained state
        # before a response can reserve any one-use material.
        core._decoder_continuation(compiled)
        graph = compiled._plan.decode
        required = sum(4 * math.prod(row["shape"]) for row in graph["state_inputs"])
        required += 4 * int(state.bundle.cfg["vocab_size"])
        if required > self.max_state_bytes:
            raise ValueError("compiled conversation KV/logits exceed max_state_bytes")

    def send(self, input: str | list[Any], **options: Any) -> Response:
        """Complete one turn; reuse exact prior state and retire the old owner."""
        return self._send(input, options, threading.Event())

    def _send(self, input, options, cancelled):
        self._validate_options(options)
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("conversation already has an active turn")
        issued: set[str] = set()
        events = None
        committed = False
        try:
            options = {**self._defaults, **options}
            self._admit(options)
            if cancelled.is_set():
                raise _TurnCancelled()
            body = {**options, "input": input, "store": True}
            if self._response_id is not None:
                body["previous_response_id"] = self._response_id
            events = self._core.events(body)
            result = None
            for event in events:
                if event.type in {"response.created", "response.completed", "response.incomplete"}:
                    issued.add(str(event.response["id"]))
                if cancelled.is_set():
                    raise _TurnCancelled()
                if event.type in {"response.completed", "response.incomplete"}:
                    result = Response.from_dict(event.response)
            if result is None:
                raise RuntimeError("conversation turn ended without a completed response")
            with self._core._transformer_conversation_lock:
                state = self._core._transformer_conversations.get(result.id)
                if state is None:
                    raise RuntimeError("conversation did not produce owned decoder state")
                arrays = [state.next_logits]
                for cache in state.snapshot.caches:
                    arrays.extend((cache.key, cache.value))
                payload = sum(value.nbytes for value in arrays if value is not None)
                if state.snapshot.shared_kv or payload > self.max_state_bytes:
                    raise RuntimeError("conversation state exceeds its admitted ownership")
            if cancelled.is_set():
                raise _TurnCancelled()
            prior = self._response_id
            self._response_id, self._state_bytes = result.id, payload
            committed = True
            if prior is not None:
                self._core.evict_response(prior)
            return result
        finally:
            try:
                try:
                    if events is not None:
                        events.close()
                finally:
                    if not committed:
                        for response_id in issued:
                            self._core.evict_response(response_id)
                            self._core._pending_response_store.pop(response_id, None)
            finally:
                self._lock.release()

    def close(self) -> None:
        """Retire this chain's response, history and KV; leave the client open."""
        with self._lock:
            self._closed = True
            if self._response_id is not None:
                self._core.evict_response(self._response_id)
            self._response_id, self._state_bytes = None, 0

    def __enter__(self):
        if self._closed:
            raise RuntimeError("conversation is closed")
        return self

    def __exit__(self, *_):
        self.close()


class AsyncConversation:
    """Async conversation with cooperative cancellation and worker draining."""

    def __init__(self, client, *, max_state_bytes: int = 32 << 20, **defaults: Any):
        self._sync = Conversation(client.sync, max_state_bytes=max_state_bytes, **defaults)
        self._lock = asyncio.Lock()

    @property
    def response_id(self) -> str | None:
        return self._sync.response_id

    @property
    def max_state_bytes(self) -> int:
        return self._sync.max_state_bytes

    @property
    def state_bytes(self) -> int:
        return self._sync.state_bytes

    async def send(self, input: str | list[Any], **options: Any) -> Response:
        if self._lock.locked():
            raise RuntimeError("conversation already has an active turn")
        async with self._lock:
            return await self._send(input, options)

    async def _send(self, input, options):
        cancelled = threading.Event()
        task = asyncio.create_task(asyncio.to_thread(self._sync._send, input, options, cancelled))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled.set()
            # Cancelling to_thread does not stop its worker. Drain it before the
            # caller can reuse/close state or issue the next one-use reservation.
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    continue
                except BaseException:
                    break
            if not task.cancelled():
                task.exception()
            # Cancellation may arrive just after the worker commits. Retire the
            # chain in either case; never expose an unseen turn to a later send.
            self._sync.close()
            raise

    async def close(self) -> None:
        async with self._lock:
            self._sync.close()

    async def __aenter__(self):
        self._sync.__enter__()
        return self

    async def __aexit__(self, *_):
        await self.close()
