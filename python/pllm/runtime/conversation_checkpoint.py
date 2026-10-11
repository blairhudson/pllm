"""Authenticated full-KV capture/restoration at the explicit SDK client boundary.

One immutable directory per checkpoint. Storage reuses bounded context-graph
pages; the ordinary prepared decoder retains its existing in-memory execution.
"""
from __future__ import annotations

import os
import math

import msgpack
import numpy as np

from pllm.state.checkpoint import CheckpointHead
from pllm.state.conversation import ConversationCheckpoint

from .context_graph import ContextGraphExecution
from .tiered_execution import _MAX_MANIFEST
from .transformer_client import LayerCache, RuntimeSnapshot
from .types import Response, new_id

_DOMAIN = b"pllm.conversation_checkpoint.v1\0"


def _encode_metadata(value):
    pending, size, items = [(value, 0)], 0, 0
    while pending:
        current, depth = pending.pop()
        items += 1
        if depth > 32 or items > 32768:
            raise ValueError("conversation checkpoint metadata exceeds structural bounds")
        if type(current) is str:
            if len(current) > _MAX_MANIFEST:
                raise ValueError("conversation history exceeds the 2 MiB checkpoint bound")
            size += 5 + len(current.encode("utf-8"))
        elif current is None or type(current) is bool:
            size += 1
        elif type(current) in (int, float):
            if (type(current) is int and not -(1 << 63) <= current < 1 << 64
                    or type(current) is float and not math.isfinite(current)):
                raise ValueError("conversation checkpoint metadata number exceeds its encoding")
            size += 9
        elif type(current) in (list, tuple, dict):
            children_count = len(current) * (2 if type(current) is dict else 1)
            if items + len(pending) + children_count > 32768:
                raise ValueError("conversation checkpoint metadata exceeds structural bounds")
            size += 5
            children = list(current.items()) if type(current) is dict else current
            for child in children:
                if type(current) is dict:
                    key, child = child
                    if type(key) is not str:
                        raise ValueError("conversation checkpoint metadata keys must be strings")
                    pending.append((key, depth + 1))
                pending.append((child, depth + 1))
        else:
            raise ValueError(f"conversation checkpoint metadata requires JSON values, got {type(current).__name__}")
        if size > _MAX_MANIFEST:
            raise ValueError("conversation history exceeds the 2 MiB checkpoint bound")
    return msgpack.packb(value, use_bin_type=True)


class _Storage(ContextGraphExecution):
    def _aad(self, metadata):
        return _DOMAIN + super()._aad(metadata)

    def capture(self, runtime, saved):
        snapshot = saved.snapshot
        runtime._validate_full_snapshot(snapshot)
        basis = snapshot.state_basis
        if (basis is None or not basis.valid() or basis.verification_failure_bits
                or basis.phase not in {"completed_prefill", "incremental"}
                or basis.position != snapshot.position):
            raise ValueError("checkpoint requires authenticated ordinary response state")
        runtime.position, runtime._snapshot_basis = snapshot.position, basis
        ids = saved.token_ids[:snapshot.position]
        if (not ids or any(type(t) is not int or not 0 <= t < runtime.cfg["vocab_size"] for t in saved.token_ids)
                or len(saved.pending_token_ids) > 1
                or len(saved.token_ids) != snapshot.position + len(saved.pending_token_ids)
                or saved.token_ids[snapshot.position:] != saved.pending_token_ids
                or saved.next_logits.shape != (runtime.cfg["vocab_size"],)
                or saved.next_logits.dtype != np.float32 or not np.all(np.isfinite(saved.next_logits))):
            raise ValueError("checkpoint executed history or pending token differs")
        # Price the complete manifest and disk payload before writing any page.
        self.begin_capture(runtime, len(ids))
        try:
            for layer, cache in enumerate(snapshot.caches):
                for start in range(0, snapshot.position, self.page_rows):
                    rows = min(self.page_rows, snapshot.position - start)
                    page = {"start": start, "rows": rows}
                    for kind in ("key", "value"):
                        page[kind] = self._write(getattr(cache, kind)[start:start + rows].tobytes(),
                            {"layer": layer, "start": start, "rows": rows, "kind": kind})
                    self.pending[layer].append(page)
                self.done.add(layer)
            self.finish(runtime, np.asarray(ids, np.int64), saved.next_logits)
        except BaseException:
            self.abort(runtime)
            raise

    def begin_capture(self, runtime, rows):
        # Reuse ordinary storage preflight while keeping the original signed
        # execution basis. No linear operator or provider callback runs here.
        position = runtime.position
        runtime.position = 0
        try:
            self.begin(runtime, range(rows), False)
        finally:
            runtime.position = position

    def snapshot(self, runtime, response_id):
        caches = []
        self.pending, self.rows, self.done = self.records, 0, set()
        try:
            for layer in range(self.layers):
                self.load(runtime, layer)
                caches.append(runtime.caches[layer])
                runtime.caches[layer] = LayerCache()
                self.active = None
            snapshot = RuntimeSnapshot(runtime.position, caches, {}, runtime._state_binding_digest,
                runtime._snapshot_basis._with_owner(response_id))
            runtime._validate_full_snapshot(snapshot)
            runtime._validate_verification_basis(snapshot.state_basis)
            return snapshot
        except BaseException:
            for cache in caches:
                cache.key.fill(0)
                cache.value.fill(0)
            raise
        finally:
            self.pending = None


def _unavailable_remote(*_):
    raise RuntimeError("checkpoint storage cannot execute a provider stage")


def _binding(chat):
    chat._admit(chat._defaults)
    core = chat._core
    state = core._transformer_state(core.experiment.model)
    budget = core._experiment_budget
    compiled = core._compiled_public_decoder(state, max_input_tokens=budget.max_input_tokens,
                                             max_new_tokens=budget.max_new_tokens)
    runtime = compiled.runtime(_unavailable_remote)
    if (runtime._verification_bits or runtime._mlp_tiling is not None
            or runtime._causal_reduction != "prefix_f32"):
        runtime.reset()
        raise ValueError("checkpoint requires ordinary prefix_f32 full-KV execution")
    return state, compiled, runtime


def save(chat, directory, key, *, page_rows):
    state, compiled, runtime = _binding(chat)
    core, owner = chat._core, None
    try:
        with core._transformer_conversation_lock:
            saved = core._transformer_conversations.get(chat.response_id)
            response = core.cache.get(chat.response_id)
            if (saved is None or response is None or not response.store
                    or saved.bundle_fingerprint != state.bundle_fingerprint
                    or saved.snapshot.state_basis.owner_response_id != chat.response_id):
                raise ValueError("checkpoint requires this conversation's completed response")
            metadata = {"schema": "pllm.conversation_metadata.v1", "model": saved.model_id,
                "bundle": saved.bundle_fingerprint, "response": response.to_dict(),
                "rendered": saved.rendered_context, "messages": core.message_histories[chat.response_id],
                "pending": saved.pending_token_ids}
            raw = _encode_metadata(metadata)
            owner = _Storage(compiled, runtime, directory, key, page_rows=page_rows)
            owner.capture(runtime, saved)
            head = CheckpointHead("context_graph", page_rows, 0, **owner.head["record"])
            record = owner._write(raw, {"conversation": head.to_spec()})
            fd = os.open(owner.root, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            result = ConversationCheckpoint(head, **record)
            owner.new_files.clear()
            return result
    finally:
        if owner is not None:
            owner.close(runtime)
        runtime.reset()


def restore(chat, directory, key, head):
    if type(head) is not ConversationCheckpoint:
        raise TypeError("restore requires a trusted ConversationCheckpoint")
    if chat.response_id is not None:
        raise ValueError("restore requires an empty conversation")
    state, compiled, runtime = _binding(chat)
    owner, snapshot, logits = None, None, None
    try:
        owner = _Storage(compiled, runtime, directory, key, head.state._record(), page_rows=head.state.page_rows)
        raw = owner._read({"name": head.name, "digest": head.digest, "bytes": head.bytes},
                          {"conversation": head.state.to_spec()}, _MAX_MANIFEST)
        doc = msgpack.unpackb(raw, raw=False, strict_map_key=True, max_bin_len=_MAX_MANIFEST,
            max_str_len=_MAX_MANIFEST, max_array_len=32768, max_map_len=32768)
        if (type(doc) is not dict or set(doc) != {"schema", "model", "bundle", "response", "rendered", "messages", "pending"}
                or doc["schema"] != "pllm.conversation_metadata.v1" or doc["model"] != state.bundle.model_id
                or doc["bundle"] != state.bundle_fingerprint or type(doc["rendered"]) is not str
                or type(doc["messages"]) is not list or any(type(r) is not dict for r in doc["messages"])
                or type(doc["pending"]) is not list or len(doc["pending"]) > 1
                or any(type(t) is not int or not 0 <= t < runtime.cfg["vocab_size"] for t in doc["pending"])):
            raise ValueError("conversation checkpoint metadata differs")
        response = Response.from_dict(doc["response"])
        if (not response.store or response.model != doc["model"]
                or response.status not in {"completed", "incomplete"}):
            raise ValueError("conversation checkpoint response is incomplete or unowned")
        response_id = new_id("resp")
        result = Response.from_dict({**response.to_dict(), "id": response_id, "previous_response_id": None})
        snapshot = owner.snapshot(runtime, response_id)
        logits = owner.logits.copy()
        payload = logits.nbytes + sum(cache.key.nbytes + cache.value.nbytes for cache in snapshot.caches)
        if payload > chat.max_state_bytes:
            raise ValueError("restored conversation exceeds max_state_bytes")
        from .client import _TransformerConversationState
        saved = _TransformerConversationState(doc["model"], doc["bundle"], [*owner.history, *doc["pending"]],
                                             doc["rendered"], snapshot, logits, doc["pending"])
        core = chat._core
        with core._transformer_conversation_lock:
            if core._closing:
                raise RuntimeError("conversation client is closed")
            # Publish only after every ciphertext and native state contract passed.
            core.cache[response_id] = result
            core.histories[response_id] = doc["rendered"]
            core.message_histories[response_id] = doc["messages"]
            core._transformer_conversations[response_id] = saved
            chat._response_id, chat._state_bytes = response_id, payload
        snapshot = None
        return result
    finally:
        if snapshot is not None:
            if logits is not None:
                logits.fill(0)
            for cache in snapshot.caches:
                cache.key.fill(0)
                cache.value.fill(0)
        if owner is not None:
            owner.close(runtime)
        runtime.reset()
