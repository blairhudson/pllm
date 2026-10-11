"""Bounded exact-context graph: immutable KV pages and authenticated manifest deltas.

Internal research owner, not an SDK conversation or rollback service. Branches
share one directory/key/context and take turns under the existing writer lock.
Caller retains each branch's newest trusted head. No page or accepted head is GC'd.
"""
from __future__ import annotations

import time

import msgpack
import numpy as np

from .tiered_execution import TieredExecution, _MAX_MANIFEST

_SCHEMA = "pllm.context_graph.v1"
_ANCHOR_INTERVAL = 16


def _unpack(raw):
    return msgpack.unpackb(raw, raw=False, strict_map_key=True,
        max_bin_len=_MAX_MANIFEST, max_str_len=256, max_array_len=32768, max_map_len=32)


class ContextGraphExecution(TieredExecution):
    """Same numeric executor; only manifest publication/restoration changes."""

    def __init__(self, *args, **kwargs):
        self.read_cpu_seconds = self.write_cpu_seconds = 0.0
        self.manifest_read_bytes = self.manifest_written_bytes = 0
        self.logical_manifest_written_bytes = 0
        super().__init__(*args, **kwargs)

    def _aad(self, metadata):
        # Separate storage-format domain, still binds compiler/numeric context.
        return _SCHEMA.encode() + b"\0" + super()._aad(metadata)

    def _write(self, raw, metadata):
        start = time.process_time()
        try:
            if "manifest" in metadata:
                doc = _unpack(raw)
                self.logical_manifest_written_bytes += len(raw)
                pending = int(np.argmax(np.frombuffer(doc["logits"], np.float32)))
                if metadata["manifest"] % _ANCHOR_INTERVAL == 0:
                    envelope = {"schema": _SCHEMA, "base": doc, "pending": pending}
                else:
                    # Committed ledgers are append-only. Do not republish prefix
                    # history or page commitments on every branch step.
                    if (doc["history"][:len(self.history)] != self.history
                            or any(new[:len(old)] != old for new, old in
                                   zip(doc["records"], self.records, strict=True))):
                        raise ValueError("context graph prefix is not immutable")
                    envelope = {"schema": _SCHEMA, "parent": self.head,
                        "context": doc["context"],
                        "records": [new[len(old):] for new, old in
                                    zip(doc["records"], self.records, strict=True)],
                        "history": doc["history"][len(self.history):],
                        "logits": doc["logits"], "basis": doc["basis"], "pending": pending}
                raw = msgpack.packb(envelope, use_bin_type=True)
                if len(raw) > _MAX_MANIFEST:
                    raise ValueError("context graph manifest bound exceeded")
            record = super()._write(raw, metadata)
            if "manifest" in metadata:
                self.manifest_written_bytes += record["bytes"]
            return record
        finally:
            self.write_cpu_seconds += time.process_time() - start

    def _read_frame(self, record, metadata, maximum):
        start = time.process_time()
        try:
            raw = super()._read(record, metadata, maximum)
            if "manifest" in metadata:
                self.manifest_read_bytes += record["bytes"]
            return raw
        finally:
            self.read_cpu_seconds += time.process_time() - start

    def _read(self, record, metadata, maximum):
        if "manifest" not in metadata:
            return self._read_frame(record, metadata, maximum)
        revision = metadata["manifest"]
        deltas = []
        for _ in range(_ANCHOR_INTERVAL):
            env = _unpack(self._read_frame(record, {"manifest": revision}, maximum))
            if type(env) is not dict or env.get("schema") != _SCHEMA:
                raise ValueError("invalid context graph schema")
            if revision % _ANCHOR_INTERVAL == 0:
                if set(env) != {"schema", "base", "pending"} or type(env["base"]) is not dict:
                    raise ValueError("invalid context graph anchor")
                doc = env["base"]
                self._validate_pending(env, doc)
                break
            if (set(env) != {"schema", "parent", "context", "records", "history",
                             "logits", "basis", "pending"}
                    or env["context"] != self.context
                    or type(env["records"]) is not list or len(env["records"]) != self.layers
                    or any(type(rows) is not list for rows in env["records"])
                    or type(env["history"]) is not list or not env["history"]):
                raise ValueError("invalid context graph delta")
            parent = env["parent"]
            if (type(parent) is not dict or set(parent) != {"revision", "record"}
                    or type(parent["revision"]) is not int or parent["revision"] != revision - 1):
                raise ValueError("context graph parent revision differs")
            deltas.append(env)
            revision, record = parent["revision"], parent["record"]
        else:
            raise ValueError("context graph exceeds authenticated chain bound")
        # Every reconstructed state passes the original full-ledger admission on
        # final restore. Limit intermediate owners before concatenation/packing.
        for delta in reversed(deltas):
            if (type(doc.get("history")) is not list or type(doc.get("records")) is not list
                    or len(doc["records"]) != self.layers
                    or any(type(rows) is not list for rows in doc["records"])
                    or len(doc["history"]) + len(delta["history"]) > self.maximum
                    or any(len(old) + len(new) > self.maximum for old, new in
                           zip(doc["records"], delta["records"], strict=True))):
                raise ValueError("context graph reconstructed extent exceeds bound")
            doc = {"context": delta["context"],
                "records": [old + new for old, new in
                            zip(doc["records"], delta["records"], strict=True)],
                "history": doc["history"] + delta["history"],
                "logits": delta["logits"], "basis": delta["basis"]}
            self._validate_pending(delta, doc)
        raw = msgpack.packb(doc, use_bin_type=True)
        if len(raw) > _MAX_MANIFEST:
            raise ValueError("context graph reconstructed manifest exceeds bound")
        return raw

    @staticmethod
    def _validate_pending(env, doc):
        scores = doc.get("logits")
        if (type(scores) is not bytes or not scores or len(scores) % 4
                or type(env["pending"]) is not int
                or not np.all(np.isfinite(np.frombuffer(scores, np.float32)))
                or env["pending"] != int(np.argmax(np.frombuffer(scores, np.float32)))):
            raise ValueError("context graph pending-token provenance differs")

    @property
    def pending_token(self):
        """Next greedy token, not yet executed or included in saved history/KV."""
        if self.closed or self.failed or self.logits is None:
            raise ValueError("context graph owner is unavailable")
        return int(np.argmax(self.logits))


def context_graph_runtime(compiled, remote, directory, key, *, head=None, page_rows=64):
    runtime = compiled.runtime(remote)
    runtime._tiered = ContextGraphExecution(compiled, runtime, directory, key, head, page_rows=page_rows)
    return runtime
