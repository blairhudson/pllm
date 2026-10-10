"""Bounded compiler-bound layer paging research, with transactional encrypted state.

The trusted caller retains the key and newest head. This is not an SDK cache or
an anti-rollback/key-custody service. No protocol material is persisted.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import secrets

import msgpack
import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from filelock import FileLock

from .transformer_client import LayerCache, _snapshot_state_basis

_DOMAIN = b"pllm.tiered_decoder_reference.v1\0"
_MAX_MANIFEST = 2 << 20
_MAX_PAGE = 1 << 20
_MAX_SLAB = 128 << 20
_MAX_DISK = 2 << 30
_MAX_FILES = 32768


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _identity(value, size):
    return type(value) is str and len(value) == size and all(c in "0123456789abcdef" for c in value)


class TieredExecution:
    """One live layer, append-only authenticated pages, atomic completed phases."""

    @property
    def head(self):
        return self._committed[0]

    @property
    def records(self):
        return self._committed[1]

    @property
    def history(self):
        return self._committed[2]

    @property
    def logits(self):
        return self._committed[3]

    def __init__(self, compiled, runtime, directory, key, head=None, *, page_rows=64):
        if type(key) is not bytes or len(key) != 32 or type(page_rows) is not int or not 1 <= page_rows <= 256:
            raise ValueError("invalid tiered state key or page geometry")
        compiled.validate()
        from pllm.configuration import Pipeline
        contract = compiled._plan.continuation_schedule(Pipeline.from_spec(json.loads(compiled._canonical_composition)))
        if (runtime.position or runtime._retained is not None or runtime._window_contracts
                or runtime._hybrid_contracts or runtime._verification_bits or runtime._mlp_tiling is not None
                or runtime.nonlinear_evaluator is not None or runtime._causal_reduction != "prefix_f32"):
            raise ValueError("tiered reference requires ordinary canonical full-KV execution")
        self.shape = (runtime.default_kv_heads, runtime.default_head_dim)
        self.maximum = int(runtime._graphs["decode"]["maximum_key_sequence"])
        self.layers, self.page_rows = runtime.layers, page_rows
        self.slab_bytes = self.maximum * math.prod(self.shape) * 8
        if (self.slab_bytes > _MAX_SLAB or self.slab_bytes * self.layers > _MAX_DISK
                or page_rows * math.prod(self.shape) * 4 > _MAX_PAGE
                or 4 * runtime.cfg["vocab_size"] + 4096 > _MAX_MANIFEST):
            raise ValueError("tiered reference exceeds layer/disk admission")
        # Native continuation admission validates the complete source/state
        # contract, independently of this storage layout and its residency bound.
        contract.admit(1, 1)
        self.last_steps = {}
        for phase in ("prefill", "decode"):
            graph, schedule = runtime._graphs[phase], runtime._schedule[phase]
            by_id = {op["id"]: op for op in graph["operations"]}
            if any(s["kind"] not in {"key", "value"} for s in graph["state_outputs"]):
                raise ValueError("tiered reference requires exclusive dense KV")
            if len(graph["state_outputs"]) != 2 * self.layers:
                raise ValueError("tiered reference requires paired layer state")
            for layer in range(self.layers):
                indexes = [i for i, step in enumerate(schedule["steps"])
                           if any(by_id[op].get("layer") == layer for op in step["operation_ids"])]
                if not indexes or indexes != list(range(min(indexes), max(indexes) + 1)):
                    raise ValueError("tiered layer execution must be contiguous")
                self.last_steps[phase, max(indexes)] = layer
        self.context = hashlib.sha256(_DOMAIN + _canonical({"binding": compiled.digest,
            "continuation": contract.digest, "shape": self.shape, "layers": self.layers,
            "maximum": self.maximum, "page_rows": page_rows})).hexdigest()
        self.root = Path(directory)
        if head is None:
            self.root.mkdir(mode=0o700)
        elif not self.root.is_dir():
            raise ValueError("tiered checkpoint directory is missing")
        self.lock = FileLock(str(self.root / ".writer"), timeout=0)
        self.lock.acquire()
        self.aead = AESGCM(key)
        self._committed = (None, [[] for _ in range(self.layers)], [], None)
        self.active = None
        self.pending = None
        self.new_files = []
        self.failed = self.closed = False
        self.peak_live_kv_bytes = self.read_bytes = self.written_bytes = 0
        try:
            self.disk_bytes = self.disk_files = 0
            for path in self.root.iterdir():
                if path.name == ".writer":
                    continue
                if not _identity(path.name, 32) or path.is_symlink() or not path.is_file():
                    raise ValueError("tiered directory contains an unexpected owner")
                self.disk_files += 1
                self.disk_bytes += path.stat().st_size
                if self.disk_files > _MAX_FILES or self.disk_bytes > _MAX_DISK:
                    raise ValueError("tiered disk ownership exceeds admission")
            if head is not None:
                self._restore(head, runtime)
        except BaseException:
            self.close(runtime)
            raise

    def _aad(self, metadata):
        return _DOMAIN + self.context.encode() + _canonical(metadata)

    def _write(self, raw, metadata):
        if self.disk_bytes + len(raw) + 28 > _MAX_DISK or self.disk_files == _MAX_FILES:
            raise ValueError("tiered disk ownership exceeds admission")
        nonce = secrets.token_bytes(12)
        frame = nonce + self.aead.encrypt(nonce, raw, self._aad(metadata))
        name = secrets.token_hex(16)
        with (self.root / name).open("xb") as out:
            self.new_files.append((name, len(frame)))
            self.disk_bytes += len(frame)
            self.disk_files += 1
            out.write(frame)
            out.flush()
            os.fsync(out.fileno())
        self.written_bytes += len(frame)
        return {"name": name, "digest": hashlib.sha256(frame).hexdigest(), "bytes": len(frame)}

    def _read(self, record, metadata, maximum):
        if (type(record) is not dict or set(record) != {"name", "digest", "bytes"}
                or not _identity(record["name"], 32) or not _identity(record["digest"], 64)
                or type(record["bytes"]) is not int or not 28 <= record["bytes"] <= maximum + 28):
            raise ValueError("invalid tiered page commitment")
        with (self.root / record["name"]).open("rb") as source:
            frame = source.read(record["bytes"] + 1)
        if len(frame) != record["bytes"] or hashlib.sha256(frame).hexdigest() != record["digest"]:
            raise ValueError("tiered page differs from trusted commitment")
        raw = self.aead.decrypt(frame[:12], frame[12:], self._aad(metadata))
        self.read_bytes += len(frame)
        return raw

    def _restore(self, head, runtime):
        if (type(head) is not dict or set(head) != {"revision", "record"}
                or type(head["revision"]) is not int or not 0 <= head["revision"] < 4096):
            raise ValueError("invalid trusted tiered head")
        raw = self._read(head["record"], {"manifest": head["revision"]}, _MAX_MANIFEST)
        doc = msgpack.unpackb(raw, raw=False, strict_map_key=True, max_bin_len=_MAX_MANIFEST,
                             max_str_len=256, max_array_len=32768, max_map_len=32)
        if (type(doc) is not dict or set(doc) != {"context", "records", "history", "logits", "basis"}
                or doc["context"] != self.context or type(doc["history"]) is not list
                or not 1 <= len(doc["history"]) <= self.maximum
                or any(type(t) is not int or not 0 <= t < runtime.cfg["vocab_size"] for t in doc["history"])
                or type(doc["records"]) is not list or len(doc["records"]) != self.layers
                or type(doc["logits"]) is not bytes or len(doc["logits"]) != 4 * runtime.cfg["vocab_size"]):
            raise ValueError("tiered checkpoint source or geometry differs")
        position = len(doc["history"])
        for records in doc["records"]:
            start = 0
            if type(records) is not list or len(records) > self.maximum:
                raise ValueError("invalid tiered page ledger")
            for page in records:
                if (type(page) is not dict or set(page) != {"start", "rows", "key", "value"}
                        or type(page["start"]) is not int or page["start"] != start
                        or type(page["rows"]) is not int or not 1 <= page["rows"] <= self.page_rows):
                    raise ValueError("tiered page ranges differ")
                start += page["rows"]
            if start != position:
                raise ValueError("tiered state extent differs")
        logits = np.frombuffer(doc["logits"], np.float32).copy()
        basis = doc["basis"]
        if (not np.all(np.isfinite(logits)) or type(basis) is not dict
                or set(basis) != {"phase", "native_phase", "execution", "prefill_extent"}
                or basis["phase"] not in {"completed_prefill", "incremental"}
                or not _identity(basis["native_phase"], 64) or not _identity(basis["execution"], 64)
                or type(basis["prefill_extent"]) is not int or not 1 <= basis["prefill_extent"] <= position):
            raise ValueError("tiered checkpoint provenance differs")
        self._committed = (json.loads(json.dumps(head)), doc["records"], doc["history"], logits)
        runtime.position = position
        runtime._snapshot_basis = _snapshot_state_basis(basis["phase"], position,
            runtime._state_binding_digest, basis["native_phase"], basis["execution"],
            secrets.token_hex(16), basis["prefill_extent"])

    def begin(self, runtime, ids, continuation):
        if (self.closed or self.failed or self.pending is not None or continuation
                or runtime.position != len(self.history)):
            raise ValueError("tiered owner is unavailable or phase differs")
        if not 1 <= len(ids) or runtime.position + len(ids) > self.maximum:
            raise ValueError("tiered step exceeds its response geometry")
        pages = sum(len(records) for records in self.records) + self.layers * math.ceil(len(ids) / self.page_rows)
        # Upper-bound page records, history and logits before tensor execution or
        # ciphertext publication; the encoded manifest is checked again at commit.
        manifest_bound = pages * 384 + (runtime.position + len(ids)) * 9 + 4 * runtime.cfg["vocab_size"] + 4096
        new_pages = self.layers * math.ceil(len(ids) / self.page_rows)
        if manifest_bound > _MAX_MANIFEST:
            raise ValueError("tiered manifest admission exceeded")
        if (self.disk_bytes + len(ids) * math.prod(self.shape) * 8 * self.layers
                + new_pages * 56 + manifest_bound + 28 > _MAX_DISK
                or self.disk_files + 2 * new_pages + 1 > _MAX_FILES
                or (self.head is not None and self.head["revision"] >= 4095)):
            raise ValueError("tiered disk or revision admission exceeded")
        self.pending = [list(records) for records in self.records]
        self.new_files = []
        self.rows = len(ids)
        self.done = set()
        self.phase = "decode" if runtime.position else "prefill"

    def load(self, runtime, layer):
        if self.active == layer:
            return
        if self.pending is None or self.active is not None or layer in self.done:
            raise ValueError("tiered layer ownership/order differs")
        shape = (runtime.position + self.rows, *self.shape)
        cache = LayerCache(np.zeros(shape, np.float32), np.zeros(shape, np.float32), runtime.position)
        try:
            for page in self.records[layer]:
                start, rows = page["start"], page["rows"]
                for kind in ("key", "value"):
                    raw = self._read(page[kind], {"layer": layer, "start": start, "rows": rows, "kind": kind}, _MAX_PAGE)
                    if len(raw) != rows * math.prod(self.shape) * 4:
                        raise ValueError("tiered page numeric extent differs")
                    values = np.frombuffer(raw, np.float32).reshape(rows, *self.shape)
                    if not np.all(np.isfinite(values)):
                        raise ValueError("tiered state is nonfinite")
                    getattr(cache, kind)[start:start + rows] = values
            runtime.caches[layer] = cache
            self.active = layer
            self.peak_live_kv_bytes = max(self.peak_live_kv_bytes, cache.key.nbytes + cache.value.nbytes)
        except BaseException:
            cache.key.fill(0)
            cache.value.fill(0)
            raise

    def retire_step(self, runtime, index, values):
        layer = self.last_steps.get((self.phase, index))
        if layer is None:
            return
        if self.active != layer:
            raise ValueError("tiered layer did not execute its complete state")
        cache = runtime.caches[layer]
        if cache.length != runtime.position + self.rows:
            raise ValueError("tiered layer has incomplete KV")
        # No surviving graph alias may outlive the slab. This catches unsupported
        # cross-layer state consumers before erasure instead of corrupting them.
        for value in values.values():
            if isinstance(value, np.ndarray) and any(np.shares_memory(value, data) for data in (cache.key, cache.value)):
                raise ValueError("tiered state still has a live graph consumer")
        for start in range(runtime.position, cache.length, self.page_rows):
            rows = min(self.page_rows, cache.length - start)
            page = {"start": start, "rows": rows}
            for kind in ("key", "value"):
                page[kind] = self._write(getattr(cache, kind)[start:start + rows].tobytes(),
                    {"layer": layer, "start": start, "rows": rows, "kind": kind})
            self.pending[layer].append(page)
        self.done.add(layer)
        self._retire(runtime)

    def _retire(self, runtime):
        if self.active is not None:
            cache = runtime.caches[self.active]
            for value in (cache.key, cache.value):
                if value is not None:
                    value.fill(0)
            runtime.caches[self.active] = LayerCache()
            self.active = None

    def finish(self, runtime, ids, logits):
        basis = runtime._snapshot_basis
        if (self.pending is None or self.done != set(range(self.layers)) or self.active is not None
                or basis is None or not basis.valid() or basis.phase not in {"completed_prefill", "incremental"}):
            raise ValueError("tiered phase lacks complete committed provenance")
        scores = np.asarray(logits, np.float32).reshape(-1, runtime.cfg["vocab_size"])[-1]
        if not np.all(np.isfinite(scores)):
            raise ValueError("tiered phase has nonfinite logits")
        history = self.history + ids.tolist()
        doc = {"context": self.context, "records": self.pending, "history": history,
            "logits": scores.tobytes(), "basis": {"phase": basis.phase,
                "native_phase": basis.native_phase_digest, "execution": basis.execution_digest,
                "prefill_extent": basis.prefill_extent}}
        raw = msgpack.packb(doc, use_bin_type=True)
        revision = 0 if self.head is None else self.head["revision"] + 1
        if len(raw) > _MAX_MANIFEST or revision >= 4096:
            raise ValueError("tiered manifest or revision bound exceeded")
        saved_scores = scores.copy()
        record = self._write(raw, {"manifest": revision})
        next_head = {"revision": revision, "record": record}
        generation = (next_head, self.pending, history, saved_scores)
        # Directory durability precedes publication of the new trusted head.
        fd = os.open(self.root, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        # Stop rollback deletion before the single in-memory publication point.
        # An interruption here leaves only charged, unreferenced ciphertext and
        # the entire previous generation. After publication the new head is valid.
        self.new_files.clear()
        self._committed = generation
        self.pending = None
        # Older manifests remain charged and immutable: a crash before the caller
        # durably accepts this head must leave its previous trusted head usable.

    def abort(self, runtime):
        self.failed = True
        self._retire(runtime)
        for name, size in self.new_files:
            try:
                (self.root / name).unlink(missing_ok=True)
                self.disk_bytes -= size
                self.disk_files -= 1
            except OSError:
                pass  # Unreferenced ciphertext may remain; the owner still burns.
        self.new_files.clear()
        self.pending = None
        runtime.position = len(self.history)
        runtime._snapshot_basis = None

    def close(self, runtime):
        if not self.closed:
            self.abort(runtime)
            if self.logits is not None:
                self.logits.fill(0)
            self.history.clear()
            self.aead = None
            self.closed = True
            self.lock.release()


def tiered_runtime(compiled, remote, directory, key, *, head=None, page_rows=64):
    """Construct the research executor; the caller owns key/head persistence."""
    runtime = compiled.runtime(remote)
    runtime._tiered = TieredExecution(compiled, runtime, directory, key, head, page_rows=page_rows)
    return runtime
