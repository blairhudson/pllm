"""Thin durable decode storage using existing full-KV authenticated frame format.

No executor is instantiated. Immutable one-row layer pages share across branches;
only an ACK-published, caller-trusted encrypted index makes nodes discoverable.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import stat

import msgpack
import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from filelock import FileLock

from pllm.state.decode import DecodeCacheHead, _DecodeCacheHit
from .tiered_execution import TieredExecution, _canonical, _identity, _MAX_PAGE
from .transformer_client import LayerCache, RuntimeSnapshot, _snapshot_state_basis

_DOMAIN = b"pllm.decode_cache.v2\0"


def _preflight_metadata(raw, maximum, budget):
    """Bound decoded object fanout/depth before msgpack creates Python containers.

    Cache metadata uses ASCII strings and binary float32 payloads exclusively.
    Price every serialized object/slot, even duplicate map keys; no dedup savings.
    """
    offset, cost, remaining = 0, 0, [1]

    def length(width):
        nonlocal offset
        if offset + width > len(raw):
            raise ValueError("truncated decode metadata")
        value = int.from_bytes(raw[offset:offset + width], "big")
        offset += width
        return value

    while remaining:
        if remaining[-1] == 0:
            remaining.pop()
            continue
        remaining[-1] -= 1
        code = length(1)
        count = children = payload = 0
        kind = "scalar"
        if code <= 0x7F or code >= 0xE0 or code in {0xC0, 0xC2, 0xC3}:
            pass
        elif 0x80 <= code <= 0x8F:
            kind, count = "map", code & 15
        elif 0x90 <= code <= 0x9F:
            kind, count = "array", code & 15
        elif 0xA0 <= code <= 0xBF:
            kind, payload = "str", code & 31
        elif code in {0xC4, 0xC5, 0xC6, 0xD9, 0xDA, 0xDB}:
            kind = "bin" if code < 0xD9 else "str"
            payload = length({0xC4: 1, 0xC5: 2, 0xC6: 4, 0xD9: 1, 0xDA: 2, 0xDB: 4}[code])
        elif code in {0xDC, 0xDD, 0xDE, 0xDF}:
            kind = "array" if code <= 0xDD else "map"
            count = length(2 if code in {0xDC, 0xDE} else 4)
        elif code in {0xCA, 0xCB, 0xCC, 0xCD, 0xCE, 0xCF, 0xD0, 0xD1, 0xD2, 0xD3}:
            payload = {0xCA: 4, 0xCB: 8, 0xCC: 1, 0xCD: 2, 0xCE: 4, 0xCF: 8,
                       0xD0: 1, 0xD1: 2, 0xD2: 4, 0xD3: 8}[code]
        else:
            raise ValueError("unsupported decode metadata encoding")
        if kind in {"array", "map"}:
            if count > 32768:
                raise ValueError("decode metadata container exceeds bound")
            children = count * (2 if kind == "map" else 1)
            cost += (256 + 96 * count) if kind == "map" else (80 + 16 * count)
        elif kind in {"str", "bin"}:
            if payload > (256 if kind == "str" else maximum) or offset + payload > len(raw):
                raise ValueError("decode metadata payload exceeds bound")
            if kind == "str" and not raw[offset:offset + payload].isascii():
                raise ValueError("decode metadata strings must be ASCII")
            cost += 80 + payload
        else:
            cost += 32
        if offset + payload > len(raw):
            raise ValueError("truncated decode metadata scalar")
        offset += payload
        if cost > budget:
            raise ValueError("decode parsed metadata exceeds transient bound")
        if children:
            if len(remaining) >= 32:
                raise ValueError("decode metadata nesting exceeds bound")
            remaining.append(children)
    if offset != len(raw):
        raise ValueError("decode metadata has trailing content")
    return cost


class DecodeStore(TieredExecution):
    """Reuse frame commitments/AEAD, not TieredExecution's runtime/linear worker."""

    def __init__(self, compiled, runtime, directory, key, head, limits):
        self.limits = limits.copy()
        self.context, self.contract = self._binding(compiled, runtime)
        self.binding_owner_bytes = getattr(self, "_candidate_binding_bytes", 0)
        self.shape = (runtime.default_kv_heads, runtime.default_head_dim)
        self.layers, self.vocab = runtime.layers, int(runtime.cfg["vocab_size"])
        token_bound = getattr(self, "_candidate_token_bound", None)
        self.maximum = min(limits["max_tokens"], self.contract.to_dict()["token_bound"] if token_bound is None else token_bound)
        self.row_bytes = math.prod(self.shape) * 4
        self.binding = runtime._state_binding_digest
        self.compiled_digest = compiled.digest
        if (self.row_bytes > _MAX_PAGE or self.vocab * 4 + 4096 > limits["max_index_bytes"]
                or self.layers * 2 > limits["max_pages"]):
            raise ValueError("decode source exceeds page/index bounds")
        if self.layers * 2 * self.row_bytes + self.vocab * 4 > limits["max_snapshot_bytes"]:
            raise ValueError("decode source exceeds snapshot bound")
        if head is not None and head.context != self.context:
            raise ValueError("decode cache head context differs")
        self.root = Path(directory)
        if head is None:
            # Never discover old contents or a disk-nominated latest head.
            self.root.mkdir(mode=0o700)
        elif self.root.is_symlink() or not self.root.is_dir():
            raise ValueError("decode cache directory is missing or unsafe")
        writer_path = self.root / ".writer"
        if writer_path.is_symlink() or writer_path.exists() and not writer_path.is_file():
            raise ValueError("decode cache writer lock is unsafe")
        self.lock = FileLock(str(self.root / ".writer"), timeout=0)
        self.lock.acquire()
        self.aead = AESGCM(key)
        self._head = head
        self.nodes, self.pages, self.staged, self._pending_pages = {}, {}, {}, {}
        self.disk_bytes = self.disk_files = self.read_bytes = self.written_bytes = 0
        self.peak_transient_bytes = self.hits = self.misses = self.publications = 0
        self.closed = False
        try:
            # scandir is incremental; Path.iterdir/listdir can materialize an
            # attacker-sized directory before the file ceiling is checked.
            with os.scandir(self.root) as entries:
                for entry in entries:
                    if entry.name == ".writer":
                        continue
                    info = entry.stat(follow_symlinks=False)
                    if not _identity(entry.name, 32) or not stat.S_ISREG(info.st_mode):
                        raise ValueError("decode directory contains an unexpected owner")
                    self.disk_files += 1
                    self.disk_bytes += info.st_size
                    self._disk_bound(0, 0)
            if head is not None:
                raw = self._read(self._record(head), {"index": head.revision}, limits["max_index_bytes"])
                doc = self._decode(raw)
                self._validate_index(doc)
                self.nodes, self.pages = doc["nodes"], doc["pages"]
        except BaseException:
            self.close()
            raise

    def _binding(self, compiled, runtime):
        from pllm.configuration import Pipeline
        from .model_binding import _state_binding_digest
        # Borrow the immutable contract already admitted by this ordinary runtime;
        # do not construct/retain a second native plan/continuation owner.
        contract = getattr(runtime, "_generated_prefix_contract", None)
        if contract is None:
            raise ValueError("decode cache requires an admitted generated-prefix contract")
        payloads = {id(value): value for value in (
            contract._canonical_bytes, contract._source_plan, contract._composition)}
        schedule = json.dumps(runtime._schedule, sort_keys=True, separators=(",", ":")).encode()
        binding_bytes = sum(len(value) for value in payloads.values()) + 4 * len(schedule) + 65536
        if binding_bytes + 4 * self.limits["max_index_bytes"] > self.limits["max_transient_bytes"]:
            raise ValueError("decode native binding metadata exceeds transient bound")
        compiled.validate()
        pipeline = Pipeline.from_spec(json.loads(compiled._canonical_composition))
        document = contract.to_dict()
        if (runtime._retained is not None or runtime._window_contracts or runtime._hybrid_contracts
                or runtime._verification_bits or runtime._mlp_tiling is not None
                or runtime.nonlinear_evaluator is not None or runtime._tiered is not None
                or runtime._causal_reduction != "prefix_f32" or runtime.shared_kv
                or getattr(runtime, "_capsule", None) is not None
                or not document.get("generated_prefix_canonical")
                or contract._composition != compiled._canonical_composition):
            raise ValueError("decode cache requires ordinary native generated-prefix full-KV execution")
        # Recheck runtime/source schedule identity without installing any state.
        if (contract._source_plan != runtime._source_plan.canonical_bytes()
                or contract._source_plan != compiled._plan.canonical_bytes()
                or runtime._state_binding_digest != _state_binding_digest(compiled._plan, compiled._bundle, pipeline.digest())
                or document["source_schedule_digest"] != hashlib.sha256(
                    b"pllm.decoder_runtime_schedule.v2\0" + schedule).hexdigest()):
            raise ValueError("decode runtime source schedule differs")
        contract.admit(1, 1)
        self._candidate_binding_bytes = binding_bytes
        self._candidate_token_bound = document["token_bound"]
        context = hashlib.sha256(_DOMAIN + _canonical({"compiled": compiled.digest,
            "binding": runtime._state_binding_digest, "continuation": contract.digest,
            "shape": [runtime.default_kv_heads, runtime.default_head_dim], "layers": runtime.layers,
            "vocab": runtime.cfg["vocab_size"], "limits": self.limits, "page_rows": 1})).hexdigest()
        return context, contract

    def validate_binding(self, compiled, runtime):
        context, contract = self._binding(compiled, runtime)
        if context != self.context:
            raise ValueError("decode cache compiled/numeric context differs")
        self.contract = contract
        self.binding_owner_bytes = getattr(self, "_candidate_binding_bytes", 0)

    @property
    def head(self):
        return self._head

    @staticmethod
    def _record(head):
        return {"name": head.name, "digest": head.digest, "bytes": head.bytes}

    def _aad(self, metadata):
        return _DOMAIN + self.context.encode() + _canonical(metadata)

    def _disk_bound(self, size, files):
        if (self.disk_bytes + size > self.limits["max_disk_bytes"]
                or self.disk_files + files > self.limits["max_files"]):
            raise ValueError("decode disk ownership exceeds bound (including orphans)")

    def _write(self, raw, metadata):
        self._disk_bound(len(raw) + 28, 1)
        # Keep the existing AEAD domains/commitment structure, but private file
        # mode and exact attempted-write charging also apply on partial failure.
        nonce = secrets.token_bytes(12)
        frame = nonce + self.aead.encrypt(nonce, raw, self._aad(metadata))
        name = secrets.token_hex(16)
        fd = os.open(self.root / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        self.disk_files += 1
        self.disk_bytes += len(frame)  # Conservative charge even if write fails.
        self.written_bytes += len(frame)
        with os.fdopen(fd, "wb") as out:
            out.write(frame)
            out.flush()
            os.fsync(out.fileno())
        return {"name": name, "digest": hashlib.sha256(frame).hexdigest(), "bytes": len(frame)}

    def _read(self, record, metadata, maximum):
        self._validate_record(record, maximum)
        path = self.root / record["name"]
        # Same full-frame commitment/AEAD contract as TieredExecution._read.
        # Unlike research counters, attempted damaged reads also stay charged;
        # no-follow opening closes the filename/regular-file check race.
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as source:
            if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
                raise ValueError("decode page is not a regular private file")
            frame = source.read(record["bytes"] + 1)
        self.read_bytes += len(frame)
        if len(frame) != record["bytes"] or hashlib.sha256(frame).hexdigest() != record["digest"]:
            raise ValueError("decode page differs from trusted commitment")
        return self.aead.decrypt(frame[:12], frame[12:], self._aad(metadata))

    @staticmethod
    def _validate_record(record, maximum):
        if (type(record) is not dict or set(record) != {"name", "digest", "bytes"}
                or not _identity(record["name"], 32) or not _identity(record["digest"], 64)
                or type(record["bytes"]) is not int or not 28 <= record["bytes"] <= maximum + 28):
            raise ValueError("invalid decode frame commitment")

    def tokens(self, ids):
        if (type(ids) not in (list, tuple) or not 1 <= len(ids) <= self.maximum
                or any(type(t) is not int or not 0 <= t < self.vocab for t in ids)):
            raise ValueError("decode executed token path exceeds bound")
        return tuple(ids)

    def _path(self, ids):
        return hashlib.sha256(_DOMAIN + self.context.encode() + np.asarray(ids, dtype="<u4").tobytes()).hexdigest()

    def _page_meta(self, page_id, page):
        return {"page": page_id, "layer": page["layer"], "start": page["start"], "rows": 1, "kind": page["kind"]}

    def _validate_index(self, doc):
        if (type(doc) is not dict or set(doc) != {"schema", "context", "nodes", "pages"}
                or doc["schema"] != "pllm.decode_index.v2" or doc["context"] != self.context
                or type(doc["nodes"]) is not dict or type(doc["pages"]) is not dict
                or len(doc["nodes"]) > self.limits["max_nodes"]
                or len(doc["pages"]) > self.limits["max_pages"]):
            raise ValueError("decode index geometry/context exceeds bound")
        for pid, page in doc["pages"].items():
            if (not _identity(pid, 64) or type(page) is not dict
                    or set(page) != {"layer", "start", "kind", "record"}
                    or type(page["layer"]) is not int or not 0 <= page["layer"] < self.layers
                    or type(page["start"]) is not int or not 0 <= page["start"] < self.maximum
                    or page["kind"] not in {"key", "value"}):
                raise ValueError("decode row page geometry differs")
            self._validate_record(page["record"], self.row_bytes)
            if page["record"]["bytes"] != self.row_bytes + 28:
                raise ValueError("decode row page extent differs")
        for path, node in doc["nodes"].items():
            if type(node) is not dict or set(node) != {"tokens", "record"}:
                raise ValueError("invalid decode node")
            ids = self.tokens(node["tokens"])
            if not _identity(path, 64) or path != self._path(ids):
                raise ValueError("decode executed path commitment differs")
            self._validate_record(node["record"], self.limits["max_index_bytes"])
            self._manifest_bound(len(ids))
        self._metadata_bound([len(n["tokens"]) for n in doc["nodes"].values()], len(doc["pages"]))
        incoming = (65536 + len(doc["pages"]) * (1536 + 96)
                    + sum(1536 + len(n["tokens"]) * 44 for n in doc["nodes"].values()))
        transient = incoming + self._owner_metadata_bytes() + self.binding_owner_bytes + 4 * self.limits["max_index_bytes"]
        if transient > self.limits["max_transient_bytes"]:
            raise ValueError("decode index reconstruction metadata exceeds transient bound")
        self.peak_transient_bytes = max(self.peak_transient_bytes, transient)

    def _manifest_bound(self, position):
        count = position * self.layers * 2
        maximum = self.vocab * 4 + count * 66 + 4096
        if count > 32768 or maximum > self.limits["max_index_bytes"]:
            raise ValueError("decode node manifest exceeds bound")
        return maximum

    def _validate_node(self, doc, ids, pages):
        if (type(doc) is not dict or set(doc) != {"schema", "path", "position", "pages", "basis", "logits"}
                or doc["schema"] != "pllm.decode_node.v2" or doc["path"] != self._path(ids)
                or type(doc["position"]) is not int or doc["position"] != len(ids)
                or type(doc["logits"]) is not bytes or len(doc["logits"]) != self.vocab * 4
                or type(doc["pages"]) is not list or len(doc["pages"]) != len(ids) * self.layers * 2):
            raise ValueError("decode node manifest source/path/full-KV extent differs")
        for index, pid in enumerate(doc["pages"]):
            if not _identity(pid, 64) or pid not in pages:
                raise ValueError("decode node row commitment is missing")
            page = pages[pid]
            layer, offset = divmod(index, len(ids) * 2)
            if (page["layer"], page["start"], page["kind"]) != (
                    layer, offset // 2, "key" if offset % 2 == 0 else "value"):
                raise ValueError("decode node row ordering differs")
        basis = doc["basis"]
        if (type(basis) is not dict or set(basis) != {"phase", "native_phase", "execution", "prefill_extent"}
                or basis["phase"] not in {"completed_prefill", "canonical_incremental"}
                or not _identity(basis["native_phase"], 64) or not _identity(basis["execution"], 64)
                or basis["phase"] == "canonical_incremental" and basis["native_phase"] != self.contract.digest
                or type(basis["prefill_extent"]) is not int
                or not len(ids) <= basis["prefill_extent"] <= self.maximum):
            raise ValueError("decode node native execution provenance differs")

    def _node_doc(self, node, pages):
        ids = self.tokens(node["tokens"])
        raw = self._read(node["record"], {"node": self._path(ids)}, self._manifest_bound(len(ids)))
        doc = self._decode(raw)
        self._validate_node(doc, ids, pages)
        return doc

    def _decode(self, raw):
        base = self._owner_metadata_bytes() + self.binding_owner_bytes + 4 * self.limits["max_index_bytes"]
        price = _preflight_metadata(raw, self.limits["max_index_bytes"], self.limits["max_transient_bytes"] - base)
        self.peak_transient_bytes = max(self.peak_transient_bytes, base + price)
        return msgpack.unpackb(raw, raw=False, strict_map_key=True, max_bin_len=self.limits["max_index_bytes"],
            max_str_len=256, max_array_len=32768, max_map_len=32768)

    def _snapshot_bound(self, position, *, extra_pages=0, extra_position=0):
        size = position * self.layers * 2 * self.row_bytes + self.vocab * 4
        manifest = self._manifest_bound(position)
        # Explicit conservative Python metadata envelope, distinct from encoded
        # bytes. Full refs/logits live on disk; only one manifest is resident.
        metadata = self._owner_metadata_bytes(extra_pages=extra_pages, extra_position=extra_position)
        transient = (size + metadata + self.binding_owner_bytes + 4 * self.limits["max_index_bytes"]
                     + 4 * manifest + position * self.layers * 2 * 128 + 4 * self.row_bytes)
        if size > self.limits["max_snapshot_bytes"] or transient > self.limits["max_transient_bytes"]:
            raise ValueError("decode snapshot/transient ownership exceeds bound")
        self.peak_transient_bytes = max(self.peak_transient_bytes, transient)

    def _all_pages(self):
        pages = self.pages.copy()
        pages.update(self._pending_pages)
        return pages

    def _owner_metadata_bytes(self, *, extra_pages=0, extra_position=0):
        unique = len(self.pages.keys() | self._pending_pages.keys()) + extra_pages
        # 1536/page covers content ID, nested page/record dictionaries, strings,
        # integers and primary map entry. Every additional dictionary slot is
        # also charged; duplicate dictionaries are not assumed free.
        slots = (len(self.pages) + len(self._pending_pages)
                 + sum(len(pages) for _, pages in self.staged.values()) + unique + 2 * extra_pages)
        positions = [len(n["tokens"]) for n in self.nodes.values()]
        positions.extend(len(n["tokens"]) for n, _ in self.staged.values())
        if extra_position:
            positions.append(extra_position)
        return 65536 + unique * 1536 + slots * 96 + sum(1536 + n * 44 for n in positions)

    def _metadata_bound(self, nodes, pages):
        # Bound Python reference fanout before page lists or encoding allocate.
        estimate = 1024 + pages * 384 + sum(512 + n * 9 for n in nodes)
        if estimate > self.limits["max_index_bytes"]:
            raise ValueError("decode index/staging metadata exceeds bound")

    def stage(self, runtime, token_ids, snapshot, logits):
        ids = self.tokens(token_ids)
        self._snapshot_bound(len(ids))
        if len(self.staged) >= self.limits["max_staged"]:
            raise ValueError("decode staged batch exceeds bound")
        if snapshot.position != len(ids):
            raise ValueError("decode path must exclude pending/unexecuted selections")
        basis = snapshot.state_basis
        if (basis is None or not basis.cache_eligible(generated_prefixes=True)
                or basis.verification_failure_bits or basis.position != len(ids)
                or basis.source_binding_digest != self.binding
                or not len(ids) <= basis.prefill_extent <= self.maximum
                or basis.phase == "canonical_incremental" and basis.native_phase_digest != self.contract.digest):
            raise ValueError("decode state lacks ordinary owner-free completed provenance")
        runtime._validate_full_snapshot(snapshot)
        runtime._validate_verification_basis(basis)
        self.contract.admit(max(1, len(ids) - 1), 1)
        values = np.asarray(logits)
        if values.dtype != np.float32 or values.shape != (self.vocab,) or not np.all(np.isfinite(values)):
            raise ValueError("decode next logits differ from source geometry")
        path = self._path(ids)
        nodes = {**self.nodes, **{self._path(n["tokens"]): n for n, _ in self.staged.values()}}
        nodes[path] = {"tokens": ids}
        if len(nodes) > self.limits["max_nodes"]:
            raise ValueError("decode node count exceeds bound")
        old_pages = self._all_pages()
        # First pass counts genuinely new pages without allocating a page ledger
        # or writing anything. Existing prefixes do not consume the bound twice.
        # We count repeated new IDs conservatively; row/layer identity makes
        # within-snapshot repeats impossible under admitted exclusive full KV.
        new_count = 0
        for layer, cache in enumerate(snapshot.caches):
            for start in range(len(ids)):
                for kind in ("key", "value"):
                    raw = getattr(cache, kind)[start:start + 1].tobytes()
                    pid = hashlib.sha256(_DOMAIN + self.context.encode() + _canonical(
                        [layer, start, kind, self.shape]) + raw).hexdigest()
                    new_count += pid not in old_pages
        if len(old_pages) + new_count > self.limits["max_pages"]:
            raise ValueError("decode staged row page count exceeds bound")
        # Resident nodes are tiny index entries, not full page reference lists.
        # Keep one shared pending page pool; stages retain newly introduced pages
        # only. Complete ordered refs and logits are encrypted disk manifests.
        positions = [len(n["tokens"]) for n in self.nodes.values()]
        positions.extend(len(n["tokens"]) for n, _ in self.staged.values())
        positions.append(len(ids))
        self._metadata_bound(positions, len(old_pages) + new_count)
        self._snapshot_bound(len(ids), extra_pages=new_count, extra_position=len(ids))
        self._disk_bound(new_count * (self.row_bytes + 28) + self._manifest_bound(len(ids)) + 28,
                         new_count + 1)
        refs, new_pages = [], {}
        try:
            for layer, cache in enumerate(snapshot.caches):
                for start in range(len(ids)):
                    for kind in ("key", "value"):
                        raw = getattr(cache, kind)[start:start + 1].tobytes()
                        pid = hashlib.sha256(_DOMAIN + self.context.encode() + _canonical(
                            [layer, start, kind, self.shape]) + raw).hexdigest()
                        page = {"layer": layer, "start": start, "kind": kind}
                        if pid in old_pages:
                            # Never trust an existing row merely by its filename.
                            existing = old_pages[pid]
                            if self._read(existing["record"], self._page_meta(pid, existing), self.row_bytes) != raw:
                                raise ValueError("decode shared row content differs")
                            page = existing
                        else:
                            page["record"] = self._write(raw, self._page_meta(pid, page))
                            new_pages[pid] = page
                        refs.append(pid)
            old = nodes.get(path)
            if path in self.nodes:
                old = self.nodes[path]
            else:
                old = next((n for n, _ in self.staged.values() if self._path(n["tokens"]) == path), None)
            if old is not None:
                old_doc = self._node_doc(old, old_pages)
                if old_doc["pages"] != refs or old_doc["logits"] != values.tobytes():
                    raise ValueError("decode path has conflicting exact state")
                # Existing exact state is not a new successful state. No ticket,
                # manifest write or publication needed (including replay).
                return None
            else:
                doc = {"schema": "pllm.decode_node.v2", "path": path, "position": len(ids),
                    "pages": refs, "logits": values.tobytes(), "basis": {"phase": basis.phase,
                    "native_phase": basis.native_phase_digest, "execution": basis.execution_digest,
                    "prefill_extent": basis.prefill_extent}}
                raw = msgpack.packb(doc, use_bin_type=True)
                if len(raw) > self._manifest_bound(len(ids)):
                    raise ValueError("decode node manifest exceeds bound")
                node = {"tokens": list(ids), "record": self._write(raw, {"node": path})}
            ticket = secrets.token_hex(16)
            self.staged[ticket] = (node, new_pages)
            self._pending_pages.update(new_pages)
            return ticket
        except BaseException:
            # No ticket escapes; attempts remain charged orphans. No deletion/GC.
            raise

    def _tickets(self, tickets):
        if (type(tickets) not in (list, tuple) or not 1 <= len(tickets) <= self.limits["max_staged"]
                or any(type(t) is not str or t not in self.staged for t in tickets)
                or len(set(tickets)) != len(tickets)):
            raise ValueError("invalid decode staged tickets")
        return tuple(tickets)

    def commit(self, tickets, *, acknowledged):
        tickets = self._tickets(tickets)
        if acknowledged is not True:
            raise ValueError("decode publication requires authenticated completion acknowledgement")
        nodes, pages = self.nodes.copy(), self.pages.copy()
        try:
            available = self._all_pages()
            for ticket in tickets:
                node, staged_pages = self.staged[ticket]
                # Selected children may depend on an unselected/discarded root's
                # introduced pages. Authenticate refs against the shared pool.
                doc = self._node_doc(node, available)
                nodes[self._path(node["tokens"])] = node
                for pid in doc["pages"]:
                    pages[pid] = available[pid]
                del doc
            doc = {"schema": "pllm.decode_index.v2", "context": self.context, "nodes": nodes, "pages": pages}
            self._validate_index(doc)
            raw = msgpack.packb(doc, use_bin_type=True)
            revision = 0 if self.head is None else self.head.revision + 1
            if len(raw) > self.limits["max_index_bytes"] or revision >= 4096:
                raise ValueError("decode index or revision exceeds bound")
            record = self._write(raw, {"index": revision})
            head = DecodeCacheHead(revision, self.context, **record)
            fd = os.open(self.root, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            # Only publication point. Caller must durably accept this new head.
            self.nodes, self.pages, self._head = nodes, pages, head
            self.publications += 1
            self.discard(tickets)
            return head
        except BaseException:
            self.discard(tickets)
            raise

    def discard(self, tickets=None):
        if tickets is None:
            self.staged.clear()
        else:
            for ticket in self._tickets(tickets):
                del self.staged[ticket]
        if not self.staged:
            self._pending_pages.clear()
        else:
            # Retain only private pool records; committed records are already in
            # the global authenticated page index. No persisted file is deleted.
            self._pending_pages = {pid: page for pid, page in self._pending_pages.items() if pid not in self.pages}

    def _restore_head(self, previous):
        # Explicit caller authority only. Never inspect an on-disk latest index.
        # Clear unfinished plaintext attempts even when a prior index is damaged.
        self.staged.clear()
        self._pending_pages.clear()
        if previous is None:
            self.nodes, self.pages, self._head = {}, {}, None
            return
        if (type(previous) is not DecodeCacheHead or previous.context != self.context
                or self.head is None or previous.revision > self.head.revision
                or previous.revision == self.head.revision and previous != self.head):
            raise ValueError("decode rollback requires an explicit trusted prior head")
        raw = self._read(self._record(previous), {"index": previous.revision}, self.limits["max_index_bytes"])
        doc = self._decode(raw)
        self._validate_index(doc)
        # Single publication point after complete bounded index authentication.
        self.nodes, self.pages, self._head = doc["nodes"], doc["pages"], previous

    def lookup(self, runtime, token_ids):
        ids = self.tokens(token_ids)
        node = self.nodes.get(self._path(ids))
        if node is None:
            self.misses += 1
            return None
        self._snapshot_bound(len(ids))
        self.contract.admit(max(1, len(ids) - 1), 1)
        # Authenticate and validate the entire disk-backed manifest BEFORE any
        # snapshot/logit array allocation. Pages must match global index ordering.
        doc = self._node_doc(node, self.pages)
        basis = doc["basis"]
        logits = np.frombuffer(doc["logits"], np.float32).copy()
        caches = []
        try:
            if not np.all(np.isfinite(logits)):
                raise ValueError("decode next logits are nonfinite")
            for layer in range(self.layers):
                cache = LayerCache(np.empty((len(ids), *self.shape), np.float32),
                                   np.empty((len(ids), *self.shape), np.float32), len(ids))
                caches.append(cache)
                for start in range(len(ids)):
                    for kind_index, kind in enumerate(("key", "value")):
                        pid = doc["pages"][(layer * len(ids) + start) * 2 + kind_index]
                        page = self.pages[pid]
                        raw = self._read(page["record"], self._page_meta(pid, page), self.row_bytes)
                        if len(raw) != self.row_bytes or hashlib.sha256(_DOMAIN + self.context.encode()
                                + _canonical([layer, start, kind, self.shape]) + raw).hexdigest() != pid:
                            raise ValueError("decode row content commitment differs")
                        values = np.frombuffer(raw, np.float32).reshape(1, *self.shape)
                        if not np.all(np.isfinite(values)):
                            raise ValueError("decode row state is nonfinite")
                        getattr(cache, kind)[start:start + 1] = values
            # No old seal is serialized. Remint only after every page authenticated
            # and native continuation admission succeeded in this process.
            reminted = _snapshot_state_basis(basis["phase"], len(ids), self.binding,
                basis["native_phase"], hashlib.sha256(_DOMAIN + secrets.token_bytes(32)
                + basis["execution"].encode()).hexdigest(), None, basis["prefill_extent"])
            snapshot = RuntimeSnapshot(len(ids), caches, {}, self.binding, reminted)
            runtime._validate_full_snapshot(snapshot)
            runtime._validate_verification_basis(reminted)
            if not runtime._cache_basis_valid(reminted):
                raise ValueError("decode restored native cache basis differs")
            self.hits += 1
            return _DecodeCacheHit(snapshot, logits)
        except BaseException:
            logits.fill(0)
            for cache in caches:
                cache.key.fill(0)
                cache.value.fill(0)
            raise

    def metrics(self):
        if self.closed:
            return self._closed_metrics.copy()
        reachable = {self.head.name} if self.head is not None else set()
        reachable.update(n["record"]["name"] for n in self.nodes.values())
        reachable.update(p["record"]["name"] for p in self.pages.values())
        reachable_bytes = (0 if self.head is None else self.head.bytes) + sum(
            n["record"]["bytes"] for n in self.nodes.values()) + sum(p["record"]["bytes"] for p in self.pages.values())
        return {"disk_bytes": self.disk_bytes, "disk_files": self.disk_files,
                "read_bytes": self.read_bytes, "written_bytes": self.written_bytes,
                "node_count": len(self.nodes), "page_count": len(self.pages),
                "staged_count": len(self.staged), "hits": self.hits, "misses": self.misses,
                "publications": self.publications, "peak_transient_bytes": self.peak_transient_bytes,
                "pending_page_count": len(self._pending_pages),
                "owner_metadata_bytes": self._owner_metadata_bytes(), "binding_owner_bytes": self.binding_owner_bytes,
                "unpublished_bytes": self.disk_bytes - reachable_bytes,
                "unpublished_files": self.disk_files - len(reachable)}

    def close(self):
        if not self.closed:
            self._closed_metrics = self.metrics()
            self.staged.clear()
            self._pending_pages.clear()
            self.nodes.clear()
            self.pages.clear()
            self.aead = self.contract = None
            self.closed = True
            self.lock.release()
