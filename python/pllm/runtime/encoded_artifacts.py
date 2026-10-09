"""App-owned, bounded pre-encoded public objects; no private request material.

Entries bind the original bundle, object, raw content and frame-codec identities.
First use charges publication. Later clients reuse verified framed files. Active
readers pin entries; capacity includes pinned files and publication staging.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import secrets
import stat
import tempfile
import threading
import time
import weakref

from .bundle_compression import CHUNK_BYTES, ENCODING, encode_bundle_frames

MAX_ENTRIES = 1024
MAX_READERS = 8
WORK_BYTES = 48 << 20  # eight queued batches, one publisher, eight frame readers, metadata
HEADER = "X-PLLM-Artifact-Cache-Bytes"


@dataclass(slots=True)
class _Entry:
    path: Path
    size: int
    digest: str
    raw_digest: str
    readers: int = 0


class _Stream:
    """An iterator with deterministic retirement even before first iteration."""

    def __init__(self, cache, *, entry=None, file=None, frames=None):
        self._cache = weakref.ref(cache)
        self._entry, self._file, self._frames = entry, file, frames
        self._offset = 0
        self._closed = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._closed:
            raise StopIteration
        try:
            if self._file is None:
                return next(self._frames)
            if self._offset == self._entry.size:
                raise StopIteration
            block = self._file.read(min(65536, self._entry.size - self._offset))
            if not block:
                raise ValueError("encoded artifact was truncated")
            self._offset += len(block)
            return block
        except BaseException:
            self.close()
            raise

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            if self._file is not None:
                self._file.close()
            if self._frames is not None:
                self._frames.close()
        finally:
            cache = self._cache()
            if cache is not None:
                with cache._lock:
                    if self._entry is not None:
                        self._entry.readers -= 1
                    cache._readers.discard(self)
            self._file = self._frames = self._entry = None

    def __del__(self):
        self.close()


class EncodedArtifactCache:
    """Private ephemeral disk cache, shared by an app's authenticated clients."""

    def __init__(self, max_bytes: int):
        if type(max_bytes) is not int or not 1 << 20 <= max_bytes <= 2 << 30:
            raise ValueError("encoded artifact cache requires 1 MiB–2 GiB")
        self.max_bytes = max_bytes
        self._root = tempfile.TemporaryDirectory(prefix="pllm-encoded-artifacts-")
        self._entries = OrderedDict()
        self._readers = weakref.WeakSet()
        self._lock = threading.RLock()
        self._publication_slots = threading.BoundedSemaphore(MAX_READERS)
        self._closed = False
        self._bytes = 0
        self._stats = dict(hits=0, misses=0, bypasses=0, evictions=0, repairs=0,
                          publication_cpu_ns=0, publication_wall_ns=0,
                          published_bytes=0, peak_disk_bytes=0)

    @property
    def directory(self):
        return Path(self._root.name)

    def reserve_publication(self):
        # Bound off-loop tasks before scheduling them. Their existing public
        # matrix views are borrowed; each queued batch owns at most 1 MiB.
        if not self._publication_slots.acquire(blocking=False):
            raise ValueError("encoded artifact publication capacity exhausted")

    def release_publication(self):
        self._publication_slots.release()

    def snapshot(self):
        with self._lock:
            return {"schema": "pllm.encoded_artifact_cache.v1", "capacity_bytes": self.max_bytes,
                    "resident_bytes": self._bytes, "entries": len(self._entries),
                    "active_readers": len(self._readers), "closed": self._closed, **self._stats,
                    "scope": "app-owned public framed files; publication is included in role CPU; "
                             "publication process-CPU intervals may overlap other app work"}

    def _remove(self, key):
        entry = self._entries.pop(key)
        entry.path.unlink(missing_ok=True)
        self._bytes -= entry.size

    def _make_room(self, size):
        if size > self.max_bytes:
            return False
        for key, entry in list(self._entries.items()):
            if self._bytes + size <= self.max_bytes and len(self._entries) < MAX_ENTRIES:
                break
            if not entry.readers:
                self._remove(key)
                self._stats["evictions"] += 1
        return self._bytes + size <= self.max_bytes and len(self._entries) < MAX_ENTRIES

    def _publish(self, payload, raw_digest):
        # Encoder uses bounded zlib frames; the decoder admits at most 64 KiB
        # overhead per frame. Reserve that upper bound before writing any file.
        bound = len(payload) + ((len(payload) + CHUNK_BYTES - 1) // CHUNK_BYTES) * (65536 + 8)
        if not self._make_room(bound):
            return None
        path = self.directory / secrets.token_hex(16)
        cpu, wall = time.process_time_ns(), time.perf_counter_ns()
        count, digest = 0, hashlib.sha256()
        try:
            with path.open("xb") as stream:
                for block in encode_bundle_frames(payload):
                    if count + len(block) > bound:
                        raise ValueError("encoded artifact exceeds its reserved byte bound")
                    stream.write(block)
                    digest.update(block)
                    count += len(block)
                    self._stats["peak_disk_bytes"] = max(self._stats["peak_disk_bytes"], self._bytes + count)
                stream.flush()
            self._stats["published_bytes"] += count
            return _Entry(path, count, digest.hexdigest(), raw_digest)
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        finally:
            self._stats["publication_cpu_ns"] += time.process_time_ns() - cpu
            self._stats["publication_wall_ns"] += time.perf_counter_ns() - wall

    @staticmethod
    def _open_verified(entry):
        fd = os.open(entry.path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        stream = os.fdopen(fd, "rb")
        try:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size != entry.size:
                raise ValueError("encoded artifact size changed")
            digest = hashlib.sha256()
            remaining = entry.size
            while remaining:
                block = stream.read(min(65536, remaining))
                if not block:
                    raise ValueError("encoded artifact is incomplete")
                digest.update(block)
                remaining -= len(block)
            if stream.read(1) or digest.hexdigest() != entry.digest:
                raise ValueError("encoded artifact commitment changed")
            stream.seek(0)
            return stream
        except BaseException:
            stream.close()
            raise

    def stream(self, payload, *, bundle_digest: str, object_digest: str):
        from .bundle_artifacts import _digest

        _digest(bundle_digest)
        _digest(object_digest)
        view = memoryview(payload)
        if not view.c_contiguous or not 0 < view.nbytes <= 4 << 30:
            raise ValueError("encoded artifact requires bounded contiguous public bytes")
        view = view.cast("B")
        raw_digest = hashlib.sha256(view).hexdigest()
        key = (bundle_digest, object_digest, ENCODING)
        with self._lock:
            if self._closed:
                raise ValueError("encoded artifact cache is closed")
            if len(self._readers) >= MAX_READERS:
                raise ValueError("encoded artifact reader capacity exhausted")
            entry = self._entries.get(key)
            file = None
            if entry is not None:
                if entry.raw_digest != raw_digest:
                    raise ValueError("public artifact source changed under the same identity")
                try:
                    file = self._open_verified(entry)
                except (OSError, ValueError):
                    if entry.readers:
                        raise ValueError("active encoded artifact commitment changed") from None
                    self._remove(key)
                    self._stats["repairs"] += 1
                    entry = None
                else:
                    self._entries.move_to_end(key)
                    self._stats["hits"] += 1
            if entry is None:
                self._stats["misses"] += 1
                entry = self._publish(view, raw_digest)
                if entry is not None:
                    self._entries[key] = entry
                    self._bytes += entry.size
                    try:
                        file = self._open_verified(entry)
                    except BaseException:
                        self._remove(key)
                        raise
                else:
                    self._stats["bypasses"] += 1
            if entry is not None:
                entry.readers += 1
            reader = _Stream(self, entry=entry, file=file,
                             frames=encode_bundle_frames(view) if entry is None else None)
            self._readers.add(reader)
            return reader

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for reader in list(self._readers):
                reader.close()
            for key in list(self._entries):
                self._remove(key)
            self._root.cleanup()
