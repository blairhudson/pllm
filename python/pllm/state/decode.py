"""Caller-owned durable exact decode branches, independent of SDK configuration."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import NamedTuple, Sequence


class _DecodeCacheHit(NamedTuple):
    """Tuple-compatible result with named fields for ordinary SDK restoration."""
    snapshot: object
    logits: object


@dataclass(frozen=True, slots=True)
class DecodeCacheHead:
    """Trusted index commitment; disk never selects the newest revision."""

    revision: int
    context: str
    name: str
    digest: str
    bytes: int

    def __post_init__(self):
        from pllm.runtime.tiered_execution import _identity
        if (type(self.revision) is not int or not 0 <= self.revision < 4096
                or not _identity(self.context, 64) or not _identity(self.name, 32)
                or not _identity(self.digest, 64) or type(self.bytes) is not int
                or not 28 <= self.bytes <= (2 << 20) + 28):
            raise ValueError("invalid decode cache head")

    def to_spec(self) -> dict:
        return {"schema": "pllm.decode_cache_head.v1", "revision": self.revision,
                "context": self.context, "name": self.name, "digest": self.digest, "bytes": self.bytes}

    @classmethod
    def from_spec(cls, spec: dict) -> DecodeCacheHead:
        if (type(spec) is not dict or set(spec) != {
                "schema", "revision", "context", "name", "digest", "bytes"}
                or spec["schema"] != "pllm.decode_cache_head.v1"):
            raise ValueError("invalid decode cache head document")
        return cls(**{k: v for k, v in spec.items() if k != "schema"})


class DecodeCache:
    """One writer, encrypted staged KV, explicit ACK publication, no automatic GC.

    Binding uses an existing thin compiled runtime, never a checkpoint executor.
    Snapshots and returned logits remain caller-owned. Close drops runtime/key
    references but leaves every persisted file (including orphans) caller-owned.
    """

    def __init__(self, directory: str | Path, key: bytes, *, head: DecodeCacheHead | None = None,
                 max_nodes: int = 512, max_staged: int = 32, max_pages: int = 8192,
                 max_files: int = 32768, max_disk_bytes: int = 256 << 20,
                 max_index_bytes: int = 2 << 20, max_snapshot_bytes: int = 64 << 20,
                 max_transient_bytes: int = 128 << 20, max_tokens: int = 4096):
        if type(key) is not bytes or len(key) != 32:
            raise ValueError("decode cache requires a 32-byte caller key")
        if head is not None and type(head) is not DecodeCacheHead:
            raise TypeError("decode cache requires a trusted DecodeCacheHead")
        limits = locals().copy()
        caps = {"max_nodes": 4096, "max_staged": 256, "max_pages": 32768,
                "max_files": 32768, "max_disk_bytes": 2 << 30, "max_index_bytes": 2 << 20,
                "max_snapshot_bytes": 128 << 20, "max_transient_bytes": 256 << 20,
                "max_tokens": 4096}
        for name, cap in caps.items():
            value = limits[name]
            if type(value) is not int or not 1 <= value <= cap:
                raise ValueError(f"invalid decode cache bound: {name}")
        if max_transient_bytes < max_snapshot_bytes + 4 * max_index_bytes + (4 << 20):
            raise ValueError("decode transient budget must include snapshot, indexes and page frames")
        self._limits = {name: limits[name] for name in caps}
        self._directory, self._key, self._head = Path(directory), key, head
        self._store = self._runtime = self._client = None
        self._lock, self._closed = RLock(), False

    def _available(self):
        if self._closed:
            raise ValueError("decode cache is closed")

    def claim(self, client: object) -> None:
        """SDK must claim once before use; another client cannot borrow this owner."""
        with self._lock:
            self._available()
            if client is None or self._client is not None and self._client is not client:
                raise ValueError("decode cache already belongs to another client")
            self._client = client

    def bind(self, compiled, runtime) -> None:
        """Bind/rebind an existing runtime of exactly the same compiled context."""
        from pllm.runtime.decode_store import DecodeStore
        with self._lock:
            self._available()
            if self._store is None:
                self._store = DecodeStore(compiled, runtime, self._directory, self._key,
                                          self._head, self._limits)
                self._key = None
            else:
                self._store.validate_binding(compiled, runtime)
            self._runtime = runtime

    def _bound(self):
        self._available()
        if self._store is None:
            raise ValueError("decode cache must bind a compiled runtime first")

    @property
    def head(self) -> DecodeCacheHead | None:
        with self._lock:
            return self._head if self._store is None else self._store.head

    @property
    def storage_metrics(self) -> dict[str, int]:
        with self._lock:
            return {} if self._store is None else self._store.metrics()

    def stage(self, token_ids: Sequence[int], snapshot, logits) -> str | None:
        """Disk-back completed executed state; never makes it discoverable."""
        with self._lock:
            self._bound()
            return self._store.stage(self._runtime, token_ids, snapshot, logits)

    def commit(self, tickets: Sequence[str], *, acknowledged: bool) -> DecodeCacheHead:
        """Publish selected bounded batch only after authenticated provider retirement."""
        with self._lock:
            self._bound()
            return self._store.commit(tickets, acknowledged=acknowledged)

    def discard(self, tickets: Sequence[str] | None = None) -> None:
        """Retire staged attempts on failure/cancel; orphan ciphertext stays charged."""
        with self._lock:
            self._bound()
            self._store.discard(tickets)

    def _restore_head(self, previous: DecodeCacheHead | None) -> None:
        """SDK-only finalization rollback to an explicitly remembered trusted head.

        Caller must hold the single-active-response guard and verify that current
        head is still this response's just-published head. No files are deleted.
        Failed authentication leaves current publication unchanged; SDK must close
        this owner rather than continue with a cancelled response's publication.
        """
        with self._lock:
            self._bound()
            self._store._restore_head(previous)

    def lookup(self, token_ids: Sequence[int]):
        """Return independent (RuntimeSnapshot, exact next logits), or None."""
        with self._lock:
            self._bound()
            return self._store.lookup(self._runtime, token_ids)

    def lookup_child(self, parent_ids: Sequence[int], token_id: int):
        """Lookup executed child before issuing fresh provider rows; never sample."""
        with self._lock:
            self._bound()
            ids = self._store.tokens(parent_ids)
            if type(token_id) is not int or not 0 <= token_id < self._store.vocab:
                raise ValueError("invalid decode child token")
            return self._store.lookup(self._runtime, (*ids, token_id))

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._closed = True
                if self._store is not None:
                    self._store.close()
                self._runtime = self._client = self._key = None

    def __enter__(self):
        self._available()
        return self

    def __exit__(self, *_):
        self.close()

    def __copy__(self):
        raise TypeError("live decode caches cannot be copied")

    def __deepcopy__(self, memo):
        return self.__copy__()

    def __reduce_ex__(self, protocol):
        raise TypeError("persist the trusted DecodeCacheHead, not a live cache")
