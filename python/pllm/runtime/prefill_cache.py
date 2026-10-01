"""Bounded client-only reuse of completed, exact decoder prefills."""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .transformer_client import LayerCache, RuntimeSnapshot


def prefill_key(binding_digest: str, bundle_fingerprint: str, token_ids: Sequence[int]) -> str:
    values = np.asarray(token_ids, dtype=np.int64)
    if values.ndim != 1 or not 0 < values.size <= 1_000_000 or np.any(values < 0) or np.any(values > 0xFFFFFFFF):
        raise ValueError("prefill cache token IDs are invalid")
    hasher = hashlib.sha256(b"pllm.client.prefill-cache.v1\0")
    hasher.update(binding_digest.encode("ascii"))
    hasher.update(bundle_fingerprint.encode("ascii"))
    hasher.update(len(values).to_bytes(4, "big"))
    hasher.update(values.astype("<u4").tobytes())
    return hasher.hexdigest()


def _erase(snapshot: RuntimeSnapshot, logits: np.ndarray | None) -> None:
    if logits is not None:
        logits.fill(0)
    for row in snapshot.caches:
        if row.key is not None:
            row.key.fill(0)
        if row.value is not None:
            row.value.fill(0)
    for key, value in snapshot.shared_kv.values():
        key.fill(0)
        value.fill(0)


@dataclass(slots=True)
class _Entry:
    snapshot: RuntimeSnapshot
    logits: np.ndarray | None
    bytes: int


class ExactPrefillCache:
    """No disk, no prompt strings, no shared mutable state between responses."""

    def __init__(self, max_bytes: int) -> None:
        if type(max_bytes) is not int or not 0 < max_bytes <= 256 << 20:
            raise ValueError("prefill cache must be bounded to at most 256 MiB")
        self.max_bytes = max_bytes
        self._bytes = 0
        self._entries: OrderedDict[str, _Entry] = OrderedDict()
        self._lock = threading.Lock()

    @property
    def size_bytes(self) -> int:
        with self._lock:
            return self._bytes

    def get(self, key: str, *, position: int, layers: int) -> tuple[RuntimeSnapshot, np.ndarray] | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            snapshot = entry.snapshot
            if snapshot.position != position or len(snapshot.caches) != layers or entry.logits is None:
                self._remove(key)
                return None
            self._entries.move_to_end(key)
            return (
                RuntimeSnapshot(
                    snapshot.position,
                    [row.copy_active() for row in snapshot.caches],
                    {name: (pair[0].copy(), pair[1].copy()) for name, pair in snapshot.shared_kv.items()},
                ),
                entry.logits.copy(),
            )

    def longest_prefix(
        self, binding_digest: str, bundle_fingerprint: str, token_ids: Sequence[int],
        *, layers: int,
    ) -> tuple[int, RuntimeSnapshot] | None:
        """Find longest retained proper prefix; never return a mutable cached share."""
        for position in range(len(token_ids) - 1, 0, -1):
            key = prefill_key(binding_digest, bundle_fingerprint, token_ids[:position])
            with self._lock:
                entry = self._entries.get(key)
                if entry is None:
                    continue
                snapshot = entry.snapshot
                if snapshot.position != position or len(snapshot.caches) != layers:
                    self._remove(key)
                    continue
                self._entries.move_to_end(key)
                return position, RuntimeSnapshot(
                    position,
                    [row.copy_active() for row in snapshot.caches],
                    {name: (pair[0].copy(), pair[1].copy()) for name, pair in snapshot.shared_kv.items()},
                )
        return None

    def _remove(self, key: str) -> None:
        entry = self._entries.pop(key)
        self._bytes -= entry.bytes
        _erase(entry.snapshot, entry.logits)

    def put(self, key: str, snapshot: RuntimeSnapshot, logits: np.ndarray) -> bool:
        if not key or snapshot.position < 1 or not snapshot.caches or snapshot.shared_kv:
            return False
        arrays = [
            value for row in snapshot.caches for value in (row.key, row.value)
            if value is not None
        ]
        values = np.asarray(logits)
        if (
            values.dtype != np.float32 or values.ndim != 1 or values.size < 1
            or any(value.dtype != np.float32 or not np.all(np.isfinite(value)) for value in arrays)
            or not np.all(np.isfinite(values))
        ):
            return False
        return self._put(key, snapshot, values)

    def put_prefixes(
        self, binding_digest: str, bundle_fingerprint: str, token_ids: Sequence[int],
        snapshot: RuntimeSnapshot,
    ) -> int:
        """Save bounded causal KV checkpoints from a completed, full-context prefill."""
        total = len(token_ids)
        if (
            snapshot.position != total or total < 2 or total > 4096 or snapshot.shared_kv
            or not snapshot.caches
            or any(
                row.length != total or row.key is None or row.value is None
                or row.key.shape[0] < total or row.value.shape[0] < total
                or row.key.dtype != np.float32 or row.value.dtype != np.float32
                or not np.all(np.isfinite(row.key[:total]))
                or not np.all(np.isfinite(row.value[:total]))
                for row in snapshot.caches
            )
        ):
            return 0
        # Geometric checkpoints bound storage while retaining useful common
        # prompt prefixes; the last token checkpoint captures small edits.
        positions = {
            total - 1,
            *(1 << bit for bit in range(total.bit_length()) if 1 << bit < total),
            *(position for position in range(8, total, 8)),
        }
        saved = 0
        for position in sorted(positions):
            clipped_caches: list[LayerCache] = []
            for row in snapshot.caches:
                assert row.key is not None and row.value is not None
                clipped_caches.append(LayerCache(
                    row.key[:position].copy(), row.value[:position].copy(), position,
                ))
            clipped = RuntimeSnapshot(
                position,
                clipped_caches,
                {},
            )
            if self._put(prefill_key(binding_digest, bundle_fingerprint, token_ids[:position]), clipped, None):
                saved += 1
        return saved

    def _put(self, key: str, snapshot: RuntimeSnapshot, values: np.ndarray | None) -> bool:
        arrays = [
            value for row in snapshot.caches for value in (row.key, row.value)
            if value is not None
        ]
        size = (0 if values is None else values.nbytes) + sum(value.nbytes for value in arrays)
        if size > self.max_bytes:
            return False
        entry = _Entry(snapshot, None if values is None else values.copy(), size)
        with self._lock:
            if key in self._entries:
                self._remove(key)
            while self._entries and self._bytes + size > self.max_bytes:
                self._remove(next(iter(self._entries)))
            self._entries[key] = entry
            self._bytes += size
        return True

    def clear(self) -> None:
        with self._lock:
            while self._entries:
                self._remove(next(iter(self._entries)))


__all__ = ["ExactPrefillCache", "prefill_key"]
