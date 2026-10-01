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
    if (
        values.ndim != 1
        or not 0 < values.size <= 1_000_000
        or np.any(values < 0)
        or np.any(values > 0xFFFFFFFF)
    ):
        raise ValueError("prefill cache token IDs are invalid")
    hasher = hashlib.sha256(b"pllm.client.prefill-cache.v1\0")
    hasher.update(binding_digest.encode("ascii"))
    hasher.update(bundle_fingerprint.encode("ascii"))
    hasher.update(len(values).to_bytes(4, "big"))
    hasher.update(values.astype("<u4").tobytes())
    return hasher.hexdigest()


@dataclass(slots=True)
class _Block:
    arrays: tuple[tuple[np.ndarray, np.ndarray], ...]
    bytes: int
    references: int = 0

    def erase(self) -> None:
        for pair in self.arrays:
            for value in pair:
                value.setflags(write=True)
                value.fill(0)
                value.setflags(write=False)


@dataclass(slots=True)
class _Entry:
    position: int
    layers: int
    blocks: tuple[tuple[str, int], ...]
    logits: np.ndarray | None


class ExactPrefillCache:
    """Private immutable eight-row KV blocks; hits return independent snapshots.

    The byte limit counts owned NumPy payloads, as before, not Python metadata
    or transient snapshot copies. A separate entry ceiling bounds metadata.
    Snapshots passed to this cache remain owned by the caller.
    """

    _BLOCK_ROWS = 8
    _MAX_ENTRIES = 4096

    def __init__(self, max_bytes: int) -> None:
        if type(max_bytes) is not int or not 0 < max_bytes <= 256 << 20:
            raise ValueError("prefill cache must be bounded to at most 256 MiB")
        self.max_bytes = max_bytes
        self._bytes = 0
        self._entries: OrderedDict[str, _Entry] = OrderedDict()
        self._blocks: dict[str, _Block] = {}
        self._lock = threading.Lock()

    @property
    def size_bytes(self) -> int:
        with self._lock:
            return self._bytes

    @property
    def entry_count(self) -> int:
        with self._lock:
            return len(self._entries)

    @property
    def block_count(self) -> int:
        with self._lock:
            return len(self._blocks)

    def _snapshot(self, entry: _Entry) -> RuntimeSnapshot:
        caches = []
        for layer in range(entry.layers):
            pairs = [self._blocks[key].arrays[layer] for key, _ in entry.blocks]
            key = np.concatenate(
                [pair[0][:length] for pair, (_, length) in zip(pairs, entry.blocks, strict=True)]
            )
            value = np.concatenate(
                [pair[1][:length] for pair, (_, length) in zip(pairs, entry.blocks, strict=True)]
            )
            caches.append(LayerCache(key, value, entry.position))
        return RuntimeSnapshot(entry.position, caches, {})

    def get(
        self, key: str, *, position: int, layers: int
    ) -> tuple[RuntimeSnapshot, np.ndarray] | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            if entry.position != position or entry.layers != layers or entry.logits is None:
                self._remove(key)
                return None
            self._entries.move_to_end(key)
            return self._snapshot(entry), entry.logits.copy()

    def longest_prefix(
        self,
        binding_digest: str,
        bundle_fingerprint: str,
        token_ids: Sequence[int],
        *,
        layers: int,
    ) -> tuple[int, RuntimeSnapshot] | None:
        """Find longest retained proper prefix; never return a mutable cached share."""
        for position in range(len(token_ids) - 1, 0, -1):
            key = prefill_key(binding_digest, bundle_fingerprint, token_ids[:position])
            with self._lock:
                entry = self._entries.get(key)
                if entry is None:
                    continue
                if entry.position != position or entry.layers != layers:
                    self._remove(key)
                    continue
                self._entries.move_to_end(key)
                return position, self._snapshot(entry)
        return None

    def _remove(self, key: str) -> None:
        entry = self._entries.pop(key)
        if entry.logits is not None:
            self._bytes -= entry.logits.nbytes
            entry.logits.fill(0)
        for block_key, _ in entry.blocks:
            block = self._blocks[block_key]
            block.references -= 1
            if not block.references:
                self._bytes -= block.bytes
                del self._blocks[block_key]
                block.erase()

    @staticmethod
    def _valid(snapshot: RuntimeSnapshot) -> bool:
        if (
            type(snapshot.position) is not int
            or not 1 <= snapshot.position <= 4096
            or not snapshot.caches
            or snapshot.shared_kv
        ):
            return False
        for row in snapshot.caches:
            if row.length != snapshot.position or row.key is None or row.value is None:
                return False
            if (
                row.key.dtype != np.float32
                or row.value.dtype != np.float32
                or row.key.ndim != 3
                or row.value.shape != row.key.shape
                or row.key.shape[0] < snapshot.position
                or min(row.key.shape[1:]) < 1
                or not np.all(np.isfinite(row.key[: snapshot.position]))
                or not np.all(np.isfinite(row.value[: snapshot.position]))
            ):
                return False
        return True

    @staticmethod
    def _row_bytes(snapshot: RuntimeSnapshot) -> int:
        size = 0
        for row in snapshot.caches:
            assert row.key is not None and row.value is not None
            size += row.key[0].nbytes + row.value[0].nbytes
        return size

    def _prepare(
        self,
        snapshot: RuntimeSnapshot,
        limit: int | None = None,
    ) -> tuple[list[str], dict[str, _Block]]:
        total = snapshot.position if limit is None else limit
        keys, blocks = [], {}
        for start in range(0, total, self._BLOCK_ROWS):
            stop = min(total, start + self._BLOCK_ROWS)
            arrays = []
            hasher = hashlib.sha256(b"pllm.client.kv-block.v1\0")
            hasher.update(len(snapshot.caches).to_bytes(4, "big"))
            for row in snapshot.caches:
                assert row.key is not None and row.value is not None
                pair = (row.key[start:stop].copy(), row.value[start:stop].copy())
                for value in pair:
                    for dimension in value.shape:
                        hasher.update(dimension.to_bytes(4, "big"))
                    hasher.update(value.tobytes(order="C"))
                    value.setflags(write=False)
                arrays.append(pair)
            key = hasher.hexdigest()
            block = _Block(tuple(arrays), sum(value.nbytes for pair in arrays for value in pair))
            keys.append(key)
            if key in blocks:
                block.erase()
            else:
                blocks[key] = block
        return keys, blocks

    def _discard(self, blocks: dict[str, _Block]) -> None:
        # Called while locked. Prepared buffers not retained by this cache are
        # erased; shared live blocks are erased only after their last reference.
        for key, block in blocks.items():
            if self._blocks.get(key) is not block:
                block.erase()

    def put(self, key: str, snapshot: RuntimeSnapshot, logits: np.ndarray) -> bool:
        if not key or not self._valid(snapshot):
            return False
        values = np.asarray(logits)
        if (
            values.dtype != np.float32
            or values.ndim != 1
            or values.size < 1
            or not np.all(np.isfinite(values))
        ):
            return False
        if values.nbytes + snapshot.position * self._row_bytes(snapshot) > self.max_bytes:
            return False
        keys, blocks = self._prepare(snapshot)
        with self._lock:
            try:
                return self._put(key, snapshot.position, len(snapshot.caches), keys, blocks, values)
            finally:
                self._discard(blocks)

    def put_prefixes(
        self,
        binding_digest: str,
        bundle_fingerprint: str,
        token_ids: Sequence[int],
        snapshot: RuntimeSnapshot,
    ) -> int:
        """Save bounded causal KV checkpoints from a completed, full-context prefill."""
        total = len(token_ids)
        if snapshot.position != total or total < 2 or not self._valid(snapshot):
            return 0
        # All checkpoints reference the same immutable full blocks. Only the
        # last reference is clipped; materialization never reveals future rows.
        positions = {
            total - 1,
            *(1 << bit for bit in range(total.bit_length()) if 1 << bit < total),
            *(position for position in range(8, total, 8)),
        }
        # Preflight candidate copies; retain a bounded prefix even when a full
        # snapshot exceeds the budget. No allocation scales beyond this limit.
        limit = min(total, self.max_bytes // self._row_bytes(snapshot))
        if limit < 1:
            return 0
        if limit < total:
            positions = {position for position in positions if position <= limit}
            positions.add(limit)
        keys, blocks = self._prepare(snapshot, limit)
        saved = 0
        with self._lock:
            try:
                for position in sorted(positions):
                    if self._put(
                        prefill_key(binding_digest, bundle_fingerprint, token_ids[:position]),
                        position,
                        len(snapshot.caches),
                        keys,
                        blocks,
                        None,
                    ):
                        saved += 1
            finally:
                self._discard(blocks)
        return saved

    def _put(
        self,
        key: str,
        position: int,
        layers: int,
        keys: list[str],
        blocks: dict[str, _Block],
        values: np.ndarray | None,
    ) -> bool:
        refs = tuple(
            (block_key, min(self._BLOCK_ROWS, position - offset))
            for offset, block_key in zip(range(0, position, self._BLOCK_ROWS), keys)
        )
        # Retain known logits when a later prefill stores this same checkpoint.
        old = self._entries.get(key)
        if (
            values is None
            and old is not None
            and old.position == position
            and old.layers == layers
            and len(old.blocks) == len(refs)
            and all(
                old_length == length
                and all(
                    np.array_equal(a[:length], b[:length])
                    for old_pair, pair in zip(
                        self._blocks[old_key].arrays, blocks[block_key].arrays, strict=True
                    )
                    for a, b in zip(old_pair, pair, strict=True)
                )
                for (old_key, old_length), (block_key, length) in zip(old.blocks, refs, strict=True)
            )
        ):
            values = old.logits
        logits = None if values is None else values.copy()
        size = (0 if logits is None else logits.nbytes) + sum(
            blocks[block_key].bytes for block_key in {item[0] for item in refs}
        )
        if size > self.max_bytes:
            return False
        if key in self._entries:
            self._remove(key)
        while True:
            additional = (0 if logits is None else logits.nbytes) + sum(
                blocks[block_key].bytes
                for block_key in {item[0] for item in refs}
                if block_key not in self._blocks
            )
            if (
                self._bytes + additional <= self.max_bytes
                and len(self._entries) < self._MAX_ENTRIES
            ):
                break
            self._remove(next(iter(self._entries)))
        for block_key, _ in refs:
            if block_key not in self._blocks:
                # Candidate buffers must survive any eviction during this batch.
                # Never register a candidate itself: eviction zeroizes live pools.
                source = blocks[block_key]
                arrays = tuple(tuple(value.copy() for value in pair) for pair in source.arrays)
                for pair in arrays:
                    for value in pair:
                        value.setflags(write=False)
                self._blocks[block_key] = _Block(arrays, source.bytes)
                self._bytes += source.bytes
            self._blocks[block_key].references += 1
        self._entries[key] = _Entry(position, layers, refs, logits)
        self._bytes += 0 if logits is None else logits.nbytes
        return True

    def clear(self) -> None:
        with self._lock:
            while self._entries:
                self._remove(next(iter(self._entries)))


__all__ = ["ExactPrefillCache", "prefill_key"]
