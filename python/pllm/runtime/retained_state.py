"""Client-local sparse KV ownership for bounded state research.

Selection happens at the trusted client. This owner is intentionally distinct
from LayerCache/RuntimeSnapshot: it cannot enter a full-KV compiled schedule.
No selected address is sent to a worker by this API.
"""

from __future__ import annotations

import numpy as np

from pllm import _native


def static_retention(scores: np.ndarray, *, keep: int, recent: int) -> np.ndarray:
    """Stable native importance ranking at the trusted client."""
    scores = np.asarray(scores)
    if scores.ndim != 1 or scores.dtype != np.float32 or not 1 <= scores.size <= 8192:
        raise ValueError("retention scores require bounded float32 input")
    if type(keep) is not int or type(recent) is not int or not 0 <= recent <= keep <= scores.size or keep == 0:
        raise ValueError("retention counts are invalid")
    return np.frombuffer(_native._cache_retain_f32(scores.astype("<f4", copy=False).tobytes(), keep, recent), dtype="<u4")


def _positions(values: np.ndarray) -> bytes:
    values = np.asarray(values)
    if (
        values.ndim != 1 or values.dtype.kind not in "iu" or not 1 <= values.size <= 8192
        or np.any(values < 0) or np.any(values >= 1 << 20)
    ):
        raise ValueError("retained positions must be bounded integer token positions")
    return values.astype("<u4", copy=False).tobytes()


def _tensors(key: np.ndarray, value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    key, value = np.asarray(key), np.asarray(value)
    if (
        key.dtype != np.float32 or value.dtype != np.float32
        or key.ndim != 3 or value.shape != key.shape or min(key.shape) < 1
        or key.shape[0] > 8192 or key.shape[1] * key.shape[2] > 8192
        or key.nbytes + value.nbytes > 128 << 20
    ):
        raise ValueError("retained KV expects bounded matching float32 [sequence, heads, feature] tensors")
    return key, value


class RetainedLayerCache:
    """Own retained rows plus a public capacity for future token appends.

    Native storage is bounded at 64 MiB per owner; constructor byte/float
    conversions, independent returned arrays, and compaction overlap are extra.
    A whole-decoder caller must price the sum of every layer and these transients.
    Counts below measure owned payload allocations, not process RSS.
    """

    __slots__ = ("_owner", "_shape")

    def __init__(self, key: np.ndarray, value: np.ndarray, retained: np.ndarray, *, capacity: int, maximum_position: int):
        key, value = _tensors(key, value)
        if type(capacity) is not int or not 1 <= capacity <= 8192 or type(maximum_position) is not int or not 1 <= maximum_position <= 1 << 20:
            raise ValueError("retained KV capacity and position bound must be positive bounded integers")
        self._shape = key.shape[1:]
        width = self._shape[0] * self._shape[1]
        if capacity * (width * 8 + 4) > 64 << 20:
            raise ValueError("retained KV storage exceeds its allocation bound")
        self._owner = _native._RetainedKvCacheF32(
            key.astype("<f4", copy=False).tobytes(), value.astype("<f4", copy=False).tobytes(),
            width, _positions(retained), capacity, maximum_position,
        )

    @property
    def length(self) -> int:
        return self._owner.stats()[0]

    @property
    def capacity(self) -> int:
        return self._owner.stats()[1]

    @property
    def position(self) -> int:
        return self._owner.stats()[3]

    @property
    def allocated_bytes(self) -> int:
        return self._owner.stats()[4]

    def append(self, position: int, key: np.ndarray, value: np.ndarray) -> None:
        key, value = _tensors(key, value)
        if type(position) is not int or position < 0 or key.shape[1:] != self._shape:
            raise ValueError("retained KV append position or head layout differs")
        self._owner.append(position, key.astype("<f4", copy=False).tobytes(), value.astype("<f4", copy=False).tobytes())

    def retain(self, positions: np.ndarray, *, capacity: int) -> None:
        if type(capacity) is not int or not 1 <= capacity <= self.capacity:
            raise ValueError("retained KV compaction capacity must shrink or stay equal")
        self._owner.retain(_positions(positions), capacity)

    def gather(self, positions: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Independent read-only [sequence, heads, feature] arrays and positions."""
        k, v, p = self._owner.export(None if positions is None else _positions(positions))
        indices = np.frombuffer(p, dtype="<u4")
        shape = (indices.size, *self._shape)
        return np.frombuffer(k, dtype="<f4").reshape(shape), np.frombuffer(v, dtype="<f4").reshape(shape), indices

    def select_clusters(self, query: np.ndarray, candidates: np.ndarray, *, cluster_rows: int, keep_clusters: int, alpha: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Client-local min/max scoring over original-position clusters."""
        query = np.asarray(query)
        if query.dtype != np.float32 or query.shape != self._shape:
            raise ValueError("retained KV cluster query shape differs")
        if type(cluster_rows) is not int or type(keep_clusters) is not int or min(cluster_rows, keep_clusters) < 1:
            raise ValueError("cluster dimensions must be positive integers")
        if alpha is not None and (type(alpha) is not float or not 0 <= alpha <= 1):
            raise ValueError("cluster alpha must be a public fraction")
        p, scores = self._owner.clusters(query.astype("<f4", copy=False).tobytes(), _positions(candidates), cluster_rows, keep_clusters, alpha)
        return np.frombuffer(p, dtype="<u4"), np.frombuffer(scores, dtype="<f8")

    def close(self) -> None:
        self._owner.close()
