"""Bounded valid-prefix KV views for semantic windowed attention."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pllm.runtime.transformer_client import LayerCache


class SemanticStateError(ValueError):
    pass


MAX_WINDOW_VIEW_ELEMENTS = 1 << 24
MAX_WINDOW_STATE_BYTES = 2 << 30


@dataclass(slots=True)
class WindowedLayerCache(LayerCache):
    """Retain W−1 tokens; materialize only public, bounded query-relative views."""

    window: int = 1
    position: int = 0

    def append_windows(
        self, key: np.ndarray, value: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if type(self.window) is not int or self.window < 1:
            raise SemanticStateError("sliding cache window must be positive")
        key = np.asarray(key)
        value = np.asarray(value)
        if (
            key.dtype != np.float32
            or value.dtype != np.float32
            or key.ndim != 3
            or value.shape != key.shape
            or min(key.shape) < 1
            or not np.all(np.isfinite(key))
            or not np.all(np.isfinite(value))
        ):
            raise SemanticStateError("sliding cache inputs must be finite matching float32 tensors")
        if self.length < 0 or self.length > self.window - 1 or self.position < self.length:
            raise SemanticStateError("sliding cache prefix is invalid")
        if self.position + key.shape[0] > np.iinfo(np.int64).max:
            raise SemanticStateError("sliding cache position overflowed")
        stored_key = self.key
        stored_value = self.value
        if self.length:
            if (
                stored_key is None
                or stored_value is None
                or stored_key.dtype != np.float32
                or stored_value.dtype != np.float32
                or stored_key.shape != stored_value.shape
                or stored_key.shape != (self.length, *key.shape[1:])
                or not np.all(np.isfinite(stored_key))
                or not np.all(np.isfinite(stored_value))
            ):
                raise SemanticStateError("sliding cache stored prefix is incompatible")
        elif stored_key is not None or stored_value is not None:
            if (
                stored_key is None
                or stored_value is None
                or stored_key.shape != (0, *key.shape[1:])
                or stored_value.shape != stored_key.shape
                or stored_key.dtype != np.float32
                or stored_value.dtype != np.float32
            ):
                raise SemanticStateError("sliding cache empty prefix is inconsistent")

        query, heads, feature = key.shape
        count = heads * query * self.window * feature
        if count > MAX_WINDOW_VIEW_ELEMENTS:
            raise SemanticStateError("sliding cache query exceeds the window-view resource bound")
        old_key = (
            stored_key[: self.length]
            if stored_key is not None and self.length
            else np.empty((0, heads, feature), dtype=np.float32)
        )
        old_value = (
            stored_value[: self.length]
            if stored_value is not None and self.length
            else np.empty((0, heads, feature), dtype=np.float32)
        )
        joined_key = np.concatenate((old_key, key))
        joined_value = np.concatenate((old_value, value))
        key_views = np.zeros((1, heads, query, self.window, feature), dtype=np.float32)
        value_views = np.zeros_like(key_views)
        for index in range(query):
            end = self.length + index + 1
            begin = max(0, end - self.window)
            width = end - begin
            key_views[0, :, index, -width:, :] = joined_key[begin:end].transpose(1, 0, 2)
            value_views[0, :, index, -width:, :] = joined_value[begin:end].transpose(1, 0, 2)
        retained = min(self.window - 1, joined_key.shape[0])
        # Commit only after both views and both new state arrays have succeeded.
        next_key = joined_key[-retained:].copy() if retained else joined_key[:0].copy()
        next_value = joined_value[-retained:].copy() if retained else joined_value[:0].copy()
        self.key, self.value = next_key, next_value
        self.length = retained
        self.position += query
        return key_views, value_views

    def copy_active(self) -> "WindowedLayerCache":
        return WindowedLayerCache(
            key=None if self.key is None else self.key[: self.length].copy(),
            value=None if self.value is None else self.value[: self.length].copy(),
            length=self.length,
            window=self.window,
            position=self.position,
        )


__all__ = ["SemanticStateError", "WindowedLayerCache"]
