"""Explicit trusted-client checkpoint ownership for admitted full-KV decoders."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Literal


@dataclass(frozen=True, slots=True)
class CheckpointHead:
    """Immutable storage commitment. The caller must retain its latest trusted head."""

    storage: Literal["layer_pages", "context_graph"]
    page_rows: int
    revision: int
    name: str
    digest: str
    bytes: int

    def __post_init__(self):
        if (self.storage not in {"layer_pages", "context_graph"}
                or type(self.page_rows) is not int or not 1 <= self.page_rows <= 256
                or type(self.revision) is not int or not 0 <= self.revision < 4096
                or type(self.bytes) is not int or not 28 <= self.bytes <= (2 << 20) + 28):
            raise ValueError("invalid checkpoint head geometry")
        for value, size in ((self.name, 32), (self.digest, 64)):
            if type(value) is not str or len(value) != size or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("invalid checkpoint head commitment")

    def to_spec(self) -> dict:
        return {"schema": "pllm.checkpoint_head.v1", "storage": self.storage,
                "page_rows": self.page_rows, "revision": self.revision,
                "name": self.name, "digest": self.digest, "bytes": self.bytes}

    @classmethod
    def from_spec(cls, spec: dict) -> CheckpointHead:
        if (type(spec) is not dict or set(spec) != {
                "schema", "storage", "page_rows", "revision", "name", "digest", "bytes"}
                or spec["schema"] != "pllm.checkpoint_head.v1"):
            raise ValueError("invalid checkpoint head document")
        return cls(**{k: v for k, v in spec.items() if k != "schema"})

    def _record(self):
        return {"revision": self.revision,
                "record": {"name": self.name, "digest": self.digest, "bytes": self.bytes}}


class CheckpointSession:
    """Bounded compiled execution with encrypted append-only state.

    This low-level session takes an already-bound model and its linear executor.
    Keys and accepted heads remain with the caller. It does not persist provider
    credentials, inventories, response IDs, or an OpenAI conversation.
    """

    def __init__(self, compiled, remote, directory: str | Path, key: bytes, *,
                 storage: Literal["layer_pages", "context_graph"] = "context_graph",
                 head: CheckpointHead | None = None, page_rows: int = 64):
        from pllm.runtime.context_graph import ContextGraphExecution
        from pllm.runtime.tiered_execution import TieredExecution

        if storage not in {"layer_pages", "context_graph"}:
            raise ValueError("unknown checkpoint storage")
        if head is not None and (type(head) is not CheckpointHead
                                 or head.storage != storage or head.page_rows != page_rows):
            raise ValueError("checkpoint head storage or page geometry differs")
        self._lock = RLock()
        self._storage, self._page_rows = storage, page_rows
        self._runtime = compiled.runtime(remote)
        self._closed = False
        owner = ContextGraphExecution if storage == "context_graph" else TieredExecution
        try:
            self._owner = owner(compiled, self._runtime, directory, key,
                                None if head is None else head._record(), page_rows=page_rows)
            self._runtime._tiered = self._owner
        except BaseException:
            self._runtime.reset()
            self._closed = True
            raise

    def _available(self):
        if self._closed or self._owner.closed or self._owner.failed:
            raise ValueError("checkpoint session is unavailable")

    @property
    def head(self) -> CheckpointHead | None:
        """Last fully published generation, still readable after close or failure."""
        with self._lock:
            head = self._owner.head
            return None if head is None else CheckpointHead(
                self._storage, self._page_rows, head["revision"], **head["record"])

    @property
    def position(self) -> int:
        with self._lock:
            self._available()
            return self._runtime.position

    @property
    def pending_token(self) -> int:
        import numpy as np
        with self._lock:
            self._available()
            if self._owner.logits is None:
                raise ValueError("checkpoint session has no completed execution")
            return int(np.argmax(self._owner.logits))

    @property
    def logits(self):
        with self._lock:
            self._available()
            if self._owner.logits is None:
                raise ValueError("checkpoint session has no completed execution")
            return self._owner.logits.copy()

    @property
    def storage_metrics(self) -> dict[str, int]:
        with self._lock:
            return {name: getattr(self._owner, name) for name in (
                "disk_bytes", "disk_files", "read_bytes", "written_bytes", "peak_live_kv_bytes")}

    def append(self, token_ids: list[int] | tuple[int, ...]):
        """Execute explicit token IDs and publish one generation; return owned logits."""
        with self._lock:
            self._available()
            try:
                if (type(token_ids) not in (list, tuple) or not token_ids
                        or len(token_ids) > self._owner.maximum - self._runtime.position
                        or any(type(t) is not int or not 0 <= t < self._runtime.cfg["vocab_size"] for t in token_ids)):
                    raise ValueError("checkpoint tokens exceed their source or workload bound")
                if self._runtime.position:
                    self._runtime.forward_ids(token_ids)
                else:
                    self._runtime.prepare_ids(token_ids)
                return self._owner.logits.copy()
            except BaseException:
                self.close()
                raise

    def advance(self):
        """Execute the current pending greedy token, excluding it until commit."""
        with self._lock:
            return self.append([self.pending_token])

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._closed = True
                self._runtime.reset()

    def __enter__(self):
        self._available()
        return self

    def __exit__(self, *_):
        self.close()

    def __copy__(self):
        raise TypeError("live checkpoint sessions cannot be copied")

    def __deepcopy__(self, memo):
        return self.__copy__()

    def __reduce_ex__(self, protocol):
        raise TypeError("persist the trusted CheckpointHead, not a live session")
