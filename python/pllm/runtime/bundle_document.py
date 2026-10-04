"""Immutable binary segments for the existing canonical MessagePack bundle.

Python owns the wire document; large binary views retain their native immutable
weight owner. No tensor arithmetic, new encoding, or disk snapshot is introduced.
"""
from __future__ import annotations

import hashlib
from collections.abc import Iterator
from typing import Any

import msgpack

CHUNK_BYTES = 1 << 20
MAX_BUNDLE_BYTES = 8 << 30


def binary_view(value: bytes | memoryview) -> memoryview:
    view = memoryview(value)
    if not view.readonly or not view.c_contiguous:
        raise ValueError("bundle binary storage must be immutable and contiguous")
    return view.cast("B")


def _snapshot_binary(value: memoryview) -> memoryview:
    view = binary_view(value)
    owner: Any = view.obj
    for _ in range(64):
        if type(owner) is bytes:
            return view
        if isinstance(owner, memoryview):
            owner = owner.obj
        elif getattr(owner, "base", None) is not None:
            owner = owner.base
        else:
            break
    from ._native_support import extension
    native = extension()
    if native is not None and isinstance(owner, native.Matrix):
        return view
    # Read-only NumPy flags alone do not seal caller-owned storage. Freeze such
    # sources; provider stage views instead terminate at the immutable Rust owner.
    return memoryview(view.tobytes())


def packed_parts(value) -> Iterator[bytes | memoryview]:
    """Use MessagePack's headers/scalars; stream canonical bin bodies separately."""
    packer = msgpack.Packer(use_bin_type=True)
    budget = 250000

    def visit(node, depth=0):
        nonlocal budget
        budget -= 1
        if depth > 64 or budget < 0:
            raise ValueError("bundle document exceeds its node/depth bound")
        if type(node) is dict:
            yield packer.pack_map_header(len(node))
            for key, item in node.items():
                if type(key) is not str:
                    raise ValueError("bundle document keys must be strings")
                yield packer.pack(key)
                yield from visit(item, depth + 1)
        elif type(node) in (list, tuple):
            yield packer.pack_array_header(len(node))
            for item in node:
                yield from visit(item, depth + 1)
        elif type(node) in (bytes, memoryview):
            view = binary_view(node)
            n = len(view)
            if n >= 1 << 32:
                raise ValueError("bundle binary field exceeds MessagePack bin32")
            yield (b"\xc4" + n.to_bytes(1, "big") if n <= 255 else
                   b"\xc5" + n.to_bytes(2, "big") if n <= 65535 else
                   b"\xc6" + n.to_bytes(4, "big"))
            yield view
        elif type(node) in (str, int, float, bool, type(None)):
            yield packer.pack(node)
        else:
            raise ValueError("unsupported bundle document value")

    yield from visit(value)


def document_identity(value, *, maximum=MAX_BUNDLE_BYTES) -> tuple[int, str]:
    digest, size = hashlib.sha256(), 0
    for part in packed_parts(value):
        size += len(part)
        if size > maximum:
            raise ValueError("bundle document exceeds its byte bound")
        digest.update(part)
    return size, digest.hexdigest()


class BundleDocument:
    """Provider-owned metadata snapshot and immutable binary views."""
    __slots__ = ("_value", "_parts", "_size", "_digest")

    def __init__(self, value):
        # Copy containers, never their immutable binary owners. The bounded
        # validation pass also rejects unsupported/mutable leaf representations.
        budget = 250000
        def snapshot(node: Any, depth=0) -> Any:
            nonlocal budget
            budget -= 1
            if depth > 64 or budget < 0:
                raise ValueError("bundle document exceeds its node/depth bound")
            if type(node) is dict:
                return {key: snapshot(item, depth + 1) for key, item in node.items()}
            if type(node) in (list, tuple):
                return [snapshot(item, depth + 1) for item in node]
            if type(node) is memoryview:
                return _snapshot_binary(node)
            return node
        if type(value) is not dict or type(value.get("v")) is not int:
            raise ValueError("bundle document requires a schema version")
        self._value = snapshot(value)
        self._parts = tuple(packed_parts(self._value))
        self._size = sum(len(part) for part in self._parts)
        if self._size > MAX_BUNDLE_BYTES:
            raise ValueError("bundle document exceeds its byte bound")
        digest = hashlib.sha256()
        for part in self._parts:
            digest.update(part)
        self._digest = digest.hexdigest()

    def __len__(self):
        return self._size

    @property
    def descriptor(self):
        return {"schema": int(self._value["v"]), "sha256": self._digest,
                "size": self._size, "etag": f'"{self._digest}"'}

    def to_bytes(self) -> bytes:
        """Compatibility API for direct callers needing a contiguous bundle."""
        return b"".join(self._parts)

    def chunks(self) -> Iterator[bytes]:
        """Preserve global 1-MiB frame boundaries, including across binary fields."""
        pending = bytearray()
        for part in self._parts:
            view = memoryview(part)
            position = 0
            while position < len(view):
                take = min(CHUNK_BYTES - len(pending), len(view) - position)
                pending.extend(view[position:position + take])
                position += take
                if len(pending) == CHUNK_BYTES:
                    yield bytes(pending)
                    pending.clear()
        if pending:
            yield bytes(pending)
