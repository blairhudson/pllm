"""Immutable binary segments for the existing canonical MessagePack bundle.

Python owns the wire document; large binary views retain their native immutable
weight owner. No tensor arithmetic, new encoding, or disk snapshot is introduced.
"""
from __future__ import annotations

import hashlib
import math
from collections.abc import Iterator
from typing import Any

import msgpack

from .bundle_storage import FilePayload

CHUNK_BYTES = 1 << 20
MAX_BUNDLE_BYTES = 8 << 30
MAX_RAW_BYTES = 4 << 30
MAX_MANIFEST_BYTES = 16 << 20
MAX_OBJECTS = 16384


def unpack_document(payload, *, maximum=MAX_BUNDLE_BYTES, artifact_limits=False):
    """Ordinary MessagePack decoder, with transport bounds before allocation."""
    if type(payload) not in (bytes, bytearray) or not 0 < len(payload) <= maximum:
        raise ValueError("bundle document exceeds its byte bound")
    n = len(payload)

    def pairs(rows):
        result = {}
        for key, value in rows:
            if type(key) is not str or key in result:
                raise ValueError("bundle maps require unique string keys")
            result[key] = value
        return result

    return msgpack.unpackb(payload, raw=False, object_pairs_hook=pairs,
        max_bin_len=maximum if artifact_limits else n,
        max_str_len=MAX_MANIFEST_BYTES if artifact_limits else n,
        max_array_len=65536 if artifact_limits else n,
        max_map_len=65536 if artifact_limits else n // 2,
        max_ext_len=32 if artifact_limits else n)


def _string_bytes(value, *, json_mode=False):
    # Count UTF-8 and JSON escapes without allocating an encoded string.
    size = 2 if json_mode else 0
    for char in value:
        code = ord(char)
        if 0xD800 <= code <= 0xDFFF:
            raise ValueError("bundle metadata contains a lone surrogate")
        if json_mode and (char in '\"\\' or char in '\b\f\n\r\t'):
            size += 2
        elif json_mode and code < 32:
            size += 6
        else:
            size += 1 if code < 128 else 2 if code < 2048 else 3 if code < 65536 else 4
        if size > MAX_MANIFEST_BYTES:
            raise ValueError("bundle metadata exceeds its byte bound")
    return size


def bounded_metadata(value, *, json_mode=False):
    """Iterative exact-type admission; never reads numerical binary bodies.

    JSON accounting conservatively dominates compact ensure_ascii=False output.
    MessagePack accounting includes binary headers, but excludes their bodies
    from the metadata budget. Cycles exhaust depth/nodes, without a seen registry.
    """
    metadata = raw = nodes = 0
    stack = [iter((value,))]
    while stack:
        try:
            node = next(stack[-1])
        except StopIteration:
            stack.pop()
            continue
        nodes += 1
        if nodes > 250000 or len(stack) > 65:
            raise ValueError("bundle document exceeds its node/depth bound")
        kind = type(node)
        body = 0
        if kind is dict:
            n = len(node)
            size = (2 + max(0, n - 1) + n) if json_mode else (1 if n <= 15 else 3 if n <= 65535 else 5)
            def entries(mapping):
                for key, item in mapping.items():
                    if type(key) is not str:
                        raise ValueError("bundle document keys must be strings")
                    yield key
                    yield item
            stack.append(iter(entries(node)))
        elif kind in ((list,) if json_mode else (list, tuple)):
            n = len(node)
            size = (2 + max(0, n - 1)) if json_mode else (1 if n <= 15 else 3 if n <= 65535 else 5)
            stack.append(iter(node))
        elif kind is str:
            size = _string_bytes(node, json_mode=json_mode)
            if not json_mode:
                size += 1 if size <= 31 else 2 if size <= 255 else 3 if size <= 65535 else 5
        elif kind is int:
            if not -(1 << 63) <= node < 1 << 64:
                raise ValueError("bundle metadata integer exceeds its bound")
            if json_mode:
                size = 21
            elif node >= 0:
                size = (1 if node <= 127 else 2 if node <= 255 else
                        3 if node <= 65535 else 5 if node <= 0xFFFFFFFF else 9)
            else:
                size = (1 if node >= -32 else 2 if node >= -128 else
                        3 if node >= -32768 else 5 if node >= -(1 << 31) else 9)
        elif kind is float:
            if not math.isfinite(node):
                raise ValueError("bundle metadata floats must be finite")
            size = 32 if json_mode else 9
        elif kind in (bool, type(None)):
            size = 5 if json_mode else 1
        elif not json_mode and kind in (bytes, memoryview, FilePayload):
            if kind is memoryview:
                if not node.readonly or not node.c_contiguous:
                    raise ValueError("bundle binary storage must be immutable and contiguous")
                body = node.nbytes
            elif kind is FilePayload:
                body = node._size
                if type(body) is not int or body <= 0:
                    raise ValueError("invalid stored bundle binary byte count")
            else:
                body = len(node)
            if body >= 1 << 32:
                raise ValueError("bundle binary field exceeds MessagePack bin32")
            size = 2 if body <= 255 else 3 if body <= 65535 else 5
        else:
            raise ValueError("unsupported bundle metadata value")
        metadata += size
        raw += size + body
        if metadata > MAX_MANIFEST_BYTES or raw > MAX_RAW_BYTES:
            raise ValueError("legacy bundle exceeds metadata/raw byte bound")
    return metadata, raw


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
        elif type(node) in (bytes, memoryview, FilePayload):
            n = len(node)
            if n >= 1 << 32:
                raise ValueError("bundle binary field exceeds MessagePack bin32")
            yield (b"\xc4" + n.to_bytes(1, "big") if n <= 255 else
                   b"\xc5" + n.to_bytes(2, "big") if n <= 65535 else
                   b"\xc6" + n.to_bytes(4, "big"))
            if type(node) is FilePayload:
                yield from node.chunks()
            else:
                yield binary_view(node)
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
