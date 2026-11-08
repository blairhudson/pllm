"""Bounded, private temporary storage for authenticated public binary fields.

These files hold complete public objects, never private token-selected downloads.
Native-backed weight payloads share their immutable snapshot with paged kernels.
"""
from __future__ import annotations

import hashlib
import tempfile
import threading
import math
import sys
from pathlib import Path

import numpy as np

CHUNK_BYTES = 1 << 20
MAX_TOKENIZER_BYTES = 64 << 20


def f32_array_nbytes(shape):
    """Check public geometry before NumPy construction; not a memory budget."""
    if (type(shape) not in (list, tuple) or not shape
            or any(type(width) is not int or width <= 0 for width in shape)):
        raise ValueError("client tensor shape requires positive integer dimensions")
    elements = math.prod(shape)
    if elements > sys.maxsize // 4 or len(shape) > getattr(np, "MAXDIMS", 64):
        raise ValueError("client tensor geometry exceeds NumPy addressable bounds")
    return elements * 4


def immutable_f32_array(data, shape):
    """Borrow exact immutable MessagePack bytes, never a mutable snapshot copy."""
    size = f32_array_nbytes(shape)
    if type(data) is not bytes or len(data) != size:
        raise ValueError("client tensor requires exact immutable bytes and byte length")
    return np.frombuffer(data, dtype="<f4").reshape(tuple(shape))


def immutable_f32_backing(array):
    """Reject readonly views of mutable owners, including buffer wrappers."""
    if (type(array) is not np.ndarray or array.dtype != np.dtype("<f4")
            or not array.flags.c_contiguous or array.flags.writeable):
        raise ValueError("client tensor requires contiguous immutable float32 storage")
    f32_array_nbytes(array.shape)
    owner = array
    while type(owner) is np.ndarray:
        if owner.flags.writeable:
            raise ValueError("client tensor has a mutable ndarray owner")
        owner = owner.base
    if type(owner) is not bytes:
        raise ValueError("client tensor backing must be exact immutable bytes")
    return owner


def immutable_f32_retained_bytes(arrays):
    """Price complete retained bytes owners once, even for alias/slice views."""
    owners = {id(root): root for root in map(immutable_f32_backing, arrays)}
    return sum(len(root) for root in owners.values())


def f32_chunks(array):
    """Zero-copy contiguous chunks; finite masks need at most CHUNK_BYTES/4."""
    flat = array.reshape(-1)
    for start in range(0, flat.size, CHUNK_BYTES // 4):
        yield flat[start:start + CHUNK_BYTES // 4]


def binary_digest(value):
    """Hash actual binary contents, including owned file snapshots."""
    digest = hashlib.sha256()
    if type(value) is FilePayload:
        for chunk in value.chunks():
            digest.update(chunk)
    else:
        view = memoryview(value)
        digest.update(view if view.c_contiguous else bytes(view))
    return digest.hexdigest()


class FilePayload:
    """Verified owned file; no complete in-memory byte representation."""

    __slots__ = ("_file", "_snapshot", "_size", "_digest", "_lock")

    def __init__(self, chunks, *, size: int, digest: str, native: bool = False,
                 temp_directory: str | Path | None = None):
        if (type(size) is not int or not 0 < size <= 4 << 30
                or type(digest) is not str or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)
                or type(native) is not bool):
            raise ValueError("file payload has invalid size or digest")
        self._file = self._snapshot = None
        self._lock = threading.Lock()
        self._size, self._digest = size, digest
        if native:
            from ._native_support import extension

            backend = extension()
            if backend is None:
                raise RuntimeError("native public snapshots require the Rust extension")
            writer = backend.PublicFileWriter(size, digest)
            try:
                for chunk in chunks:
                    if not 0 < len(chunk) <= CHUNK_BYTES:
                        raise ValueError("file payload chunk exceeds its bounds")
                    writer.write(bytes(chunk))
                self._snapshot = writer.finish()
            finally:
                writer.close()
            return
        self._file = tempfile.NamedTemporaryFile(prefix="pllm-public-object-", mode="w+b",
                                                 dir=temp_directory)
        actual, count = hashlib.sha256(), 0
        try:
            for chunk in chunks:
                if not 0 < len(chunk) <= CHUNK_BYTES or count + len(chunk) > size:
                    raise ValueError("file payload chunk exceeds its bounds")
                self._file.write(chunk)
                actual.update(chunk)
                count += len(chunk)
            if count != size or actual.hexdigest() != digest:
                raise ValueError("file payload digest or size mismatch")
            self._file.flush()
        except BaseException:
            self._file.close()
            raise

    def __len__(self):
        return self._size

    def __bytes__(self):
        # Explicit compatibility boundary for ordinary tokenizer libraries.
        # Indexed tokenization and binding only need a streaming digest.
        return b"".join(self.chunks())

    @property
    def digest(self):
        return self._digest

    @property
    def path(self):
        if self._file is None:
            raise ValueError("native public snapshot has no mutable filesystem path")
        return self._file.name

    def chunks(self):
        with self._lock:
            if self._snapshot is not None:
                for offset in range(0, self._size, CHUNK_BYTES):
                    yield self._snapshot.read(offset, min(CHUNK_BYTES, self._size - offset))
                return
            self._file.seek(0)
            count = 0
            while count < self._size:
                block = self._file.read(min(CHUNK_BYTES, self._size - count))
                if not block:
                    raise ValueError("private file payload was truncated")
                count += len(block)
                yield block
            if self._file.read(1):
                raise ValueError("private file payload has trailing bytes")

    def read_at(self, offset: int, size: int) -> bytes:
        """Read the owned descriptor/snapshot, never reopen a mutable pathname."""
        if (type(offset) is not int or type(size) is not int
                or not 0 <= offset <= self._size or not 0 <= size <= self._size - offset):
            raise ValueError("private file read exceeds snapshot bounds")
        with self._lock:
            if self._snapshot is not None:
                data = self._snapshot.read(offset, size)
            else:
                self._file.seek(offset)
                data = self._file.read(size)
            if len(data) != size:
                raise ValueError("private file payload was truncated")
            return data

    def close(self):
        with self._lock:
            if self._snapshot is not None:
                self._snapshot.close()
            if self._file is not None:
                self._file.close()
