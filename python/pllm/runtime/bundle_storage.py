"""Bounded, private temporary storage for authenticated public binary fields.

These files hold complete public objects, never private token-selected downloads.
Native-backed weight payloads share their immutable snapshot with paged kernels.
"""
from __future__ import annotations

import hashlib
import tempfile
import threading

CHUNK_BYTES = 1 << 20
MAX_TOKENIZER_BYTES = 64 << 20


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

    def __init__(self, chunks, *, size: int, digest: str, native: bool = False):
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
        self._file = tempfile.NamedTemporaryFile(prefix="pllm-public-object-", mode="w+b")
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

    def close(self):
        with self._lock:
            if self._snapshot is not None:
                self._snapshot.close()
            if self._file is not None:
                self._file.close()
