"""Bounded, private temporary storage for authenticated public binary fields.

These files hold complete public objects, never private token-selected downloads.
The native paged kernel takes its own immutable snapshot before execution.
"""
from __future__ import annotations

import hashlib
import tempfile
import threading

CHUNK_BYTES = 1 << 20


class FilePayload:
    """Verified owned file; no complete in-memory byte representation."""

    __slots__ = ("_file", "_size", "_digest", "_lock")

    def __init__(self, chunks, *, size: int, digest: str):
        if (type(size) is not int or not 0 < size <= 4 << 30
                or type(digest) is not str or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)):
            raise ValueError("file payload has invalid size or digest")
        self._file = tempfile.NamedTemporaryFile(prefix="pllm-public-object-", mode="w+b")
        self._lock = threading.Lock()
        self._size, self._digest = size, digest
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

    @property
    def digest(self):
        return self._digest

    @property
    def path(self):
        return self._file.name

    def chunks(self):
        with self._lock:
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
            self._file.close()
