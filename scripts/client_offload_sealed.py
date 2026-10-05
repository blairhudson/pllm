"""In-process encrypted-blob reference, not a runtime protocol or remote client.

The object represents the trusted owner. An untrusted store receives only frame
bytes. Fresh keys, monotonically allocated nonces and context-bound AAD protect
each immutable slot. Restart/resumption and durable anti-rollback are absent.
"""
from __future__ import annotations

import hashlib
import struct

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


class SealedSlots:
    MAX_SLOTS = 4096
    MAX_BYTES = 8 << 20

    def __init__(self, context: bytes):
        if not context or len(context) > 4096:
            raise ValueError("invalid sealed-slot context")
        self._cipher = AESGCM(AESGCM.generate_key(bit_length=256))
        self.context = hashlib.sha256(b"pllm/client-offload-probe/v1\0" + context).digest()
        self.lengths: list[int] = []
        self.burned: set[int] = set()
        self.closed = False

    def seal(self, data: bytes) -> bytes:
        if self.closed or len(self.lengths) >= self.MAX_SLOTS or not 0 < len(data) <= self.MAX_BYTES:
            raise ValueError("sealed-slot issuance closed or over capacity")
        index = len(self.lengths)
        self.lengths.append(len(data))  # Allocate nonce before encryption; never retry it.
        header = struct.pack(">IQ", index, len(data))
        return header + self._cipher.encrypt(index.to_bytes(12, "big"), data, self.context + header)

    def open(self, index: int, frame: bytes, *, consume: bool = False) -> bytes:
        if self.closed or type(index) is not int or not 0 <= index < len(self.lengths):
            raise ValueError("invalid sealed-slot index or retired owner")
        if index in self.burned:
            raise ValueError("sealed slot already consumed")
        if consume:
            self.burned.add(index)  # Failure, malformed frame and cancellation burn material.
        header = struct.pack(">IQ", index, self.lengths[index])
        if len(frame) != 28 + self.lengths[index] or frame[:12] != header:
            raise ValueError("sealed-slot frame binding mismatch")
        return self._cipher.decrypt(index.to_bytes(12, "big"), frame[12:], self.context + header)

    def close(self):
        self.closed = True
        self._cipher = None
