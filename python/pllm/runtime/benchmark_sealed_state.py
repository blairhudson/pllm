"""Bounded streaming codec for the existing synthetic AEAD state fixture.

This shares the exact AES-GCM frames, transactional generations and trusted-head
contract with the one-shot control. No runtime snapshot authorization is implied.
"""
import hashlib
import os
import secrets

import numpy as np
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from .benchmark_state_targets import SealedStateFixture


class StreamingSealedStateFixture(SealedStateFixture):
    _CHUNK = 1 << 20

    def __init__(self, directory, key, context, layout):
        super().__init__(directory, key, context, layout)
        self._algorithm = algorithms.AES(key)
        self.aead = None

    def _write_record(self, out, value, revision, index):
        if not value.flags.c_contiguous:
            raise ValueError("streaming state requires contiguous records")
        nonce = secrets.token_bytes(12)
        encryptor = Cipher(self._algorithm, modes.GCM(nonce)).encryptor()
        encryptor.authenticate_additional_data(self._aad(revision, index))
        out.write(nonce)
        hashed = hashlib.sha256(nonce)
        plaintext = memoryview(value).cast("B")
        buffer = bytearray(min(self._CHUNK, len(plaintext)) + 15)
        scratch = memoryview(buffer)
        for start in range(0, len(plaintext), self._CHUNK):
            count = encryptor.update_into(plaintext[start:start + self._CHUNK], buffer)
            out.write(scratch[:count])
            hashed.update(scratch[:count])
        final = encryptor.finalize()
        out.write(final)
        out.write(encryptor.tag)
        hashed.update(final)
        hashed.update(encryptor.tag)
        return hashed.hexdigest()

    def _read_record(self, source, revision, index, expected):
        size = self.sizes[index]
        if os.fstat(source.fileno()).st_size != size + 28:
            raise ValueError("sealed state frame differs")
        nonce = source.read(12)
        source.seek(size + 12)
        tag = source.read(16)
        source.seek(12)
        decryptor = Cipher(self._algorithm, modes.GCM(nonce, tag)).decryptor()
        decryptor.authenticate_additional_data(self._aad(revision, index))
        plaintext = np.empty(size + 15, dtype=np.uint8)
        destination = memoryview(plaintext)
        buffer = bytearray(min(self._CHUNK, size))
        scratch = memoryview(buffer)
        hashed = hashlib.sha256(nonce)
        try:
            offset = 0
            while offset < size:
                count = min(len(buffer), size - offset)
                if source.readinto(scratch[:count]) != count:
                    raise ValueError("truncated sealed state frame")
                hashed.update(scratch[:count])
                written = decryptor.update_into(scratch[:count], destination[offset:offset + count + 15])
                if written != count:
                    raise ValueError("unexpected GCM plaintext extent")
                offset += count
            if decryptor.finalize():
                raise ValueError("unexpected GCM final plaintext")
            hashed.update(tag)
            if hashed.hexdigest() != expected:
                raise ValueError("sealed state frame differs")
            value = plaintext[:size].view(np.float32).reshape(self.layout[index]["shape"])
            value.flags.writeable = False
            return value
        except BaseException:
            plaintext.fill(0)
            raise
