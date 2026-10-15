"""Exact output-only head residues from an independently admitted descriptor.

No peer-provided layout, weight scan, or certificate is accepted. Signed-i8
weights (including -128) and symmetric A4/A8 quantization prove
``abs(dot) <= hidden * 128 * 127``. Both activation profiles deliberately use
the same robust public width. Inputs remain full ring32 shares.
Checksums detect accidental corruption, not malicious-worker forgery; existing
authenticated transport and trusted source/compiled admission remain required.
"""

from __future__ import annotations

import hashlib
import struct

import numpy as np

from .private_boundary_protocol import (
    BoundaryAdmission, BoundaryDescriptor, BoundaryError, MAX_BODY_BYTES,
    MAX_HEADS, MAX_LOOKUP_ROWS, ROLES, canonical, context_bytes, integer,
)

MAGIC = b"PLLMBH01"  # Distinct from probe and all existing native frames.
HEADER = struct.Struct("<8sBB32s32s")
DOMAIN = b"pllm/private-boundary/uniform-head-residues/v1\0"


class HeadOutputCodec:
    """One bounded layout owner; construct only from trusted compiled descriptor.

    ``encode(raw_u32, *, admission, role, paired_session, epoch, sequence)``
    returns one response subframe. ``decode(a, b, *, admission, paired_session,
    epoch, sequence)`` expects A then B and returns shape (1,V), dtype int64.
    Outer frames must still pass the existing transport frame validators.

    Memory pricing (V=vocabulary, P=ceil(V*width/8), F=HEADER.size+P):
    residue layout retains V bytes, with no weight/scales/certificate owner.
    Worker codec peak includes 4V input bytes plus two P-sized native/Python
    packing owners; its caller additionally holds the 4V wrap32 ndarray.
    Price at least 9V+2P+HEADER.size before outer msgpack/HTTP framing.
    Decode with two caller-owned subframes and two sliced payloads can hold
    2F+2P+17V during native-to-Python i64 copying (two 8V outputs and V layout).
    Returned ndarray views the immutable 8V bytes owner, without another copy.
    Outer envelopes, dequantization temporaries, allocator and HTTP/TLS buffers
    are additional; these formulas are owner accounting, not measured RSS.
    """

    def __init__(self, descriptor: BoundaryDescriptor):
        if type(descriptor) is not BoundaryDescriptor:
            raise BoundaryError("head codec requires trusted boundary descriptor")
        self.descriptor = descriptor
        self.bound = descriptor.hidden * 128 * 127
        self.width = (2 * self.bound).bit_length()
        integer(self.width, 1, 32, "head residue width")
        self.widths = bytes([self.width]) * descriptor.vocabulary if descriptor.head_output_encoding == "uniform_residues" else b""
        self.payload_bytes = (descriptor.vocabulary * self.width + 7) // 8
        if HEADER.size + self.payload_bytes > MAX_BODY_BYTES:
            raise BoundaryError("head residue frame exceeds limit")

    def _binding(self, *, admission, role, paired_session, epoch, sequence):
        if type(admission) is not BoundaryAdmission or role not in ROLES:
            raise BoundaryError("invalid head output ownership")
        integer(sequence, 0, MAX_LOOKUP_ROWS + MAX_HEADS - 1, "operation sequence")
        context = context_bytes(self.descriptor, admission, paired_session, epoch)
        return hashlib.sha256(DOMAIN + context + canonical({
            "descriptor": self.descriptor.digest, "role": role,
            "sequence": sequence, "encoding": self.descriptor.head_output_encoding,
            "width": self.width,
        })).digest()

    def encode(self, raw_u32: bytes, *, admission: BoundaryAdmission, role: str,
               paired_session: str, epoch: int, sequence: int) -> bytes:
        binding = self._binding(admission=admission, role=role, paired_session=paired_session,
                                epoch=epoch, sequence=sequence)
        if type(raw_u32) is not bytes or len(raw_u32) != self.descriptor.vocabulary * 4:
            raise BoundaryError("head output geometry differs")
        if self.descriptor.head_output_encoding == "u32":
            return raw_u32
        from pllm import _native

        payload = _native.offset_pack_rows(raw_u32, self.widths, 1)
        checksum = hashlib.sha256(DOMAIN + binding + payload).digest()
        return HEADER.pack(MAGIC, ROLES.index(role), self.width, binding, checksum) + payload

    def _payload(self, frame, *, admission, role, paired_session, epoch, sequence):
        binding = self._binding(admission=admission, role=role, paired_session=paired_session,
                                epoch=epoch, sequence=sequence)
        if type(frame) is not bytes:
            raise BoundaryError("invalid head output frame")
        if self.descriptor.head_output_encoding == "u32":
            if len(frame) != self.descriptor.vocabulary * 4:
                raise BoundaryError("head output geometry differs")
            return frame
        if len(frame) != HEADER.size + self.payload_bytes:
            raise BoundaryError("head residue length differs")
        magic, party, width, actual_binding, checksum = HEADER.unpack_from(frame)
        if (magic, party, width, actual_binding) != (MAGIC, ROLES.index(role), self.width, binding):
            raise BoundaryError("head residue context/profile differs")
        payload = frame[HEADER.size:]
        used = self.descriptor.vocabulary * self.width % 8
        if used and payload[-1] >> used:
            raise BoundaryError("head residue padding differs")
        if hashlib.sha256(DOMAIN + binding + payload).digest() != checksum:
            raise BoundaryError("head residue checksum differs")
        return payload

    def decode(self, a: bytes, b: bytes, *, admission: BoundaryAdmission,
               paired_session: str, epoch: int, sequence: int) -> np.ndarray:
        args = dict(admission=admission, paired_session=paired_session, epoch=epoch, sequence=sequence)
        a = self._payload(a, role=ROLES[0], **args)
        b = self._payload(b, role=ROLES[1], **args)
        if self.descriptor.head_output_encoding == "u32":
            # Preserve exact historical ring32 centering; no output bound
            # enforcement added to the legacy path.
            combined = (np.frombuffer(a, "<u4").astype(np.int64)
                        + np.frombuffer(b, "<u4")) % (1 << 32)
            values = np.where(combined >= 1 << 31, combined - (1 << 32), combined)
        else:
            from pllm import _native

            raw = _native.offset_reconstruct_rows(a, b, self.widths, 1)
            values = np.frombuffer(raw, "<i8")
            bound = min(self.bound, self.descriptor.signed_output_bound)
            if np.any(values < -bound) or np.any(values > bound):
                raise BoundaryError("head output exceeds admitted signed bound")
        return values.reshape(1, self.descriptor.vocabulary)
