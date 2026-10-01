"""Bounded local cost references, not a shared decoder or reviewed protocol.

One-use sum-of-squares correlations and a two-party additive DPF lookup share
the existing point-tree construction. Dealer/client sees both issued keys only
in this reference. Each evaluator receives one material object. There is no
transport, compiler admission, malicious security or float-scale conversion.
"""

from __future__ import annotations

import hashlib
import secrets
import struct
from dataclasses import dataclass

import numpy as np

from .shared_dpf import PointKeyDealer, _Correction, _ring_leaf, _stretch, _xor


class InteractionReferenceError(ValueError):
    pass


def _words(value: np.ndarray, width: int) -> np.ndarray:
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != np.uint32
        or value.shape != (width,)
        or not value.flags.c_contiguous
    ):
        raise InteractionReferenceError("reference share must be a bounded uint32 vector")
    return value


def fresh_shares(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(value, np.ndarray) or value.dtype != np.uint32 or value.ndim != 1:
        raise InteractionReferenceError("reference value must be a uint32 vector")
    if not 0 < value.size <= 4096:
        raise InteractionReferenceError("reference vector exceeds capacity")
    first = np.frombuffer(secrets.token_bytes(value.size * 4), dtype="<u4").copy()
    return first, np.subtract(value, first, dtype=np.uint32)


@dataclass(frozen=True)
class SquareOpening:
    session: str
    operation: str
    party: int
    body: bytes


class SumSquaresMaterial:
    """One party's r share and scalar <r,r> share. Burn before validation."""

    def __init__(
        self,
        session: str,
        operation: str,
        party: int,
        mask: np.ndarray,
        square: int,
    ) -> None:
        self.session, self.operation, self.party = session, operation, party
        self._mask = mask.copy()
        self._square = square
        self._used = False
        self._pending: np.ndarray | None = None

    @property
    def body_bytes(self) -> int:
        return self._mask.nbytes + 4

    def cancel(self) -> None:
        self._used = True
        self._pending = None

    def open(self, value: np.ndarray) -> SquareOpening:
        if self._used:
            raise InteractionReferenceError("sum-squares material already consumed")
        self._used = True
        value = _words(value, self._mask.size)
        pending = np.subtract(value, self._mask, dtype=np.uint32)
        self._pending = pending
        return SquareOpening(
            self.session, self.operation, self.party, pending.astype("<u4").tobytes()
        )

    def finish(self, peer: SquareOpening) -> int:
        pending, self._pending = self._pending, None
        if pending is None:
            raise InteractionReferenceError("sum-squares opening already consumed or cancelled")
        if (
            not isinstance(peer, SquareOpening)
            or peer.session != self.session
            or peer.operation != self.operation
            or peer.party != 1 - self.party
            or type(peer.body) is not bytes
            or len(peer.body) != self._mask.nbytes
        ):
            raise InteractionReferenceError("peer sum-squares opening binding or size mismatch")
        public = np.add(pending, np.frombuffer(peer.body, dtype="<u4"), dtype=np.uint32)
        # uint64 sums/products intentionally reduce modulo 2^32 at return.
        cross = int(np.dot(public.astype(np.uint64), self._mask.astype(np.uint64)))
        public_square = int(np.dot(public.astype(np.uint64), public.astype(np.uint64)))
        return (self._square + 2 * cross + (public_square if self.party == 0 else 0)) % (1 << 32)


def issue_sum_squares(
    session: str,
    operation: str,
    width: int,
) -> tuple[SumSquaresMaterial, SumSquaresMaterial]:
    if (
        type(session) is not str
        or not session
        or type(operation) is not str
        or not operation
        or type(width) is not int
        or not 1 <= width <= 4096
    ):
        raise InteractionReferenceError("sum-squares identity or capacity invalid")
    mask = np.frombuffer(secrets.token_bytes(width * 4), dtype="<u4").copy()
    masks = fresh_shares(mask)
    square = int(np.dot(mask.astype(np.uint64), mask.astype(np.uint64))) % (1 << 32)
    first = secrets.randbits(32)
    return (
        SumSquaresMaterial(session, operation, 0, masks[0], first),
        SumSquaresMaterial(session, operation, 1, masks[1], (square - first) % (1 << 32)),
    )


class BroadcastMaterial:
    """One vector/scalar triple, avoiding n repeated copies of scalar b."""

    def __init__(
        self,
        session: str,
        operation: str,
        party: int,
        a: np.ndarray,
        b: np.ndarray,
        product: np.ndarray,
    ) -> None:
        self.session, self.operation, self.party = session, operation, party
        self._a, self._b, self._product = a.copy(), int(b[0]), product.copy()
        self._used = False
        self._pending: np.ndarray | None = None

    @property
    def body_bytes(self) -> int:
        return self._a.nbytes + self._product.nbytes + 4

    def cancel(self) -> None:
        self._used = True
        self._pending = None

    def open(self, vector: np.ndarray, scalar: np.ndarray) -> SquareOpening:
        if self._used:
            raise InteractionReferenceError("broadcast material already consumed")
        self._used = True
        vector, scalar = _words(vector, self._a.size), _words(scalar, 1)
        pending = np.concatenate(
            (
                np.subtract(vector, self._a, dtype=np.uint32),
                np.asarray([(int(scalar[0]) - self._b) % (1 << 32)], dtype=np.uint32),
            )
        )
        self._pending = pending
        return SquareOpening(
            self.session, self.operation, self.party, pending.astype("<u4").tobytes()
        )

    def finish(self, peer: SquareOpening) -> np.ndarray:
        pending, self._pending = self._pending, None
        if pending is None:
            raise InteractionReferenceError("broadcast opening already consumed or cancelled")
        if (
            not isinstance(peer, SquareOpening)
            or peer.session != self.session
            or peer.operation != self.operation
            or peer.party != 1 - self.party
            or type(peer.body) is not bytes
            or len(peer.body) != pending.nbytes
        ):
            raise InteractionReferenceError("peer broadcast opening binding or size mismatch")
        public = np.add(pending, np.frombuffer(peer.body, dtype="<u4"), dtype=np.uint32)
        vector, scalar = public[:-1], public[-1]
        output = self._product + np.multiply(vector, np.uint32(self._b), dtype=np.uint32)
        output += np.multiply(self._a, scalar, dtype=np.uint32)
        if self.party == 0:
            output += np.multiply(vector, scalar, dtype=np.uint32)
        return output


def issue_broadcast(
    session: str,
    operation: str,
    width: int,
) -> tuple[BroadcastMaterial, BroadcastMaterial]:
    if (
        type(session) is not str
        or not session
        or type(operation) is not str
        or not operation
        or type(width) is not int
        or not 1 <= width <= 4096
    ):
        raise InteractionReferenceError("broadcast identity or capacity invalid")
    a = np.frombuffer(secrets.token_bytes(width * 4), dtype="<u4").copy()
    b = np.asarray([secrets.randbits(32)], dtype=np.uint32)
    product = np.multiply(a, b[0], dtype=np.uint32)
    aa, bb, cc = fresh_shares(a), fresh_shares(b), fresh_shares(product)
    return (
        BroadcastMaterial(session, operation, 0, aa[0], bb[0], cc[0]),
        BroadcastMaterial(session, operation, 1, aa[1], bb[1], cc[1]),
    )


class LookupPointKey:
    """One-use 32-bit additive selector, bounded to an 18-bit public table.

    Body serialization is measured; importing these bytes is deliberately not a
    runtime API. Table binding is checked before expanding one party's key.
    """

    def __init__(
        self,
        session: str,
        query: str,
        party: int,
        bits: int,
        vocabulary: int,
        fingerprint: bytes,
        root: bytes,
        words: tuple[_Correction, ...],
        final: int,
    ) -> None:
        self.session, self.query, self.party = session, query, party
        self.bits, self.vocabulary, self.fingerprint = bits, vocabulary, fingerprint
        self._root, self._words, self._final = root, words, final
        self._used = False

    def serialize(self) -> bytes:
        if self._used:
            raise InteractionReferenceError("lookup key already consumed")
        return (
            b"PLLMPLK1"
            + struct.pack("<BBI", self.party, self.bits, self.vocabulary)
            + hashlib.sha256(self.session.encode()).digest()
            + hashlib.sha256(self.query.encode()).digest()
            + self.fingerprint
            + self._root
            + b"".join(w.seed + bytes((w.left | (w.right << 1),)) for w in self._words)
            + struct.pack("<I", self._final)
        )

    def cancel(self) -> None:
        self._used = True

    def _expand(self) -> np.ndarray:
        output = np.empty(self.vocabulary, dtype=np.uint32)

        def visit(seed: bytes, control: int, level: int, prefix: int) -> None:
            if prefix << (self.bits - level) >= self.vocabulary:
                return
            if level == self.bits:
                base = _ring_leaf(seed, 1 << 32)[0]
                value = (base + control * self._final) % (1 << 32)
                output[prefix] = value if self.party == 0 else (-value) % (1 << 32)
                return
            correction = self._words[level]
            for direction, (child_seed, child_control) in enumerate(_stretch(seed)):
                if control:
                    child_seed = _xor(child_seed, correction.seed)
                    child_control ^= correction.right if direction else correction.left
                visit(child_seed, child_control, level + 1, prefix * 2 + direction)

        visit(self._root, self.party, 0, 0)
        return output

    def evaluate(
        self,
        table: np.ndarray,
        *,
        session: str,
        query: str,
        fingerprint: bytes,
    ) -> np.ndarray:
        if self._used:
            raise InteractionReferenceError("lookup key already consumed")
        self._used = True
        if (
            session != self.session
            or query != self.query
            or fingerprint != self.fingerprint
            or not isinstance(table, np.ndarray)
            or table.dtype != np.int8
            or table.ndim != 2
            or table.shape[0] != self.vocabulary
            or not 1 <= table.shape[1] <= 4096
            or table.nbytes > 256 << 20
            or not table.flags.c_contiguous
        ):
            raise InteractionReferenceError("lookup binding, table or capacity mismatch")
        if hashlib.sha256(memoryview(table)).digest() != fingerprint:
            raise InteractionReferenceError("lookup table fingerprint mismatch")
        selector = self._expand()
        output = np.zeros(table.shape[1], dtype=np.uint32)
        # Bounded 4 MiB conversion window, never a vocabulary*width u32 snapshot.
        for start in range(0, self.vocabulary, 1024):
            chunk = table[start : start + 1024].astype(np.uint32)
            output += np.matmul(selector[start : start + len(chunk)], chunk, dtype=np.uint32)
        return output


def issue_lookup(
    session: str,
    query: str,
    token: int,
    vocabulary: int,
    fingerprint: bytes,
) -> tuple[LookupPointKey, LookupPointKey]:
    if (
        type(session) is not str
        or not session
        or type(query) is not str
        or not query
        or type(vocabulary) is not int
        or not 2 <= vocabulary <= 1 << 18
        or type(token) is not int
        or not 0 <= token < vocabulary
        or type(fingerprint) is not bytes
        or len(fingerprint) != 32
    ):
        raise InteractionReferenceError("lookup identity, token or vocabulary invalid")
    bits = (vocabulary - 1).bit_length()
    roots, seeds, controls, words = PointKeyDealer(session)._issue_tree(
        query, token, bits, maximum_bits=18
    )
    first, second = (_ring_leaf(seed, 1 << 32)[0] for seed in seeds)
    sign = 1 if controls[0] else -1
    final = (sign * (1 - first + second)) % (1 << 32)
    keys = tuple(
        LookupPointKey(
            session, query, party, bits, vocabulary, fingerprint, roots[party], words, final
        )
        for party in (0, 1)
    )
    return keys[0], keys[1]
