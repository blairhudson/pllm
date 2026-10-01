"""Bounded two-party point sharing for studying FSS comparison costs.

Reference only. Keys are issued together inside a trusted dealer, handed to
different parties, and consumed once. No compiler admission or transport.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass


class DpfReferenceError(ValueError):
    pass


_DOMAIN = b"pllm.two_party_point_fss.v1"
_SEED_BYTES = 16
_MAX_BITS = 10


def _xor(left: bytes, right: bytes) -> bytes:
    return bytes(a ^ b for a, b in zip(left, right, strict=True))


def _stretch(seed: bytes) -> tuple[tuple[bytes, int], tuple[bytes, int]]:
    children = []
    for direction in (0, 1):
        digest = hmac.digest(seed, _DOMAIN + bytes((direction,)), hashlib.sha256)
        children.append((digest[:_SEED_BYTES], digest[_SEED_BYTES] & 1))
    return children[0], children[1]


def _leaf(seed: bytes) -> int:
    return hmac.digest(seed, _DOMAIN + b"\x02", hashlib.sha256)[0] & 1


def _ring_leaf(seed: bytes, modulus: int) -> tuple[int, int]:
    output = hmac.digest(seed, _DOMAIN + b"\x03", hashlib.sha256)
    return (
        int.from_bytes(output[:8], "little") % modulus,
        int.from_bytes(output[8:16], "little") % modulus,
    )


@dataclass(frozen=True, slots=True)
class _Correction:
    seed: bytes
    left: int
    right: int


class PointKeyShare:
    """One role's opaque point key; cannot be evaluated again after any call."""

    def __init__(
        self,
        session_id: str,
        gate_id: str,
        party: int,
        bits: int,
        seed: bytes,
        corrections: tuple[_Correction, ...],
        final: int,
    ) -> None:
        if not session_id or not gate_id or party not in (0, 1) or not 1 <= bits <= _MAX_BITS:
            raise DpfReferenceError("point-key identity or bounded domain is invalid")
        if len(seed) != _SEED_BYTES or len(corrections) != bits or final not in (0, 1):
            raise DpfReferenceError("point-key structure is invalid")
        if any(
            len(word.seed) != _SEED_BYTES or word.left not in (0, 1)
            or word.right not in (0, 1)
            for word in corrections
        ):
            raise DpfReferenceError("point-key correction word is invalid")
        self.session_id = session_id
        self.gate_id = gate_id
        self.party = party
        self.bits = bits
        self._seed = seed
        self._corrections = corrections
        self._final = final
        self._used = False
        self._prefix: bytearray | None = None

    @property
    def key_bytes(self) -> int:
        """Cryptographic body only; does not count metadata or wire framing."""
        return _SEED_BYTES + self.bits * (_SEED_BYTES + 1) + 1

    def cancel(self) -> None:
        self._used = True
        self._prefix = None

    @property
    def expanded_bytes(self) -> int:
        return len(self._prefix) if self._prefix is not None else 0

    def prepare(self) -> None:
        """Expand party-local function share offline; fresh key still consumed once."""
        if self._used or self._prefix is not None:
            raise DpfReferenceError("point-key share was already prepared or consumed")
        size = 1 << self.bits
        prefix = bytearray((size + 7) // 8)
        running = 0
        for index in range(size - 1, -1, -1):
            running ^= self._eval_leaf(index)
            prefix[index // 8] |= running << (index % 8)
        self._prefix = prefix

    def _eval_leaf(self, index: int) -> int:
        seed = self._seed
        control = self.party
        for level, correction in enumerate(self._corrections):
            direction = (index >> (self.bits - level - 1)) & 1
            child_seed, child_control = _stretch(seed)[direction]
            if control:
                child_seed = _xor(child_seed, correction.seed)
                child_control ^= correction.right if direction else correction.left
            seed, control = child_seed, child_control
        return _leaf(seed) ^ (control & self._final)

    def less_than_share(self, public_value: int) -> int:
        """XOR share of 1{public_value < hidden point}; O(2**bits) PRG leaves."""
        if self._used:
            raise DpfReferenceError("point-key share was already consumed")
        self._used = True
        if type(public_value) is not int or not 0 <= public_value < 1 << self.bits:
            self._prefix = None
            raise DpfReferenceError("public comparison input is out of range")
        if self._prefix is not None:
            index = public_value + 1
            result = (self._prefix[index // 8] >> (index % 8)) & 1 if index < 1 << self.bits else 0
            self._prefix = None
            return result
        output = 0
        for index in range(public_value + 1, 1 << self.bits):
            output ^= self._eval_leaf(index)
        return output


class RingPointKeyShare:
    """Two-coordinate, additive point-function share for ring-correlation research."""

    def __init__(
        self,
        session_id: str,
        gate_id: str,
        party: int,
        bits: int,
        ring_bits: int,
        seed: bytes,
        corrections: tuple[_Correction, ...],
        final: tuple[int, int],
    ) -> None:
        if (
            not session_id or not gate_id or party not in (0, 1)
            or type(bits) is not int or not 1 <= bits <= _MAX_BITS
            or type(ring_bits) is not int or ring_bits not in (8, 16, 32, 64)
            or len(seed) != _SEED_BYTES or len(corrections) != bits
            or len(final) != 2 or any(type(x) is not int or not 0 <= x < 1 << ring_bits for x in final)
            or any(len(c.seed) != _SEED_BYTES or c.left not in (0, 1) or c.right not in (0, 1)
                   for c in corrections)
        ):
            raise DpfReferenceError("additive point-key structure is invalid")
        self.session_id = session_id
        self.gate_id = gate_id
        self.party = party
        self.bits = bits
        self.ring_bits = ring_bits
        self._seed = seed
        self._corrections = corrections
        self._final = final
        self._used = False

    @property
    def key_bytes(self) -> int:
        return _SEED_BYTES + self.bits * (_SEED_BYTES + 1) + 2 * (self.ring_bits // 8)

    def cancel(self) -> None:
        self._used = True

    def eval_all_once(self, count: int) -> tuple[tuple[int, int], ...]:
        """Expand only this party's share of the point function over a public prefix."""
        if self._used:
            raise DpfReferenceError("additive point-key was already consumed")
        self._used = True
        if type(count) is not int or not 0 < count <= 1 << self.bits:
            raise DpfReferenceError("additive point-key domain prefix is invalid")
        modulus = 1 << self.ring_bits
        output: list[tuple[int, int]] = []
        for index in range(count):
            seed = self._seed
            control = self.party
            for level, correction in enumerate(self._corrections):
                direction = (index >> (self.bits - level - 1)) & 1
                child_seed, child_control = _stretch(seed)[direction]
                if control:
                    child_seed = _xor(child_seed, correction.seed)
                    child_control ^= correction.right if direction else correction.left
                seed, control = child_seed, child_control
            base = _ring_leaf(seed, modulus)
            sign = 1 if self.party == 0 else -1
            output.append((
                (sign * (base[0] + control * self._final[0])) % modulus,
                (sign * (base[1] + control * self._final[1])) % modulus,
            ))
        return tuple(output)


def pack_bit_shares(bits: tuple[int, ...]) -> bytes:
    if not bits or any(type(value) is not int or value not in (0, 1) for value in bits):
        raise DpfReferenceError("Boolean opening shares must be nonempty bits")
    packed = bytearray((len(bits) + 7) // 8)
    for index, value in enumerate(bits):
        packed[index // 8] |= value << (index % 8)
    return bytes(packed)


def unpack_bit_shares(payload: bytes, count: int) -> tuple[int, ...]:
    if (
        type(payload) is not bytes or type(count) is not int or count <= 0
        or len(payload) != (count + 7) // 8
        or payload[-1] >> (((count - 1) % 8) + 1)
    ):
        raise DpfReferenceError("Boolean opening has invalid length or padding")
    return tuple((payload[index // 8] >> (index % 8)) & 1 for index in range(count))


class PointKeyDealer:
    """Trusted in-process issuance. Keys must be placed in separate roles."""

    def __init__(self, session_id: str) -> None:
        if not session_id:
            raise DpfReferenceError("point-key session is missing")
        self.session_id = session_id
        self._issued: set[str] = set()

    def _issue_tree(
        self, gate_id: str, point: int, bits: int, *, maximum_bits: int = _MAX_BITS,
    ) -> tuple[list[bytes], list[bytes], list[int], tuple[_Correction, ...]]:
        if not gate_id or gate_id in self._issued:
            raise DpfReferenceError("point-key gate was already issued")
        if (
            type(maximum_bits) is not int or not 1 <= maximum_bits <= 18
            or type(bits) is not int or not 1 <= bits <= maximum_bits
        ):
            raise DpfReferenceError("point-key domain exceeds reference limit")
        if type(point) is not int or not 0 <= point < 1 << bits:
            raise DpfReferenceError("point-key point is out of range")
        # Reserve identity before sampling or generating any key prefix.
        self._issued.add(gate_id)
        roots = [secrets.token_bytes(_SEED_BYTES) for _ in range(2)]
        seeds = list(roots)
        controls = [0, 1]
        corrections: list[_Correction] = []
        for level in range(bits):
            direction = (point >> (bits - level - 1)) & 1
            children = [_stretch(seed) for seed in seeds]
            word = _Correction(
                _xor(children[0][1 - direction][0], children[1][1 - direction][0]),
                children[0][0][1] ^ children[1][0][1] ^ (1 - direction),
                children[0][1][1] ^ children[1][1][1] ^ direction,
            )
            for party in (0, 1):
                child_seed, child_control = children[party][direction]
                if controls[party]:
                    child_seed = _xor(child_seed, word.seed)
                    child_control ^= word.right if direction else word.left
                seeds[party], controls[party] = child_seed, child_control
            corrections.append(word)
        return roots, seeds, controls, tuple(corrections)

    def issue(self, gate_id: str, point: int, *, bits: int) -> tuple[PointKeyShare, PointKeyShare]:
        roots, seeds, _, words = self._issue_tree(gate_id, point, bits)
        final = _leaf(seeds[0]) ^ _leaf(seeds[1]) ^ 1
        return (
            PointKeyShare(self.session_id, gate_id, 0, bits, roots[0], words, final),
            PointKeyShare(self.session_id, gate_id, 1, bits, roots[1], words, final),
        )

    def issue_ring(
        self,
        gate_id: str,
        point: int,
        payload: tuple[int, int],
        *,
        bits: int,
        ring_bits: int,
    ) -> tuple[RingPointKeyShare, RingPointKeyShare]:
        if (
            type(ring_bits) is not int or ring_bits not in (8, 16, 32, 64)
            or type(payload) is not tuple or len(payload) != 2
            or any(type(x) is not int or not 0 <= x < 1 << ring_bits for x in payload)
        ):
            raise DpfReferenceError("additive point payload is outside ring")
        roots, seeds, controls, words = self._issue_tree(gate_id, point, bits)
        if controls[0] == controls[1]:
            raise DpfReferenceError("point-key control correction lost its active path")
        modulus = 1 << ring_bits
        value0, value1 = _ring_leaf(seeds[0], modulus), _ring_leaf(seeds[1], modulus)
        sign = 1 if controls[0] else -1
        final = (
            (sign * (payload[0] - value0[0] + value1[0])) % modulus,
            (sign * (payload[1] - value0[1] + value1[1])) % modulus,
        )
        return (
            RingPointKeyShare(self.session_id, gate_id, 0, bits, ring_bits, roots[0], words, final),
            RingPointKeyShare(self.session_id, gate_id, 1, bits, ring_bits, roots[1], words, final),
        )
