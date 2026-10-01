"""Bounded, in-process Ring PCG OLE algebra reference (not a secure PCG).

Checks the paper's four sparse cross-terms, additive function shares, and
evaluation in GR(2**k, 2)[X1,...]/(Xi**3-1). Its tiny parameters have no
QA-SD hardness, distributed seed setup, or compiler/runtime admission.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass

from pllm.runtime.shared_dpf import PointKeyDealer, RingPointKeyShare

Pair = tuple[int, int]
Polynomial = tuple[Pair, ...]
Monomial = tuple[int, Pair]


class RingPcgReferenceError(ValueError):
    pass


def _add(left: Pair, right: Pair, modulus: int) -> Pair:
    return ((left[0] + right[0]) % modulus, (left[1] + right[1]) % modulus)


def _mul(left: Pair, right: Pair, modulus: int) -> Pair:
    # zeta**2 = -zeta - 1 in GR(2**k, 2).
    return (
        (left[0] * right[0] - left[1] * right[1]) % modulus,
        (left[0] * right[1] + left[1] * right[0] - left[1] * right[1]) % modulus,
    )


def _index_sum(left: int, right: int, digits: int) -> int:
    place = 1
    result = 0
    for _ in range(digits):
        result += (((left // place) % 3 + (right // place) % 3) % 3) * place
        place *= 3
    return result


def _poly_mul(left: Polynomial, right: Polynomial, digits: int, modulus: int) -> Polynomial:
    result: list[Pair] = [(0, 0) for _ in left]
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            idx = _index_sum(i, j, digits)
            result[idx] = _add(result[idx], _mul(a, b, modulus), modulus)
    return tuple(result)


def _poly_add(left: Polynomial, right: Polynomial, modulus: int) -> Polynomial:
    return tuple(_add(a, b, modulus) for a, b in zip(left, right, strict=True))


def _crt(coefficients: Polynomial, digits: int, modulus: int) -> Polynomial:
    roots: tuple[Pair, ...] = ((1, 0), (0, 1), ((modulus - 1), (modulus - 1)))
    evaluations = []
    for point in range(3**digits):
        total: Pair = (0, 0)
        for index, coefficient in enumerate(coefficients):
            term = coefficient
            place = 1
            for _ in range(digits):
                root = roots[(point // place) % 3]
                for _ in range((index // place) % 3):
                    term = _mul(term, root, modulus)
                place *= 3
            total = _add(total, term, modulus)
        evaluations.append(total)
    return tuple(evaluations)


def _frobenius(value: Pair, modulus: int) -> Pair:
    # Frobenius fixes Z/2**k and takes zeta to zeta**2 = -1 - zeta.
    return ((value[0] - value[1]) % modulus, -value[1] % modulus)


def _sigma_sparse(items: tuple[Monomial, ...], digits: int, modulus: int) -> tuple[Monomial, ...]:
    return tuple((_index_sum(index, index, digits), _frobenius(value, modulus))
                 for index, value in items)


def _sigma_poly(items: Polynomial, digits: int, modulus: int) -> Polynomial:
    result: list[Pair] = [(0, 0) for _ in items]
    for index, value in enumerate(items):
        result[_index_sum(index, index, digits)] = _frobenius(value, modulus)
    return tuple(result)


def _zeta(exponent: int, modulus: int) -> Pair:
    return ((1, 0), (0, 1), (modulus - 1, modulus - 1))[exponent % 3]


def _poly_scale(items: Polynomial, scalar: Pair, modulus: int) -> Polynomial:
    return tuple(_mul(scalar, value, modulus) for value in items)


def _trace(value: Pair, modulus: int) -> int:
    return (2 * value[0] - value[1]) % modulus


@dataclass(frozen=True, slots=True)
class RingPcgOleShare:
    party: int
    public_a: Polynomial
    sparse_s: tuple[Monomial, ...]
    sparse_e: tuple[Monomial, ...]
    cross_terms: tuple[tuple[RingPointKeyShare, ...], ...]
    digits: int
    ring_bits: int
    base_ring_trace: bool = False

    @property
    def key_bytes(self) -> int:
        return sum(key.key_bytes for group in self.cross_terms for key in group)


class RingPcgParty:
    """One correlation holder, with no access to the other party's sparse terms."""

    def __init__(self, share: RingPcgOleShare) -> None:
        self._share = share
        self._used = False

    @property
    def key_bytes(self) -> int:
        return self._share.key_bytes

    def cancel(self) -> None:
        self._used = True
        for group in self._share.cross_terms:
            for key in group:
                key.cancel()

    def _expand_raw(self) -> tuple[Polynomial, list[Polynomial]]:
        if self._used:
            raise RingPcgReferenceError("ring correlation was already expanded or cancelled")
        self._used = True
        share = self._share
        modulus = 1 << share.ring_bits
        count = 3**share.digits
        try:
            sparse_s = [(0, 0) for _ in range(count)]
            sparse_e = [(0, 0) for _ in range(count)]
            for index, coefficient in share.sparse_s:
                sparse_s[index] = coefficient
            for index, coefficient in share.sparse_e:
                sparse_e[index] = coefficient
            b = _poly_add(
                _poly_mul(share.public_a, tuple(sparse_s), share.digits, modulus),
                tuple(sparse_e), modulus,
            )
            cross: list[Polynomial] = []
            for group in share.cross_terms:
                result: list[Pair] = [(0, 0) for _ in range(count)]
                for key in group:
                    for index, element in enumerate(key.eval_all_once(count)):
                        result[index] = _add(result[index], element, modulus)
                cross.append(tuple(result))
            return b, cross
        except Exception:
            self.cancel()
            raise

    def expand_once(self) -> tuple[Polynomial, Polynomial]:
        if self._share.base_ring_trace:
            raise RingPcgReferenceError("use base-ring expansion for a trace-bound share")
        b, cross = self._expand_raw()
        share = self._share
        modulus = 1 << share.ring_bits
        aa = _poly_mul(share.public_a, share.public_a, share.digits, modulus)
        z = _poly_add(
            _poly_add(_poly_mul(aa, cross[0], share.digits, modulus),
                      _poly_mul(share.public_a, cross[1], share.digits, modulus), modulus),
            _poly_add(_poly_mul(share.public_a, cross[2], share.digits, modulus),
                      cross[3], modulus), modulus,
        )
        return _crt(b, share.digits, modulus), _crt(z, share.digits, modulus)

    def expand_base_ring_once(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Paper's trace extraction to two Z/2**k OLE lanes per CRT point."""
        if not self._share.base_ring_trace:
            raise RingPcgReferenceError("trace expansion requires trace-bound material")
        b, cross = self._expand_raw()
        share = self._share
        modulus = 1 << share.ring_bits
        a = share.public_a
        sigma_a = _sigma_poly(a, share.digits, modulus)

        def term(factor: Polynomial, key: Polynomial) -> Polynomial:
            return _poly_mul(factor, key, share.digits, modulus)

        z_regular = _poly_add(
            _poly_add(term(term(a, a), cross[0]), term(a, cross[1]), modulus),
            _poly_add(term(a, cross[2]), cross[3], modulus), modulus,
        )
        z_sigma = _poly_add(
            _poly_add(term(term(a, sigma_a), cross[4]), term(sigma_a, cross[5]), modulus),
            _poly_add(term(a, cross[6]), cross[7], modulus), modulus,
        )
        x_lanes: list[int] = []
        z_lanes: list[int] = []
        for basis_exponent in (1, 2):
            x_poly = _poly_scale(b, _zeta(basis_exponent, modulus), modulus)
            z_poly = _poly_add(
                _poly_scale(z_regular, _zeta(2 * basis_exponent, modulus), modulus),
                _poly_scale(z_sigma, _zeta(3 * basis_exponent, modulus), modulus),
                modulus,
            )
            x_lanes.extend(_trace(value, modulus) for value in _crt(x_poly, share.digits, modulus))
            z_lanes.extend(_trace(value, modulus) for value in _crt(z_poly, share.digits, modulus))
        return tuple(x_lanes), tuple(z_lanes)


class RingPcgReferenceDealer:
    """Trusted test-only issuer; sparse terms are never copied into peer shares."""

    def __init__(self, session_id: str, *, digits: int, ring_bits: int) -> None:
        if not session_id or type(digits) is not int or digits not in (1, 2):
            raise RingPcgReferenceError("ring correlation session/dimension is invalid")
        if type(ring_bits) is not int or ring_bits not in (8, 16, 32, 64):
            raise RingPcgReferenceError("ring correlation word width is unsupported")
        self.session_id = session_id
        self.digits = digits
        self.ring_bits = ring_bits
        self._issued: set[str] = set()
        self._keys = PointKeyDealer(session_id)

    def issue(
        self,
        operation_id: str,
        first: tuple[tuple[Monomial, ...], tuple[Monomial, ...]],
        second: tuple[tuple[Monomial, ...], tuple[Monomial, ...]],
        *,
        max_key_bytes_per_party: int = 64 * 1024,
    ) -> tuple[RingPcgParty, RingPcgParty]:
        return self._issue(
            operation_id, first, second,
            max_key_bytes_per_party=max_key_bytes_per_party, base_ring_trace=False,
        )

    def issue_base_ring(
        self,
        operation_id: str,
        first: tuple[tuple[Monomial, ...], tuple[Monomial, ...]],
        second: tuple[tuple[Monomial, ...], tuple[Monomial, ...]],
        *,
        max_key_bytes_per_party: int = 64 * 1024,
    ) -> tuple[RingPcgParty, RingPcgParty]:
        return self._issue(
            operation_id, first, second,
            max_key_bytes_per_party=max_key_bytes_per_party, base_ring_trace=True,
        )

    def _issue(
        self,
        operation_id: str,
        first: tuple[tuple[Monomial, ...], tuple[Monomial, ...]],
        second: tuple[tuple[Monomial, ...], tuple[Monomial, ...]],
        *,
        max_key_bytes_per_party: int,
        base_ring_trace: bool,
    ) -> tuple[RingPcgParty, RingPcgParty]:
        if not operation_id or operation_id in self._issued:
            raise RingPcgReferenceError("ring correlation was already issued")
        count = 3**self.digits
        modulus = 1 << self.ring_bits
        terms = (first, second)
        for term_pair in terms:
            if len(term_pair) != 2 or any(not 0 < len(group) <= 2 for group in term_pair):
                raise RingPcgReferenceError("sparse correlation terms are invalid")
            for group in term_pair:
                if len({index for index, _ in group}) != len(group):
                    raise RingPcgReferenceError("sparse correlation positions must be unique")
                if any(
                    type(index) is not int or not 0 <= index < count
                    or type(value) is not tuple or len(value) != 2
                    or any(type(item) is not int or not 0 <= item < modulus for item in value)
                    or value == (0, 0)
                    for index, value in group
                ):
                    raise RingPcgReferenceError("sparse correlation coefficient is invalid")
        cross: tuple[tuple[tuple[Monomial, ...], tuple[Monomial, ...]], ...] = (
            (first[0], second[0]), (first[1], second[0]),
            (first[0], second[1]), (first[1], second[1]),
        )
        if base_ring_trace:
            cross += (
                (first[0], _sigma_sparse(second[0], self.digits, modulus)),
                (first[1], _sigma_sparse(second[0], self.digits, modulus)),
                (first[0], _sigma_sparse(second[1], self.digits, modulus)),
                (first[1], _sigma_sparse(second[1], self.digits, modulus)),
            )
        width = (count - 1).bit_length()
        body = sum(len(left) * len(right) for left, right in cross)
        key_bytes = body * (16 + 17 * width + 2 * (self.ring_bits // 8))
        if (
            type(max_key_bytes_per_party) is not int or max_key_bytes_per_party <= 0
            or key_bytes > max_key_bytes_per_party
        ):
            raise RingPcgReferenceError("ring correlation key budget exceeded before issuance")
        self._issued.add(operation_id)
        public_a: Polynomial = tuple(
            (secrets.randbelow(modulus), secrets.randbelow(modulus)) for _ in range(count)
        )
        grouped: list[list[tuple[RingPointKeyShare, ...]]] = [[], []]
        try:
            for group_index, (left, right) in enumerate(cross):
                keys = [[], []]
                for left_index, a in left:
                    for right_index, b in right:
                        point = _index_sum(left_index, right_index, self.digits)
                        pair = self._keys.issue_ring(
                            f"{operation_id}.cross.{group_index}.{left_index}.{right_index}",
                            point, _mul(a, b, modulus), bits=width, ring_bits=self.ring_bits,
                        )
                        keys[0].append(pair[0])
                        keys[1].append(pair[1])
                grouped[0].append(tuple(keys[0]))
                grouped[1].append(tuple(keys[1]))
            def party_material(party: int) -> RingPcgParty:
                return RingPcgParty(RingPcgOleShare(
                    party, public_a, terms[party][0], terms[party][1],
                    tuple(grouped[party]), self.digits, self.ring_bits, base_ring_trace,
                ))

            return party_material(0), party_material(1)
        except Exception:
            for party_groups in grouped:
                for group in party_groups:
                    for key in group:
                        key.cancel()
            raise
