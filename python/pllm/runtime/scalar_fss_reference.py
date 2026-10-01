"""Bounded, one-use scalar FSS compiler reference with constant public shapes.

The interval backend is a *dense*, information-theoretic function sharing of
the translated coefficient table. This executes the two-call contract on a tiny
ring but is not FuseFSS's compact GPU vector-interval backend or a model path.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass

import numpy as np

from pllm.runtime.shared_dpf import PointKeyDealer, PointKeyShare


class ScalarFssError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class AffineScalarSpec:
    bits: int
    boundaries: tuple[int, ...]
    coefficients: tuple[tuple[int, int], ...]
    predicates: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if type(self.bits) is not int or not 2 <= self.bits <= 8:
            raise ScalarFssError("dense scalar reference supports only 2–8-bit rings")
        if (
            len(self.boundaries) < 2 or len(self.boundaries) > 9
            or self.boundaries[0] != 0 or self.boundaries[-1] != 1 << self.bits
            or any(type(v) is not int for v in self.boundaries)
            or any(a >= b for a, b in zip(self.boundaries, self.boundaries[1:], strict=False))
        ):
            raise ScalarFssError("scalar intervals must form a bounded full partition")
        if len(self.coefficients) != len(self.boundaries) - 1 or any(
            len(pair) != 2 or any(type(v) is not int or not 0 <= v < 1 << 64 for v in pair)
            for pair in self.coefficients
        ):
            raise ScalarFssError("every interval needs exactly two uint64 affine coefficients")
        if len(self.predicates) > 8 or any(
            type(p) is not int or not 0 < p < 1 << self.bits for p in self.predicates
        ):
            raise ScalarFssError("scalar predicate threshold is unsupported")

    @property
    def public_shape(self) -> tuple[int, int, int, int]:
        """Ring width, predicate atoms, interval slots, payload words."""
        return self.bits, 2 * len(self.predicates), min(len(self.coefficients) + 1, 1 << self.bits), 2


class ScalarGateShare:
    def __init__(
        self,
        session_id: str,
        gate_id: str,
        party: int,
        spec: AffineScalarSpec,
        mask_share: np.uint64,
        coefficient_table: np.ndarray,
        predicate_keys: tuple[tuple[PointKeyShare, PointKeyShare], ...],
        carry_shares: tuple[int, ...],
    ) -> None:
        self.session_id = session_id
        self.gate_id = gate_id
        self.party = party
        self.spec = spec
        self.mask_share = mask_share
        self._table = coefficient_table
        self._keys = predicate_keys
        self._carry_shares = carry_shares
        self._consumed = False
        if (
            not session_id or not gate_id or party not in (0, 1)
            or coefficient_table.dtype != np.uint64
            or coefficient_table.shape != (1 << spec.bits, 2)
            or len(predicate_keys) != len(spec.predicates)
            or len(carry_shares) != len(spec.predicates)
            or any(c not in (0, 1) for c in carry_shares)
            or any(
                left.session_id != session_id or right.session_id != session_id
                or left.party != party or right.party != party
                or left.gate_id != f"{gate_id}.theta.{index}"
                or right.gate_id != f"{gate_id}.mask.{index}"
                for index, (left, right) in enumerate(predicate_keys)
            )
        ):
            raise ScalarFssError("scalar FSS gate has invalid shape or identity")

    @property
    def offline_bytes(self) -> int:
        return int(self._table.nbytes) + sum(
            left.key_bytes + right.key_bytes for left, right in self._keys
        ) + len(self._carry_shares)

    def cancel(self) -> None:
        self._consumed = True
        for left, right in self._keys:
            left.cancel()
            right.cancel()

    def evaluate(self, public_masked_input: int) -> tuple[tuple[np.uint64, np.uint64], tuple[int, ...]]:
        if self._consumed:
            raise ScalarFssError("scalar FSS gate already consumed")
        self._consumed = True
        try:
            if type(public_masked_input) is not int or not 0 <= public_masked_input < 1 << self.spec.bits:
                raise ScalarFssError("public masked scalar is outside the gate domain")
            coefficients = self._table[public_masked_input]
            predicates = tuple(
                left.less_than_share(public_masked_input)
                ^ right.less_than_share(public_masked_input) ^ carry
                for (left, right), carry in zip(self._keys, self._carry_shares, strict=True)
            )
            return (np.uint64(coefficients[0]), np.uint64(coefficients[1])), predicates
        finally:
            self.cancel()


class ScalarGateIssuer:
    """Input-independent, trusted in-process dealer for one scalar gate."""

    def __init__(self, session_id: str) -> None:
        self._dealer = PointKeyDealer(session_id)
        self._issued: set[str] = set()
        self.session_id = session_id

    def issue(self, spec: AffineScalarSpec, gate_id: str) -> tuple[ScalarGateShare, ScalarGateShare]:
        if not isinstance(gate_id, str) or not gate_id or gate_id in self._issued:
            raise ScalarFssError("scalar gate identity was already issued")
        self._issued.add(gate_id)
        modulus = 1 << spec.bits
        mask = secrets.randbelow(modulus)
        share0 = np.frombuffer(secrets.token_bytes(8), dtype=np.uint64)[0]
        mask_shares = (share0, np.uint64((mask - int(share0)) % (1 << 64)))

        table = np.empty((modulus, 2), dtype=np.uint64)
        for public_value in range(modulus):
            private_value = (public_value - mask) % modulus
            interval = next(i for i, upper in enumerate(spec.boundaries[1:]) if private_value < upper)
            table[public_value] = spec.coefficients[interval]
        first = np.frombuffer(secrets.token_bytes(table.nbytes), dtype=np.uint64).copy().reshape(table.shape)
        tables = first, table - first

        keys0: list[tuple[PointKeyShare, PointKeyShare]] = []
        keys1: list[tuple[PointKeyShare, PointKeyShare]] = []
        carries0: list[int] = []
        carries1: list[int] = []
        for index, boundary in enumerate(spec.predicates):
            theta = (mask + boundary) % modulus
            left = self._dealer.issue(f"{gate_id}.theta.{index}", theta, bits=spec.bits)
            right = self._dealer.issue(f"{gate_id}.mask.{index}", mask, bits=spec.bits)
            keys0.append((left[0], right[0]))
            keys1.append((left[1], right[1]))
            first_carry = secrets.randbits(1)
            carries0.append(first_carry)
            carries1.append(first_carry ^ int(mask + boundary >= modulus))
        return (
            ScalarGateShare(
                self.session_id, gate_id, 0, spec, mask_shares[0], tables[0],
                tuple(keys0), tuple(carries0),
            ),
            ScalarGateShare(
                self.session_id, gate_id, 1, spec, mask_shares[1], tables[1],
                tuple(keys1), tuple(carries1),
            ),
        )
