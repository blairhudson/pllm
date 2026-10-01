"""Bounded one-use quadratic Q7 gated-product shares (research reference).

An offline dealer shifts the public polynomial in two secret linear projections
of the same masked hidden input. Each party receives five independent shares of
the shifted coefficients per output; online parties exchange only their masked
hidden shares. Arithmetic is exact in the 24-bit ring for bounded Q7 gate/up.

This is NOT a Qwen layer: public bound admission, fixed-scale edge conversion,
protected truncation, RMSNorm and attention are not implemented here.
"""

from __future__ import annotations

import hashlib
import secrets

import numpy as np

from .shared_gate_reference import FusedGateError, FusedOpening, _digest, _identity


_MASK = (1 << 24) - 1
_SIGNED_LIMIT = 1 << 23
_PROFILE_DOMAIN = b"pllm.resident_q7_quadratic_gated_product.v1\0"


def _pack_u24(values: np.ndarray) -> bytes:
    row = np.ascontiguousarray(values, dtype="<u4").reshape(-1)
    if np.any(row > _MASK):
        raise FusedGateError("Q7 polynomial residue is outside the 24-bit ring")
    return row.view(np.uint8).reshape(-1, 4)[:, :3].tobytes()


def _unpack_u24(payload: bytes, count: int) -> np.ndarray:
    if type(payload) is not bytes or len(payload) != count * 3:
        raise FusedGateError("Q7 polynomial u24 body has invalid length")
    row = np.frombuffer(payload, dtype=np.uint8).reshape(count, 3).astype(np.uint32)
    return row[:, 0] | (row[:, 1] << 8) | (row[:, 2] << 16)


class QuadraticQ7Gate:
    """Numerator (256g+g²)u, scale 65536, with signed Q7 gate/up."""

    __slots__ = ()
    fingerprint = hashlib.sha256(
        _PROFILE_DOMAIN + b"g[-128,127];u[-128,127];ring24;scale65536"
    ).digest()

    def numerator(self, gate: int, up: int) -> int:
        if (
            type(gate) is not int or type(up) is not int
            or not -128 <= gate <= 127 or not -128 <= up <= 127
        ):
            raise FusedGateError("quadratic Q7 gate/up must lie in [-128, 127]")
        result = (256 * gate + gate * gate) * up
        if not -_SIGNED_LIMIT <= result < _SIGNED_LIMIT:
            raise FusedGateError("quadratic Q7 numerator exceeds signed ring")
        return result


def _evaluate(masked_gate: np.ndarray, masked_up: np.ndarray, coeffs: np.ndarray, party: int) -> np.ndarray:
    gate = masked_gate.astype(np.uint64)
    up = masked_up.astype(np.uint64)
    coefficients = coeffs.astype(np.uint64)
    gate2 = (gate * gate) & _MASK
    gate_up = (gate * up) & _MASK
    # First term has public coefficient one and belongs to exactly one party.
    shared = gate2 * up if party == 0 else np.zeros_like(gate2)
    result = (
        shared
        + coefficients[:, :, 0] * gate_up
        + coefficients[:, :, 1] * up
        + coefficients[:, :, 2] * gate2
        + coefficients[:, :, 3] * gate
        + coefficients[:, :, 4]
    ) & _MASK
    return result.astype(np.uint32)


class QuadraticQ7Party:
    """Consumes one party's 24-bit masked-source and polynomial coefficients."""

    __slots__ = (
        "session_id", "operation_id", "graph_digest", "channel_id", "profile_digest",
        "party", "rows", "hidden", "outputs", "_gate_weights", "_up_weights",
        "_source_mask", "_coefficients", "_opening", "_spent",
    )

    def __init__(
        self, session_id: str, operation_id: str, graph_digest: bytes, channel_id: bytes,
        profile_digest: bytes, party: int, gate_weights: np.ndarray, up_weights: np.ndarray,
        source_mask: np.ndarray, coefficients: np.ndarray,
    ) -> None:
        self.session_id = _identity(session_id)
        self.operation_id = _identity(operation_id)
        self.graph_digest = _digest(graph_digest)
        self.channel_id = _digest(channel_id)
        self.profile_digest = _digest(profile_digest)
        if type(party) is not int or party not in (0, 1):
            raise FusedGateError("quadratic party identity must be zero or one")
        self.party = party
        self._gate_weights = np.array(gate_weights, dtype=np.int8, order="C", copy=True)
        self._up_weights = np.array(up_weights, dtype=np.int8, order="C", copy=True)
        self.outputs, self.hidden = self._gate_weights.shape
        self._source_mask = np.array(source_mask, dtype=np.uint32, order="C", copy=True)
        self.rows = self._source_mask.shape[0]
        self._coefficients = np.array(coefficients, dtype=np.uint32, order="C", copy=True)
        if (
            self._up_weights.shape != self._gate_weights.shape
            or self._source_mask.shape != (self.rows, self.hidden)
            or self._coefficients.shape != (self.rows, self.outputs, 5)
            or np.any(self._source_mask > _MASK) or np.any(self._coefficients > _MASK)
        ):
            raise FusedGateError("quadratic material shape or residue is invalid")
        self._opening: bytes | None = None
        self._spent = False

    @property
    def material_body_bytes(self) -> int:
        return 3 * (self._source_mask.size + self._coefficients.size)

    def cancel(self) -> None:
        self._spent = True
        self._opening = None
        self._source_mask.fill(0)
        self._coefficients.fill(0)

    def begin(self, hidden_share: np.ndarray) -> bytes:
        if self._spent or self._opening is not None:
            raise FusedGateError("quadratic Q7 material is spent")
        try:
            row = np.asarray(hidden_share)
            if (
                row.shape != (self.rows, self.hidden) or row.dtype != np.uint32
                or np.any(row > _MASK)
            ):
                raise FusedGateError("quadratic hidden share has invalid shape or ring")
            masked = (row + self._source_mask) & np.uint32(_MASK)
            self._opening = FusedOpening(
                self.session_id, self.operation_id, self.graph_digest, self.channel_id,
                self.profile_digest, self.party, self.rows * self.hidden,
                _pack_u24(masked),
            ).pack()
            return self._opening
        except BaseException:
            self.cancel()
            raise

    def finish(self, peer_payload: bytes) -> np.ndarray:
        if self._spent or self._opening is None:
            raise FusedGateError("quadratic Q7 material is unavailable or spent")
        limit = 256 + 3 * self.rows * self.hidden
        try:
            peer = FusedOpening.unpack(peer_payload, maximum_bytes=limit)
            own = FusedOpening.unpack(self._opening, maximum_bytes=limit)
            if (
                peer.session_id != self.session_id or peer.operation_id != self.operation_id
                or peer.graph_digest != self.graph_digest or peer.channel_id != self.channel_id
                or peer.profile_digest != self.profile_digest or peer.party != 1 - self.party
                or peer.elements != self.rows * self.hidden
            ):
                raise FusedGateError("quadratic peer opening has different commitments")
            hidden = (
                _unpack_u24(own.body, own.elements)
                + _unpack_u24(peer.body, peer.elements)
            ).reshape(self.rows, self.hidden) & np.uint32(_MASK)
            # Modular projection is correct without disclosing either mask.
            gate = (hidden.astype(np.int64) @ self._gate_weights.astype(np.int64).T) & _MASK
            up = (hidden.astype(np.int64) @ self._up_weights.astype(np.int64).T) & _MASK
            output = _evaluate(gate, up, self._coefficients, self.party)
            self.cancel()
            return output
        except BaseException:
            self.cancel()
            raise


class QuadraticQ7Dealer:
    """Test-only offline dealer with aggregate per-party issuance admission."""

    def __init__(
        self, session_id: str, graph_digest: bytes, channel_id: bytes,
        *, maximum_material_bytes_per_party: int,
    ) -> None:
        self.session_id = _identity(session_id)
        self.graph_digest = _digest(graph_digest)
        self.channel_id = _digest(channel_id)
        if (
            type(maximum_material_bytes_per_party) is not int
            or not 0 < maximum_material_bytes_per_party <= 16 << 20
        ):
            raise FusedGateError("quadratic reference cap exceeds 16 MiB per party")
        self.limit = maximum_material_bytes_per_party
        self.issued_bytes_per_party = 0
        self._issued: dict[str, tuple[QuadraticQ7Party, QuadraticQ7Party]] = {}

    def issue(
        self, operation_id: str, gate_weights: np.ndarray, up_weights: np.ndarray,
        *, rows: int, hidden_bound: int,
    ) -> tuple[QuadraticQ7Party, QuadraticQ7Party]:
        operation_id = _identity(operation_id)
        if operation_id in self._issued:
            raise FusedGateError("quadratic issuance cannot replay an operation")
        if type(rows) is not int or not 0 < rows <= 16:
            raise FusedGateError("quadratic row count is outside test reference")
        gate = np.asarray(gate_weights)
        up = np.asarray(up_weights)
        if gate.dtype != np.int8 or up.dtype != np.int8 or gate.ndim != 2 or up.shape != gate.shape:
            raise FusedGateError("quadratic weights need equal-shape signed int8 rows")
        outputs, hidden = gate.shape
        if not 0 < outputs <= 32 or not 0 < hidden <= 32:
            raise FusedGateError("quadratic weights exceed bounded toy dimensions")
        if (
            type(hidden_bound) is not int or not 0 <= hidden_bound <= 127
            or hidden_bound * max(
                int(np.max(np.sum(np.abs(gate.astype(np.int16)), axis=1))),
                int(np.max(np.sum(np.abs(up.astype(np.int16)), axis=1))),
            ) > 127
        ):
            raise FusedGateError("quadratic gate/up lack a public signed Q7 range bound")
        material = 3 * (rows * hidden + 5 * rows * outputs)
        if material > self.limit - self.issued_bytes_per_party:
            raise FusedGateError("quadratic aggregate material cap fails before issuance")
        elements = rows * hidden
        masks = tuple(
            _unpack_u24(secrets.token_bytes(3 * elements), elements).reshape(rows, hidden)
            for _ in range(2)
        )
        combined = (masks[0] + masks[1]) & np.uint32(_MASK)
        gate_mask = (combined.astype(np.int64) @ gate.astype(np.int64).T) & _MASK
        up_mask = (combined.astype(np.int64) @ up.astype(np.int64).T) & _MASK
        squared = (gate_mask * gate_mask) & _MASK
        coefficients = np.stack(
            (
                (256 - 2 * gate_mask) & _MASK,
                (squared - 256 * gate_mask) & _MASK,
                (-up_mask) & _MASK,
                (((2 * gate_mask - 256) & _MASK) * up_mask) & _MASK,
                (((256 * gate_mask - squared) & _MASK) * up_mask) & _MASK,
            ), axis=2,
        ).astype(np.uint32)
        output_elements = rows * outputs * 5
        first_coefficients = _unpack_u24(
            secrets.token_bytes(3 * output_elements), output_elements,
        ).reshape(rows, outputs, 5)
        second_coefficients = (coefficients - first_coefficients) & np.uint32(_MASK)
        profile_digest = hashlib.sha256(
            _PROFILE_DOMAIN + bytes([rows, hidden, outputs, hidden_bound])
            + gate.tobytes() + up.tobytes()
        ).digest()
        first = QuadraticQ7Party(
            self.session_id, operation_id, self.graph_digest, self.channel_id,
            profile_digest, 0, gate, up, masks[0], first_coefficients,
        )
        second = QuadraticQ7Party(
            self.session_id, operation_id, self.graph_digest, self.channel_id,
            profile_digest, 1, gate, up, masks[1], second_coefficients,
        )
        self._issued[operation_id] = first, second
        self.issued_bytes_per_party += material
        return first, second

    def cancel(self) -> None:
        for parties in self._issued.values():
            for party in parties:
                party.cancel()
        self._issued.clear()
