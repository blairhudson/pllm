"""One-use, dense two-input function-sharing reference for resident tensors.

This test-only dealer distributes independent random shares of a shifted public
table. Neither party learns the other mask share or a plaintext input; each
online party sends only its share of the two masked inputs to its peer. The
dense table is intentionally charged in full. It is not compressed FSS, a
production dealer, or an executable semantic decoder operator.
"""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass

import msgpack
import numpy as np


class FusedGateError(ValueError):
    """Malformed, replayed, or resource-inadmissible reference material."""


_DOMAIN = b"pllm.resident_fused_gate.v1\0"


def _identity(value: str) -> str:
    if type(value) is not str or not 0 < len(value) <= 128 or not value.isascii():
        raise FusedGateError("fused gate identity must be bounded ASCII")
    return value


def _digest(value: bytes) -> bytes:
    if type(value) is not bytes or len(value) != 32:
        raise FusedGateError("fused gate graph and channel digests must be 32 bytes")
    return value


@dataclass(frozen=True, slots=True)
class PublicFusedGate:
    """Immutable signed Q3 or Q7 SiLU(gate)×up truth table."""

    bits: int
    table: np.ndarray
    fingerprint: bytes

    def __post_init__(self) -> None:
        if type(self.bits) is not int or self.bits not in (4, 8):
            raise FusedGateError("public fused gate width is unsupported")
        domain = 1 << self.bits
        values = np.asarray(self.table)
        if values.dtype != np.uint64 or values.shape != (domain, domain):
            raise FusedGateError("public fused gate table has invalid width or shape")
        snapshot = np.array(values, dtype=np.uint64, order="C", copy=True)
        expected = hashlib.sha256(
            _DOMAIN + bytes([self.bits]) + snapshot.view("<i8").tobytes()
        ).digest()
        if type(self.fingerprint) is not bytes or self.fingerprint != expected:
            raise FusedGateError("public fused gate table fingerprint differs from its values")
        snapshot.setflags(write=False)
        object.__setattr__(self, "table", snapshot)

    @classmethod
    def silu_product(cls, bits: int) -> "PublicFusedGate":
        if type(bits) is not int or bits not in (4, 8):
            raise FusedGateError("fused lookup admits only bounded Q3 or Q7 inputs")
        domain = 1 << bits
        scale = domain // 2
        signed = np.arange(domain, dtype=np.int64)
        signed[signed >= scale] -= domain
        gate = signed[:, None].astype(np.float64) / scale
        up = signed[None, :].astype(np.float64) / scale
        # Rounding is ties-to-even; this table is a bounded numeric reference,
        # not the float32 semantic decoder SiLU contract.
        table = np.rint(scale * (gate / (1.0 + np.exp(-gate))) * up).astype(np.int64)
        raw = table.astype("<i8", copy=False).tobytes()
        table = np.ascontiguousarray(table.view(np.uint64))
        table.setflags(write=False)
        fingerprint = hashlib.sha256(_DOMAIN + bytes([bits]) + raw).digest()
        return cls(bits, table, fingerprint)

    def value(self, gate: int, up: int) -> int:
        if type(gate) is not int or type(up) is not int:
            raise FusedGateError("fused gate values must be signed integers")
        bound = 1 << (self.bits - 1)
        if not -bound <= gate < bound or not -bound <= up < bound:
            raise FusedGateError("fused gate value is outside public Q domain")
        return int(self.table[gate % (2 * bound), up % (2 * bound)].view(np.int64))


def _read_fields(payload: bytes, bits: int, elements: int) -> tuple[np.ndarray, np.ndarray]:
    if type(payload) is not bytes or len(payload) != elements * (1 if bits == 4 else 2):
        raise FusedGateError("fused opening body has invalid length")
    values = np.frombuffer(payload, dtype=np.uint8)
    if bits == 4:
        return values & np.uint8(15), values >> np.uint8(4)
    return values[::2], values[1::2]


@dataclass(frozen=True, slots=True)
class FusedOpening:
    session_id: str
    operation_id: str
    graph_digest: bytes
    channel_id: bytes
    profile_digest: bytes
    party: int
    elements: int
    body: bytes

    def pack(self) -> bytes:
        return msgpack.packb(
            (1, self.session_id, self.operation_id, self.graph_digest,
             self.channel_id, self.profile_digest, self.party, self.elements, self.body),
            use_bin_type=True,
        )

    @classmethod
    def unpack(cls, payload: bytes, *, maximum_bytes: int) -> "FusedOpening":
        if type(payload) is not bytes or not 0 < len(payload) <= maximum_bytes:
            raise FusedGateError("fused opening exceeds body limit")
        try:
            row = msgpack.unpackb(
                payload, raw=False, max_array_len=9, max_bin_len=maximum_bytes,
                max_str_len=128,
            )
            if not isinstance(row, list) or len(row) != 9 or type(row[0]) is not int or row[0] != 1:
                raise ValueError("unknown opening schema")
            _, session, operation, graph, channel, profile, party, elements, body = row
            _identity(session)
            _identity(operation)
            _digest(graph)
            _digest(channel)
            _digest(profile)
            if party not in (0, 1) or type(party) is not int or type(elements) is not int or elements <= 0:
                raise ValueError("invalid party or element count")
            if type(body) is not bytes:
                raise ValueError("invalid opening body")
            return cls(session, operation, graph, channel, profile, party, elements, body)
        except (
            ValueError, TypeError, UnicodeDecodeError, OverflowError,
            msgpack.ExtraData, msgpack.FormatError, msgpack.OutOfData, msgpack.StackError,
        ) as exc:
            raise FusedGateError("invalid fused opening frame") from exc


class FusedGateParty:
    """One party's opaque table share; begin/finish or cancellation burns it."""

    __slots__ = (
        "session_id", "operation_id", "graph_digest", "channel_id", "profile_digest",
        "party", "bits", "elements", "_table", "_gate_mask", "_up_mask", "_opening",
        "_spent",
    )

    def __init__(
        self, session_id: str, operation_id: str, graph_digest: bytes, channel_id: bytes,
        profile_digest: bytes, party: int, bits: int, table: np.ndarray,
        gate_mask: np.ndarray, up_mask: np.ndarray,
    ) -> None:
        self.session_id = _identity(session_id)
        self.operation_id = _identity(operation_id)
        self.graph_digest = _digest(graph_digest)
        self.channel_id = _digest(channel_id)
        self.profile_digest = _digest(profile_digest)
        if party not in (0, 1) or type(party) is not int or bits not in (4, 8):
            raise FusedGateError("invalid party or fused gate width")
        self.party = party
        self.bits = bits
        self._table = np.array(table, dtype=np.uint64, order="C", copy=True)
        self.elements = int(self._table.shape[0])
        n = 1 << bits
        self._gate_mask = np.array(gate_mask, dtype=np.uint8, order="C", copy=True)
        self._up_mask = np.array(up_mask, dtype=np.uint8, order="C", copy=True)
        if (
            not 0 < self.elements <= 128 or self._table.shape != (self.elements, n, n)
            or self._gate_mask.shape != (self.elements,)
            or self._up_mask.shape != (self.elements,)
            or np.any(self._gate_mask >= n) or np.any(self._up_mask >= n)
        ):
            raise FusedGateError("fused table material has invalid shape or mask")
        self._opening: bytes | None = None
        self._spent = False

    @property
    def material_body_bytes(self) -> int:
        return self._table.nbytes + self._gate_mask.nbytes + self._up_mask.nbytes

    def cancel(self) -> None:
        self._spent = True
        self._opening = None
        self._table.fill(0)
        self._gate_mask.fill(0)
        self._up_mask.fill(0)

    def begin(self, gate_share: np.ndarray, up_share: np.ndarray) -> bytes:
        if self._spent or self._opening is not None:
            raise FusedGateError("fused table material is spent")
        try:
            gate = np.asarray(gate_share)
            up = np.asarray(up_share)
            n = 1 << self.bits
            if (
                gate.shape != (self.elements,) or up.shape != (self.elements,)
                or gate.dtype != np.uint8 or up.dtype != np.uint8
                or np.any(gate >= n) or np.any(up >= n)
            ):
                raise FusedGateError("fused input shares do not match the bounded ring")
            first = (gate + self._gate_mask) & np.uint8(n - 1)
            second = (up + self._up_mask) & np.uint8(n - 1)
            if self.bits == 4:
                body = (first | (second << np.uint8(4))).tobytes()
            else:
                body = np.stack((first, second), axis=1).tobytes()
            self._opening = FusedOpening(
                self.session_id, self.operation_id, self.graph_digest, self.channel_id,
                self.profile_digest, self.party, self.elements, body,
            ).pack()
            return self._opening
        except BaseException:
            self.cancel()
            raise

    def finish(self, peer_payload: bytes) -> np.ndarray:
        if self._spent or self._opening is None:
            raise FusedGateError("fused table material is unavailable or spent")
        try:
            peer = FusedOpening.unpack(
                peer_payload,
                maximum_bytes=256 + self.elements * (1 if self.bits == 4 else 2),
            )
            if (
                peer.session_id != self.session_id or peer.operation_id != self.operation_id
                or peer.graph_digest != self.graph_digest or peer.channel_id != self.channel_id
                or peer.profile_digest != self.profile_digest or peer.party != 1 - self.party
                or peer.elements != self.elements
            ):
                raise FusedGateError("fused peer opening has different commitments")
            own = FusedOpening.unpack(
                self._opening,
                maximum_bytes=256 + self.elements * (1 if self.bits == 4 else 2),
            )
            own_gate, own_up = _read_fields(own.body, self.bits, self.elements)
            peer_gate, peer_up = _read_fields(peer.body, self.bits, self.elements)
            idx = np.arange(self.elements)
            gate = (own_gate + peer_gate) & np.uint8((1 << self.bits) - 1)
            up = (own_up + peer_up) & np.uint8((1 << self.bits) - 1)
            result = self._table[idx, gate, up].copy()
            self.cancel()
            return result
        except BaseException:
            self.cancel()
            raise


class FusedGateDealer:
    """Bounded, local *test-only* source of one-use party-specific material."""

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
            raise FusedGateError("fused reference material cap exceeds 16 MiB per party")
        self.limit = maximum_material_bytes_per_party
        self.issued_bytes_per_party = 0
        self._issued: dict[str, tuple[FusedGateParty, FusedGateParty]] = {}

    def issue(
        self, operation_id: str, profile: PublicFusedGate, *, elements: int,
    ) -> tuple[FusedGateParty, FusedGateParty]:
        operation_id = _identity(operation_id)
        if (
            operation_id in self._issued or type(profile) is not PublicFusedGate
            or type(elements) is not int or not 0 < elements <= 128
        ):
            raise FusedGateError("fused issuance identity or shape is invalid")
        n = 1 << profile.bits
        material = elements * (n * n * 8 + 2)
        if material > self.limit - self.issued_bytes_per_party:
            raise FusedGateError("fused table exceeds aggregate one-use material cap before issuance")
        masks = tuple(
            np.frombuffer(secrets.token_bytes(elements), dtype=np.uint8).copy() & np.uint8(n - 1)
            for _ in range(4)
        )
        gate_mask = (masks[0] + masks[1]) & np.uint8(n - 1)
        up_mask = (masks[2] + masks[3]) & np.uint8(n - 1)
        random_share = np.frombuffer(
            secrets.token_bytes(elements * n * n * 8), dtype="<u8",
        ).reshape(elements, n, n)
        indices = np.arange(n, dtype=np.uint16)
        rows = (indices[None, :] - gate_mask[:, None]) % n
        cols = (indices[None, :] - up_mask[:, None]) % n
        shifted = profile.table[rows[:, :, None], cols[:, None, :]]
        other_share = shifted - random_share
        first = FusedGateParty(
            self.session_id, operation_id, self.graph_digest, self.channel_id,
            profile.fingerprint, 0, profile.bits, random_share, masks[0], masks[2],
        )
        second = FusedGateParty(
            self.session_id, operation_id, self.graph_digest, self.channel_id,
            profile.fingerprint, 1, profile.bits, other_share, masks[1], masks[3],
        )
        self._issued[operation_id] = first, second
        self.issued_bytes_per_party += material
        return first, second

    def cancel(self) -> None:
        for parties in self._issued.values():
            for party in parties:
                party.cancel()
        self._issued.clear()
