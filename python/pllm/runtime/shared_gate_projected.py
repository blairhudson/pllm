"""Test-local correlated-source mask for one-use fused shared gate lookup.

Gate and up are public linear images of the *same* secret-shared input. A
dealer can mask that input once and pre-shift each output lookup by its
projected mask. This removes repeated intermediate-width online openings;
it does not compress the per-output FSS table or supply a Qwen scale bridge.
"""

from __future__ import annotations

import hashlib
import secrets

import numpy as np

from .shared_gate_reference import FusedGateError, FusedOpening, PublicFusedGate, _digest, _identity


def _source_body(values: np.ndarray, bits: int) -> bytes:
    flat = values.reshape(-1)
    if bits == 8:
        return flat.tobytes()
    packed = np.zeros((flat.size + 1) // 2, dtype=np.uint8)
    packed[:flat.size // 2] = flat[::2][:flat.size // 2] | (flat[1::2] << np.uint8(4))
    if flat.size % 2:
        packed[-1] = flat[-1]
    return packed.tobytes()


def _read_source(payload: bytes, bits: int, elements: int) -> np.ndarray:
    expected = (elements * bits + 7) // 8
    if type(payload) is not bytes or len(payload) != expected:
        raise FusedGateError("projected source body has invalid length")
    data = np.frombuffer(payload, dtype=np.uint8)
    if bits == 8:
        return data.copy()
    if elements % 2 and data[-1] & 0xf0:
        raise FusedGateError("projected source body has nonzero terminal padding")
    output = np.empty(elements, dtype=np.uint8)
    output[::2] = data[:(elements + 1) // 2] & np.uint8(15)
    output[1::2] = data[:elements // 2] >> np.uint8(4)
    return output


class ProjectedFusedGateParty:
    """One local table and one hidden-mask share; no peer's share or plaintext."""

    __slots__ = (
        "session_id", "operation_id", "graph_digest", "channel_id", "profile_digest",
        "party", "bits", "rows", "hidden", "outputs", "_gate_weights", "_up_weights",
        "_source_mask", "_table", "_opening", "_spent",
    )

    def __init__(
        self, session_id: str, operation_id: str, graph_digest: bytes, channel_id: bytes,
        profile_digest: bytes, party: int, bits: int,
        gate_weights: np.ndarray, up_weights: np.ndarray,
        source_mask: np.ndarray, table: np.ndarray,
    ) -> None:
        self.session_id = _identity(session_id)
        self.operation_id = _identity(operation_id)
        self.graph_digest = _digest(graph_digest)
        self.channel_id = _digest(channel_id)
        self.profile_digest = _digest(profile_digest)
        if party not in (0, 1) or type(party) is not int or bits not in (4, 8):
            raise FusedGateError("invalid projected party or domain")
        self.party = party
        self.bits = bits
        self._gate_weights = np.array(gate_weights, dtype=np.int8, order="C", copy=True)
        self._up_weights = np.array(up_weights, dtype=np.int8, order="C", copy=True)
        self.outputs, self.hidden = self._gate_weights.shape
        self._source_mask = np.array(source_mask, dtype=np.uint8, order="C", copy=True)
        self.rows = self._source_mask.shape[0]
        self._table = np.array(table, dtype=np.uint64, order="C", copy=True)
        if (
            self._up_weights.shape != (self.outputs, self.hidden)
            or self._source_mask.shape != (self.rows, self.hidden)
            or self._table.shape != (self.rows, self.outputs, 1 << bits, 1 << bits)
        ):
            raise FusedGateError("projected material shape differs from public weights")
        self._opening: bytes | None = None
        self._spent = False

    @property
    def material_body_bytes(self) -> int:
        return self._table.nbytes + self._source_mask.nbytes

    def cancel(self) -> None:
        self._spent = True
        self._opening = None
        self._table.fill(0)
        self._source_mask.fill(0)

    def begin(self, hidden_share: np.ndarray) -> bytes:
        if self._spent or self._opening is not None:
            raise FusedGateError("projected material is spent")
        try:
            values = np.asarray(hidden_share)
            if (
                values.dtype != np.uint8 or values.shape != (self.rows, self.hidden)
                or np.any(values >= 1 << self.bits)
            ):
                raise FusedGateError("projected source share has invalid ring or shape")
            masked = (values + self._source_mask) & np.uint8((1 << self.bits) - 1)
            self._opening = FusedOpening(
                self.session_id, self.operation_id, self.graph_digest, self.channel_id,
                self.profile_digest, self.party, self.rows * self.hidden,
                _source_body(masked, self.bits),
            ).pack()
            return self._opening
        except BaseException:
            self.cancel()
            raise

    def finish(self, peer_payload: bytes) -> np.ndarray:
        if self._spent or self._opening is None:
            raise FusedGateError("projected material is unavailable or spent")
        maximum = 256 + self.rows * self.hidden
        try:
            peer = FusedOpening.unpack(peer_payload, maximum_bytes=maximum)
            own = FusedOpening.unpack(self._opening, maximum_bytes=maximum)
            if (
                peer.session_id != self.session_id or peer.operation_id != self.operation_id
                or peer.graph_digest != self.graph_digest or peer.channel_id != self.channel_id
                or peer.profile_digest != self.profile_digest or peer.party != 1 - self.party
                or peer.elements != self.rows * self.hidden
            ):
                raise FusedGateError("projected peer opening has different commitments")
            source = (
                _read_source(own.body, self.bits, own.elements)
                + _read_source(peer.body, self.bits, peer.elements)
            ).reshape(self.rows, self.hidden).astype(np.int32) & ((1 << self.bits) - 1)
            gate = (source @ self._gate_weights.astype(np.int32).T) % (1 << self.bits)
            up = (source @ self._up_weights.astype(np.int32).T) % (1 << self.bits)
            result = self._table[
                np.arange(self.rows)[:, None], np.arange(self.outputs)[None, :], gate, up,
            ].copy()
            self.cancel()
            return result
        except BaseException:
            self.cancel()
            raise


class ProjectedFusedGateDealer:
    """Local test-only issuer: admits full key cost before sharing hidden masks."""

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
            raise FusedGateError("projected reference material cap exceeds 16 MiB per party")
        self.limit = maximum_material_bytes_per_party
        self.issued_bytes_per_party = 0
        self._issued: dict[str, tuple[ProjectedFusedGateParty, ProjectedFusedGateParty]] = {}

    def issue(
        self, operation_id: str, profile: PublicFusedGate,
        gate_weights: np.ndarray, up_weights: np.ndarray,
        *, rows: int, hidden_bound: int,
    ) -> tuple[ProjectedFusedGateParty, ProjectedFusedGateParty]:
        operation_id = _identity(operation_id)
        if operation_id in self._issued or type(profile) is not PublicFusedGate:
            raise FusedGateError("projected issuance identity is repeated or invalid")
        if type(rows) is not int or not 0 < rows <= 8:
            raise FusedGateError("projected row count exceeds bounded reference")
        gate_raw = np.asarray(gate_weights)
        up_raw = np.asarray(up_weights)
        if (
            gate_raw.ndim != 2 or gate_raw.shape != up_raw.shape
            or gate_raw.dtype != np.int8 or up_raw.dtype != np.int8
        ):
            raise FusedGateError("projected weights must be equal-shape signed int8")
        outputs, hidden = gate_raw.shape
        n = 1 << profile.bits
        if not 0 < hidden <= 16 or not 0 < outputs <= 16 or np.any(abs(gate_raw.astype(np.int16)) > 1) or np.any(abs(up_raw.astype(np.int16)) > 1):
            raise FusedGateError("projected weights exceed the toy public bound")
        if (
            type(hidden_bound) is not int or hidden_bound < 0
            or hidden_bound * max(
                int(np.max(np.sum(abs(gate_raw.astype(np.int16)), axis=1))),
                int(np.max(np.sum(abs(up_raw.astype(np.int16)), axis=1))),
            ) >= n // 2
        ):
            raise FusedGateError("projected gate/up could exceed the public Q input domain")
        material = rows * (outputs * n * n * 8 + hidden)
        if material > self.limit - self.issued_bytes_per_party:
            raise FusedGateError("projected full table exceeds material cap before issuance")
        first_mask = np.frombuffer(secrets.token_bytes(rows * hidden), dtype=np.uint8).reshape(rows, hidden).copy() & (n - 1)
        second_mask = np.frombuffer(secrets.token_bytes(rows * hidden), dtype=np.uint8).reshape(rows, hidden).copy() & (n - 1)
        combined = (first_mask + second_mask).astype(np.int32) & (n - 1)
        gate_mask = (combined @ gate_raw.astype(np.int32).T) % n
        up_mask = (combined @ up_raw.astype(np.int32).T) % n
        random_share = np.frombuffer(
            secrets.token_bytes(rows * outputs * n * n * 8), dtype="<u8",
        ).reshape(rows, outputs, n, n)
        indices = np.arange(n, dtype=np.int32)
        shifted_gate = (indices[None, None, :] - gate_mask[:, :, None]) % n
        shifted_up = (indices[None, None, :] - up_mask[:, :, None]) % n
        shifted = profile.table[
            shifted_gate[:, :, :, None], shifted_up[:, :, None, :],
        ]
        other_share = shifted - random_share
        profile_digest = hashlib.sha256(
            b"pllm.projected_fused_gate.v1\0" + profile.fingerprint
            + bytes([profile.bits, hidden_bound]) + bytes([rows, hidden, outputs])
            + gate_raw.tobytes() + up_raw.tobytes()
        ).digest()
        first = ProjectedFusedGateParty(
            self.session_id, operation_id, self.graph_digest, self.channel_id,
            profile_digest, 0, profile.bits, gate_raw, up_raw, first_mask, random_share,
        )
        second = ProjectedFusedGateParty(
            self.session_id, operation_id, self.graph_digest, self.channel_id,
            profile_digest, 1, profile.bits, gate_raw, up_raw, second_mask, other_share,
        )
        self._issued[operation_id] = first, second
        self.issued_bytes_per_party += material
        return first, second

    def cancel(self) -> None:
        for parties in self._issued.values():
            for party in parties:
                party.cancel()
        self._issued.clear()
