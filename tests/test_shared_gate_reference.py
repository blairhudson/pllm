"""One-use, party-local dense fused gate: numeric and admission checks."""

from __future__ import annotations

import hashlib
import secrets

import msgpack
import numpy as np
import pytest

from pllm.runtime.shared_gate_reference import (
    FusedGateDealer,
    FusedGateError,
    FusedOpening,
    PublicFusedGate,
)


def _digest(value: str) -> bytes:
    return hashlib.sha256(value.encode()).digest()


def _shares(values: np.ndarray, bits: int) -> tuple[np.ndarray, np.ndarray]:
    n = 1 << bits
    first = np.frombuffer(secrets.token_bytes(values.size), dtype=np.uint8).copy() & (n - 1)
    second = (values.astype(np.uint8) - first) & (n - 1)
    return first, second


@pytest.mark.parametrize("bits,elements", [(4, 32), (8, 4)])
def test_fused_gate_hidden_two_inputs_exact_and_one_use(bits: int, elements: int) -> None:
    profile = PublicFusedGate.silu_product(bits)
    rng = np.random.default_rng(256 + bits)
    n = 1 << bits
    gates = rng.integers(-n // 2, n // 2, size=elements, dtype=np.int64)
    ups = rng.integers(-n // 2, n // 2, size=elements, dtype=np.int64)
    gate_shares = _shares(gates, bits)
    up_shares = _shares(ups, bits)
    issuer = FusedGateDealer(
        "session", _digest("graph"), _digest("channel"),
        maximum_material_bytes_per_party=4 << 20,
    )
    first, second = issuer.issue("mlp.fused", profile, elements=elements)
    assert first.material_body_bytes == second.material_body_bytes == issuer.issued_bytes_per_party
    left = first.begin(gate_shares[0], up_shares[0])
    right = second.begin(gate_shares[1], up_shares[1])
    lframe = FusedOpening.unpack(left, maximum_bytes=1024)
    rframe = FusedOpening.unpack(right, maximum_bytes=1024)
    assert lframe.body != gates.astype(np.uint8).tobytes()
    assert rframe.body != ups.astype(np.uint8).tobytes()
    outputs = first.finish(right), second.finish(left)
    actual = (outputs[0] + outputs[1]).view(np.int64)
    expected = np.array(
        [profile.value(int(g), int(u)) for g, u in zip(gates, ups, strict=True)],
        dtype=np.int64,
    )
    np.testing.assert_array_equal(actual, expected)
    assert len(left) + len(right) >= elements * (2 if bits == 4 else 4)
    with pytest.raises(FusedGateError, match="spent"):
        first.finish(right)
    with pytest.raises(FusedGateError, match="spent"):
        second.begin(gate_shares[1], up_shares[1])
    with pytest.raises(FusedGateError, match="issuance"):
        issuer.issue("mlp.fused", profile, elements=elements)


def test_fused_gate_public_q7_table_is_bounded_not_whole_decoder_silu() -> None:
    profile = PublicFusedGate.silu_product(8)
    assert profile.table.shape == (256, 256)
    assert len(profile.fingerprint) == 32
    assert profile.value(0, 127) == 0
    assert profile.value(127, 127) == round(127 * 127 / (1 + np.exp(-127 / 128)) / 128)
    with pytest.raises(FusedGateError, match="outside"):
        profile.value(128, 0)


def test_fused_gate_mismatched_frame_burns_only_local_material() -> None:
    profile = PublicFusedGate.silu_product(4)
    issuer = FusedGateDealer(
        "session", _digest("graph"), _digest("channel"),
        maximum_material_bytes_per_party=1 << 20,
    )
    first, second = issuer.issue("gate", profile, elements=1)
    first.begin(np.array([7], dtype=np.uint8), np.array([6], dtype=np.uint8))
    frame = FusedOpening.unpack(
        second.begin(np.array([1], dtype=np.uint8), np.array([2], dtype=np.uint8)),
        maximum_bytes=1024,
    )
    forged = FusedOpening(
        frame.session_id, frame.operation_id, frame.graph_digest,
        _digest("wrong-channel"), frame.profile_digest, frame.party,
        frame.elements, frame.body,
    ).pack()
    with pytest.raises(FusedGateError, match="commitments"):
        first.finish(forged)
    with pytest.raises(FusedGateError, match="spent"):
        first.finish(frame.pack())
    second.cancel()
    with pytest.raises(FusedGateError, match="spent"):
        second.finish(b"invalid")


def test_fused_gate_allocation_preflights_full_party_inventory(monkeypatch: pytest.MonkeyPatch) -> None:
    issuer = FusedGateDealer(
        "session", _digest("graph"), _digest("channel"),
        maximum_material_bytes_per_party=2049,
    )
    profile = PublicFusedGate.silu_product(4)
    called = False

    def forbid_sampling(_size: int) -> bytes:
        nonlocal called
        called = True
        raise AssertionError("preflight must precede sampling")

    monkeypatch.setattr("pllm.runtime.shared_gate_reference.secrets.token_bytes", forbid_sampling)
    with pytest.raises(FusedGateError, match="before issuance"):
        issuer.issue("gate", profile, elements=1)  # 16²×8 + two mask bytes
    assert not called
    assert issuer.issued_bytes_per_party == 0


def test_fused_opening_rejects_extra_fields_and_oversize() -> None:
    forged = msgpack.packb([1, "a", "b", _digest("g"), _digest("c"), _digest("p"), 0, 1, b"\x00", 42])
    with pytest.raises(FusedGateError, match="invalid"):
        FusedOpening.unpack(forged, maximum_bytes=512)
    with pytest.raises(FusedGateError, match="body limit"):
        FusedOpening.unpack(forged, maximum_bytes=5)
    with pytest.raises(FusedGateError, match="invalid"):
        FusedOpening.unpack(b"\x91", maximum_bytes=512)


def test_public_profile_is_immutable_and_rejects_forged_table() -> None:
    profile = PublicFusedGate.silu_product(4)
    with pytest.raises(ValueError):
        profile.table[0, 0] = 2
    table = profile.table.copy()
    table[0, 0] += np.uint64(1)
    with pytest.raises(FusedGateError, match="fingerprint"):
        PublicFusedGate(4, table, profile.fingerprint)
    with pytest.raises(FusedGateError, match="width or shape"):
        PublicFusedGate(4, table[:1], profile.fingerprint)
