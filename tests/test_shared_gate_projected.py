"""Correlated hidden mask avoids repeated intermediate-width peer openings."""

from __future__ import annotations

import hashlib
import secrets

import numpy as np
import pytest

from pllm.runtime.shared_gate_projected import ProjectedFusedGateDealer
from pllm.runtime.shared_gate_reference import FusedGateError, FusedOpening, PublicFusedGate


def _digest(value: str) -> bytes:
    return hashlib.sha256(value.encode()).digest()


@pytest.mark.parametrize("bits", [4, 8])
def test_two_parties_reuse_one_masked_hidden_across_gate_and_up(bits: int) -> None:
    profile = PublicFusedGate.silu_product(bits)
    gate = np.array([[1, 1, 0], [-1, 0, 1]], dtype=np.int8)
    up = np.array([[0, -1, 1], [1, 0, 1]], dtype=np.int8)
    hidden = np.array([[-2, 1, 2], [1, -2, 0]], dtype=np.int16)
    n = 1 << bits
    mask = np.frombuffer(secrets.token_bytes(hidden.size), dtype=np.uint8).reshape(hidden.shape) & (n - 1)
    shares = mask, ((hidden - mask.astype(np.int16)) % n).astype(np.uint8)
    dealer = ProjectedFusedGateDealer(
        "session", _digest("graph"), _digest("channel"),
        maximum_material_bytes_per_party=4 << 20,
    )
    parties = dealer.issue("layer.gated", profile, gate, up, rows=2, hidden_bound=2)
    openings = tuple(party.begin(share) for party, share in zip(parties, shares, strict=True))
    assert len(openings[0]) == len(openings[1])
    assert dealer.issued_bytes_per_party == parties[0].material_body_bytes
    output = (
        parties[0].finish(openings[1]) + parties[1].finish(openings[0])
    ).view(np.int64)
    reference_gate = hidden @ gate.astype(np.int16).T
    reference_up = hidden @ up.astype(np.int16).T
    expected = np.array(
        [
            [profile.value(int(a), int(b)) for a, b in zip(row_g, row_u, strict=True)]
            for row_g, row_u in zip(reference_gate, reference_up, strict=True)
        ], dtype=np.int64,
    )
    np.testing.assert_array_equal(output, expected)
    raw = sum(len(FusedOpening.unpack(body, maximum_bytes=1024).body) for body in openings)
    assert raw == (hidden.size if bits == 4 else 2 * hidden.size)
    assert raw < hidden.shape[0] * gate.shape[0] * (2 if bits == 4 else 4)
    for party in parties:
        with pytest.raises(FusedGateError, match="spent"):
            party.begin(shares[party.party])


def test_projected_session_replay_and_tampered_peer_burn_material() -> None:
    profile = PublicFusedGate.silu_product(4)
    weights = np.array([[1, 1]], dtype=np.int8)
    dealer = ProjectedFusedGateDealer(
        "session", _digest("graph"), _digest("channel"),
        maximum_material_bytes_per_party=1 << 20,
    )
    first, second = dealer.issue("op", profile, weights, weights, rows=1, hidden_bound=1)
    first.begin(np.array([[1, 0]], dtype=np.uint8))
    peer = FusedOpening.unpack(
        second.begin(np.array([[0, 2]], dtype=np.uint8)), maximum_bytes=1024,
    )
    forged = FusedOpening(
        peer.session_id, peer.operation_id, peer.graph_digest, _digest("forged"),
        peer.profile_digest, peer.party, peer.elements, peer.body,
    ).pack()
    with pytest.raises(FusedGateError, match="commitments"):
        first.finish(forged)
    with pytest.raises(FusedGateError, match="spent"):
        first.finish(peer.pack())
    second.cancel()
    with pytest.raises(FusedGateError, match="spent"):
        second.finish(b"invalid")
    with pytest.raises(FusedGateError, match="repeated"):
        dealer.issue("op", profile, weights, weights, rows=1, hidden_bound=1)


def test_projected_key_cap_and_input_domain_fail_before_sampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = PublicFusedGate.silu_product(8)
    dealer = ProjectedFusedGateDealer(
        "session", _digest("graph"), _digest("channel"),
        maximum_material_bytes_per_party=1 << 20,
    )
    weights = np.ones((3, 4), dtype=np.int8)
    sampled = False

    def forbid_sampling(_length: int) -> bytes:
        nonlocal sampled
        sampled = True
        raise AssertionError("resource gate must precede sampling")

    monkeypatch.setattr("pllm.runtime.shared_gate_projected.secrets.token_bytes", forbid_sampling)
    with pytest.raises(FusedGateError, match="before issuance"):
        dealer.issue("expensive", profile, weights, weights, rows=1, hidden_bound=1)
    with pytest.raises(FusedGateError, match="public Q input domain"):
        dealer.issue("domain", profile, weights, weights, rows=1, hidden_bound=32)
    assert not sampled and dealer.issued_bytes_per_party == 0


def test_projected_terminal_padding_failure_burns_key() -> None:
    profile = PublicFusedGate.silu_product(4)
    weights = np.array([[1, 0, 0]], dtype=np.int8)
    dealer = ProjectedFusedGateDealer(
        "session", _digest("graph"), _digest("channel"),
        maximum_material_bytes_per_party=1 << 20,
    )
    first, second = dealer.issue("op", profile, weights, weights, rows=1, hidden_bound=1)
    first.begin(np.array([[1, 0, 0]], dtype=np.uint8))
    peer = FusedOpening.unpack(
        second.begin(np.array([[0, 1, 0]], dtype=np.uint8)), maximum_bytes=1024,
    )
    forged = FusedOpening(
        peer.session_id, peer.operation_id, peer.graph_digest, peer.channel_id,
        peer.profile_digest, peer.party, peer.elements,
        peer.body[:-1] + bytes([peer.body[-1] | 0xf0]),
    ).pack()
    with pytest.raises(FusedGateError, match="terminal padding"):
        first.finish(forged)
    with pytest.raises(FusedGateError, match="spent"):
        first.finish(peer.pack())
