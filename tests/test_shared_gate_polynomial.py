"""Exact 24-bit one-use shares of Q7 quadratic gated-product numerator."""

from __future__ import annotations

import hashlib
import secrets

import numpy as np
import pytest

from pllm.runtime.shared_gate_polynomial import QuadraticQ7Dealer, QuadraticQ7Gate
from pllm.runtime.shared_gate_reference import FusedGateError, FusedOpening


_MASK = (1 << 24) - 1


def _digest(text: str) -> bytes:
    return hashlib.sha256(text.encode()).digest()


def _dealer(cap: int = 16 << 20) -> QuadraticQ7Dealer:
    return QuadraticQ7Dealer(
        "session", _digest("graph"), _digest("peer channel"),
        maximum_material_bytes_per_party=cap,
    )


@pytest.mark.parametrize("rows,hidden,outputs", [(1, 1, 1), (3, 7, 9), (16, 16, 12)])
def test_polynomial_masked_shared_gate_exact_numerator(rows: int, hidden: int, outputs: int) -> None:
    rng = np.random.default_rng(rows * hidden + outputs)
    h = rng.integers(-4, 5, size=(rows, hidden), dtype=np.int64)
    gate = rng.integers(-1, 2, size=(outputs, hidden), dtype=np.int8)
    up = rng.integers(-1, 2, size=(outputs, hidden), dtype=np.int8)
    profile = QuadraticQ7Gate()
    weight_bound = max(
        int(np.max(np.sum(np.abs(gate.astype(np.int16)), axis=1))),
        int(np.max(np.sum(np.abs(up.astype(np.int16)), axis=1))),
    )
    assert weight_bound * 4 <= 127
    issuer = _dealer()
    parties = issuer.issue("mlp.gated", gate, up, rows=rows, hidden_bound=4)
    expected_material = 3 * (rows * hidden + 5 * rows * outputs)
    assert issuer.issued_bytes_per_party == expected_material
    assert [party.material_body_bytes for party in parties] == [expected_material] * 2

    left = np.frombuffer(secrets.token_bytes(3 * h.size), dtype=np.uint8).reshape(-1, 3).astype(np.uint32)
    left = (left[:, 0] | left[:, 1] << 8 | left[:, 2] << 16).reshape(h.shape)
    right = ((h - left.astype(np.int64)) & _MASK).astype(np.uint32)
    opening0 = parties[0].begin(left)
    opening1 = parties[1].begin(right)
    assert len(FusedOpening.unpack(opening0, maximum_bytes=1024).body) == 3 * h.size
    actual_ring = (parties[0].finish(opening1) + parties[1].finish(opening0)) & _MASK
    actual = actual_ring.astype(np.int64)
    actual[actual >= 1 << 23] -= 1 << 24
    reference_g = h @ gate.astype(np.int64).T
    reference_u = h @ up.astype(np.int64).T
    expected = np.array(
        [
            [profile.numerator(int(g), int(u)) for g, u in zip(gate_row, up_row, strict=True)]
            for gate_row, up_row in zip(reference_g, reference_u, strict=True)
        ], dtype=np.int64,
    )
    np.testing.assert_array_equal(actual, expected)
    with pytest.raises(FusedGateError, match="spent"):
        parties[0].finish(opening1)
    with pytest.raises(FusedGateError, match="replay"):
        issuer.issue("mlp.gated", gate, up, rows=rows, hidden_bound=4)


def test_polynomial_q7_domain_and_signed_24_bit_bound() -> None:
    profile = QuadraticQ7Gate()
    assert profile.numerator(127, 127) == (256 * 127 + 127**2) * 127
    assert -1 << 23 < profile.numerator(-128, -128) < 1 << 23
    with pytest.raises(FusedGateError, match=r"\[-128, 127\]"):
        profile.numerator(128, 0)
    # The shared numerator is not the existing two-step rounded Q7 composite.
    assert round(profile.numerator(1, 127) / 65536) == 0
    assert round(round((256 * 1 + 1) / 512) * 127 / 128) == 1


def test_polynomial_ring_wrap_matches_independent_extreme_q7_gate_and_up() -> None:
    profile = QuadraticQ7Gate()
    hidden = np.array(
        [
            [-127, -127], [-127, 127], [127, -127], [127, 127],
            [-126, -1], [-1, -126], [1, 127], [0, -127],
        ], dtype=np.int64,
    )
    gate = np.array([[1, 0]], dtype=np.int8)
    up = np.array([[0, 1]], dtype=np.int8)
    parties = _dealer().issue("boundary", gate, up, rows=hidden.shape[0], hidden_bound=127)
    first = np.frombuffer(secrets.token_bytes(3 * hidden.size), dtype=np.uint8).reshape(-1, 3).astype(np.uint32)
    first = (first[:, 0] | first[:, 1] << 8 | first[:, 2] << 16).reshape(hidden.shape)
    second = ((hidden - first.astype(np.int64)) & _MASK).astype(np.uint32)
    frames = parties[0].begin(first), parties[1].begin(second)
    encoded = (parties[0].finish(frames[1]) + parties[1].finish(frames[0])) & _MASK
    values = encoded.astype(np.int64).reshape(-1)
    values[values >= 1 << 23] -= 1 << 24
    expected = np.array(
        [profile.numerator(int(g), int(u)) for g, u in hidden], dtype=np.int64,
    )
    np.testing.assert_array_equal(values, expected)


def test_polynomial_tampering_and_cancel_burn_material() -> None:
    weight = np.array([[1, 0, -1]], dtype=np.int8)
    parties = _dealer().issue("layer", weight, weight, rows=1, hidden_bound=2)
    parties[0].begin(np.array([[1, 0, 1]], dtype=np.uint32))
    peer = FusedOpening.unpack(
        parties[1].begin(np.array([[0, 1, 0]], dtype=np.uint32)), maximum_bytes=512,
    )
    forged = FusedOpening(
        peer.session_id, peer.operation_id, peer.graph_digest, _digest("wrong channel"),
        peer.profile_digest, peer.party, peer.elements, peer.body,
    ).pack()
    with pytest.raises(FusedGateError, match="commitments"):
        parties[0].finish(forged)
    with pytest.raises(FusedGateError, match="spent"):
        parties[0].finish(peer.pack())
    parties[1].cancel()
    with pytest.raises(FusedGateError, match="spent"):
        parties[1].finish(b"invalid")


def test_polynomial_aggregate_preflight_before_sampling(monkeypatch: pytest.MonkeyPatch) -> None:
    issuer = _dealer(77)
    weights = np.array([[1, 0, 1], [1, -1, 0]], dtype=np.int8)
    sampled = False

    def forbid_sampling(_size: int) -> bytes:
        nonlocal sampled
        sampled = True
        raise AssertionError("one-use dealer must preflight before sampling")

    monkeypatch.setattr("pllm.runtime.shared_gate_polynomial.secrets.token_bytes", forbid_sampling)
    with pytest.raises(FusedGateError, match="before issuance"):
        issuer.issue("op", weights, weights, rows=2, hidden_bound=3)  # 3×(6+5×4)=78
    assert not sampled and issuer.issued_bytes_per_party == 0
    with pytest.raises(FusedGateError, match="range bound"):
        issuer.issue("bad", weights, weights, rows=1, hidden_bound=127)
    assert not sampled
