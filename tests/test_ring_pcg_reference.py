import pytest

from pllm.runtime.ring_pcg_reference import (
    RingPcgReferenceDealer,
    RingPcgReferenceError,
)


@pytest.mark.parametrize("digits,ring_bits", [(1, 8), (2, 8), (2, 64)])
def test_four_sparse_cross_terms_expand_into_party_local_oles(
    digits: int, ring_bits: int,
) -> None:
    modulus = 1 << ring_bits
    first = (((0, (modulus - 1, 1)), (1, (2, 3))), ((2, (4, 1)),))
    second = (((2, (7, 2)), (0, (1, 5))), ((1, (3, 6)),))
    dealer = RingPcgReferenceDealer(f"ring-{digits}-{ring_bits}", digits=digits, ring_bits=ring_bits)
    worker0, worker1 = dealer.issue("ole", first, second)
    result0 = worker0.expand_once()
    result1 = worker1.expand_once()
    assert len(result0[0]) == len(result1[0]) == 3**digits
    for index in range(3**digits):
        a, b = result0[0][index], result1[0][index]
        c, d = result0[1][index], result1[1][index]
        assert (
            (a[0] * b[0] - a[1] * b[1]) % modulus,
            (a[0] * b[1] + a[1] * b[0] - a[1] * b[1]) % modulus,
        ) == ((c[0] + d[0]) % modulus, (c[1] + d[1]) % modulus)
    with pytest.raises(RingPcgReferenceError, match="already expanded"):
        worker0.expand_once()
    with pytest.raises(RingPcgReferenceError, match="already issued"):
        dealer.issue("ole", first, second)


@pytest.mark.parametrize("digits,ring_bits", [(1, 8), (2, 16), (2, 64)])
def test_trace_variant_extracts_two_base_ring_ole_lanes_per_point(
    digits: int, ring_bits: int,
) -> None:
    modulus = 1 << ring_bits
    first = (((0, (modulus - 1, 1)), (1, (2, 3))), ((2, (4, 1)),))
    second = (((2, (7, 2)), (0, (1, 5))), ((1, (3, 6)),))
    dealer = RingPcgReferenceDealer(f"trace-{digits}-{ring_bits}", digits=digits, ring_bits=ring_bits)
    worker0, worker1 = dealer.issue_base_ring("trace-ole", first, second)
    x0, z0 = worker0.expand_base_ring_once()
    x1, z1 = worker1.expand_base_ring_once()
    assert len(x0) == len(x1) == 2 * 3**digits
    for lane in range(2 * 3**digits):
        assert x0[lane] * x1[lane] % modulus == (z0[lane] + z1[lane]) % modulus
    assert x0[:3**digits] != x0[3**digits:]
    with pytest.raises(RingPcgReferenceError, match="already expanded"):
        worker0.expand_base_ring_once()


def test_trace_key_budget_includes_second_frobenius_cross_term_set() -> None:
    dealer = RingPcgReferenceDealer("trace-budget", digits=1, ring_bits=64)
    first = (((0, (1, 0)),), ((1, (2, 0)),))
    second = (((1, (0, 1)),), ((0, (1, 0)),))
    with pytest.raises(RingPcgReferenceError, match="budget exceeded before issuance"):
        dealer.issue_base_ring("trace", first, second, max_key_bytes_per_party=4 * 83)
    party0, party1 = dealer.issue_base_ring("trace", first, second)
    assert party0.key_bytes == party1.key_bytes == 8 * (16 + 17 * 2 + 16)


def test_ring_pcg_cost_gate_precedes_any_key_issuance() -> None:
    dealer = RingPcgReferenceDealer("budget", digits=2, ring_bits=64)
    first = (((0, (1, 0)),), ((1, (2, 0)),))
    second = (((1, (0, 1)),), ((0, (1, 0)),))
    with pytest.raises(RingPcgReferenceError, match="budget exceeded before issuance"):
        dealer.issue("try", first, second, max_key_bytes_per_party=1)
    pair = dealer.issue("try", first, second)
    assert pair[0].key_bytes == pair[1].key_bytes == 4 * (16 + 17 * 4 + 16)
    pair[0].cancel()
    with pytest.raises(RingPcgReferenceError, match="cancelled"):
        pair[0].expand_once()
    pair[1].cancel()


@pytest.mark.parametrize(
    "bad",
    [(((0, (1, 0)), (0, (1, 0))), ((1, (1, 0)),)),
     (((3, (1, 0)),), ((1, (1, 0)),)),
     (((0, (0, 0)),), ((1, (1, 0)),))],
)
def test_ring_pcg_rejects_duplicate_missing_and_zero_terms(bad: tuple) -> None:
    dealer = RingPcgReferenceDealer("invalid", digits=1, ring_bits=8)
    good = (((0, (1, 0)),), ((1, (1, 0)),))
    with pytest.raises(RingPcgReferenceError, match="invalid|unique"):
        dealer.issue("bad", bad, good)
