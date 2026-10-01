import pytest

from pllm.runtime.shared_dpf import (
    DpfReferenceError,
    PointKeyDealer,
    pack_bit_shares,
    unpack_bit_shares,
)


@pytest.mark.parametrize("bits", [1, 4, 10])
def test_two_party_point_comparison_with_boundary_and_middle_points(bits: int) -> None:
    domain = 1 << bits
    for point in (0, domain // 2, domain - 1):
        for query in sorted({0, point - 1, point, point + 1, domain - 1}):
            if not 0 <= query < domain:
                continue
            keys = PointKeyDealer(f"test-{bits}-{point}-{query}").issue(
                "compare", point, bits=bits
            )
            assert keys[0].key_bytes == keys[1].key_bytes == 17 + 17 * bits
            if (point + query) % 2 == 0:
                for key in keys:
                    key.prepare()
                    assert key.expanded_bytes == (domain + 7) // 8
            shares = tuple(key.less_than_share(query) for key in keys)
            assert shares[0] ^ shares[1] == int(query < point)
            assert all(key.expanded_bytes == 0 for key in keys)
            with pytest.raises(DpfReferenceError, match="already consumed"):
                keys[0].less_than_share(query)


def test_point_key_fails_closed_on_bad_query_and_reissued_gate() -> None:
    dealer = PointKeyDealer("session")
    first, second = dealer.issue("gate", 3, bits=4)
    with pytest.raises(DpfReferenceError, match="out of range"):
        first.less_than_share(16)
    with pytest.raises(DpfReferenceError, match="already consumed"):
        first.less_than_share(0)
    second.cancel()
    with pytest.raises(DpfReferenceError, match="already consumed"):
        second.less_than_share(0)
    with pytest.raises(DpfReferenceError, match="already issued"):
        dealer.issue("gate", 3, bits=4)
    with pytest.raises(DpfReferenceError, match="reference limit"):
        dealer.issue("too-wide", 1, bits=11)


def test_small_domain_all_hidden_points_and_public_queries() -> None:
    dealer = PointKeyDealer("exhaustive")
    for point in range(1 << 4):
        for query in range(1 << 4):
            keys = dealer.issue(f"{point}.{query}", point, bits=4)
            assert keys[0].less_than_share(query) ^ keys[1].less_than_share(query) == int(
                query < point
            )


def test_bit_shares_reject_padding_and_trailing_bytes() -> None:
    original = (0, 1, 1, 0, 1, 0, 0, 1, 1)
    payload = pack_bit_shares(original)
    assert unpack_bit_shares(payload, len(original)) == original
    with pytest.raises(DpfReferenceError, match="padding"):
        unpack_bit_shares(payload[:-1] + b"\x03", len(original))
    with pytest.raises(DpfReferenceError, match="padding"):
        unpack_bit_shares(payload + b"\x00", len(original))


@pytest.mark.parametrize("bits,ring_bits", [(3, 8), (4, 16), (5, 64)])
def test_additive_point_keys_generate_only_the_requested_ring_coordinate(
    bits: int, ring_bits: int,
) -> None:
    dealer = PointKeyDealer(f"ring-{bits}-{ring_bits}")
    modulus = 1 << ring_bits
    for point in (0, 1, (1 << bits) - 1):
        payload = (modulus - 1, 27)
        pair = dealer.issue_ring(f"point-{point}", point, payload, bits=bits, ring_bits=ring_bits)
        shares = [key.eval_all_once(1 << bits) for key in pair]
        for index in range(1 << bits):
            assert (
                (shares[0][index][0] + shares[1][index][0]) % modulus,
                (shares[0][index][1] + shares[1][index][1]) % modulus,
            ) == (payload if index == point else (0, 0))
        with pytest.raises(DpfReferenceError, match="consumed"):
            pair[0].eval_all_once(1)
    with pytest.raises(DpfReferenceError, match="already issued"):
        dealer.issue("point-0", 0, bits=bits)
