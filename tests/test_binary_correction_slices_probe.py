"""Independent bounded controls, not parameter/security certification."""
from collections import Counter
from fractions import Fraction
from itertools import product

import pytest

from scripts import probe_binary_correction_slices as probe


def test_direct_binary_oracle():
    for w in product(range(-2, 2), repeat=2):
        for a in product((0, 1), repeat=4):
            matrix = [list(a[:2]), list(a[2:])]
            for u, e in product(product((0, 1), repeat=2), repeat=2):
                low, image = probe.binary_correction([list(w)], matrix, u, e)
                oracle_low = [((a[2*i] * u[0]) ^ (a[2*i+1] * u[1]) ^ e[i]) for i in range(2)]
                assert low == oracle_low
                assert image == [(w[0] * low[0] + w[1] * low[1]) & 1]


def test_signed_i8_endpoints_heterogeneous_output_widths():
    w = [[-128, 127], [127, -128]]
    for x in product((-128, -1, 0, 127), repeat=2):
        for low in product((0, 1), repeat=2):
            r, s, q = probe.join_correction(w, low, [255, 128], [1, 0], [17, 33], 17)
            d = [(x[i] - r[i]) % (1 << 17) for i in range(2)]
            for j, width in enumerate((16, 17)):
                assert (sum(w[j][i] * d[i] for i in range(2)) + q[j] + s[j]) % (1 << width) == sum(w[j][i] * x[i] for i in range(2)) % (1 << width)


def test_carry_and_borrow_independent_oracles():
    _, s, q = probe.join_correction([[1, 1]], [1, 1], [0, 0], [0], [0], 3)
    assert s == [0] and q == [2]  # Naive W*high gives zero.
    _, s, q = probe.join_correction([[1]], [0], [0], [1], [0], 3)
    assert s == [1] and q == [7]  # q_high=3, not zero.


def test_joint_view_and_custody():
    base = probe.toy_view((0, 0))
    assert len(base) == 512 and len(set(base.values())) == 1
    for x in ((0, 1), (1, 0), (7, 7)):
        assert probe.tv(base, probe.toy_view(x)) == 0
    assert probe.tv(probe.toy_view((0, 0), noise="biased"), probe.toy_view((1, 0), noise="biased")) == Fraction(1, 4)
    assert probe.tv(probe.toy_view((0, 0), disclose_low=True), probe.toy_view((1, 0), disclose_low=True)) == 1
    assert probe.tv(probe.toy_view((0, 0), disclose_high_share=True), probe.toy_view((1, 0), disclose_high_share=True)) > 0


def test_reuse_and_subspace_annihilator():
    # Reused full mask discloses exact ring difference on every input pair.
    for x0, x1, r in product(range(8), repeat=3):
        assert (((x1-r) % 8) - ((x0-r) % 8)) % 8 == (x1-x0) % 8
    # A=(1,1), reused e: dual vector (1,1) cancels every fresh u.
    for e, u0, u1, x0, x1 in product(product((0, 1), repeat=2), range(2), range(2), product((0, 1), repeat=2), product((0, 1), repeat=2)):
        observed = [x0[i] ^ u0 ^ e[i] ^ x1[i] ^ u1 ^ e[i] for i in range(2)]
        assert observed[0] ^ observed[1] == x0[0] ^ x0[1] ^ x1[0] ^ x1[1]
    # No noise reveals input parity even for fresh u.
    distributions = []
    for x in ((0, 0), (1, 0)):
        distributions.append(Counter(tuple(v ^ u for v in x) for u in (0, 1)))
    assert probe.tv(*distributions) == 1


def test_fresh_two_query_full_views_independent():
    # Separate direct scalar oracle: no join_correction or provider_view calls.
    def single(x):
        counts = Counter()
        for r, high_share, qlow in product(range(4), range(2), range(2)):
            slow = ((-r) - qlow) % 2
            s = slow + 2 * high_share
            q = (-r - s) % 4
            counts[((x-r) % 4, q)] += 1
        return counts

    def pair(x0, x1):
        return Counter({v0 + v1: n0 * n1 for v0, n0 in single(x0).items()
                        for v1, n1 in single(x1).items()})

    base = pair(0, 0)
    assert len(base) == 256
    for x0, x1 in product(range(4), repeat=2):
        assert pair(x0, x1) == base


def test_frame_rounding_and_horizon_cost():
    for bits in range(1, 65):
        for lanes in range(1, bits+1):
            oracle = len(bytes((bits+7)//8)) - len(bytes((bits-lanes+7)//8))
            assert probe.split_saving(bits, lanes) == oracle
            assert oracle <= probe.packed(lanes)
    assert probe.finite_cost(2, 3, 4, 1, 11, 13, 17) == {"rows": 7, "pools": 2, "unused": 1, "extra_bytes": 156}
    assert probe.finite_cost(1, 4, 4, 1, 0, 13, 0)["pools"] == 2
    with pytest.raises(ValueError):
        probe.finite_cost(1, 1, 0, 0, 0, 0, 0)
    control = {"public_delivery_bytes": 20, "covered_pair_bytes": 100}
    assert probe.budget(control, 80, 2, Fraction(1, 5), 7) == -23


def test_pinned_budget_and_drift(tmp_path):
    source = probe.json.loads((probe.ROOT / next(iter(probe.LOCKS))).read_text())
    costs = probe.cost_report(source)
    assert [r["optimistic_one_bit_removed_bytes"] for r in costs["bulk_controls"]] == [4195200, 4195200, 8054784, 8054784]
    assert costs["bulk_binary_w_publication_bytes"] == 25059328
    for case in costs["bulk_controls"]:
        assert Fraction(case["warm_zero_cost_saving_fraction"]) < Fraction(1, 50)
        assert all(Fraction(h["20pct_extra_ceiling"]) < 0 for h in case["horizons"])
    for path in probe.LOCKS:
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}")
    with pytest.raises(ValueError, match="source hash drift"):
        probe.report(tmp_path)
