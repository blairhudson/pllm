from fractions import Fraction

from scripts.probe_narrow_ingress import (
    algebra_checks, distributions, geometry_projection, lift, privacy_checks,
    repaired_product, tv,
)


def test_signed_byte_lift_and_independent_products():
    result = algebra_checks()
    assert result["signed_byte_lifts"] == 65536
    assert result["tiny_product_outputs"] == 24576
    assert result["distributed_security_proof"] is False


def test_exhaustive_joint_openings_not_marginals_only():
    for quotient_bits in (1, 2, 3):
        views = distributions(bits=3, quotient_bits=quotient_bits)
        assert all(tv(views[0], view) == 0 for view in views)
    bad = distributions(q_domain=range(2))
    assert any(tv(bad[0], view) > 0 for view in bad)
    checks = privacy_checks()
    assert checks["disclosed_carry"]["maximum_tv"] == "7/8"
    assert checks["single_pad_carry_lookup"]["distinct_patterns"] == 8


def test_missing_carry_changes_original_integer():
    x, r = -128, 127
    a, b = lift(x, r)
    assert b == 1
    assert a + r != x
    assert a + r - 256 * b == x


def test_seeded_binary_share_is_not_wide_uniform_share():
    full = distributions(q_domain=range(8))
    binary = distributions(q_domain=range(2))
    assert tv(full[0], full[-1]) == 0
    assert tv(binary[0], binary[-1]) == Fraction(7, 16)


def test_actual_varying_geometry_cost_conservation():
    report = geometry_projection()
    assert len(report["stages"]) == 96
    assert set(report["geometry_stage_width_histogram"]) == {"23", "24", "26"}
    sums = report["historical_46_row_projection"]
    assert sums["byte_input_payload"] + sums["repair_input_payload"] == sums["baseline_input_payload"]
    assert sums["output_fused_net_extra_payload"] == sums["extra_repair_correction_payload"] > 0
    assert sums["fully_fused_net_extra_payload"] == 0
    control = report["measured_control"]
    assert control["pair_bytes"] == 493575247
    assert control["classes"]["inputs"] == 90443518
    assert control["classes"]["public_bundle"] == 124905091


def test_exact_bit_vector_bound_cannot_beat_quotient():
    for w0 in range(-128, 128):
        for w1 in range(-128, 128):
            positive = max(w0, 0) + max(w1, 0)
            negative = -min(w0, 0) - min(w1, 0)
            d = max(1, (max(positive + 1, negative) - 1).bit_length() + 1)
            h = max(1, (127 * (abs(w0) + abs(w1))).bit_length() + 1)
            assert d >= h - 8
            total = abs(w0) + abs(w1)
            unsigned_d = total.bit_length()
            assert unsigned_d >= h - 8
            assert 127 * total < (1 << (unsigned_d + 7))
            assert (unsigned_d == 0) == (total == 0)
            for b0, b1 in ((0, 0), (0, 1), (1, 0), (1, 1)):
                product = w0 * b0 + w1 * b1
                assert -(1 << (d - 1)) <= product < (1 << (d - 1))
                shifted = product + negative
                assert 0 <= shifted <= total
                assert shifted < (1 << unsigned_d)
