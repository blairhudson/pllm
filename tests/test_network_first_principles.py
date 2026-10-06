"""Independent algebra, domain and privacy gates for non-selectable hypotheses."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from network_first_principles import (  # noqa: E402
    batch_inverse,
    coset_oracle,
    hasse_jet,
    hermite_oracle,
    koopman_lift,
    koopman_oracle,
    masked_inverse_oracle,
    permutation_oracle,
    ramp_oracle,
    rational_silu,
    rational_silu_tails,
    silu32,
    solve_field,
    span_oracle,
    sparse_syndrome_decode,
    wallace_counts,
    wallace_evaluate,
    zech_oracle,
)


def test_coset_coding_needs_an_independent_domain_certificate():
    result = coset_oracle()
    assert result["nonzero_domain_words_checked"] == 8192
    assert result["component_symbol_reduction"] == 16
    assert result["two_sparse_aliases_one_sparse"]
    assert not result["domain_membership_certified_by_syndrome"]
    with pytest.raises(ValueError, match="outside"):
        sparse_syndrome_decode([0, 1])


def test_span_reconstruction_and_inconsistent_membership():
    assert span_oracle()["exact_field_outputs"]
    assert solve_field([[1, 0], [0, 1], [1, 1]], [2, 3, 6]) is None
    np.testing.assert_array_equal(solve_field([[0, 1], [0, 2]], [3, 6]), [0, 3])


def test_continued_fraction_is_close_but_not_original_numeric_identity():
    x = np.linspace(-32, 32, 32769, dtype=np.float32)
    reference, rational = silu32(x), rational_silu(x)
    assert np.max(np.abs(reference.astype(np.float64) - rational)) < 0.00001
    assert np.any(reference.view(np.uint32) != rational.view(np.uint32))
    assert np.all(np.isfinite(rational))


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf, -32.01, 32.01])
def test_rational_has_no_private_range_extension_or_fallback(invalid):
    with pytest.raises(ValueError, match="domain"):
        rational_silu([invalid])


def test_batch_inverse_matches_independent_scalar_pow_and_rejects_zero():
    values = np.random.default_rng(4905).integers(1, 65537, 1024, dtype=np.int64)
    np.testing.assert_array_equal(batch_inverse(values), [pow(int(x), -1, 65537) for x in values])
    with pytest.raises(ValueError, match="nonzero"):
        batch_inverse([1, 0, 3])


def test_masked_inverse_checks_nonzero_views_and_exact_results():
    result = masked_inverse_oracle()
    assert result["denominators_checked"] == 4096
    assert result["exhaustive_nonzero_mask_views_per_checked_denominator"] == 65536
    assert result["nonzero_domain_only"]


def test_explicit_tail_profile_preserves_its_separate_approximation_contract():
    x = np.array([-1000, -104, -33, -32, 0, 32, 33, 1000], np.float32)
    value = rational_silu_tails(x)
    assert np.max(np.abs(value.astype(np.float64) - silu32(x))) < 0.000002
    np.testing.assert_array_equal(value[x > 32], x[x > 32])
    assert np.all(value[x < -32] == 0)
    assert np.all(np.signbit(value[x < -32]))
    with pytest.raises(ValueError):
        rational_silu_tails([np.inf])


def test_ramp_gate_resharing_and_one_worker_view():
    result = ramp_oracle(33)
    assert result["exact_gate_and_reshare"] and result["one_worker_view_exhausted"]
    for row in result["layouts"]:
        assert row["public_linear_work_vs_two_offset"] > 1
        assert row["one_gated_product_per_element_peer_bytes"] > 33 * 4


def test_permutation_blinding_telescopes_without_claiming_branch_privacy():
    result = permutation_oracle()
    assert result["and_truth_table_and_telescoping"]
    assert result["recursive_formula_compiler"][2]["two_branch_16byte_objects_bytes"] == 32 * 4**12


def test_koopman_operator_closes_on_every_state():
    basis, relation, transition = koopman_lift(61)
    np.testing.assert_array_equal(basis[-1][transition], relation @ basis % 61)
    result = koopman_oracle()
    assert all(row["linear_share_updates_without_peer_messages"] for row in result["examples"])
    assert result["qwen_lift_width"] is None
    controlled = result["controlled_example"]
    assert controlled["exact_both_operator_closures"]
    assert controlled["lift_width"] > len(basis)
    assert not controlled["private_operator_selection_implemented"]


def test_hermite_derivatives_do_not_supply_free_private_parallelism():
    result = hermite_oracle()
    assert result["degree8_hermite_interpolation_exact"]
    assert result["two_jet_degree1_mask_recovers_secret"]
    point, p = 3, 17
    for secret in (0, 9):
        # Two fresh mask coefficients protect both jets from ONE worker.
        views = {tuple(hasse_jet([secret, a, b], point, 2, p)) for a in range(p) for b in range(p)}
        assert len(views) == p**2
    for row in result["private_layouts"]:
        assert row["workers_for_generic_interpolation"] == 2 ** row["quadratic_layers"] + 1


def test_zech_handles_every_pair_including_zero_and_cancellation():
    result = zech_oracle()
    assert result["input_pairs_exhausted"] == 257**2
    assert result["exact_addition_and_multiplication"]


@pytest.mark.parametrize("bits", [8, 16, 24, 32])
def test_bit_heap_matches_integer_dot_across_signed_extremes(bits):
    rng = np.random.default_rng(4906)
    weights = np.array([-128, -127, -1, 0, 1, 63, 127], np.int8)
    inputs = [
        np.full(7, -128, np.int8),
        np.full(7, 127, np.int8),
        *rng.integers(-128, 128, (30, 7), dtype=np.int8),
    ]
    for values in inputs:
        expected = int(weights.astype(np.int64) @ values.astype(np.int64)) % (1 << bits)
        assert wallace_evaluate(weights, values, bits) == expected
    counts = wallace_counts(weights.reshape(1, -1), bits)
    assert counts["and_gates_per_row"][0] >= 2 * bits


def test_bit_heap_counts_reject_unbounded_or_ambiguous_inputs():
    with pytest.raises(ValueError):
        wallace_counts(np.zeros((2, 2), dtype=np.int16), 24)
    with pytest.raises(ValueError):
        wallace_counts(np.zeros((2, 2), dtype=np.int8), 64)
