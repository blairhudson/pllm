"""Counterexamples prevent invalid 100-fold promotion claims."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from network_frontier_hypotheses import (  # noqa: E402
    composed_polynomial_budget, correction_side_information, nonlinear_tensor_factorization,
    observational_state_quotient, prime_rank, reusable_label_witness,
)


def test_correction_side_information_cannot_decode_a_shorter_uniform_input():
    result = correction_side_information()
    assert result["private_inputs_checked"] == 16
    assert result["conditional_entropy_given_correction_bits"] == result["masked_input_bits"]


def test_public_label_refresh_preserves_private_cross_request_linkage():
    result = reusable_label_witness()
    assert result["exact_private_equality_relations_learned"] == result["wire_labels"]
    assert not result["public_epoch_refresh_prevents_linkage"]


def test_scalar_composition_already_exceeds_whole_response_dense_key_budget():
    result = composed_polynomial_budget()
    assert result["records"][-1]["dense_shifted_coefficient_bytes_two_parties"] > 178_970_558 // 100


def test_nonlinear_table_factorization_checks_exact_function_rank():
    assert prime_rank(np.eye(16, dtype=np.int64)) == 16
    assert prime_rank(np.ones((16, 16), dtype=np.int64)) == 1
    result = nonlinear_tensor_factorization()
    assert result["unfolding_ranks"][4] > 4
    assert result["minimal_dense_tensor_train_bytes"] > result["flat_public_table_bytes"]


def test_next_token_equality_is_not_persistent_state_equivalence():
    result = observational_state_quotient()
    assert not result["same_next_token_merger_is_sound"]
    counts = result["finite_horizon_equivalence_classes"]
    assert counts["8"] > counts["1"]
