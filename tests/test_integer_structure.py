from __future__ import annotations

import importlib.util
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from pllm.runtime.integer_structure_reference import (
    centered_lift,
    factor_apply,
    factor_cost_gate,
    minimum_signed_bits,
    odd_minor_witness,
    verify_factorization,
    verify_odd_minor,
)
from pllm.runtime.quantization import dequantize_matmul


ROOT = Path(__file__).resolve().parents[1]


def scalar_product(left, right):
    """Independent scalar oracle: Python integers, no NumPy matmul."""
    return np.asarray(
        [
            [
                sum(int(left[i, k]) * int(right[k, j]) for k in range(left.shape[1]))
                for j in range(right.shape[1])
            ]
            for i in range(left.shape[0])
        ],
        dtype=object,
    )


@pytest.mark.parametrize("bits", [8, 16, 24, 32, 64])
def test_exact_integer_factors_residual_and_original_scale_bias(bits):
    a = np.array([[2, -1], [-3, 4], [1, 2]], dtype=np.int8)
    b = np.array([[3, 1, -2, 0], [-1, 2, 1, 3]], dtype=np.int8)
    residual = np.array([[0, 1, 0, 0], [0, 0, -2, 0], [0, 0, 0, 0]], dtype=np.int8)
    weight = np.asarray(scalar_product(a, b) + residual.astype(object), np.int8)
    inputs = np.array([[-127, 127, -63, 2], [1, -2, 3, -4]], dtype=np.int8)
    assert verify_factorization(weight, a, b, residual=residual, bits=bits)
    expected = scalar_product(inputs, weight.T)
    actual = factor_apply(a, b, inputs, residual=residual, bits=bits)
    np.testing.assert_array_equal(actual.astype(object), expected % (1 << bits))
    if bits == 8:
        # Mod256 coefficients/products exact, yet signed W8A8 accumulator not liftable.
        with pytest.raises(ValueError, match="half the modulus"):
            centered_lift(actual, bits=bits, output_bound=4000)
        return
    integer = centered_lift(actual, bits=bits, output_bound=4000)
    np.testing.assert_array_equal(integer, expected)
    # Unequal output scales detect applying scale to factor components or twice.
    activation_scales = np.array([0.125, 0.25], np.float32)
    row_scales = np.array([0.03125, 0.5, 0.0625], np.float32)
    bias = np.array([0.25, -0.75, 1.5], np.float32)
    result = dequantize_matmul(integer, activation_scales, row_scales) + bias
    oracle = np.empty((2, 3), dtype=np.float32)
    for i in range(2):
        for j in range(3):
            oracle[i, j] = (
                np.float32(expected[i, j]) * activation_scales[i] * row_scales[j] + bias[j]
            )
    np.testing.assert_array_equal(result, oracle)


def test_64bit_products_use_arbitrary_precision_before_modular_reduction():
    a = np.array([[2**64 - 1, 2**63 + 5]], dtype=np.uint64)
    b = np.array([[2**63 - 1, 3], [2**64 - 7, 2**63]], dtype=np.uint64)
    inputs = np.array([[2**64 - 1, 2**64 - 2]], dtype=np.uint64)
    weight = np.asarray(scalar_product(a, b) % (1 << 64), dtype=np.uint64)
    assert verify_factorization(weight, a, b, bits=64)
    np.testing.assert_array_equal(
        factor_apply(a, b, inputs, bits=64).astype(object),
        scalar_product(inputs, weight.T) % (1 << 64),
    )
    bad = weight.copy()
    bad[0, 0] = int(bad[0, 0]) ^ 1
    assert not verify_factorization(bad, a, b, bits=64)


def test_small_modulus_factor_equality_is_not_integer_or_wider_ring_equality():
    a = np.array([[256]], dtype=np.int64)
    b = np.array([[1]], dtype=np.int8)
    weight = np.zeros((1, 1), dtype=np.int8)
    assert verify_factorization(weight, a, b, bits=8)
    assert not verify_factorization(weight, a, b, bits=16)
    assert centered_lift(np.array([[65535]], np.uint32), bits=16, output_bound=1)[0, 0] == -1
    with pytest.raises(ValueError, match="violates"):
        centered_lift(np.array([[2]], np.uint32), bits=16, output_bound=1)


@pytest.mark.parametrize("transpose", [False, True])
def test_rectangular_odd_minor_witness_never_exceeds_min_dimension(transpose):
    weight = np.array([[2, 4], [1, 0], [0, 3], [1, 1]], dtype=np.int8)
    if transpose:
        weight = weight.T
    witness = odd_minor_witness(weight)
    assert witness["size"] == 2 == min(weight.shape)
    assert verify_odd_minor(weight, witness)
    assert len(witness["rows"]) == len(witness["columns"]) == 2
    # Wide matrix: full ROW rank; a third independent column cannot be claimed.
    assert not verify_odd_minor(weight, {"size": 3, "rows": [0, 1, 2], "columns": [0, 1, 2]})


def test_deficient_gf2_rank_does_not_imply_low_width_factorization_over_256():
    weight = np.array([[2, 0], [0, 2]], dtype=np.int8)
    assert odd_minor_witness(weight)["size"] == 0
    # Rank modulo2 is zero, but W is not zero modulo256 and cannot have width1:
    # its image has 128**2 distinct elements, while one generator over Z/256
    # has at most 256. Never silently replace W with a rank0 GF2 result.
    assert not verify_factorization(
        weight, np.zeros((2, 1), np.int8), np.zeros((1, 2), np.int8), bits=8
    )
    assert len({(2 * x % 256, 2 * y % 256) for x in range(128) for y in range(128)}) > 256


def test_exhaustive_small_determinants_independently_validate_witnesses():
    for bits in itertools.product((0, 1), repeat=9):
        matrix = np.array(bits, np.int8).reshape(3, 3)
        # Leibniz determinant modulo2: signs disappear, no elimination oracle.
        determinant_parity = (
            sum(
                int(matrix[0, p[0]]) * int(matrix[1, p[1]]) * int(matrix[2, p[2]])
                for p in itertools.permutations(range(3))
            )
            % 2
        )
        assert verify_odd_minor(
            matrix, {"size": 3, "rows": [0, 1, 2], "columns": [0, 1, 2]}
        ) == bool(determinant_parity)
        witness = odd_minor_witness(matrix)
        if witness["size"]:
            assert verify_odd_minor(matrix, witness)
        assert (witness["size"] == 3) == bool(determinant_parity)


@pytest.mark.parametrize(
    "witness",
    [
        {},
        {"size": 0, "rows": [], "columns": []},
        {"size": 2, "rows": [0, 0], "columns": [0, 1]},
        {"size": 2, "rows": [0, 1], "columns": [0, 0]},
        {"size": 1, "rows": [-1], "columns": [0]},
        {"size": 1, "rows": [0], "columns": [2]},
        {"size": 1, "rows": [0.0], "columns": [0]},
        {"size": True, "rows": [0], "columns": [0]},
    ],
)
def test_witness_checker_rejects_unsound_metadata(witness):
    assert not verify_odd_minor(np.eye(2, dtype=np.int8), witness)


@pytest.mark.parametrize(
    "bad",
    [
        np.array([1]),
        np.empty((1, 0), np.int8),
        np.ones((1, 1), np.float64),
        np.array([[1.5]], object),
    ],
)
def test_references_reject_invalid_matrix_shapes_and_dtypes(bad):
    with pytest.raises(ValueError, match="integer"):
        odd_minor_witness(bad)
    with pytest.raises(ValueError, match="integer"):
        factor_apply(np.ones((1, 1), np.int8), np.ones((1, 1), np.int8), bad, bits=8)


def test_factor_shape_ring_and_residual_checks():
    one = np.ones((1, 1), np.int8)
    for bits in (0, 65, True, 8.0):
        with pytest.raises(ValueError, match="ring bits"):
            factor_apply(one, one, one, bits=bits)
    with pytest.raises(ValueError, match="shapes"):
        verify_factorization(one, np.ones((2, 1), np.int8), one, bits=8)
    with pytest.raises(ValueError, match="residual shape"):
        factor_apply(one, one, one, bits=8, residual=np.ones((2, 1), np.int8))


def test_cost_gate_prices_client_projection_fullwidth_residual_and_fresh_masks():
    qkv = factor_cost_gate(inputs=896, outputs=1152, rank=896, ring_bits=32)
    assert qkv["client_dense_projection_mac_fraction"] == pytest.approx(896 / 1152)
    assert qkv["optimistic_online_bytes"] == 70 * 4 * (896 + 1152)
    assert qkv["optimistic_all_link_bytes"] == 70 * 4 * (896 + 2 * 1152)
    down = factor_cost_gate(inputs=4864, outputs=896, rank=896, ring_bits=32)
    assert down["client_dense_projection_mac_fraction"] == 1
    assert down["client_dense_projection_macs"] == 70 * 4864 * 896
    sparse = factor_cost_gate(
        inputs=896, outputs=1152, rank=1, ring_bits=32, residual_nnz=896, residual_input_columns=896
    )
    assert sparse["fresh_mask_input_elements_per_row"] == 897
    assert sparse["fresh_mask_output_elements_per_row"] == 1152
    assert sparse["optimistic_online_bytes"] > qkv["optimistic_online_bytes"]
    assert sparse["client_dense_factor_weight_bytes"] == 896
    for kwargs in ({"residual_nnz": 1}, {"residual_nnz": 1, "residual_input_columns": 2}):
        with pytest.raises(ValueError, match="support"):
            factor_cost_gate(inputs=4, outputs=3, rank=1, ring_bits=24, **kwargs)
    assert [minimum_signed_bits(x) for x in (0, 127, 128, 32767, 32768)] == [2, 8, 9, 16, 17]


def load_probe():
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        spec = importlib.util.spec_from_file_location(
            "integer_probe", ROOT / "scripts/probe_integer_structure.py"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def test_structure_and_vector_classes_find_planted_exact_structure():
    probe = load_probe()
    weight = np.array([[2, -2, 3, 3], [4, -4, 6, 6], [0, 0, 0, 0], [-2, 2, -3, -3]], np.int8)
    rows = probe.vector_classes(weight, axis=0)
    assert rows["zero_vectors"] == 1
    assert rows["rational_proportional_nonzero_vectors_beyond_first"] == 2
    columns = probe.vector_classes(weight, axis=1)
    assert columns["duplicate_nonzero_vectors_beyond_first"] == 1
    assert columns["rational_proportional_nonzero_vectors_beyond_first"] == 3
    cse = probe.pair_cse(weight)
    assert cse["remote_products_saved_per_row"] == 6
    assert cse["remote_shared_additions_per_row"] == 2
    assert cse["network_saving_bytes"] == 0
    constant = np.array([[3, 3, 3], [-2, -2, -2]], np.int8)
    screened = probe.residual_screen(constant, 1)
    row_mode = next(r for r in screened if r["structure"] == "row_mode_rank1")
    assert row_mode["exact_all_entries_verified"]
    assert row_mode["residual_nnz"] == row_mode["residual_active_input_columns"] == 0


def test_measured_evidence_gates_have_sound_rectangular_scope_and_no_sparse_win():
    evidence = json.loads(
        (ROOT / "docs/evidence/integer-structure-screen-2026-10-01.json").read_text()
    )
    assert evidence["sampled_layers"] == [0, 12, 23]
    assert evidence["sampled_stages"] == len(evidence["matrices"]) == 9
    assert evidence["resource_limits"]["resident_selected_stages"] == 1
    for matrix in evidence["matrices"]:
        m, n = matrix["shape_output_by_input"]
        witness = matrix["minor_witness"]
        assert witness["size"] == min(m, n) == 896
        assert len(set(witness["rows"])) == len(set(witness["columns"])) == 896
        assert max(witness["rows"]) < m and max(witness["columns"]) < n
        assert matrix["full_column_rank_certified"] == (n == 896)
        assert matrix["full_row_rank_certified"] == (m == 896)
        assert matrix["minor_determinant_parity_independently_verified"] == 1
        for axis in ("row_classes", "column_classes"):
            classes = matrix[axis]
            assert classes["zero_vectors"] == 0
            assert classes["duplicate_nonzero_vectors_beyond_first"] == 0
            assert classes["rational_proportional_nonzero_vectors_beyond_first"] == 0
            assert classes["unit_proportional_mod256_vectors_beyond_first"] == 0
        for candidate in matrix["structured_residuals"]:
            assert candidate["exact_all_entries_verified"]
            assert candidate["residual_fraction"] > 0.96
            offset = int(candidate["structure"] == "anchored_additive_rank2")
            assert candidate["residual_active_input_columns"] == n - offset
            assert candidate["residual_active_output_rows"] == m - offset
            assert candidate["residual_minimum_signed_ring_bits"] >= 23
    for control_name, gates in evidence["shape_only_network_gates"].items():
        control = evidence["controls"][control_name]
        assert (
            gates["unattained_uniform_u16_zero_ingress_output_only_online_bytes"]
            > control["tenfold_online_budget_bytes"]
        )
        assert (
            gates["unattained_uniform_u16_zero_ingress_output_plus_fresh_correction_all_link_bytes"]
            > control["tenfold_all_link_budget_bytes"]
        )
