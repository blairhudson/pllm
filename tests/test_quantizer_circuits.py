from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "probe_quantizer_circuits", ROOT / "scripts/probe_quantizer_circuits.py"
)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def test_all_two_input_boolean_functions_and_complemented_outputs() -> None:
    # Complete independent primitive truth tables, including asymmetric mux
    # branches and output complements. Reverse order must preserve bit indexing.
    for function in range(16):
        bit = np.asarray([(function >> index) & 1 for index in range(4)], np.uint64)
        table = bit | ((bit ^ 1) << 1)
        for shared in (False, True):
            for order in ([0, 1], [1, 0]):
                circuit = probe.synthesize(table, 2, order, shared=shared)
                np.testing.assert_array_equal(circuit.evaluate_all(chunk_bits=3), table)


def test_full_q7_encoded_function_is_bit_exact_under_same_order_cse() -> None:
    table, contract = probe.function_table(bits=8)
    assert contract["truth_assignments"] == 65536
    assert contract["independent_product_bit_mismatches"] == 0
    assert contract["independent_code_and_scale_mismatches"] == 0
    order = list(reversed(range(16)))
    generic = probe.synthesize(table, 8, order, shared=False)
    optimized = probe.synthesize(table, 8, order, shared=True)
    np.testing.assert_array_equal(generic.evaluate_all(), table)
    np.testing.assert_array_equal(optimized.evaluate_all(), table)
    assert optimized.ands < generic.ands
    assert optimized.summary()["and_depth"] <= generic.summary()["and_depth"]
    # Domain includes both non-symmetric -128 inputs; no don't-care shortcut.
    for gate, up in ((-128, -128), (-128, 127), (0, -128), (127, -128)):
        index = ((gate & 255) << 8) | (up & 255)
        expected = probe.scalar_quantize(probe.scalar_product(gate, up, 8), np.float32(1 / 128))
        assert int(table[index]) == expected & 255


def test_dynamic_vector_includes_exact_scale_not_only_output_codes() -> None:
    table, contract = probe.function_table(bits=4, dynamic_vector=True)
    order = [7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8]
    circuit = probe.synthesize(table, 48, order, shared=True)
    np.testing.assert_array_equal(circuit.evaluate_all(), table)
    assert contract["distinct_scale_words"] > 100
    assert contract["scalar_float32_product_comparisons"] == 131072
    assert int(table[0]) == 0x3F800000 << 16  # zero row => scale 1, codes 0
    # Max from either feature; swapping complete input pairs swaps only codes.
    for index in (0x127F, 0xF721, 0x7777, 0x8008, 0x00F1):
        swapped = ((index & 255) << 8) | (index >> 8)
        word, other = int(table[index]), int(table[swapped])
        assert word >> 16 == other >> 16
        assert word & 255 == (other >> 8) & 255
        assert (word >> 8) & 255 == other & 255


def test_independent_rounding_edges_and_entropy_cost_do_not_hide_unknowns() -> None:
    edges = probe.rounding_edges()
    assert edges["fixed_scale_tie_neighbor_saturation_cases"] == 780
    assert edges["dynamic_rounding_counterexample"]["before"] == [127, 4]
    assert edges["dynamic_rounding_counterexample"]["after"] == [127, 5]
    cost = probe.material_cost({"input_bits": 16, "output_bits": 8, "and_non_xor_gates": 10})
    assert cost["half_gate_fresh_ciphertext_body_bytes"] == 340
    assert cost["selected_input_label_payload_bytes"] == 272
    assert cost["output_labels_to_authorized_recipient_payload_bytes"] == 136
    assert cost["ot_extension_and_base_ot_body_bytes"] is None
    assert cost["required_hidden_label_entropy_bits"] == 128
    assert cost["security_implementation"] is False


def test_synthesis_rejects_width_overflow_and_incomplete_domains() -> None:
    for table, outputs, order in (
        ([0, 1, 2], 2, [1, 0]),
        ([0, 1, 2, 4], 2, [1, 0]),
        ([0, 1, 2, 3], 2, [0, 0]),
    ):
        with pytest.raises(ValueError):
            probe.synthesize(np.asarray(table, np.uint64), outputs, order, shared=True)
    with pytest.raises(ValueError):
        probe.Circuit(17, shared=True)


def test_locked_evidence_covers_both_cohorts_and_retains_unpriced_region_work() -> None:
    evidence = json.loads(
        (ROOT / "docs/evidence/quantizer-circuit-screen-2026-10-01.json").read_text()
    )
    assert evidence["schema"] == "pllm.quantizer_circuit_screen.v1"
    assert not evidence["security_implementation"] and not evidence["runtime_activated"]
    for case in (evidence["fixed_public_scale"], evidence["dynamic_scale_two_element_vector"]):
        assert case["contract"]["truth_assignments"] == 65536
        assert len(case["circuits"]) == 3
        assert all(row["exhaustive_bit_evaluator_mismatches"] == 0 for row in case["circuits"])
    cohorts = evidence["pinned_semantic_qwen_cohorts"]
    assert [row["output_tokens"] for row in cohorts] == [8, 32]
    for cohort in cohorts:
        assert cohort["executed_rows"] == 39 + cohort["output_tokens"] - 1
        assert cohort["demonstrated_stage_crossings_removed"] == 0
        assert not cohort["complete_encoded_qwen_region_available"]
        for price in cohort["quantizer_circuit_prices"]:
            assert price["hypothetical_fixed_domain_element_invocations"] == (
                cohort["executed_rows"] * cohort["semantic_layers"] * cohort["intermediate"]
            )
            assert price["complete_protected_region_cost_bytes"] is None
            assert price["ot_body_bytes"] is None
            assert (
                price["fresh_garbler_to_evaluator_ciphertext_body_bytes"]
                > (cohort["prepared_control"]["covered_all_link_body_bytes"])
            )
