from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "complete_mlp", ROOT / "scripts/probe_complete_mlp_contract.py"
)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


@pytest.mark.parametrize("width", [2, 3, 4, 5, 6])
@pytest.mark.parametrize("subtract", [False, True])
def test_exhaustive_conversion_carry_netlist(width: int, subtract: bool) -> None:
    circuit = probe.Bits(width)
    outputs = circuit.add(subtract=subtract)
    assert circuit.ands == 2 * width - 3
    for a in range(1 << width):
        for b in range(1 << width):
            expected = (a - b if subtract else a + b) % (1 << width)
            assert circuit.evaluate(a, b, outputs) == expected


def test_real_width_a2b_b2a_and_negative_sign_extension() -> None:
    a2b, b2a = probe.Bits(32), probe.Bits(32)
    add_out, sub_out = a2b.add(), b2a.add(subtract=True)
    assert a2b.ands == b2a.ands == 61
    masks = (0, 1, 127, 128, 255, 1 << 23, 1 << 31, (1 << 32) - 1)
    for q in range(-128, 128):
        for mask in masks:
            other = b2a.evaluate(q & 0xFFFFFFFF, mask, sub_out)
            reconstructed = a2b.evaluate(mask, other, add_out)
            assert probe.signed(reconstructed, 32) == q
    # A8 shares may NOT be independently signed-lifted to R32.
    assert probe.signed((5 + 124) % 256, 8) == -127
    assert probe.signed(5, 8) + probe.signed(124, 8) == 129


def test_public_linear_shares_and_target_ring_before_down_projection() -> None:
    # Independent integer dot oracle, including wrap and negative W8 entries.
    matrix = [[127, -127, 3, -5], [-11, 9, -127, 127]]
    for codes in ([127, -127, 0, 1], [-128, 3, 127, -1], [0, 0, 0, 0]):
        for masks in ([0, 0, 0, 0], [0xFFFFFFFF, 3, 0x80000000, 65537]):
            complement = [(q - a) % (1 << 32) for q, a in zip(codes, masks, strict=True)]
            for weight in matrix:
                left = sum(w * a for w, a in zip(weight, masks, strict=True)) % (1 << 32)
                right = sum(w * b for w, b in zip(weight, complement, strict=True)) % (1 << 32)
                expected = sum(w * q for w, q in zip(weight, codes, strict=True))
                assert probe.signed((left + right) % (1 << 32), 32) == expected


def test_integer_cast_rounding_is_not_exact_real_scale_cancellation() -> None:
    for base in (1 << 23, 1 << 24, 1 << 25, 1 << 26):
        for offset in range(-9, 10):
            for sign in (-1, 1):
                n = sign * (base + offset)
                assert probe.int_f32_word(n) == int(np.asarray(np.float32(n)).view(np.uint32))
    assert probe.int_f32_word((1 << 24) + 1) == probe.int_f32_word(1 << 24)
    assert probe.int_f32_word((1 << 24) + 3) == probe.int_f32_word((1 << 24) + 4)


@pytest.mark.parametrize("rows", [46, 70])
@pytest.mark.parametrize("label", [16, 32])
def test_independent_conversion_and_directed_cost_accounting(rows: int, label: int) -> None:
    cost = probe.known_cost(rows, label_bytes=label)
    n = 24 * rows
    assert cost["conversion_ANDs"] == (3 * 4864 + 896) * 61 * n
    links = cost["known_payload_bytes_by_link_and_kind"]
    assert links["A->B:A2B_half_gate_tables"] == (2 * 4864 + 896) * 61 * n * 2 * label
    assert links["A->B:B2A_half_gate_tables"] == 4864 * 61 * n * 2 * label
    assert links["A->B:garbler_selected_input_labels"] == (3 * (4864 + 896) * 32 + 96) * n * label
    assert cost["evaluator_input_OTs"] == (2 * (4864 + 896) * 32 + 64) * n
    assert cost["known_payload_subtotal_bytes"] == sum(links.values())
    assert cost["OT_transcript_bytes"] is None
    assert cost["complete_recurring_body_bytes"] is None
    assert cost["numeric_function_tables_bytes"] is None
    assert probe.cost_gate(cost, 148_297_986) == "veto_known_payload_alone"
    assert probe.cost_gate(cost, 10**15) == "reject_missing_executable_coverage"
    assert cost["admitted"] is False


def test_unknown_cannot_be_promoted_to_zero_or_coverage_pass() -> None:
    cost = probe.known_cost(70)
    cost["complete_recurring_body_bytes"] = cost["known_payload_subtotal_bytes"]
    assert probe.cost_gate(cost, 10**15) == "reject_missing_executable_coverage"


def test_exact_removal_and_scope_conservation() -> None:
    saved = json.loads(probe.EVIDENCE.read_text())
    probe.validate_controls(saved["archived_controls"])
    c = saved["archived_controls"]["32"]
    groups = c["directed_stage_body_bytes"]
    narrow = probe.remove_bodies(
        148_297_986,
        groups,
        [("mlp_gate_up", "preparation->inference"), ("mlp_gate_up", "inference->client")],
    )
    assert narrow == {
        "removed_body_bytes": 98_183_080,
        "fixed_other_body_bytes": 50_114_906,
        "replacement_budget_25pct_bytes": 61_108_583,
        "replacement_budget_50pct_bytes": 24_034_087,
    }
    full = probe.remove_bodies(
        148_297_986, groups, [(role, edge) for role, edges in groups.items() for edge in edges]
    )
    assert full["removed_body_bytes"] == 148_296_496
    assert full["fixed_other_body_bytes"] == 1490
    assert full["removed_body_bytes"] + full["fixed_other_body_bytes"] == 148_297_986
    assert full["removed_body_bytes"] - narrow["removed_body_bytes"] == 50_113_416
    for count in ("8", "32"):
        for scope in (
            "gate_up_output_plus_correction_only_scope",
            "specified_full_MLP_region_retirement_scope",
        ):
            row = saved["cohorts"][count][scope]
            assert (
                row["removed_body_bytes"] + row["fixed_other_body_bytes"]
                == saved["archived_controls"][count]["covered_all_link_body_bytes"]
            )


@pytest.mark.parametrize("bad", ["duplicate", "negative", "unknown", "excess"])
def test_reject_bad_or_double_removal(bad: str) -> None:
    groups, total = {"mlp_gate_up": {"A->B": 10}}, 12
    removal = [("mlp_gate_up", "A->B")]
    if bad == "duplicate":
        removal *= 2
    elif bad == "negative":
        groups["mlp_gate_up"]["A->B"] = -1
    elif bad == "unknown":
        removal = [("mlp_down", "A->B")]
    else:
        total = 9
    with pytest.raises(ValueError):
        probe.remove_bodies(total, groups, removal)


def test_recorded_evidence_recomputes_known_costs_and_keeps_missing_coverage() -> None:
    saved = json.loads(probe.EVIDENCE.read_text())
    assert saved["archived_controls_sha256"] == probe.digest(saved["archived_controls"])
    assert saved["complete_executable_coverage"] is False
    assert saved["complete_protocol_cost_bytes"] is None
    for count in (8, 32):
        cohort = saved["cohorts"][str(count)]
        rows = 39 + count - 1
        assert cohort["inventory"]["executed_rows"] == rows
        assert cohort["complete_truth_table_log2_assignments_GC1"] == 311328
        for value in cohort["costs"].values():
            recomputed = probe.known_cost(
                rows, label_bytes=value["label_bytes"], width=value["accumulator_ring_bits"]
            )
            for key, expected in recomputed.items():
                assert value[key] == expected
    assert saved["arithmetic_controls"] == probe.arithmetic_controls()
