"""Independent integer identities, dependency liveness, and privacy vetoes."""
import numpy as np
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from aggregate_network_hypotheses import (
    anchor_output, integer_anchors, lifting_forest, residue_widths,
    restore_lifted, rounding_debt_witness, symmetry_privacy_witness, terminal_groups,
)


@pytest.mark.parametrize("block", [1, 4, 16])
def test_anchor_preserves_extremes_and_every_integer_accumulator(block):
    rng = np.random.default_rng(111)
    weight = rng.integers(-128, 128, (32, 32), dtype=np.int8)
    weight[0, :2] = [-128, 127]
    weight[1] = -128
    weight[2] = 127
    residual, anchor = integer_anchors(weight, block)
    inputs = rng.integers(-127, 128, (17, 32), dtype=np.int64)
    actual = inputs @ residual.astype(np.int64).T + anchor_output(anchor, inputs, block)
    np.testing.assert_array_equal(actual, inputs @ weight.astype(np.int64).T)
    assert residual.dtype == np.int8


def test_lifting_recovers_negated_and_equal_rows_with_public_signed_bounds():
    weight = np.tile(np.arange(-32, 32, dtype=np.int8), (12, 1))
    weight[1::2] *= -1
    residual, edges = lifting_forest(weight)
    assert edges and residue_widths(residual).sum() < residue_widths(weight).sum()
    inputs = np.array([np.full(64, -127), np.arange(-32, 32), np.full(64, 127)], np.int64)
    actual = restore_lifted(inputs @ residual.astype(np.int64).T, edges)
    np.testing.assert_array_equal(actual, inputs @ weight.astype(np.int64).T)
    with pytest.raises(ValueError, match="acyclic"):
        restore_lifted(actual, [(1, 1, 1)])


def test_persistent_state_and_unknown_operators_block_last_row_demand():
    def op(identity, kind, inputs):
        return {"id": identity, "operator": kind, "inputs": inputs}
    graph = {"output": "choose", "state_outputs": [{"id": "saved"}], "operations": [
        op("prior", "linear", ["input"]), op("saved", "kv_cache_append", ["prior"]),
        op("attention", "unknown_operator", ["prior"]),
        op("out", "linear", ["attention"]), op("add", "residual_add", ["out", "prior"]),
        op("gate", "linear", ["add"]), op("silu", "silu", ["gate"]),
        op("last", "last_token", ["silu"]), op("head", "output_head", ["last"]),
        op("choose", "greedy_token_selection", ["head"])]}
    schedule = {"steps": [{"executor": "remote_stage", "operation_ids": [key]}
                          for key in ("prior", "out", "gate", "head")]}
    bindings = {f"prefill:{key}": key for key in ("prior", "out", "gate", "head")}
    assert terminal_groups(graph, schedule, bindings) == {"out", "gate"}
    graph["state_outputs"].append({"id": "gate"})
    assert terminal_groups(graph, schedule, bindings) == set()


def test_group_with_full_demand_sibling_cannot_be_partially_elided():
    graph = {"output": "last", "state_outputs": [{"id": "saved"}], "operations": [
        {"id": "a", "operator": "linear", "inputs": ["input"]},
        {"id": "b", "operator": "linear", "inputs": ["input"]},
        {"id": "saved", "operator": "kv_cache_append", "inputs": ["b"]},
        {"id": "last", "operator": "last_token", "inputs": ["a"]}]}
    schedule = {"steps": [{"executor": "remote_stage", "operation_ids": ["a", "b"]}]}
    assert not terminal_groups(graph, schedule, {"prefill:a": "both", "prefill:b": "both"})


def test_century_ideas_fail_independent_privacy_and_rounding_witnesses():
    assert symmetry_privacy_witness()["private_input_recovery_max_error"] < 1e-10
    assert rounding_debt_witness()["exact_fusion_without_debt_is_false"]
