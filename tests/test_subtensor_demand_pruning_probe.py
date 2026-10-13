"""Independent semantic roots, numerical witnesses and bounded accounting."""
import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "subtensor_probe", Path(__file__).resolve().parents[1]/"scripts/probe_subtensor_demand_pruning.py")
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def fixture(family="qwen2"):
    _, plan, schedule = probe.lower(family, 7, small=True)
    return plan["prefill"], schedule["prefill"]


def query_slice(graph, schedule):
    d = probe.liveness(graph)
    c = probe.cost(graph, schedule, d, rows=7)["candidates"]
    assert len(c) == 1
    slices = c[0]["slices"]
    assert [s["demanded_rows"] for s in slices] == [1, 7, 7]
    return slices[0]["operation_id"]


@pytest.mark.parametrize("family", ["qwen2", "qwen3", "llama"])
def test_independent_logits_all_kv_and_compiler_comparator(family):
    graph, schedule = fixture(family)
    old = probe.liveness(graph, extended=False)
    new = probe.liveness(graph)
    for step in schedule["steps"]:
        if set(step["operators"]) == {"linear"}:
            assert bool(step.get("terminal_row_only")) == all(old.get(k) == 1 for k in step["operation_ids"])
    for seed in (1, 7, 19):
        full, kv = probe.array_oracle(graph, {}, seed=seed)
        for demand in (old, new):
            actual, state = probe.array_oracle(graph, demand, seed=seed)
            np.testing.assert_array_equal(full, actual)
            assert len(state) == 4
            for key in kv:
                np.testing.assert_array_equal(kv[key], state[key])


def test_q_root_hidden_export_unknown_consumer_reject_pruning():
    graph, schedule = fixture()
    q = query_slice(graph, schedule)
    assert probe.liveness(graph, extra_roots=[q])[q] == 2
    assert not probe.cost(graph, schedule, probe.liveness(graph, extra_roots=[q]), rows=7)["candidates"]
    retained = copy.deepcopy(graph)
    retained["state_outputs"].append(dict(id=q, kind="custom"))
    assert probe.liveness(retained)[q] == 2
    unknown = copy.deepcopy(graph)
    unknown["operations"].append(dict(id="opaque-consumer", operator="unknown",
        inputs=[q], output_shape=[1, 7, 16], attributes={}))
    assert probe.liveness(unknown, extra_roots=["opaque-consumer"])[q] == 2
    last = next(o for o in graph["operations"] if o["operator"] == "last_token")
    assert probe.liveness(graph, extra_roots=[last["inputs"][0]])[q] == 2


@pytest.mark.parametrize("mutation", ["softmax_axis", "reshape_layout", "reshape_shape", "selection"])
def test_uncertified_transfer_widens_to_all_rows(mutation):
    graph, schedule = fixture()
    q = query_slice(graph, schedule)
    if mutation == "softmax_axis":
        for op in graph["operations"]:
            if op["operator"] == "softmax":
                op["attributes"]["axis"] = -2
    elif mutation.startswith("reshape"):
        for op in graph["operations"]:
            if op["operator"] == "reshape" and op["inputs"] == [q]:
                if mutation == "reshape_layout":
                    op["attributes"]["layout"] = "query_mixing"
                else:
                    op["output_shape"][2] += 1
    else:
        for op in graph["operations"]:
            if op["operator"] == "last_token":
                op["attributes"]["selection"] = "arbitrary"
    assert probe.liveness(graph)[q] == 2


def test_kv_input_demand_and_exact_saved_lane_accounting():
    graph, schedule = fixture()
    report = probe.cost(graph, schedule, probe.liveness(graph), rows=7)
    candidate = report["candidates"][0]
    assert candidate["saved_input_bytes"] == 0
    assert candidate["saved_output_lanes"] == 6*16
    assert candidate["saved_reply_bytes"] == candidate["saved_correction_bytes"] == 4*6*16
    assert candidate["saved_integer_macs"] == 6*16*16
    assert report["remote_body_mac_retention"] >= .9
    # All-row consumers invalidate comparator's existing terminal pruning too.
    with pytest.raises(ValueError, match="exceeds terminal comparator"):
        probe.cost(graph, schedule, probe.liveness(graph, all_rows=True), rows=7)


def test_decode_and_continuation_not_inherit_prefill_slice_contract():
    _, plan, schedule = probe.lower("qwen3", 7, small=True)
    assert not probe.cost(plan["decode"], schedule["decode"],
                          probe.liveness(plan["decode"]), rows=1, decode_steps=0)["candidates"]
    q = query_slice(plan["prefill"], schedule["prefill"])
    assert probe.liveness(plan["prefill"], all_rows=True)[q] == 2


def test_one_prepared_execution_has_one_body_per_directed_link():
    # Unequal input/output widths expose lane-role mistakes, not just a total.
    assert probe.directed_bodies(3, 5) == {
        "Client->Inference": 12, "Inference->Client": 20, "Preparation->Inference": 20}
    graph = {"operations": [dict(id="source", output_shape=[1, 3, 3]) ]}
    schedule = {"steps": [dict(executor="remote_stage", operators=["linear"],
        input_ids=["source"], operation_ids=["projection"],
        outputs=[dict(operation_id="projection", stage_width=5)])]}
    # Three prefill rows plus two decode rows; no doubled cold/repeat workload.
    report = probe.cost(graph, schedule, {"projection": 2}, rows=3, decode_steps=2)
    assert report["prepared_response_executions"] == 1
    assert report["directed_link_multiplicity"] == {
        "Client->Inference": 1, "Inference->Client": 1, "Preparation->Inference": 1}
    assert report["terminal_comparator_directed_body_bytes"] == {
        "Client->Inference": 4*5*3,
        "Inference->Client": 4*5*5,
        "Preparation->Inference": 4*5*5}
    assert report["terminal_comparator_body_bytes"] == 4*5*(3+5+5)


def test_partition_preserves_quantization_scales_bias_and_float_order():
    rng = np.random.default_rng(88)
    x = rng.normal(size=(1, 7, 16)).astype(np.float32)
    w = rng.integers(-127, 128, (32, 16), dtype=np.int8)
    scales = rng.uniform(.01, .1, 32).astype(np.float32)
    bias = rng.uniform(-.1, .1, 32).astype(np.float32)
    full = probe.quantized_linear(x, w, scales, bias)
    split = np.concatenate([probe.quantized_linear(x, w[:16], scales[:16], bias[:16], terminal=True),
                            probe.quantized_linear(x, w[16:], scales[16:], bias[16:])], axis=-1)
    np.testing.assert_array_equal(full[:, -1], split[:, -1])
    np.testing.assert_array_equal(full[..., 16:], split[..., 16:])
    # Input partition recalibrates each reduction domain and changes integer math.
    witness = np.array([[[127., .6]]], np.float32)
    ww = np.array([[0, 1]], np.int8)
    correct = probe.quantized_linear(witness, ww, np.ones(1, np.float32), np.zeros(1, np.float32))
    wrong = probe.quantized_linear(witness[..., 1:], ww[:, 1:], np.ones(1, np.float32), np.zeros(1, np.float32))
    assert not np.array_equal(correct, wrong)
    assert probe.numeric_admission(original_input_width=16, proposed_input_width=16)
    for kwargs in (dict(proposed_input_width=8), dict(same_quantized_input=False),
                   dict(same_scales_bias=False), dict(same_f32_order=False)):
        args = dict(original_input_width=16, proposed_input_width=16)
        args.update(kwargs)
        assert not probe.numeric_admission(**args)
    n = np.float32(16777216)
    assert np.float32(np.float32(n+np.float32(1))-n) != np.float32(n+np.float32(np.float32(1)-n))


def test_renamed_ids_and_weight_paths_do_not_change_demand_cost():
    graph, schedule = fixture()
    expected = probe.cost(graph, schedule, probe.liveness(graph), rows=7)
    renamed, steps = copy.deepcopy(graph), copy.deepcopy(schedule)
    names = {o["id"]: f"opaque-{i}" for i, o in enumerate(graph["operations"])}
    for op in renamed["operations"]:
        op["id"] = names[op["id"]]
        op["inputs"] = [names.get(i, i) for i in op["inputs"]]
        if "weight" in op["attributes"]:
            op["attributes"]["weight"] = "opaque-weight"
    renamed["output"] = names[renamed["output"]]
    for state in renamed["state_outputs"]:
        state["id"] = names[state["id"]]
    for step in steps["steps"]:
        step["operation_ids"] = [names[i] for i in step["operation_ids"]]
        step["input_ids"] = [names.get(i, i) for i in step["input_ids"]]
        for out in step["outputs"]:
            out["operation_id"] = names[out["operation_id"]]
    actual = probe.cost(renamed, steps, probe.liveness(renamed), rows=7)
    assert actual["incremental_arithmetic_upper_bound_bytes"] == expected["incremental_arithmetic_upper_bound_bytes"]


def test_public_shape_and_run_bounds_rejected():
    graph, _ = fixture()
    graph["batch"] = 2
    with pytest.raises(ValueError, match="batch-one"):
        probe.liveness(graph)
    for rows, outputs in ((513, 8), (1, 8), (39, 9)):
        with pytest.raises(ValueError, match="bounded"):
            probe.run(rows, outputs)


def test_dead_output_slice_removed_without_touching_live_kv_inputs():
    graph, schedule = fixture()
    ops = {o["id"]: o for o in graph["operations"]}
    grouped = next(s for s in schedule["steps"] if len(s["operation_ids"]) == 3)
    source = ops[grouped["operation_ids"][0]]
    dead = copy.deepcopy(source)
    dead["id"] = "unconsumed-projection"
    graph["operations"].append(dead)
    width = dead["output_shape"][-1]
    grouped["operation_ids"].append(dead["id"])
    grouped["operators"].append("linear")
    grouped["outputs"].append(dict(operation_id=dead["id"], stage_offset=sum(
        r["stage_width"] for r in grouped["outputs"]), stage_width=width))
    d = probe.liveness(graph)
    assert dead["id"] not in d
    candidates = probe.cost(graph, schedule, d, rows=7)["candidates"]
    added = next(c for c in candidates if c["order"] == grouped["order"])
    assert added["slices"][-1]["demanded_rows"] == 0
    assert added["saved_output_lanes"] == 7*width
    assert added["saved_input_bytes"] == 0
