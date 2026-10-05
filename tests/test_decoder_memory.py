from copy import deepcopy

import pytest

from pllm import Model, lower_model
from pllm.profiles import MaskedLinearCpu
from pllm.runtime.decoder_memory import _phase_working_bytes, decoder_memory
from pllm.sources import _tiny_model_config


def _plan(layers=2, rows=64):
    plan = lower_model(_tiny_model_config() | {"num_hidden_layers": layers},
                      batch=1, max_input_tokens=rows, max_new_tokens=8)
    return plan, plan.runtime_schedule(MaskedLinearCpu(Model.tiny()))


def test_layer_depth_prices_state_but_releases_finished_tensor_roots():
    shallow = decoder_memory(*_plan(2))
    deep = decoder_memory(*_plan(16))
    assert deep["mode"] == "full_kv_live_roots_v1"
    assert deep["state_bytes"] == 8 * shallow["state_bytes"]
    assert deep["working_bytes"] < 2 * shallow["working_bytes"]
    assert deep["legacy_working_bytes"] > 6 * shallow["legacy_working_bytes"]
    assert deep["working_bytes"] + deep["state_bytes"] < deep["legacy_working_bytes"]


def test_view_keeps_wide_allocation_alive_beyond_base_last_use():
    graph = {"query_sequence": 64, "state_inputs": [], "operations": [
        {"id": "wide", "operator": "linear", "inputs": ["input.tokens"], "output_shape": [64, 128]},
        {"id": "narrow", "operator": "slice", "inputs": ["wide"], "output_shape": [64, 1]},
        {"id": "new_wide", "operator": "linear", "inputs": ["narrow"], "output_shape": [64, 128]},
        {"id": "result", "operator": "greedy_token_selection", "inputs": ["new_wide"], "output_shape": [64]},
    ]}
    schedule = {"steps": [{"input_ids": op["inputs"], "outputs": [
        {"operation_id": op["id"], "output_shape": op["output_shape"]}]} for op in graph["operations"]]}
    with_view = _phase_working_bytes(graph, schedule)
    allocated = deepcopy(graph)
    allocated["operations"][1]["operator"] = "linear"
    assert with_view >= _phase_working_bytes(allocated, schedule) + 8 * 64 * 128


def test_unknown_operator_keeps_conservative_unreleased_bound():
    from types import SimpleNamespace
    plan, schedule = _plan()
    a, b = deepcopy(plan.to_dict()["prefill"]), deepcopy(plan.to_dict()["decode"])
    a["operations"][0]["operator"] = "unpriced_operator"
    bound = decoder_memory(SimpleNamespace(prefill=a, decode=b), schedule)
    assert bound["mode"] == "unreleased_outputs"
    assert bound["working_bytes"] == bound["legacy_working_bytes"] + bound["rotary_coefficient_bytes"]


def test_missing_schedule_dependency_fails_instead_of_pricing_zero():
    plan, schedule = _plan()
    phase = schedule.to_dict()["prefill"]
    phase["steps"][1]["input_ids"] = ["missing.source"]
    with pytest.raises(ValueError, match="unavailable"):
        _phase_working_bytes(plan.prefill, phase)
