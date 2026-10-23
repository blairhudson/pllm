"""Declared tiny operator fixtures, NOT a trained run or native factory proof."""
import builtins
from collections import Counter
from copy import deepcopy
from fractions import Fraction as F
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.research import decoder_numeric_reference as runner


def fixture(heads=2, kv_heads=1):
    d = runner.Dimensions(1, 2, 2, heads, kv_heads, 2, 4, 1000000, "synthetic_operator_fixture")
    tensors = {"embedding-original": runner.SourceTensor((4, 2), "BF16", (0x3f80, 0, 0, 0x3f80, 0xbf80, 0, 0, 0xbf80))}
    tensors["head-separate"] = tensors["embedding-original"]
    operations, state_outputs = [], []

    def add(kind, inputs, shape, attrs, layer=0, state=None):
        identifier = f"opaque-{len(operations)}"
        operations.append({"id": identifier, "operator": kind, "layer": layer, "state_kind": state,
                           "inputs": inputs, "output_shape": list(shape), "attributes": attrs})
        return identifier

    def norm(source, ref, shape, layer=0):
        tensors[ref] = runner.SourceTensor((shape[-1],), "BF16", (0x3f80,)*shape[-1])
        return add("rms_norm", [source], shape, {"epsilon": F(1, 1000000), "weight": ref, "weight_offset": 0}, layer)

    def linear(source, ref, rows, columns):
        tensors[ref] = runner.SourceTensor((rows, columns), "synthetic_Q8", (0,)*(rows*columns))
        return add("linear", [source], (1, 8, rows), {"weight": ref, "bias": None})

    h = (1, 8, 2)
    hidden = add("token_lookup", ["input.tokens"], h, {"weight": "embedding-original"}, None)
    initial = norm(hidden, "gamma-input", h)
    q = linear(initial, "original-q", heads*2, 2)
    k = linear(initial, "original-k", kv_heads*2, 2)
    v = linear(initial, "original-v", kv_heads*2, 2)
    q = add("reshape", [q], (1, heads, 8, 2), {"layout": "batch_heads_sequence_feature"})
    k = add("reshape", [k], (1, kv_heads, 8, 2), {"layout": "batch_heads_sequence_feature"})
    v = add("reshape", [v], (1, kv_heads, 8, 2), {"layout": "batch_heads_sequence_feature"})
    q = norm(q, "gamma-query", (1, heads, 8, 2))
    k = norm(k, "gamma-key", (1, kv_heads, 8, 2))
    attrs = {"theta": 1000000, "rotary_dimensions": 2, "pairing": "split_half", "position_policy": "sequential_absolute",
             "coefficient_profile": "pllm.numeric.rope.q30.libm.v1", "input_layout": "batch_heads_sequence_feature",
             "output_layout": "batch_heads_sequence_feature", "tail_policy": "unchanged"}
    q = add("rotary_embedding", [q, "input.positions"], (1, heads, 8, 2), dict(attrs))
    k = add("rotary_embedding", [k, "input.positions"], (1, kv_heads, 8, 2), dict(attrs))
    extra = ["input.positions", "input.attention_mask", "input.sequence_lengths"]
    cache = []
    for source, state in [(k, "key"), (v, "value")]:
        attrs = {"state": "local-"+state, "mode": "initialize", "state_capacity": 8, "state_layout": "batch_kv_heads_sequence_feature",
                 "absolute_write_positions_input": "input.positions", "padding_mask_input": "input.attention_mask",
                 "valid_lengths_input": "input.sequence_lengths", "attention_domain": {
                     "layout": "batch_kv_heads_sequence_feature", "maximum_sequence": 8, "includes_current": True}}
        cache.append(add("kv_cache_append", [source]+extra, (1, kv_heads, 8, 2), attrs, state=state))
        state_outputs.append({"id": cache[-1], "layer": 0, "kind": state, "shape": [1, kv_heads, 8, 2], "maximum_sequence": 8})
    views = []
    for source, state in zip(cache, ("key", "value")):
        attrs = {"axis": 2, "maximum_sequence": 8, "semantics": "visible_valid_prefix",
                 "absolute_write_positions_input": "input.positions", "padding_mask_input": "input.attention_mask",
                 "valid_lengths_input": "input.sequence_lengths"}
        views.append(add("cache_suffix", [source]+extra, (1, kv_heads, 8, 2), attrs, state=state))
    scores = add("attention_scores", [q, views[0]], (1, heads, 8, 8), {"group_size": heads//kv_heads})
    scores = add("attention_scale", [scores], (1, heads, 8, 8), {"head_dim": 2})
    scores = add("causal_mask", [scores, "input.positions"], (1, heads, 8, 8), {})
    probabilities = add("softmax", [scores], (1, heads, 8, 8), {"axis": -1})
    context = add("attention_values", [probabilities, views[1]], (1, heads, 8, 2), {"group_size": heads//kv_heads})
    context = add("reshape", [context], (1, 8, heads*2), {"layout": "batch_sequence_hidden"})
    attention = linear(context, "original-o", 2, heads*2)
    residual = add("residual_add", [hidden, attention], h, {})
    post = norm(residual, "gamma-post", h)
    gate = linear(post, "original-gate", 2, 2)
    up = linear(post, "original-up", 2, 2)
    gate = add("silu", [gate], h, {})
    product = add("multiply", [gate, up], h, {})
    down = linear(product, "original-down", 2, 2)
    residual = add("residual_add", [residual, down], h, {})
    final = norm(residual, "gamma-final", h, None)
    final = add("last_token", [final, "input.sequence_lengths"], (1, 2), {
        "axis": 1, "selection": "last_valid", "valid_lengths_input": "input.sequence_lengths"}, None)
    head = add("output_head", [final], (1, 4), {"weight": "embedding-original", "tied": True,
               "input_layout": "batch_hidden", "output_layout": "batch_vocabulary",
               "compute_dtype": "model_native", "output_dtype": "model_native"}, None)
    selected = add("greedy_token_selection", [head], (1,), {"policy": "pllm.greedy.v1", "source": "execution_policy"}, None)
    feedback = add("token_feedback", [selected], (1, 1), {"policy": "pllm.greedy.v1", "source": "execution_policy"}, None)
    plan = {"schema_version": "pllm.decoder_plan.v1", "model_family": "declared_synthetic_fixture",
            "adapter": "synthetic_operator_fixture", "config_digest": "not-a-factory-proof", "transformations": [],
            "prefill": {"mode": "prefill", "batch": 1, "query_sequence": 8, "maximum_key_sequence": 8,
                        "operations": operations, "state_inputs": [], "state_outputs": state_outputs, "output": feedback},
            "decode": {}, "token_feedback": True}
    return plan, tensors, d


def run(plan, tensors, d, tokens=(0,)*8):
    return runner.run_fixed8(plan, tensors, tokens, dimensions=d, scale_refs={},
                             head_pair=("embedding-original", "head-separate"), numeric_binding=runner.NUMERICS)


def test_whole_synthetic_operator_schedule_full_head_and_request_local_counts():
    plan, tensors, d = fixture()
    before = deepcopy(plan)
    result = run(plan, tensors, d)
    # Independent closed fixture: zero projections preserve embedding; final RMS
    # [1,0] rounds to BF16 [sqrt(2),0], then direct BF16-derived Q8 full head.
    assert result.logits_f32_bits == (0x3fb50000, 0, 0xbfb50000, 0)
    assert result.scope == "synthetic_operator_fixture" and result.source_and_no_wrap_status == "UNPROVED"
    assert len(result.linear_calls) == 57
    assert [call[2] for call in result.linear_calls[:-1]] == list(range(8))*7
    assert result.linear_calls[-1][2:] == (7, 4)
    assert len(result.little_endian_logits()) == 16
    assert plan == before and not hasattr(result, "kv")
    assert run(plan, tensors, d, tokens=(3, 2, 1, 0, 3, 2, 1, 1)).logits_f32_bits == (0, 0x3fb50000, 0, 0xbfb50000)


def test_ids_are_opaque_and_parameter_source_refs_not_derived_from_names():
    plan, tensors, d = fixture()
    operations = plan["prefill"]["operations"]
    names = {op["id"]: f"input.not-reserved-{i}" for i, op in enumerate(operations)}
    for op in operations:
        op["id"] = names[op["id"]]
        op["inputs"] = [names.get(name, name) for name in op["inputs"]]
    plan["prefill"]["output"] = names[plan["prefill"]["output"]]
    for state in plan["prefill"]["state_outputs"]:
        state["id"] = names[state["id"]]
    assert run(plan, tensors, d).logits_f32_bits == (0x3fb50000, 0, 0xbfb50000, 0)


def test_norm_rounds_before_gamma_and_d32_is_ordered_without_fma():
    # Squared norm of [1,0] gives inverse sqrt approximately sqrt2; expected
    # intermediate BF16 exactly181/128, times gamma3 -> R16(543/128)=4.25.
    assert runner.norm((0x3f80, 0), (0x4040, 0x3f80)) == (0x4088, 0)
    assert runner.dot32((0x3f80, 0xbf80, 0x8000), (0x3f80, 0x3f80, 0x3f80)) == 0
    assert runner._op16(0x8000, 0x8000, "add") == 0x8000
    assert runner._op16(0x8000, 0, "add") == 0
    # This gamma makes removal of the pre-gamma BF16 cast observably wrong.
    assert runner.norm((0x3f80, 0), (0x3fe3, 0x3f80)) == (0x4020, 0)
    math = runner.scalar_owner()
    a = runner._op32(0x3f000000, math.round_f32(F(1, 1000000)), "add")
    wrong = runner._r16(runner._op32(runner._promote(0x3fe3), math.rsqrt_f32(a), "mul"))
    assert wrong == 0x4021


def test_causal_dot_counts_and_only_last_final_norm(monkeypatch):
    plan, tensors, d = fixture()
    original = runner.dot32
    lengths = Counter()

    def measured(a, b):
        lengths[len(a)] += 1
        return original(a, b)

    monkeypatch.setattr(runner, "dot32", measured)
    run(plan, tensors, d)
    # 41 norm dots +72 causal scores +32 context dots. Full future scores or
    # final-norm evaluation on all8 rows would change these independent counts.
    assert sum(lengths.values()) == 145
    assert lengths == Counter({2: 117, 1: 4, 3: 4, 4: 4, 5: 4, 6: 4, 7: 4, 8: 4})


def test_nonzero_mlp_uses_actual_stable_silu_backend(monkeypatch):
    plan, tensors, d = fixture()
    for ref in ("original-gate", "original-up", "original-down"):
        tensors[ref] = runner.SourceTensor((2, 2), "synthetic_Q8", (256, 0, 0, 256))
    math = runner.scalar_owner()
    original = math.silu_f32
    calls = []

    def measured(word):
        calls.append(word)
        return original(word)

    monkeypatch.setattr(math, "silu_f32", measured)
    assert run(plan, tensors, d).logits_f32_bits == (0x3fb50000, 0, 0xbfb50000, 0)
    assert len(calls) == 16 and sum(word != 0 for word in calls) == 8


def test_fourfold_grouped_attention_uses_correct_second_kv_head():
    # Independent relational fixture: selecting second KV's negative V must equal
    # a one-KV negative-V fixture, not its opposite-sign first KV.
    a, ta, da = fixture(heads=4, kv_heads=1)
    b, tb, db = fixture(heads=8, kv_heads=2)
    ta["original-v"] = runner.SourceTensor((2, 2), "synthetic_Q8", (-256, 0, 0, -256))
    tb["original-v"] = runner.SourceTensor((4, 2), "synthetic_Q8", (256, 0, 0, 256, -256, 0, 0, -256))
    for tensors, dims, selected_head in ((ta, da, 0), (tb, db, 4)):
        columns = dims.heads*2
        words = [0]*(2*columns)
        words[selected_head*2] = 256
        words[columns+selected_head*2+1] = 256
        tensors["original-o"] = runner.SourceTensor((2, columns), "synthetic_Q8", tuple(words))
    tokens = (0,)*7+(1,)
    reference = run(a, ta, da, tokens).logits_f32_bits
    actual = run(b, tb, db, tokens).logits_f32_bits
    assert actual == reference and actual[0] & 0x80000000
    assert actual != (0, 0x3fb50000, 0, 0xbfb50000)


def test_softmax_bf16_probability_mass_is_not_renormalized():
    probabilities = runner.softmax((0, 0, 0))
    assert probabilities == (0x3eab,)*3
    assert sum(runner.encoding.decode_bf16(w).value for w in probabilities) == F(513, 512)


def test_split_half_rotary_not_interleaved_pairs():
    row = (0x3f80, 0x4000, 0x4040, 0x4080)  # exact1,2,3,4
    # Declared synthetic coefficients isolate the rotation/cast operator itself.
    assert runner.rope(row, ((0,)*4, (0x3f80,)*4)) == (0xc040, 0xc080, 0x3f80, 0x4000)


def test_exact_direct_field_decode_in_linear_not_f32_bridge():
    n = 16842753
    # x=1 has xq256; choose inverse256 modp to construct the exact residue n.
    coefficient = n*pow(256, -1, runner.encoding.BABYBEAR) % runner.encoding.BABYBEAR
    assert runner._linear((coefficient, -coefficient), 2, (0x3f80,)) == (0x4381, 0xc381)


def test_minimal_real_block_import_orientation_and_no_scale_premise_widening():
    source = runner.SourceTensor((128, 128), "F8_E4M3", (0x38,)*16384)
    scale = runner.SourceTensor((1, 1), "BF16", (0x3f80,))
    assert runner._projection(source, scale, False) == (256,)*16384
    with pytest.raises(runner.PlanError):
        runner._projection(runner.SourceTensor((2, 2), "F8_E4M3", (0,)*4), scale, True)
    with pytest.raises(runner.PlanError):
        runner._projection(runner.SourceTensor((1, 1), "synthetic_Q8", (256,)), None, False)


@pytest.mark.parametrize("tokens", [(0,)*7, (0,)*9, (True,)+(0,)*7, (-1,)+(0,)*7, (4,)+(0,)*7])
def test_original_id_domain_never_pads_truncates_or_tokenizes(tokens):
    plan, tensors, d = fixture()
    with pytest.raises(runner.PlanError):
        run(plan, tensors, d, tokens)


@pytest.mark.parametrize("change", ["tie", "dtype", "extra-source", "bias", "links", "profile", "norm-epsilon", "state"])
def test_source_and_numeric_mismatches_reject_not_repair(change):
    plan, tensors, d = fixture()
    ops = plan["prefill"]["operations"]
    if change == "tie":
        tensors["head-separate"] = runner.SourceTensor((4, 2), "BF16", (0,)*8)
    elif change == "dtype":
        tensors["embedding-original"] = runner.SourceTensor((4, 2), "synthetic_Q8", (0,)*8)
    elif change == "extra-source":
        tensors["extra"] = runner.SourceTensor((1,), "BF16", (0,))
    elif change == "bias":
        ops[2]["attributes"]["bias"] = "invented-bias"
    elif change == "links":
        ops[23]["inputs"].reverse()
    elif change == "profile":
        ops[10]["attributes"]["coefficient_profile"] = "unselected-rope"
    elif change == "norm-epsilon":
        ops[1]["attributes"]["epsilon"] = "1/1000"
    else:
        plan["prefill"]["state_inputs"] = [{"external": "persistent-KV"}]
    with pytest.raises(runner.PlanError):
        run(plan, tensors, d)


def test_selected_dimensions_cannot_be_a_tiny_full_model_claim():
    assert runner.selected_dimensions().vocabulary == 151936
    assert runner.selected_dimensions().layers*7*8+1 == 2017  # contract count, NOT execution evidence
    with pytest.raises(runner.PlanError):
        runner.Dimensions(1, 2, 2, 2, 1, 2, 4, 1000000, "selected_fixed8")


def test_binding_is_explicit_and_original_map_name_is_not_reversioned():
    plan, tensors, d = fixture()
    assert runner.NUMERICS == "M-N/Qwen3-4B-8prompt-candidate-1"
    with pytest.raises(runner.PlanError):
        runner.run_fixed8(plan, tensors, (0,)*8, dimensions=d, scale_refs={},
                          head_pair=("embedding-original", "head-separate"), numeric_binding="model_native")


def test_exact_json_epsilon_and_typed_unresolved_never_fallback(monkeypatch):
    assert runner.parse_plan('{"epsilon": 1e-6}')["epsilon"] == F(1, 1000000)
    with pytest.raises(runner.PlanError):
        runner.parse_plan('{"epsilon": NaN}')
    plan, tensors, d = fixture()
    math = runner.scalar_owner()
    original = math.sincos_f32

    def limited(word):
        return original(word, math.Limits(terms=1, dyadic_bits=16))

    monkeypatch.setattr(math, "sincos_f32", limited)
    with pytest.raises(math.Unresolved):
        run(plan, tensors, d)


def test_no_framework_native_or_host_transcendental_imports(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"pllm", "torch", "numpy", "transformers", "mpmath"}:
            raise AssertionError("reference attempted framework/native import")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    plan, tensors, d = fixture()
    assert run(plan, tensors, d).logits_f32_bits[0] == 0x3fb50000
    serialized = json.dumps({"epsilon": "1e-6"})
    assert runner.parse_plan(serialized) == {"epsilon": "1e-6"}
